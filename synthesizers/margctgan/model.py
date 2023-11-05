"""MargCTGAN module. Backbone model is CTGAN. Code moderately adapted. Refer to https://github.com/sdv-dev/CTGAN"""

import warnings

import numpy as np
import pandas as pd
import torch
from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state
from packaging import version
from torch import optim
from torch.nn import (
    BatchNorm1d,
    Dropout,
    LeakyReLU,
    Linear,
    Module,
    ReLU,
    Sequential,
    functional,
)

from utils.logger import get_logger
from utils.pca import PCA
from utils.orthorgonal_vectors import random_orthogonal_matrix

# Setup logger.
LOGGER = get_logger(__name__)


class Discriminator(Module):
    """Discriminator for the CTGAN."""

    def __init__(self, input_dim, discriminator_dim, pac=10):
        super(Discriminator, self).__init__()
        dim = input_dim * pac
        self.pac = pac
        self.pacdim = dim
        seq = []
        for item in list(discriminator_dim):
            seq += [Linear(dim, item), LeakyReLU(0.2), Dropout(0.5)]
            dim = item

        seq += [Linear(dim, 1)]
        self.seq = Sequential(*seq)

    def calc_gradient_penalty(self, real, fake, device="cpu", pac=10, lambda_=10):
        """Compute the gradient penalty."""
        alpha = torch.rand(real.size(0) // pac, 1, 1, device=device)
        alpha = alpha.repeat(1, pac, real.size(1))
        alpha = alpha.view(-1, real.size(1))

        interpolates = alpha * real + ((1 - alpha) * fake)

        disc_interpolates = self(interpolates)

        gradients = torch.autograd.grad(
            outputs=disc_interpolates,
            inputs=interpolates,
            grad_outputs=torch.ones(disc_interpolates.size(), device=device),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]

        gradients_view = gradients.view(-1, pac * real.size(1)).norm(2, dim=1) - 1
        gradient_penalty = ((gradients_view) ** 2).mean() * lambda_

        return gradient_penalty

    def forward(self, input_):
        """Apply the Discriminator to the `input_`."""
        assert input_.size()[0] % self.pac == 0
        return self.seq(input_.view(-1, self.pacdim))


class Residual(Module):
    """Residual layer for the CTGAN."""

    def __init__(self, i, o):
        super(Residual, self).__init__()
        self.fc = Linear(i, o)
        self.bn = BatchNorm1d(o)
        self.relu = ReLU()

    def forward(self, input_):
        """Apply the Residual layer to the `input_`."""
        out = self.fc(input_)
        out = self.bn(out)
        out = self.relu(out)
        return torch.cat([out, input_], dim=1)


class Generator(Module):
    """Generator for the CTGAN."""

    def __init__(self, embedding_dim, generator_dim, data_dim):
        super(Generator, self).__init__()
        dim = embedding_dim
        seq = []
        for item in list(generator_dim):
            seq += [Residual(dim, item)]
            dim += item
        seq.append(Linear(dim, data_dim))
        self.seq = Sequential(*seq)

    def forward(self, input_):
        """Apply the Generator to the `input_`."""
        data = self.seq(input_)
        return data


class MargCTGAN(BaseSynthesizer):
    """Conditional Table GAN Synthesizer.
    This is the core class of the CTGAN project, where the different components
    are orchestrated together.
    For more details about the process, please check the [Modeling Tabular data using
    Conditional GAN](https://arxiv.org/abs/1907.00503) paper.
    Args:
        embedding_dim (int):
            Size of the random sample passed to the Generator. Defaults to 128.
        generator_dim (tuple or list of ints):
            Size of the output samples for each one of the Residuals. A Residual Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        discriminator_dim (tuple or list of ints):
            Size of the output samples for each one of the Discriminator Layers. A Linear Layer
            will be created for each one of the values provided. Defaults to (256, 256).
        generator_lr (float):
            Learning rate for the generator. Defaults to 2e-4.
        generator_decay (float):
            Generator weight decay for the Adam Optimizer. Defaults to 1e-6.
        discriminator_lr (float):
            Learning rate for the discriminator. Defaults to 2e-4.
        discriminator_decay (float):
            Discriminator weight decay for the Adam Optimizer. Defaults to 1e-6.
        batch_size (int):
            Number of data samples to process in each step.
        discriminator_steps (int):
            Number of discriminator updates to do for each generator update.
            From the WGAN paper: https://arxiv.org/abs/1701.07875. WGAN paper
            default is 5. Default used is 1 to match original CTGAN implementation.
        log_frequency (boolean):
            Whether to use log frequency of categorical levels in conditional
            sampling. Defaults to ``True``.
        verbose (boolean):
            Whether to have print statements for progress results. Defaults to ``False``.
        epochs (int):
            Number of training epochs. Defaults to 300.
        pac (int):
            Number of samples to group together when applying the discriminator.
            Defaults to 10.
        cuda (bool):
            Whether to attempt to use cuda for GPU computation.
            If this is False or CUDA is not available, CPU will be used.
            Defaults to ``True``.
    """

    def __init__(
        self,
        embedding_dim=128,
        generator_dim=(256, 256),
        discriminator_dim=(256, 256),
        generator_lr=2e-4,
        generator_decay=1e-6,
        discriminator_lr=2e-4,
        discriminator_decay=1e-6,
        batch_size=500,
        discriminator_steps=1,
        log_frequency=True,
        verbose=False,
        epochs=300,
        pac=10,
        cuda=True,
        condition_vector=True,
        variant="pca",
        stats="stddev",
        n_components=-1,
        save_dir="./",
    ):
        assert batch_size % 2 == 0

        self._embedding_dim = embedding_dim
        self._generator_dim = generator_dim
        self._discriminator_dim = discriminator_dim

        self._generator_lr = generator_lr
        self._generator_decay = generator_decay
        self._discriminator_lr = discriminator_lr
        self._discriminator_decay = discriminator_decay

        self._batch_size = batch_size
        self._discriminator_steps = discriminator_steps
        self._log_frequency = log_frequency
        self._verbose = verbose
        self._epochs = epochs
        self.pac = pac

        if not cuda or not torch.cuda.is_available():
            device = "cpu"

        elif isinstance(cuda, str):
            device = cuda
        else:
            device = "cuda"

        self._device = torch.device(device)

        self._transformer = None
        self._data_sampler = None
        self._generator = None

        self.condition_vector = condition_vector
        self.n_components = n_components
        self.stats = stats
        self.variant = variant
        self.save_dir = save_dir
        self._pca = None

        # Params from MargCTGAN.
        # self.gen_loss_type = extra_param_dict["gen_loss_type"]
        # self.feature_type = extra_param_dict["feature_type"]
        # self.condition_model = extra_param_dict["condition_model"]
        # self.marg_stats = extra_param_dict["marg_stats"]
        # self.save_dir = extra_param_dict["save_dir"]

    @staticmethod
    def _gumbel_softmax(logits, tau=1, hard=False, eps=1e-10, dim=-1):
        """Deals with the instability of the gumbel_softmax for older versions of torch.
        For more details about the issue:
        https://drive.google.com/file/d/1AA5wPfZ1kquaRtVruCd6BiYZGcDeNxyP/view?usp=sharing
        Args:
            logits […, num_features]:
                Unnormalized log probabilities
            tau:
                Non-negative scalar temperature
            hard (bool):
                If True, the returned samples will be discretized as one-hot vectors,
                but will be differentiated as if it is the soft sample in autograd
            dim (int):
                A dimension along which softmax will be computed. Default: -1.
        Returns:
            Sampled tensor of same shape as logits from the Gumbel-Softmax distribution.
        """
        if version.parse(torch.__version__) < version.parse("1.2.0"):
            for i in range(10):
                transformed = functional.gumbel_softmax(
                    logits, tau=tau, hard=hard, eps=eps, dim=dim
                )
                if not torch.isnan(transformed).any():
                    return transformed
            raise ValueError("gumbel_softmax returning NaN.")

        return functional.gumbel_softmax(logits, tau=tau, hard=hard, eps=eps, dim=dim)

    def _apply_activate(self, data):
        """Apply proper activation function to the output of the generator."""
        data_t = []
        st = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if span_info.activation_fn == "tanh":
                    ed = st + span_info.dim
                    data_t.append(torch.tanh(data[:, st:ed]))
                    st = ed
                elif span_info.activation_fn == "softmax":
                    ed = st + span_info.dim
                    transformed = self._gumbel_softmax(data[:, st:ed], tau=0.2)
                    data_t.append(transformed)
                    st = ed
                else:
                    raise ValueError(
                        f"Unexpected activation function {span_info.activation_fn}."
                    )

        return torch.cat(data_t, dim=1)

    def _cond_loss(self, data, c, m):
        """Compute the cross entropy loss on the fixed discrete column."""
        loss = []
        st = 0
        st_c = 0
        for column_info in self._transformer.output_info_list:
            for span_info in column_info:
                if len(column_info) != 1 or span_info.activation_fn != "softmax":
                    # not discrete column
                    st += span_info.dim
                else:
                    ed = st + span_info.dim
                    ed_c = st_c + span_info.dim
                    tmp = functional.cross_entropy(
                        data[:, st:ed],
                        torch.argmax(c[:, st_c:ed_c], dim=1),
                        reduction="none",
                    )
                    loss.append(tmp)
                    st = ed
                    st_c = ed_c

        loss = torch.stack(loss, dim=1)  # noqa: PD013

        return (loss * m).sum() / data.size()[0]

    def _validate_discrete_columns(self, data, discrete_columns):
        """Check whether ``discrete_columns`` exists in ``data``.
        Args:
            data (numpy.ndarray or pandas.DataFrame):
                 Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        if isinstance(data, pd.DataFrame):
            invalid_columns = set(discrete_columns) - set(data.columns)
        elif isinstance(data, np.ndarray):
            invalid_columns = []
            for column in discrete_columns:
                if column < 0 or column >= data.shape[1]:
                    invalid_columns.append(column)
        else:
            raise TypeError("``data`` should be either pd.DataFrame or np.array.")

        if invalid_columns:
            raise ValueError(f"Invalid columns found: {invalid_columns}")

    @random_state
    def fit(self, data, discrete_columns=()):
        """Fit the MargCTGAN Synthesizer model to the training data.
        Args:
            data (numpy.ndarray or pandas.DataFrame):
                Data. It must be a 2-dimensional numpy array or a pandas.DataFrame.
            discrete_columns (list-like):
                List of discrete columns to be used to generate the Conditional
                Vector. If ``data`` is a Numpy array, this list should
                contain the integer indices of the columns. Otherwise, if it is
                a ``pandas.DataFrame``, this list should contain the column names.
        """
        self._validate_discrete_columns(data, discrete_columns)

        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_columns)

        data_ = self._transformer.transform(data)

        self._data_sampler = DataSampler(
            data_, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

        if self.condition_vector:
            self._generator = Generator(
                self._embedding_dim + self._data_sampler.dim_cond_vec(),
                self._generator_dim,
                data_dim,
            ).to(self._device)

            self._discriminator = Discriminator(
                data_dim + self._data_sampler.dim_cond_vec(),
                self._discriminator_dim,
                pac=self.pac,
            ).to(self._device)

        else:

            self._generator = Generator(
                self._embedding_dim,
                self._generator_dim,
                data_dim,
            ).to(self._device)

            self._discriminator = Discriminator(
                data_dim,
                self._discriminator_dim,
                pac=self.pac,
            ).to(self._device)

        # Variants
        if self.variant == "pca":
            if self.n_components == -1:
                self.n_components = None
            self._pca = PCA(n_components=self.n_components, device=self._device)
            self._pca.set_random_state(self.random_states)
            self._pca.fit(torch.from_numpy(data_.astype("float32")))

            self.n_components = self._pca.n_components

            LOGGER.info(f"{self.variant} feature size: {self.n_components}")

        elif self.variant == "fixed_random_matrix":
            n = data_.shape[1]
            if self.n_components == -1:
                self.n_components = n

            self.fixed_random_matrix = torch.randn(n, self.n_components)

            LOGGER.info(f"{self.variant} feature size: {self.n_components}")

        elif (
            self.variant == "random_matrix"
            or self.variant == "random_orthogonal_matrix"
        ):

            if self.n_components == -1:
                self.n_components = data_.shape[1]

            LOGGER.info(f"{self.variant} feature size: {self.n_components}")

        elif self.variant == "raw":
            LOGGER.info(f"{self.variant} feature size: {data_.shape[1]}")

        elif self.variant == "fixed_random_orthogonal_matrix":
            n = data_.shape[1]
            if self.n_components == -1:
                self.n_components = n

            fixed_random_ortho_matrix = random_orthogonal_matrix(
                n=n, k=self.n_components
            )
            self.fixed_random_ortho_matrix = torch.from_numpy(
                fixed_random_ortho_matrix.astype("float32")
            )
            LOGGER.info(f"{self.variant} feature size: {self.n_components}")

        else:
            self.variant = None

        optimizerG = optim.Adam(
            self._generator.parameters(),
            lr=self._generator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._generator_decay,
        )

        optimizerD = optim.Adam(
            self._discriminator.parameters(),
            lr=self._discriminator_lr,
            betas=(0.5, 0.9),
            weight_decay=self._discriminator_decay,
        )

        mean = torch.zeros(self._batch_size, self._embedding_dim, device=self._device)
        std = mean + 1

        self.loss_values = pd.DataFrame(
            columns=[
                "Epoch",
                "Generator Loss",
                "Discriminator Loss",
                "Adversarial Loss",
                "Condition Loss",
                "Marginal Loss",
            ]
        )

        steps_per_epoch = max(len(data_) // self._batch_size, 1)

        # Initialize the Losses with zero.
        loss_adv = torch.tensor(0)
        loss_cond = torch.tensor(0)
        loss_marg = torch.tensor(0)

        for i in range(self._epochs):
            for id_ in range(steps_per_epoch):
                for n in range(self._discriminator_steps):
                    # Update the Discriminator
                    fakez = torch.normal(mean=mean, std=std)

                    if self.condition_vector:
                        condvec = self._data_sampler.sample_condvec(self._batch_size)
                        c1, m1, col, opt = condvec

                        c1 = torch.from_numpy(c1).to(self._device)
                        m1 = torch.from_numpy(m1).to(self._device)

                        perm = np.arange(self._batch_size)
                        np.random.shuffle(perm)

                        c2, _, col2, opt2 = c1[perm], m1[perm], col[perm], opt[perm]

                        real = self._data_sampler.sample_data(
                            self._batch_size, col2, opt2
                        )
                        real = torch.from_numpy(real.astype("float32")).to(self._device)
                        real_cat = torch.cat([real, c2], dim=1)

                        fakez = torch.cat([fakez, c1], dim=1)
                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)
                        fake_cat = torch.cat([fakeact, c1], dim=1)

                    else:
                        real = self._data_sampler.sample_data(
                            self._batch_size, None, None
                        )
                        real = torch.from_numpy(real.astype("float32")).to(self._device)
                        real_cat = real

                        fake = self._generator(fakez)
                        fakeact = self._apply_activate(fake)
                        fake_cat = fakeact

                    y_fake = self._discriminator(fake_cat)
                    y_real = self._discriminator(real_cat)

                    pen = self._discriminator.calc_gradient_penalty(
                        real_cat, fake_cat, self._device, self.pac
                    )
                    loss_d = -(torch.mean(y_real) - torch.mean(y_fake))

                    optimizerD.zero_grad()
                    pen.backward(retain_graph=True)
                    loss_d.backward(retain_graph=True)

                    optimizerD.step()

                # Update the Generator
                fakez = torch.normal(mean=mean, std=std)
                if self.condition_vector:
                    condvec = self._data_sampler.sample_condvec(self._batch_size)
                    c, m, col, opt = condvec

                    c = torch.from_numpy(c).to(self._device)
                    m = torch.from_numpy(m).to(self._device)

                    fakez = torch.cat([fakez, c], dim=1)

                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    y_fake = self._discriminator(torch.cat([fakeact, c], dim=1))
                    loss_adv = -torch.mean(y_fake)

                    loss_cond = self._cond_loss(fake, c, m)

                else:
                    fake = self._generator(fakez)
                    fakeact = self._apply_activate(fake)

                    y_fake = self._discriminator(fakeact)
                    loss_adv = -torch.mean(y_fake)

                    loss_cond = torch.tensor(0)

                if self.variant is not None:
                    if self.condition_vector:
                        real = self._data_sampler.sample_data(
                            self._batch_size, col, opt
                        )
                        real = torch.from_numpy(real.astype("float32")).to(self._device)
                    else:
                        real = self._data_sampler.sample_data(
                            self._batch_size, None, None
                        )
                        real = torch.from_numpy(real.astype("float32")).to(self._device)

                    loss_mean = torch.tensor(0)
                    loss_std = torch.tensor(0)

                    real_bar, fake_bar = self.extract_feature(
                        real, fakeact, variant=self.variant
                    )

                    if self.stats == "mean":
                        loss_mean = self._mean_norm(real_bar, fake_bar, p=2)

                    if self.stats == "stddev":
                        loss_std = self._stddev_norm(real_bar, fake_bar, p=2)

                    if self.stats == "mean_and_stddev":
                        loss_mean = self._mean_norm(real_bar, fake_bar, p=2)
                        loss_std = self._stddev_norm(real_bar, fake_bar, p=2)

                    loss_marg = loss_mean + loss_std

                else:

                    loss_marg = torch.tensor(0)

                loss_g = loss_adv + loss_cond + loss_marg

                optimizerG.zero_grad()
                loss_g.backward()
                optimizerG.step()

            discriminator_loss = loss_d.detach().cpu().item()
            generator_loss = loss_g.detach().cpu().item()
            adversarial_loss = loss_adv.detach().cpu().item()
            condition_loss = loss_cond.detach().cpu().item()
            marginal_loss = loss_marg.detach().cpu().item()

            epoch_loss_df = pd.DataFrame(
                {
                    "Epoch": [i + 1],
                    "Discriminator Loss": [discriminator_loss],
                    "Generator Loss": [generator_loss],
                    "Adversarial Loss": [adversarial_loss],
                    "Condition Loss": [condition_loss],
                    "Marginal Loss": [marginal_loss],
                }
            )
            if not self.loss_values.empty:
                self.loss_values = pd.concat(
                    [self.loss_values, epoch_loss_df]
                ).reset_index(drop=True)
            else:
                self.loss_values = epoch_loss_df

            self.loss_values.to_csv(self.save_dir + "/log.csv", index=None)

        # Plot Loss
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, axs = plt.subplots(1, 4, figsize=(16, 3))
        df = self.loss_values
        g = sns.lineplot(
            df, x="Epoch", y="Discriminator Loss", ax=axs[0], c="C0", label="disc"
        )
        sns.lineplot(df, x="Epoch", y="Generator Loss", ax=g, c="C1", label="gen")
        sns.lineplot(
            df, x="Epoch", y="Adversarial Loss", ax=axs[1], c="C2", label="adv"
        )
        sns.lineplot(df, x="Epoch", y="Condition Loss", ax=axs[2], c="C3", label="cond")
        sns.lineplot(df, x="Epoch", y="Marginal Loss", ax=axs[3], c="C4", label="marg")
        plt.subplots_adjust()
        fig.tight_layout()
        fig.savefig(
            f"{self.save_dir}/loss.png",
            dpi=500,
            bbox_inches="tight",
        )

    def extract_feature(self, real, fakeact, variant):

        if variant == "raw":
            real_bar = real.clone()
            fake_bar = fakeact.clone()

        elif variant == "pca":
            real_bar = self._pca.transform(real)
            fake_bar = self._pca.transform(fakeact)

        elif variant == "random_matrix":
            n = real.shape[1]
            random_matrix = torch.randn(n, self.n_components)
            real_bar = torch.matmul(real, random_matrix)
            fake_bar = torch.matmul(fakeact, random_matrix)

        elif variant == "fixed_random_matrix":
            real_bar = torch.matmul(real, self.fixed_random_matrix)
            fake_bar = torch.matmul(fakeact, self.fixed_random_matrix)

        elif variant == "fixed_random_orthogonal_matrix":
            real_bar = torch.matmul(real, self.random_ortho_matrix)
            fake_bar = torch.matmul(fakeact, self.random_ortho_matrix)

        elif variant == "random_orthogonal_matrix":
            n = real.shape[1]
            random_ortho_matrix = random_orthogonal_matrix(n=n, k=self.n_components)
            random_ortho_matrix = torch.from_numpy(
                random_ortho_matrix.astype("float32")
            )
            real_bar = torch.matmul(real, random_ortho_matrix)
            fake_bar = torch.matmul(fakeact, random_ortho_matrix)

        return real_bar, fake_bar

    def _mean_norm(self, real_bar, fake_bar, p):
        real_bar_mean = real_bar.mean(dim=0)
        fake_bar_mean = fake_bar.mean(dim=0)

        loss_mean = torch.norm(real_bar_mean - fake_bar_mean, p=p)
        return loss_mean

    def _stddev_norm(self, real_bar, fake_bar, p):
        real_bar_std = real_bar.std(dim=0)
        fake_bar_std = fake_bar.std(dim=0)

        loss_std = torch.norm(real_bar_std - fake_bar_std, p=p)
        return loss_std

    @random_state
    def sample(self, n, condition_column=None, condition_value=None):
        """Sample data similar to the training data.
        Choosing a condition_column and condition_value will increase the probability of the
        discrete condition_value happening in the condition_column.
        Args:
            n (int):
                Number of rows to sample.
            condition_column (string):
                Name of a discrete column.
            condition_value (string):
                Name of the category in the condition_column which we wish to increase the
                probability of happening.
        Returns:
            numpy.ndarray or pandas.DataFrame
        """
        if condition_column is not None and condition_value is not None:
            condition_info = self._transformer.convert_column_name_value_to_id(
                condition_column, condition_value
            )
            global_condition_vec = (
                self._data_sampler.generate_cond_from_condition_column_info(
                    condition_info, self._batch_size
                )
            )
        else:
            global_condition_vec = None

        steps = n // self._batch_size + 1
        data = []
        for i in range(steps):
            mean = torch.zeros(self._batch_size, self._embedding_dim)
            std = mean + 1
            fakez = torch.normal(mean=mean, std=std).to(self._device)

            if global_condition_vec is not None:
                condvec = global_condition_vec.copy()
            else:
                condvec = self._data_sampler.sample_original_condvec(self._batch_size)

            if condvec is None or not self.condition_vector:
                pass
            else:
                c1 = condvec
                c1 = torch.from_numpy(c1).to(self._device)
                fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            fakeact = self._apply_activate(fake)
            data.append(fakeact.detach().cpu().numpy())

        data = np.concatenate(data, axis=0)
        data = data[:n]

        return self._transformer.inverse_transform(data)

    def set_device(self, device):
        """Set the `device` to be used ('GPU' or 'CPU)."""
        self._device = device
        if self._generator is not None:
            self._generator.to(self._device)
