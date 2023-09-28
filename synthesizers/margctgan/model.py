"""MargCTGAN module. Backbone model is CTGAN. Code moderately adapted. Refer to https://github.com/sdv-dev/CTGAN"""

import warnings

import numpy as np
import pandas as pd
import torch
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

from ctgan.data_sampler import DataSampler
from ctgan.data_transformer import DataTransformer
from ctgan.synthesizers.base import BaseSynthesizer, random_state

import sys
sys.path.append("../..")
from utils.pca import PCA


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
        extra_param_dict=None,
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

        self.loss_type = extra_param_dict["loss_type"]
        self.loss_weight = extra_param_dict["loss_weight"]
        self.weight_scheme = extra_param_dict["weight_scheme"]
        self.variant = extra_param_dict["variant"]

        if self.variant == "pca":
            n_components = extra_param_dict.get("pca_components", None)
            if n_components == -1:
                n_components = None
            self._pca = PCA(n_components=n_components, device=self._device)

        self.real = None
        self.fake = None

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
    def fit(self, data, discrete_columns=(), epochs=None):
        """Fit the CTGAN Synthesizer models to the training data.
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

        if epochs is None:
            epochs = self._epochs
        else:
            warnings.warn(
                (
                    "`epochs` argument in `fit` method has been deprecated and will be removed "
                    "in a future version. Please pass `epochs` to the constructor instead"
                ),
                DeprecationWarning,
            )

        self._transformer = DataTransformer()
        self._transformer.fit(data, discrete_columns)

        data_ = self._transformer.transform(data)

        self._data_sampler = DataSampler(
            data_, self._transformer.output_info_list, self._log_frequency
        )

        data_dim = self._transformer.output_dimensions

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

        if self.variant == "pca":
            self._pca.set_random_state(self.random_states)
            self._pca.fit(torch.from_numpy(data_.astype("float32")))

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

        steps_per_epoch = max(len(data_) // self._batch_size, 1)

        for i in range(epochs):
            for id_ in range(steps_per_epoch):
                loss_d = self._discriminator_train(optimizerD, mean, std)
                loss_g, loss_adv, loss_cond, loss_marg = self._generator_train(
                    optimizerG, mean, std
                )

    def _generator_train(self, optimizerG, mean, std):
        fakez = torch.normal(mean=mean, std=std)
        condvec = self._data_sampler.sample_condvec(self._batch_size)

        if condvec is not None:
            c1, m1, _, _ = self._cond_and_mask(condvec)
            fakez = torch.cat([fakez, c1], dim=1)

            fake = self._generator(fakez)
            cross_entropy = self._cond_loss(fake, c1, m1)

            fakeact = self._apply_activate(fake)
            y_fake = self._discriminator(torch.cat([fakeact, c1], dim=1))

        else:
            fake = self._generator(fakez)
            cross_entropy = torch.tensor(0)

            fakeact = self._apply_activate(fake)
            y_fake = self._discriminator(fakeact)

        loss_adv = -torch.mean(y_fake)

        # Default ctgan loss.
        loss_g = loss_adv + cross_entropy

        # Feature matching loss
        if self.loss_type == "mean_and_stddev":
            loss_marg = self._mean_and_stddev_matching_loss()
        else:
            raise ValueError(f"Loss type={self.loss_type} not supported!")

        if self.weight_scheme == "linear":
            loss_g = ((1.0 - self.loss_weight) * loss_g) + (
                self.loss_weight * loss_marg
            )

        elif self.weight_scheme == "weighted":
            loss_g = loss_g + (self.loss_weight * loss_marg)

        else:
            raise NotImplementedError(
                f"Weight scheme = {self.weight_scheme} is not recognized."
            )

        optimizerG.zero_grad()
        loss_g.backward()
        optimizerG.step()

        return loss_g, loss_adv, cross_entropy, loss_marg

    def _mean_and_stddev_matching_loss(self):
        if self.variant == "pca":            
            real = self._pca.transform(self.real)
            fake = self._pca.transform(self.fake)
        elif self.variant == "marg":
            real = self.real
            fake = self.fake
        else:
            raise NotImplementedError(f"{self.variant} not recognized!")

        real_mean = real.mean(dim=0)
        fake_mean = fake.mean(dim=0)

        real_std = real.std(dim=0)
        fake_std = fake.std(dim=0)

        loss_mean = torch.norm(real_mean - fake_mean, p=2)
        loss_std = torch.norm(real_std - fake_std, p=2)
        
        loss = loss_mean + loss_std
        return loss

    def _cond_and_mask(self, condvec):
        c1, m1, col, opt = condvec

        c1 = torch.from_numpy(c1).to(self._device)
        m1 = torch.from_numpy(m1).to(self._device)

        return c1, m1, col, opt

    def _discriminator_train(self, optimizerD, mean, std):
        for n in range(self._discriminator_steps):
            fakez = torch.normal(mean=mean, std=std)

            condvec = self._data_sampler.sample_condvec(self._batch_size)

            if condvec is not None:
                c1, m1, col, opt = self._cond_and_mask(condvec)

                perm = np.arange(self._batch_size)
                np.random.shuffle(perm)

                c2, _, col2, opt2 = c1[perm], m1[perm], col[perm], opt[perm]

                real = self._data_sampler.sample_data(self._batch_size, col2, opt2)
                real = torch.from_numpy(real.astype("float32")).to(self._device)
                real_cat = torch.cat([real, c2], dim=1)

                fakez = torch.cat([fakez, c1], dim=1)
                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = torch.cat([fakeact, c1], dim=1)

            else:
                real = self._data_sampler.sample_data(self._batch_size, None, None)
                real = torch.from_numpy(real.astype("float32")).to(self._device)
                real_cat = real

                fake = self._generator(fakez)
                fakeact = self._apply_activate(fake)
                fake_cat = fakeact

            self.real = real
            self.fake = fakeact

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

        return loss_d

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

            if condvec is None:
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
