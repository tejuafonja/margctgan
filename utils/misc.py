import os
import random
import sys
import argparse

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

sys.path.append("..")

import torch
from torch.nn import Linear, LogSoftmax, Module, ReLU, Sequential
from torch.utils.data import TensorDataset

from .params import Params
from .transformer import DataTransformer
from .dataset import Dataset

__all__ = [
    "reproducibility",
    "mkdir",
    "savefig",
    "str2bool",
    "write_csv",
    "geometric_sequence",
    "arithmetic_sequence",
    "split_feature_target",
    "duplicate_data",
    "filter_duplicates",
    "get_duplicates",
    "discrete_column_wise_relationship",
    "plot_hist",
    "get_real_data",
    "load_ctgan_model",
    "get_column_names",
    "sample_fake_data",
    "trim_axs",
    "sturge_binning",
    "freedman_diaconis_binning",
]


def reproducibility(seed, use_cuda=False):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if use_cuda:
        torch.cuda.manual_seed_all(seed)


def mkdir(path):
    """Make dir if not exist"""
    os.makedirs(path, exist_ok=True)


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def write_csv(file_path, exp_name, data, index_names):
    if os.path.exists(file_path):
        df = pd.read_csv(file_path, index_col=0)
        df[exp_name] = data
    else:
        df = pd.DataFrame(data, columns=[exp_name], index=index_names)
    df.to_csv(file_path)

def geometric_sequence(start_value=40, common_ratio=2, size=9, reverse=True):
    """Sequence of non-zero numbers which is a multiple of its common ratio."""
    sequence = [int(start_value * (common_ratio**i)) for i in range(size + 1)]
    if reverse:
        sequence = sequence[::-1]
    return sequence


def arithmetic_sequence(start_value=500, common_difference=2000, size=10):
    """Sequence such that the difference between the consecutive terms is constant."""
    sequence = [start_value + (i * common_difference) for i in range(size + 1)]
    return sequence

class DotDict(dict):
    def __getattr__(self, key):
        return self[key]


def split_feature_target(data, target_name):
    """Splits the data into features and target vectors"""
    data_df = data.copy()
    target_data = data_df[[target_name]]
    data_df.pop(target_name)
    return data_df, target_data


def duplicate_data(data, n, seed=1000):
    """
    Args:
        data (pd.DataFrame):
            data to be duplicated.
        n (int):
            new size of the data.
    """
    np.random.seed(seed)
    if n < len(data):
        duplicated = data.sample(n)
    else:
        steps = n // len(data) + 1
        dup = []
        for i in range(steps):
            dup.append(data)
        duplicated = pd.concat(dup).sample(n)

    duplicated.reset_index(inplace=True, drop=True)
    return duplicated


def filter_duplicates(data):
    assert isinstance(data, np.ndarray) or isinstance(
        data, pd.DataFrame
    ), "Unrecognized datatype"

    if isinstance(data, np.ndarray):
        _, index = np.unique(data, axis=0, return_index=True)
        filtered_data = data[index]
    else:
        filtered_data = data[~data.duplicated()]
        filtered_data.reset_index(inplace=True, drop=True)

    return filtered_data


def get_duplicates(data: pd.DataFrame):
    """
    Returns:
        DataFrame with duplicated rows.
    """
    duplicates = data[data.duplicated()]
    return duplicates


def discrete_column_wise_relationship(
    data, column_A, column_B, value, plot_bar=False, title=None
):
    """
    Args:
        data (pd.DataFrame):
            Data
        col_A (str):
            First column to filter data by.
            Column must exist in data
        col_B (str):
            Second column to filter data that is
            already filtered by col_A.
            Column must exist in data
        value (str):
            Filter value
    Returns:
        Value counts if plot_bar is False
    """

    series = data[data[column_A] == value]
    if plot_bar:
        ax = series[column_B].value_counts().plot(kind="bar")
        ax.set_ylabel("Count")
        if title is not None:
            ax.set_title(title)
        return ax
    else:
        return series[column_B].value_counts()


def plot_hist(data, column, title=None):
    """
    Args:
        data (pd.DataFrame):
            Data
        column (str):
            Column to plot.
            Must be name of a continuous column present
            in data.

    Returns:
        Histogram plot with relative frequency
    """
    series = data[column]
    weights = np.zeros_like(series.values) + 1.0 / series.size
    ax = series.plot(kind="hist", weights=weights)
    ax.set_ylabel("Relative Frequency")
    ax.set_xlabel(column.capitalize())

    if title is not None:
        ax.set_title(title)

    return ax


def get_real_data(dataset_name, dataset_subset, test_size=None, seed=1000):
    """Helper function to get data subset.

    Args:
        dataset_name (str):
            Name of the dataset.
        dataset_subset (str):
            Subset of the dataset. Must be any of (``train``, ``test``, NoneType).
        test_size (int):
            Size of the data to return. Defaults to None.
        Seed (int):
            Seed for reproducibility.

    Returns:
        A tuple of (train_data, target_name, discrete_columns, numerical_columns)
    """
    train_data, target_name, discrete_columns, numerical_columns = dataset.load(
        dataset_name=dataset_name, subset=dataset_subset
    )

    if test_size is not None and test_size < len(train_data):
        _, train_data = train_test_split(
            train_data,
            test_size=test_size,
            stratify=train_data[target_name],
            random_state=seed,
        )
    return train_data, target_name, discrete_columns, numerical_columns


def load_ctgan_model(args, filepath, train_data, discrete_columns):
    """Loads trained ctgan model in filepath.

    Args:
        args (json):
            model arguments.
        filepath (str):
            filepath with saved models.
        discrete_columns (Union(List, Tuple)):
            List or tuples containing discrete column names.

    Returns:
        Trained ctgan model.
    """
    from fakeTable.synthesizers.ctgan import CTGANSynthesizer

    args.path_to_discriminator = f"{filepath}/disc_best.pk"
    args.path_to_generator = f"{filepath}/gen_best.pk"
    args.epochs = -1
    model = CTGANSynthesizer(args=args)
    model.fit(train_data, discrete_columns=discrete_columns)

    return model


def get_column_names(transformer):
    """Function extract column names from data transformer."""
    column_names = []
    for column_info in transformer._column_transform_info_list:
        if hasattr(column_info.transform, "categories_"):
            column = column_info.column_name
            categories = column_info.transform.categories_[0]
            columns = [f"{column}_{i}" for i in categories]
            column_names.extend(columns)
        elif hasattr(column_info.transform, "n_components"):
            column = column_info.column_name
            dim = column_info.output_dimensions
            columns = [f"{column}_{i}" for i in range(dim)]
            column_names.extend(columns)
        else:
            column_names.append(column_info.column_name)
    return column_names


def feature_importance(model, column_names):
    """Returns pandas dataframe with two columns ``cols` and ``imp``.
    cols: column names.
    imp: important features from an interpretable classifier
        with ``feature_importances_`` attribute.
    """
    df = pd.DataFrame(
        {"cols": column_names, "imp": model.feature_importances_}
    ).sort_values("imp", ascending=False)
    return df


def sample_fake_data(
    subset,
    filepath,
    dataset_name="adult",
    dataset_subset="train",
    seed=1000,
    size=20000,
    noise_param=None,
    is_duplicate=False,
):
    """Samples fake data.

    Args:
        subset (int):
            Train subset to sample from.
        filepath (str):
            Model (fake) args filepath.
        dataset_name (str):
            Name of the dataset. Defaults to ``adult``.
        dataset_subset (str):
            Subset of the dataset to use. Defaults to ``train``.
        seed (int):
            Reproducibility seed used to train the model. Defaults to 1000.
        size (int):
            Size of fake data to sample. Defaults to 20000.
        noise_param (tuple):
            Tuple of (noise_args_filepath, noise_scale). Defaults to None.
        is_duplicate (bool):
            Whether the model uses duplicate dataset or not. Defaults to False.

    Returns:
        A tuple of (fake_data, train_subset)
    """

    args = Params(f"{filepath}/args.json")
    train_subset, _, discrete_columns, _ = get_real_data(
        dataset_name=dataset_name,
        dataset_subset=dataset_subset,
        test_size=subset,
        seed=seed,
    )
    if noise_param is not None:
        from fakeTable.synthesizers.noise import NumericalNoiseSynthesizer

        noise_args_filepath, noise_scale = noise_param
        noise_args = Params(noise_args_filepath)
        noise_args.seed = seed
        noise_args.include_real = True
        noise_model = NumericalNoiseSynthesizer(noise_args)
        noise_model.fit(
            train_subset,
            discrete_columns=discrete_columns,
            noise_threshold=noise_scale,
        )
        # train_subset = noise_model.sample(args.train_size)
        train_subset = noise_model.sample(20480 + len(train_subset))

    if is_duplicate:
        train_subset = duplicate_data(train_subset, n=args.train_size, seed=seed)

    ctgan = load_ctgan_model(args, filepath, train_subset, discrete_columns)

    np.random.seed(np.random.randint(seed))
    fake_data = ctgan.sample(size)

    return fake_data, train_subset


def transformed_train_test_data(
    subset,
    dataset_name="adult",
    filepath=None,
    numerical_preprocess="standard",
    discrete_encode="onehot",
    seed=1000,
    noise_param=None,
    is_duplicate=None,
    fake_sample_size=None,
):
    """Returns transformed train and test data. If `filepath` is None,
    train=real_data else train=fake_data.

    Args:
        subset (int):
            Train subset to sample from.
        dataset_name (str):
            Name of the dataset. Defaults to ``adult``.
        filepath (str):
            Model (fake) args filepath.
        numerical_preprocess (str):
            Preprocessing type for the numerical columns. Defaults to ``standard``.
        discrete_encode (str):
            Encode type for categorical columns. Defaults to ``onehot``.
        seed (int):
            Reproducibility seed. Defaults to 1000.
        noise_param (tuple):
            Tuple of (noise_args_filepath, noise_scale). Defaults to None.
        is_duplicate (bool):
            Whether the model uses duplicate dataset or not. Defaults to False.
        fake_sample_size (int):
            Size of fake data to sample.

    Returns:
        A tuple of (train_data, test_data, target_index, data_transformer)
    """

    if filepath is not None:
        fake_sample_size = fake_sample_size or 20000
        train_subset, _ = sample_fake_data(
            subset,
            filepath,
            seed=seed,
            noise_param=noise_param,
            is_duplicate=is_duplicate,
            size=fake_sample_size,
        )
    else:
        train_subset, target_name, discrete_columns, _ = get_real_data(
            dataset_name=dataset_name,
            dataset_subset="train",
            test_size=subset,
            seed=seed,
        )

    test_subset, target_name, discrete_columns, _ = get_real_data(
        dataset_name=dataset_name, dataset_subset="test", seed=seed
    )

    # transform the dataset
    data_transformer = DataTransformer(
        numerical_preprocess=numerical_preprocess,
        discrete_encode=discrete_encode,
        target=target_name,
    )

    data_transformer.fit(train_subset, discrete_columns=discrete_columns)

    train_data = data_transformer.transform(train_subset)
    test_data = data_transformer.transform(test_subset)
    target_index = data_transformer._target_index - 1

    return train_data, test_data, target_index, data_transformer


def create_tensor_dataset(
    train_data,
    test_data,
    target_index=-1,
    seed=1000,
    valid_size=None,
    return_class_weight=False,
):
    """Function creates tensor dataset.

    Args:
        train_data (np.array):
            Train data.
        test_data (np.array):
            Test data.
        target_index (int):
            index of the target column. Defaults to -1.
        seed (int):
            Reproducibility seed.
        valid_size (int):
            Size of validation data. Defaults to None.
        return_class_weight (bool):
            Whether or not to compute class_weight.

    Returns:
        A tuple of (train_dataset, test_dataset, valid_dataset, class_weight).
        If `valid_size` is not None, valid_dataset is [].
        If `class_weight` is False, class_weight is []. The computed class_weight
        can be used to penalize the loss of the highly frequent class.
        This is especially useful if the classes are imbalanced and it's computed
        as `n_samples / (n_classes * np.bincount(train_y))`.
    """

    class_weight = []
    valid_dataset = []

    if valid_size is not None:
        train_data, valid_data = train_test_split(
            train_data,
            test_size=valid_size,
            random_state=seed,
            stratify=train_data[:, target_index],
        )
        valid_y = torch.Tensor(valid_data[:, target_index]).long()
        valid_X = torch.Tensor(np.delete(valid_data, target_index, axis=1))
        valid_dataset = TensorDataset(valid_X, valid_y)

    train_y = torch.Tensor(train_data[:, target_index]).long()
    train_X = torch.Tensor(np.delete(train_data, target_index, axis=1))
    train_dataset = TensorDataset(train_X, train_y)

    test_y = torch.Tensor(test_data[:, target_index]).long()
    test_X = torch.Tensor(np.delete(test_data, target_index, axis=1))
    test_dataset = TensorDataset(test_X, test_y)

    assert train_X.shape[1] == test_X.shape[1]

    if return_class_weight:
        class_weight = compute_class_weight(
            class_weight="balanced",
            classes=np.unique(train_y.numpy()),
            y=train_y.numpy(),
        )

    return train_dataset, test_dataset, valid_dataset, class_weight


class MLPTrainer(object):
    def __init__(self, args):
        self._batch_size = args["batch_size"]
        self._device = args["device"]
        self._lr_schedule = args["lr_schedule"]
        self._verbose = args["verbose"]
        self._epochs = args["epochs"]
        self._lr = args["lr"]
        self._max_lr = args["max_lr"]
        self._class_weight = args["class_weight"]
        self.model = args["model"]

        self.model.to(self._device)
        self._loss_fn = torch.nn.NLLLoss(weight=self._class_weight)

        optimizer = args["optimizer"] or torch.optim.Adam
        self._optimizer = optimizer(self.model.parameters(), lr=self._lr)

    def fit(self, train_dataset, valid_dataset=None):
        self._batch_size = min(len(train_dataset), self._batch_size)
        if valid_dataset is not None:
            valid_dataloader = torch.utils.data.DataLoader(
                dataset=valid_dataset, batch_size=self._batch_size, shuffle=False
            )

        train_dataloader = torch.utils.data.DataLoader(
            dataset=train_dataset, batch_size=self._batch_size, shuffle=True
        )

        if self._lr_schedule:
            scheduler = torch.optim.lr_scheduler.CyclicLR(
                self._optimizer,
                cycle_momentum=False,
                base_lr=self._lr,
                max_lr=self._max_lr,
            )

        for epoch in range(self._epochs):
            train_acc = 0
            train_loss = 0

            for i, (inputs, target) in enumerate(train_dataloader):

                # Zero the gradient
                self._optimizer.zero_grad()

                inputs = inputs.to(self._device)
                target = target.to(self._device)

                # Performed the forward pass
                output = self.model(inputs)
                train_acc += sum(torch.argmax(output, dim=1) == target)

                loss = self._loss_fn(output, target)
                train_loss += loss.item()

                # Performed the backward pass
                loss.backward()

                # Scheduler
                if self._lr_schedule:
                    scheduler.step()

                # Update the parameters
                self._optimizer.step()

            if self._verbose and valid_dataset is not None:
                print(
                    f"Epoch: {epoch} "
                    f"Train Loss: {round(train_loss/len(train_dataset), 3)} "
                    f"Train Acc: {round(train_acc.item()/len(train_dataset), 3)} ",
                    end=" | ",
                )
            elif self._verbose:
                print(
                    f"Epoch: {epoch} "
                    f"Train Loss: {round(train_loss/len(train_dataset), 3)} "
                    f"Train Acc: {round(train_acc.item()/len(train_dataset), 3)} "
                )

            # Validate the trained model
            if valid_dataset is not None:
                valid_acc = 0
                valid_loss = 0
                for i, (inputs, target) in enumerate(valid_dataloader):
                    self.model.eval()

                    inputs = inputs.to(self._device)
                    target = target.to(self._device)

                    output = self.model(inputs)
                    valid_acc += sum(torch.argmax(output, dim=1) == target)
                    loss = self._loss_fn(output, target)
                    valid_loss += loss.item()

                    self.model.train()
                if self._verbose:
                    print(
                        f"Valid Loss: {round(valid_loss/len(valid_dataset), 3)} "
                        f"Valid Acc: {round(valid_acc.item()/len(valid_dataset), 3)} "
                    )

    def score(self, test_dataset, scorer=None):
        y_pred = []
        y_true = []
        test_dataloader = torch.utils.data.DataLoader(
            dataset=test_dataset, batch_size=self._batch_size, shuffle=False
        )

        for i, (inputs, target) in enumerate(test_dataloader):
            self.model.to("cpu")
            output = self.model(inputs)
            y_pred.append(torch.argmax(output, dim=1).numpy())
            y_true.append(target)

        y_pred = np.concatenate(y_pred)
        y_true = np.concatenate(y_true)

        if self._verbose:
            print(f"Test Accuracy: {accuracy_score(y_true, y_pred)}")
            print(f"Test F1: {f1_score(y_true, y_pred)}")

        self.model.to(self._device)

        if scorer is not None:
            score = [s(y_true, y_pred) for s in scorer]
        else:
            score = f1_score(y_true, y_pred)
        return score


class PyTorchMLP(Module):
    def __init__(self, input_dim, label_dim, hidden_dim=[32, 32]):
        super(PyTorchMLP, self).__init__()
        seq = []
        for item in list(hidden_dim):
            seq += [Linear(input_dim, item), ReLU()]
            input_dim = item
        seq += [Linear(input_dim, label_dim), LogSoftmax(dim=-1)]
        self.seq = Sequential(*seq)
        self.seq.apply(self.init_weights)

    def init_weights(self, m):
        if isinstance(m, Linear):
            torch.nn.init.xavier_uniform_(m.weight)
            m.bias.data.fill_(0.01)

    def forward(self, input):
        return self.seq(input)


def trim_axs(axs, N):
    """
    Reduce *axs* to *N* Axes. All further Axes are removed from the figure.
    """
    axs = axs.flat
    for ax in axs[N:]:
        ax.remove()
    return axs[:N]


def sturge_binning(x):
    """Calculates number of bins based on Sturges rule."""
    bins = int(1 + np.ceil(np.log2(len(x))))
    return bins


def freedman_diaconis_binning(x):
    """Calculates number of bins based on Freedman-Diaconis rule."""
    import scipy

    iqr = scipy.stats.iqr(x)
    bin_width = (2 * iqr) / (len(x) ** (1 / 3))
    bins = int(np.ceil((x.max() - x.min()) / bin_width))
    return bins

def get_realdata(dataset,  subset="train", data_frac=None, data_dir="../data", handle_discrete_as="categorial", return_cols=False, random_state=1000):
    dset = Dataset(
            dataset_name=dataset,
            dataset_dir=data_dir,
            subset=subset,
            data_frac=data_frac,
            random_state=random_state,
            handle_discrete_as=handle_discrete_as
        )

    if return_cols:
        return dset.train_data[0], dset.cat_cols, dset.num_cols
    else:
        return dset.train_data[0]

def get_fakedata(dataset, synthesizer, exp_name, subset=-1, index=0, random_state=1000, data_dir="../data"):
    fakedata_pth = (
        f"{data_dir}/fake_samples/{synthesizer}/{dataset}/RS{random_state}/subset{subset}/{exp_name}/fakedata_{index}.csv"
    )
    
    fakedata = pd.read_csv(fakedata_pth)
    return fakedata