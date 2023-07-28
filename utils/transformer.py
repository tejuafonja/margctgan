import warnings

import sklearn

warnings.filterwarnings("ignore", category=sklearn.exceptions.ConvergenceWarning)


from collections import namedtuple

import numpy as np
import pandas as pd
from sklearn.mixture import BayesianGaussianMixture
from sklearn.preprocessing import (
    LabelEncoder,
    MinMaxScaler,
    OneHotEncoder,
    StandardScaler,
)

SpanInfo = namedtuple("SpanInfo", ["dim", "activation_fn"])
ColumnTransformInfo = namedtuple(
    "ColumnTransformInfo",
    [
        "column_name",
        "column_type",
        "transform",
        "transform_aux",
        "output_info",
        "output_dimensions",
    ],
)


class DataTransformer(object):
    """Data Transformer.

    Model continuous columns with a BayessianGMM and normalized to a scalar [0, 1] and a vector.
    Discrete columns are encoded using a scikit-learn OneHotEncoder.
    """

    def __init__(
        self,
        max_clusters=10,
        covariance_type="full",
        weight_threshold=0.005,
        discrete_encode="onehot",
        numerical_preprocess="none",
        target=None,
        random_state=1000,
    ):
        """Create a data transformer.

        Args:
            max_clusters (int):
                Maximum number of Gaussian distributions in Bayessian GMM.
            covariance_type (str):
                String describing the type of covariance parameters to use.
                Must be one of ("full", "tied").
            weight_threshold (float):
                Weight threshold for a Gaussian distribution to be kept.
            discrete_encode (str):
                Discrete encoding type. Must be one of ("label", "onehot").
                Defaults to "onehot".
            numerical_preprocess (str):
                Preprocess type for numerical columns.
                Must be one of ("standard", "minmax", "none", "bayesian").
                Defaults to "none".
            target (str):
                Data target column name. If discrete, to be transformed
                as label encoding vector instead of onehot encoding matrix.
            random_state (int):
                Random state for Bayessian GMM.
        """
        self._max_clusters = max_clusters
        self._covariance_type = covariance_type
        self._weight_threshold = weight_threshold
        self._discrete_encode = discrete_encode
        self._numerical_preprocess = numerical_preprocess
        self._target = target
        self._random_state = random_state

    def _fit_continuous(self, column_name):
        """Return column transform info"""

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=None,
            transform_aux=None,
            output_info=[SpanInfo(1, "tanh")],
            output_dimensions=1,
        )

    def _fit_continuous_gaussian_mixture(self, column_name, raw_column_data):
        """Train Bayessian GMM for continuous column."""
        gm = BayesianGaussianMixture(
            n_components=self._max_clusters,
            weight_concentration_prior=0.001,
            weight_concentration_prior_type="dirichlet_process",
            covariance_type=self._covariance_type,
            n_init=1,
            random_state=self._random_state,
        )

        gm.fit(raw_column_data.reshape(-1, 1))
        valid_component_indicator = gm.weights_ > self._weight_threshold
        num_components = valid_component_indicator.sum()

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=gm,
            transform_aux=valid_component_indicator,
            output_info=[SpanInfo(1, "tanh"), SpanInfo(num_components, "softmax")],
            output_dimensions=1 + num_components,
        )

    def _fit_continuous_standard(self, column_name, raw_column_data):
        """Fit standard scaler for continuous column."""
        ss = StandardScaler()
        ss.fit(raw_column_data.reshape(-1, 1))

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=ss,
            transform_aux=None,
            output_info=[SpanInfo(1, "tanh")],
            output_dimensions=1,
        )

    def _fit_continuous_minmax(self, column_name, raw_column_data):
        """Fit minmax scaler for continuous column."""
        minmax = MinMaxScaler()
        minmax.fit(raw_column_data.reshape(-1, 1))

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="continuous",
            transform=minmax,
            transform_aux=None,
            output_info=[SpanInfo(1, "tanh")],
            output_dimensions=1,
        )

    def _fit_discrete_onehot_encode(self, column_name, raw_column_data):
        """Fit one hot encoder for discrete column."""
        raw_column_data = self.cast_as_string(raw_column_data)

        ohe = OneHotEncoder(sparse=False, handle_unknown="ignore")
        ohe.fit(raw_column_data.reshape(-1, 1))
        num_categories = len(ohe.categories_[0])

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=ohe,
            transform_aux=None,
            output_info=[SpanInfo(num_categories, "softmax")],
            output_dimensions=num_categories,
        )

    def _fit_discrete_label_encode(self, column_name, raw_column_data):
        """Fit label encoder for discrete column."""
        raw_column_data = self.cast_as_string(raw_column_data)

        le = LabelEncoder()
        le.fit(raw_column_data)

        return ColumnTransformInfo(
            column_name=column_name,
            column_type="discrete",
            transform=le,
            transform_aux=None,
            output_info=[SpanInfo(1, "softmax")],
            output_dimensions=1,
        )

    def fit(self, raw_data, discrete_columns=tuple()):
        """Fit continuous columns and discrete columns.

        This step also counts the #columns in matrix data, and span information.
        """
        self.output_info_list = []
        self.output_dimensions = 0

        if not isinstance(raw_data, pd.DataFrame):
            self.dataframe = False
            raw_data = pd.DataFrame(raw_data)
            self._column_names = None
        else:
            self.dataframe = True
            self._column_names = raw_data.columns

        self._column_raw_dtypes = raw_data.dtypes
        self._target_index_in_raw_data = None
        self._target_index = None

        self._column_transform_info_list = []
        for column_name in raw_data.columns:
            raw_column_data = raw_data[column_name].values
            if column_name in discrete_columns:
                if self._discrete_encode == "label":
                    column_transform_info = self._fit_discrete_label_encode(
                        column_name, raw_column_data
                    )
                elif self._target is not None and column_name == self._target:
                    column_transform_info = self._fit_discrete_label_encode(
                        column_name, raw_column_data
                    )
                else:
                    column_transform_info = self._fit_discrete_onehot_encode(
                        column_name, raw_column_data
                    )

            else:
                if self._numerical_preprocess == "standard":
                    column_transform_info = self._fit_continuous_standard(
                        column_name, raw_column_data
                    )
                elif self._numerical_preprocess == "minmax":
                    column_transform_info = self._fit_continuous_minmax(
                        column_name, raw_column_data
                    )
                elif self._target is not None and column_name == self._target:
                    column_transform_info = self._fit_continuous(column_name)
                elif self._numerical_preprocess == "bayesian":
                    column_transform_info = self._fit_continuous_gaussian_mixture(
                        column_name, raw_column_data
                    )
                else:
                    column_transform_info = self._fit_continuous(column_name)

            self.output_info_list.append(column_transform_info.output_info)
            self.output_dimensions += column_transform_info.output_dimensions
            if self._target is not None and column_name == self._target:
                self._target_index = self.output_dimensions - 1
                self._target_index_in_raw_data = (
                    self._column_names.tolist().index(self._target)
                    if self.dataframe
                    else None
                )
            self._column_transform_info_list.append(column_transform_info)

    def _transform_continuous(self, raw_column_data):
        """Return raw column data."""
        return [raw_column_data.reshape(-1, 1)]

    def _transform_continuous_gaussian_mixture(
        self, column_transform_info, raw_column_data
    ):
        """Transform continuous column using variational guassian mixture technique.

        It renormalizes the column and samples new data component probability.
        """
        gm = column_transform_info.transform

        valid_component_indicator = column_transform_info.transform_aux
        num_components = valid_component_indicator.sum()

        means = gm.means_.reshape((1, self._max_clusters))

        if self._covariance_type == "full":
            stds = np.sqrt(gm.covariances_).reshape((1, self._max_clusters))
        else:
            stds = np.sqrt(gm.covariances_)

        normalized_values = ((raw_column_data - means) / (4 * stds))[
            :, valid_component_indicator
        ]
        component_probs = gm.predict_proba(raw_column_data)[
            :, valid_component_indicator
        ]

        selected_component = np.zeros(len(raw_column_data), dtype="int")
        for i in range(len(raw_column_data)):
            component_prob_t = component_probs[i] + 1e-6
            component_prob_t = component_prob_t / component_prob_t.sum()
            selected_component[i] = np.random.choice(
                np.arange(num_components), p=component_prob_t
            )

        selected_normalized_value = normalized_values[
            np.arange(len(raw_column_data)), selected_component
        ].reshape([-1, 1])

        selected_normalized_value = np.clip(selected_normalized_value, -0.99, 0.99)

        selected_component_onehot = np.zeros_like(component_probs)
        selected_component_onehot[
            np.arange(len(raw_column_data)), selected_component
        ] = 1

        return [selected_normalized_value, selected_component_onehot]

    def _transform_continuous_standard(self, column_transform_info, raw_column_data):
        """Standardize continuous column by removing the mean and scaling to unit variance."""
        ss = column_transform_info.transform

        return [ss.transform(raw_column_data).reshape(-1, 1)]

    def _transform_continuous_minmax(self, column_transform_info, raw_column_data):
        """Transform continuous column by scaling to a given range."""
        minmax = column_transform_info.transform

        return [minmax.transform(raw_column_data).reshape(-1, 1)]

    def _transform_discrete_ohehot_encode(self, column_transform_info, raw_column_data):
        """Transform discrete column by one hot encoding its categories."""

        # cast raw_column_data as string: incase the categories are integers
        raw_column_data = self.cast_as_string(raw_column_data)

        ohe = column_transform_info.transform

        return [ohe.transform(raw_column_data.reshape(-1, 1))]

    def _transform_discrete_label_encode(self, column_transform_info, raw_column_data):
        """Transform discrete column by label encoding its categories."""

        # cast raw_column_data as string: incase the categories are integers
        raw_column_data = self.cast_as_string(raw_column_data)

        le = column_transform_info.transform

        return [le.transform(raw_column_data.ravel()).reshape(-1, 1)]

    def transform(self, raw_data):
        """Take raw data and output a matrix transformed data."""
        if not isinstance(raw_data, pd.DataFrame):
            raw_data = pd.DataFrame(raw_data)

        column_data_list = []
        for column_transform_info in self._column_transform_info_list:
            column_data = raw_data[[column_transform_info.column_name]].values
            if column_transform_info.column_type == "continuous":
                if self._numerical_preprocess == "standard":
                    column_data_list += self._transform_continuous_standard(
                        column_transform_info, column_data
                    )
                elif self._numerical_preprocess == "minmax":
                    column_data_list += self._transform_continuous_minmax(
                        column_transform_info, column_data
                    )
                elif (
                    self._target is not None
                    and column_transform_info.column_name == self._target
                ):
                    column_data_list += self._transform_continuous(column_data)
                elif self._numerical_preprocess == "bayesian":
                    column_data_list += self._transform_continuous_gaussian_mixture(
                        column_transform_info, column_data
                    )
                else:
                    column_data_list += self._transform_continuous(column_data)
            else:
                assert column_transform_info.column_type == "discrete"
                if self._discrete_encode == "label":
                    column_data_list += self._transform_discrete_label_encode(
                        column_transform_info, column_data
                    )
                elif (
                    self._target is not None
                    and column_transform_info.column_name == self._target
                ):
                    column_data_list += self._transform_discrete_label_encode(
                        column_transform_info, column_data
                    )
                else:
                    column_data_list += self._transform_discrete_ohehot_encode(
                        column_transform_info, column_data
                    )

        return np.concatenate(column_data_list, axis=1).astype(float)

    def _inverse_transform_continuous(self, column_data):
        """Inverse transform continuous column."""
        return column_data

    def _inverse_transform_continuous_gaussian_mixture(
        self, column_transform_info, column_data, sigmas, st
    ):
        """Inverse transform gaussian mixture transformed continuous column."""
        gm = column_transform_info.transform
        valid_component_indicator = column_transform_info.transform_aux

        selected_normalized_value = column_data[:, 0]
        selected_component_probs = column_data[:, 1:]

        if sigmas is not None:
            sig = sigmas[st]
            selected_normalized_value = np.random.normal(selected_normalized_value, sig)

        selected_normalized_value = np.clip(selected_normalized_value, -1, 1)
        component_probs = np.ones((len(column_data), self._max_clusters)) * -100
        component_probs[:, valid_component_indicator] = selected_component_probs

        means = gm.means_.reshape([-1])
        selected_component = np.argmax(component_probs, axis=1)

        if self._covariance_type == "full":
            stds = np.sqrt(gm.covariances_).reshape([-1])
            std_t = stds[selected_component]
        else:
            std_t = np.sqrt(gm.covariances_)[0]

        mean_t = means[selected_component]
        column = selected_normalized_value * 4 * std_t + mean_t

        return column

    def _inverse_transform_continuous_standard(
        self, column_transform_info, column_data
    ):
        """Inverse transform standard scaler transformed continous column."""
        ss = column_transform_info.transform

        return ss.inverse_transform(column_data)

    def _inverse_transform_continuous_minmax(self, column_transform_info, column_data):
        """Inverse transform min max scaled continous column."""
        minmax = column_transform_info.transform

        return minmax.inverse_transform(column_data)

    def _inverse_transform_discrete_onehot_encode(
        self, column_transform_info, column_data
    ):
        """Inverse transform one hot encoded discrete column."""
        ohe = column_transform_info.transform
        mask = np.all(column_data == 0, axis=1)

        dummies = ohe.categories_[0]
        indices = np.argmax(column_data, axis=1)
        array = pd.Series(indices).map(dummies.__getitem__).values.astype("object")

        # if all elements are exactly zero, then default to None
        array[mask] = None

        return array

    def _inverse_transform_discrete_label_encode(
        self, column_transform_info, column_data
    ):
        """Inverse transform label encoded discrete column."""
        le = column_transform_info.transform

        return le.inverse_transform(column_data.ravel().astype(int))

    def inverse_transform(self, data, sigmas=None):
        """Take matrix data and output raw data.

        Output uses the same type as input to the transform function.
        Either np array or pd dataframe.
        """
        st = 0
        recovered_column_data_list = []
        column_names = []
        for column_transform_info in self._column_transform_info_list:
            dim = column_transform_info.output_dimensions
            column_data = data[:, st : st + dim]

            if column_transform_info.column_type == "continuous":
                if self._numerical_preprocess == "bayesian":
                    recovered_column_data = (
                        self._inverse_transform_continuous_gaussian_mixture(
                            column_transform_info, column_data, sigmas, st
                        )
                    )
                elif self._numerical_preprocess == "standard":
                    recovered_column_data = self._inverse_transform_continuous_standard(
                        column_transform_info, column_data
                    )
                elif self._numerical_preprocess == "minmax":
                    recovered_column_data = self._inverse_transform_continuous_minmax(
                        column_transform_info, column_data
                    )
                else:
                    recovered_column_data = self._inverse_transform_continuous(
                        column_data
                    )

            else:
                assert column_transform_info.column_type == "discrete"
                if self._discrete_encode == "label":
                    recovered_column_data = (
                        self._inverse_transform_discrete_label_encode(
                            column_transform_info, column_data
                        )
                    )
                elif (
                    self._target is not None
                    and column_transform_info.column_name == self._target
                ):
                    recovered_column_data = (
                        self._inverse_transform_discrete_label_encode(
                            column_transform_info, column_data
                        )
                    )
                else:
                    recovered_column_data = (
                        self._inverse_transform_discrete_onehot_encode(
                            column_transform_info, column_data
                        )
                    )

            recovered_column_data_list.append(recovered_column_data)
            column_names.append(column_transform_info.column_name)
            st += dim

        recovered_data = np.column_stack(recovered_column_data_list)
        recovered_data = pd.DataFrame(recovered_data, columns=column_names)

        for k, v in self._column_raw_dtypes.items():
            if str(v).startswith("i"):
                try:
                    recovered_data[k] = recovered_data[k].astype(
                        self._column_raw_dtypes[k]
                    )
                except:
                    recovered_data[k] = (
                        recovered_data[k]
                        .astype(float)
                        .astype(self._column_raw_dtypes[k])
                    )
            else:
                recovered_data[k] = (
                            recovered_data[k]
                            .astype(self._column_raw_dtypes[k])
                        )

        if not self.dataframe:
            recovered_data = recovered_data.values

        return recovered_data

    def convert_column_name_value_to_id(self, column_name, value):
        """Transform the input value to corresponding id of the column."""
        discrete_counter = 0
        column_id = 0
        for column_transform_info in self._column_transform_info_list:
            if column_transform_info.column_name == column_name:
                break
            if column_transform_info.column_type == "discrete":
                discrete_counter += 1

            column_id += 1

        else:
            raise ValueError(
                f"The column_name `{column_name}` does't exist in the data."
            )

        one_hot = column_transform_info.transform.transform(np.array([[value]]))[0]
        if sum(one_hot) == 0:
            raise ValueError(
                f"The value `{value}` doesn't exist in the column `{column_name}`."
            )

        return {
            "discrete_column_id": discrete_counter,
            "column_id": column_id,
            "value_id": np.argmax(one_hot),
        }

    def cast_as_string(self, raw_column_data):
        if type(raw_column_data) == np.ndarray:
            raw_column_data = raw_column_data.astype("str")
        else:
            raw_column_data = raw_column_data.astype("object")
        return raw_column_data


class TableGANTransformer(object):
    """TableGAN Transformer

    Args:
        side (int): side of the image.
    """

    def __init__(self, side):
        self.side = side

    def fit(self, meta):
        self.meta = meta
        self.c = len(self.meta)
        self.minn = np.zeros(self.c)
        self.maxx = np.zeros(self.c)

        for i in range(self.c):
            if self.meta[i]["type"] == "continuous":
                self.minn[i] = self.meta[i]["min"] - 1e-3
                self.maxx[i] = self.meta[i]["max"] + 1e-3
            else:
                self.minn[i] = -1e-3
                self.maxx[i] = self.meta[i]["size"] - 1 + 1e-3

    def transform(self, data):
        data = data.copy().astype("float32")

        data = (data - self.minn) / (self.maxx - self.minn) * 2 - 1
        if self.side * self.side >= data.shape[1]:
            padding = np.zeros((len(data), self.side * self.side - data.shape[1]))
            data = np.concatenate([data, padding], axis=1)
        else:
            assert (
                data.shape[1] % self.side == 0
            ), "data.shape[1] should be divisible by side. or side*side>=data.shape[1]"

        return data.reshape(-1, 1, self.side, self.side)

    def inverse_transform(self, data):
        if self.side * self.side < self.c:
            data = data.reshape(-1, self.c)
        else:
            data = data.reshape(-1, self.side * self.side)
            data = data[:, : self.c]

        assert data.shape[1] == self.c

        data_t = np.zeros([len(data), self.c])

        for id_, info in enumerate(self.meta):
            numerator = data[:, id_].reshape([-1]) + 1
            data_t[:, id_] = (numerator / 2) * (
                self.maxx[id_] - self.minn[id_]
            ) + self.minn[id_]
            if info["type"] == "categorical":
                data_t[:, id_] = np.round(data_t[:, id_])

        return data_t
