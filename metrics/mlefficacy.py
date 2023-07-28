import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.metrics import f1_score, r2_score
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


from utils.transformer import DataTransformer


## Models for Categorical Targets
class LogisticRegressionWrapper(object):
    """Binary LogisticRegression Efficacy based metric.

    This fits a LogisticRegression to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = LogisticRegression
    MODEL_KWARGS = {
        "solver": "lbfgs",
        "n_jobs": 2,
        "class_weight": "balanced",
        "max_iter": 50,
    }


class DecisionTreeClassifierWrapper(object):
    """DecisionTreeClassifier Efficacy based metric.

    This fits a DecisionTreeClassifier to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = DecisionTreeClassifier
    MODEL_KWARGS = {"max_depth": 15, "class_weight": "balanced", "random_state": 1000}


class MLPClassifierWrapper(object):
    """MLPClassifier Efficacy based metric.

    This fits a MLPClassifier to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = MLPClassifier
    MODEL_KWARGS = {
        "hidden_layer_sizes": (100),
        "batch_size": 100,
        "activation": "relu",
        "solver": "adam",
        "max_iter": 100,
        "random_state": 1000,
    }


## Models for Numerical Targets
class LinearRegressionWrapper:
    """LinearRegression Efficacy based metric.
    This fits a LinearRegression to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = LinearRegression
    MODEL_KWARGS = {}


class DecisionTreeRegressorWrapper(object):
    """DecisionTreeRegressor Efficacy based metric.

    This fits a DecisionTreeRegressor to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = DecisionTreeRegressor
    MODEL_KWARGS = {"max_depth": 15, "random_state": 1000}


class MLPRegressorWrapper:
    """MLPRegressor Efficacy based metric.
    This fits a MLPRegressor to the synthetic data and
    then evaluates it making predictions on the real data.
    """

    MODEL = MLPRegressor
    MODEL_KWARGS = {"hidden_layer_sizes": (100,), "max_iter": 50, "random_state": 1000}


class MLEfficacy(object):
    """Perform model compatibility test.

    Trains a machine learning model on the synthetic data
    and evaluate the performance of the model on real data.
    """

    def __init__(
        self,
        model_name="logistic",
        task="classification",
        preprocess_in_pipeline=False,
    ):
        """
        Args:
            model_name (str):
                The name of model to train with. Must be one of (`logistic`, `tree`, `mlp`).
            task (str):
                Machine learning task to predict. Must be one of (`classification`, `regression`).
            preprocess_in_pipe (bool):
                Whether or not to preprocess the data in the training pipeline.
                This include will SimpleImputer and RobutScaler.
        """
        self._model_name = model_name
        self._task = task
        self._preprocess_in_pipeline = preprocess_in_pipeline
        self.predictions = None

    def _score(self, scorer, real_target, predictions):
        """Score the trained model."""

        if scorer is None:
            if self._task == "classification":
                unique_labels, class_counts = np.unique(real_target, return_counts=True)
                if len(unique_labels) > 2:
                    # multiclass
                    scorer = lambda **kwargs: f1_score(**kwargs, average="micro")
                else:
                    class_to_report = np.argmin(class_counts)
                    scorer = lambda **kwargs: f1_score(
                        **kwargs, pos_label=class_to_report
                    )
            else:
                scorer = lambda **kwargs: r2_score(**kwargs)

        if isinstance(scorer, (list, tuple)):
            scorers = scorer
            return tuple(
                (scorer(y_true=real_target, y_pred=predictions) for scorer in scorers)
            )
        else:
            return scorer(y_true=real_target, y_pred=predictions)

    def _fit_predict(self, synthetic_data, synthetic_target, real_data):
        """Fit a model on the synthetic data and make predictions for the real data."""

        unique_labels = np.unique(synthetic_target)

        if len(unique_labels) == 1:
            predictions = np.full(len(real_data), unique_labels[0])

        else:

            real_data[np.isin(real_data, [np.inf, -np.inf])] = None
            synthetic_data[np.isin(synthetic_data, [np.inf, -np.inf])] = None

            if self._task == "classification":
                if self._model_name == "logistic":
                    model_class = LogisticRegressionWrapper()
                elif self._model_name == "tree":
                    model_class = DecisionTreeClassifierWrapper()
                elif self._model_name == "mlp":
                    model_class = MLPClassifierWrapper()
                else:
                    raise ValueError(
                        "`model_class` must be one of (`logistic`, `tree`, `mlp`)."
                    )
            elif self._task == "regression":
                if self._model_name == "linear":
                    model_class = LinearRegressionWrapper()
                elif self._model_name == "tree":
                    model_class = DecisionTreeRegressorWrapper()
                elif self._model_name == "mlp":
                    model_class = MLPRegressorWrapper()
                else:
                    raise ValueError(
                        "`model_class` must be one of (`linear`, `tree`, `mlp`)."
                    )
            else:
                raise NotImplementedError(f"Unsupported task: {self._task}")

            model_kwargs = (
                model_class.MODEL_KWARGS.copy() if model_class.MODEL_KWARGS else {}
            )
            model = model_class.MODEL(**model_kwargs)

            if self._preprocess_in_pipeline:
                pipeline = Pipeline(
                    [
                        ("imputer", SimpleImputer()),
                        ("scaler", RobustScaler()),
                        ("model", model),
                    ]
                )
            else:
                pipeline = Pipeline([("model", model)])

            pipeline.fit(synthetic_data, synthetic_target.ravel())
            predictions = pipeline.predict(real_data)

        return predictions

    def compute(
        self,
        real_data,
        synthetic_data,
        target=None,
        scorer=None,
        transformer=None,
        fit_data=None,
    ):
        """Compute the given metric.

        This fits a Machine Learning model on the synthetic data and
        then evaluates it making predictions on the real data.

        A ``target`` column name must be given, which will be used as the target column for the
        Machine Learning prediction.

        A ``scorer`` or lists of ``scorers`` name must be given. It must be any of (`f1`, `acc`).
        Defaults is `f1`

        Args:
            real_data (Union[pandas.DataFrame, tuple[numpy.array], list[numpy.array]]):
                The values from the real dataset.
            synthetic_data (Union[pandas.DataFrame, tuple[numpy.array], list[numpy.array]]):
                The values form the synthetic dataset.
            target (str):
                Name of the column to use as target.
            scorer (Union[callable, list[callable], NoneType]):
                Scorer (or list of scorers) to apply. If not passed, default to `f1`.
            transformer (callable, NoneType]):
                Data Transformer to apply. If not passed, default to `None`.
            fit_data (pd.DataFrame, NoneType):
                Data to fit the data transformer on. Defaults to `None`.
                Fits the data transformer on `realdata` if `transformer` is None.
        """

        real_data, synthetic_data, real_target, synthetic_target = self._preprocess(
            real_data, synthetic_data, target, transformer, fit_data
        )

        predictions = self._fit_predict(
            synthetic_data,
            synthetic_target,
            real_data,
        )

        return self._score(scorer, real_target, predictions)

    def _preprocess(self, real_data, synthetic_data, target, transformer, fit_data):
        if isinstance(real_data, (tuple, list)) and isinstance(
            synthetic_data, (tuple, list)
        ):
            assert (
                real_data[0].shape[1] == synthetic_data[0].shape[1]
            ), "`real_data` and `synthetic_data` must have the same columns."

            real_data, real_target = real_data
            synthetic_data, synthetic_target = synthetic_data

        else:
            assert set(real_data.columns) == set(
                synthetic_data.columns
            ), "`real_data` and `synthetic_data` must have the same columns."
            assert target is not None, "`target name` must be provided!"
            assert target in set(real_data.columns) and target in set(
                synthetic_data.columns
            ), "`target name` must be in real_data and synthetic_data."

            real_data = real_data.copy()
            synthetic_data = synthetic_data.copy()

            real_data = real_data[real_data[target].notnull()]
            synthetic_data = synthetic_data[synthetic_data[target].notnull()]

            if transformer is None:
                discrete_columns = set(real_data.columns) - set(
                    real_data._get_numeric_data().columns
                )

                if self._task == "classification":
                    data_transformer = DataTransformer(
                        numerical_preprocess="standard",
                        discrete_encode="onehot",
                        target=target,
                    )
                else:
                    data_transformer = DataTransformer(
                        numerical_preprocess="none",
                        discrete_encode="onehot",
                        target=target,
                    )
                if fit_data is not None:
                    data_transformer.fit(fit_data, discrete_columns=discrete_columns)
                else:
                    data_transformer.fit(real_data, discrete_columns=discrete_columns)
            else:
                data_transformer = transformer

            real_data = data_transformer.transform(real_data)
            real_target = real_data[:, data_transformer._target_index]
            real_data = np.delete(real_data, data_transformer._target_index, axis=1)

            synthetic_data = data_transformer.transform(synthetic_data)
            synthetic_target = synthetic_data[:, data_transformer._target_index]
            synthetic_data = np.delete(
                synthetic_data, data_transformer._target_index, axis=1
            )

        return real_data, synthetic_data, real_target, synthetic_target
