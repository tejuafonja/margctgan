import joblib
import itertools
import pandas as pd
import psutil

from .mlefficacy import MLEfficacy

import sys

sys.path.append("..")
from utils.metrics_utils import make_size_equal, normalize_score

__all__ = ["efficacy_test", "all_models_test", "efficacy_test_wrapper"]

N_JOBS=-1

def efficacy_test(
    realdata,
    fakedata,
    target_name,
    cat_cols=None,
    model_name=None,
    task=None,
    scorer=None,
    return_dataframe=False,
    keep_default_size=True,
    transformer=None,
    fit_data=None,
    random_state=1000,
):
    """Trains a machine learning model on the synthetic data
    and evaluate the performance of the model on real data.

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate.
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        target_name (str, optional):
            Target column name.
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        model_name (str, optional):
            Machine learning model to evaluate.
            Must be one of (`logisitc`, `tree`, `mlp`).
            Defaults to "logistic" if `target_name` in `cat_cols` else "linear"
        task (str):
                Machine learning task to predict. Must be one of (`classification`, `regression`).
                Defaults to "classification" if `target_name` in `cat_cols` else "regression".
        scorer (str or func, optional):
            Scorer to use. Defaults to `f1`. Defaults to None.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`
        keep_default_size (bool, optional):
            Whether or not to keep default size.
                If `False`, `realdata` and `fakedata` will have equal size.
        transformer (object, optional):
            Transformer object to apply on columns. Defaults to None.
        fit_data (pd.DataFrame, NoneType):
                Data to fit the data transformer on. Defaults to `None`.
                Fits the data transformer on `realdata` if `transformer` is None.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.

    Returns:
        int or array-like:
            scores
    """

    __name__ = "efficacy_test"

    if not keep_default_size:
        realdata, fakedata = make_size_equal(realdata, fakedata, random_state)

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns.to_list()

    if target_name in cat_cols:
        task = "classification"
        if model_name == None:
            model_name = "logistic"
    else:
        task = "regression"
        if model_name == None:
            model_name = "linear"

    mlefficacy = MLEfficacy(model_name=model_name, task=task)
    result = mlefficacy.compute(
        real_data=realdata,
        synthetic_data=fakedata,
        target=target_name,
        scorer=scorer,
        transformer=transformer,
        fit_data=fit_data,
    )

    if task == "regression":
        normalized_result = normalize_score(
            result, min_value="-inf", max_value=1, goal="maximize"
        )
    else:
        normalized_result = result

    if return_dataframe:
        column_type = "categorical" if target_name in cat_cols else "numerical"
        result = {
            "metric": __name__,
            "model_name": model_name,
            "column_name": target_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": normalized_result,
        }
        result = pd.DataFrame([result])

    return result


def efficacy_test_wrapper(
    realdata,
    fakedata,
    target_name,
    cat_cols=None,
    model_names=None,
    task=None,
    transformer=None,
    fit_data=None,
    random_state=1000,
    psutil_terminate=False,
):
    """Efficacy Test Wrapper

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate.
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        target_name (str, optional):
            Target column name.
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        model_names (list, optional):
            Machine learning model to evaluate.
        task (str):
                Machine learning task to predict. Must be one of (`classification`, `regression`).
                Defaults to "classification" if `target_name` in `cat_cols` else "regression".
        transformer (object, optional):
            Transformer object to apply on columns. Defaults to None.
        fit_data (pd.DataFrame, NoneType):
                Data to fit the data transformer on. Defaults to `None`.
                Fits the data transformer on `realdata` if `transformer` is None.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
        psutil_terminate (bool, optional):
            Whether or not to terminate processes spawn by joblib `loky` backend.
            Caution, turning this on might kill other processes spawn within the
            timeframe of running the code.
    """

    if model_names is None:
        results = efficacy_test(
            realdata,
            fakedata,
            target_name,
            cat_cols=cat_cols,
            model_name=model_names,
            task=task,
            return_dataframe=True,
            keep_default_size=True,
            transformer=transformer,
            fit_data=fit_data,
            random_state=random_state,
        )

    else:
        if psutil_terminate:
            current_process = psutil.Process()
            subproc_before = set(
                [p.pid for p in current_process.children(recursive=True)]
            )

        with joblib.Parallel(n_jobs=N_JOBS) as parallel:
            results = parallel(
                joblib.delayed(efficacy_test)(
                    realdata=realdata,
                    fakedata=fakedata,
                    target_name=target_name,
                    cat_cols=cat_cols,
                    model_name=model_name,
                    task=task,
                    transformer=transformer,
                    fit_data=fit_data,
                    keep_default_size=True,
                    return_dataframe=True,
                    random_state=random_state,
                )
                for model_name in model_names
            )
        results = pd.concat(results)
        results.reset_index(drop=True, inplace=True)

        if psutil_terminate:
            subproc_after = set(
                [p.pid for p in current_process.children(recursive=True)]
            )
            for subproc in subproc_after - subproc_before:
                psutil.Process(subproc).terminate()

    return results


def all_models_test(
    realdata,
    fakedata,
    cat_cols=None,
    cat_model_names=None,
    num_model_names=None,
    return_mean=True,
    return_dataframe=False,
    fit_data=None,
    random_state=1000,
    psutil_terminate=False,
):
    """All models Test. Trains a Machine Learning model on
    all the attributes

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
            If `None`, infers categorical columns from `realdata`
        cat_model_names (array-like, optional):
            List of ML models to evaluate for classification tasks.
            Defaults to `[logistic]` if None.
        num_model_names (array-like, optional):
            List of ML models to evaluate for regression tasks.
            Defaults to `[linear]` if None.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
        random_state (int, optional):
            Random state number for reproducibility.
            Defaults to `1000`.
        psutil_terminate (bool, optional):
            Whether or not to terminate processes spawn by joblib `loky` backend.
            Caution, turning this on might kill other processes spawn within the
            timeframe of running the code.
    """
    __name__ = "all_models_test"

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns

    num_cols = list(set(realdata.columns) - set(cat_cols))

    if psutil_terminate:
        current_process = psutil.Process()
        subproc_before = set([p.pid for p in current_process.children(recursive=True)])

    if cat_model_names is None and num_model_names is None:
        results = joblib.Parallel(n_jobs=N_JOBS)(
            joblib.delayed(efficacy_test)(
                realdata=realdata,
                fakedata=fakedata,
                target_name=col_name,
                cat_cols=cat_cols,
                model_name="logistic" if col_name in cat_cols else "linear",
                task="classification" if col_name in cat_cols else "regression",
                transformer=None,
                fit_data=fit_data,
                keep_default_size=True,
                return_dataframe=True,
                random_state=random_state,
            )
            for col_name in realdata.columns
        )
    else:
        if cat_model_names == None:
            cat_model_names = ["logistic"]

        if num_model_names == None:
            num_model_names = ["linear"]

        cat_combinations = itertools.product(
            cat_cols, cat_model_names, ["classification"]
        )
        num_combinations = itertools.product(num_cols, num_model_names, ["regression"])

        with joblib.Parallel(n_jobs=N_JOBS) as parallel:
            results = parallel(
                joblib.delayed(efficacy_test)(
                    realdata=realdata,
                    fakedata=fakedata,
                    target_name=col_name,
                    cat_cols=cat_cols,
                    model_name=model_name,
                    task=task,
                    transformer=None,
                    fit_data=fit_data,
                    keep_default_size=True,
                    return_dataframe=True,
                    random_state=random_state,
                )
                for col_name, model_name, task in list(cat_combinations)
                + list(num_combinations)
            )

    result = pd.concat(results)
    result.reset_index(drop=True, inplace=True)
    result.loc[:, "metric"] = __name__
    # result = result.rename(columns={efficacy_test.__name__: __name__})

    if return_mean:
        result = result.all_models_test.mean()
    else:
        return_dataframe = False

    if return_dataframe:
        result = {"score": result, "metric": __name__}
        result = pd.DataFrame([result])

    if psutil_terminate:
        subproc_after = set([p.pid for p in current_process.children(recursive=True)])
        for subproc in subproc_after - subproc_before:
            psutil.Process(subproc).terminate()

    return result
