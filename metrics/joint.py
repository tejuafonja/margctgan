import joblib
import pandas as pd
from sklearn.neighbors import NearestNeighbors

from utils.transformer import DataTransformer

__all__ = [
    "likelihood_approximation",
    "closeness_approximation",
    "nearest_neighbors_wrapper",
]

N_JOBS=-1


def likelihood_approximation(
    realdata,
    fakedata,
    cat_cols=None,
    return_mean=True,
    return_dataframe=False,
    realdata_subsample=None,
    fakedata_subsample=None,
    transformer=None,
    fit_data=None,
    kind="nearest_neighbors",
    neighbors=1,
    random_state=1000,
):
    """Calculates the likelihood approximation of realdata to fakedata.

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        return_mean (bool, optional):
            Whether or not to return mean distance. Defaults to `True`.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`
        realdata_subsample (int, optional):
            Number of real datapoint to evaluate. Defaults to None.
        fakedata_subsample (int, optional):
            Number of fake datapoint to evaluate. Defaults to None.
        transformer (object, optional):
            Transformer object to apply on columns. Defaults to None.
        fit_data (pd.DataFrame, optional):
            Data to fit the data transformer on. Defaults to `None`.
        kind (str, optional):
            Approximation method to use. Defaults to `nearest_neighbors`
        neighbors (int, optional):
            Number of neigbors to calculate distance to.
            Defaults to 1. Only if `kind`=`nearest_neighbors`
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.

    Returns:
        dict or float:
            dict if `return_mean=True`
    """

    __name__ = "likelihood_approximation"

    if realdata_subsample is not None and len(realdata) > realdata_subsample:
        realdata = realdata.sample(realdata_subsample, random_state=random_state)
        realdata.reset_index(drop=True, inplace=True)

    if fakedata_subsample is not None and len(fakedata) > fakedata_subsample:
        fakedata = fakedata.sample(fakedata_subsample, random_state=random_state)
        fakedata.reset_index(drop=True, inplace=True)

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns.to_list()

    if fit_data is not None:
        transformer = DataTransformer(
            discrete_encode="onehot",
            numerical_preprocess="minmax",
        )
        transformer.fit(fit_data, discrete_columns=cat_cols)
        realdata = transformer.transform(realdata)
        fakedata = transformer.transform(fakedata)

    elif transformer is not None:
        realdata = transformer.transform(realdata)
        fakedata = transformer.transform(fakedata)

    else:
        transformer = DataTransformer(
            discrete_encode="onehot",
            numerical_preprocess="minmax",
        )
        transformer.fit(realdata, discrete_columns=cat_cols)
        realdata = transformer.transform(realdata)
        fakedata = transformer.transform(fakedata)

    if kind == "nearest_neighbors":
        # fit on fakedata
        neigh = NearestNeighbors(n_neighbors=neighbors)
        neigh.fit(fakedata)

        # compute the distance of each realdata sample to fakedata
        result, _ = neigh.kneighbors(realdata, return_distance=True)

    else:
        raise NotImplementedError()

    if return_mean:
        result = result.mean()
    else:
        return_dataframe = False

    if return_dataframe:
        result = {"score": result, "metric": __name__, "normalized_score": result}
        result = pd.DataFrame([result])
        result["neighbors"] = neighbors

    return result


def closeness_approximation(
    realdata,
    fakedata,
    cat_cols=None,
    return_mean=True,
    return_dataframe=False,
    realdata_subsample=None,
    fakedata_subsample=None,
    transformer=None,
    fit_data=None,
    kind="nearest_neighbors",
    neighbors=1,
    random_state=1000,
):
    """Calculates the closeness of fakedata to realdata.

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        return_mean (bool, optional):
            Whether or not to return mean distance. Defaults to `True`.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`
        realdata_subsample (int, optional):
            Number of real datapoint to evaluate. Defaults to None.
        fakedata_subsample (int, optional):
            Number of fake datapoint to evaluate. Defaults to None.
        transformer (object, optional):
            Transformer object to apply on columns. Defaults to None.
        fit_data (pd.DataFrame, optional):
            Data to fit the data transformer on. Defaults to `None`.
        kind (str, optional):
            Approximation method to use. Defaults to `nearest_neighbors`
        neighbors (int, optional):
            Number of neigbors to calculate distance to.
            Defaults to 1. Only if `kind`=`nearest_neighbors`
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.

    Returns:
        dict or float:
            dict if `return_mean=True`
    """

    __name__ = "closeness_approximation"

    if realdata_subsample is not None and len(realdata) > realdata_subsample:
        realdata = realdata.sample(realdata_subsample, random_state=random_state)
        realdata.reset_index(drop=True, inplace=True)

    if fakedata_subsample is not None and len(fakedata) > fakedata_subsample:
        fakedata = fakedata.sample(fakedata_subsample, random_state=random_state)
        fakedata.reset_index(drop=True, inplace=True)

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns.to_list()

    if fit_data is not None:
        transformer = DataTransformer(
            discrete_encode="onehot",
            numerical_preprocess="minmax",
        )
        transformer.fit(fit_data, discrete_columns=cat_cols)
        realdata = transformer.transform(realdata)
        fakedata = transformer.transform(fakedata)

    elif transformer is not None:
        realdata = transformer.transform(realdata)
        fakedata = transformer.transform(fakedata)

    else:
        transformer = DataTransformer(
            discrete_encode="onehot",
            numerical_preprocess="minmax",
        )
        transformer.fit(realdata, discrete_columns=cat_cols)
        realdata = transformer.transform(realdata)
        fakedata = transformer.transform(fakedata)

    if kind == "nearest_neighbors":
        # fit on fakedata
        neigh = NearestNeighbors(n_neighbors=neighbors)
        neigh.fit(realdata)

        # compute the distance of each fakedata sample to realdata
        result, _ = neigh.kneighbors(fakedata, return_distance=True)

    else:
        raise NotImplementedError()

    if return_mean:
        result = result.mean()
    else:
        return_dataframe = False

    if return_dataframe:
        result = {"score": result, "metric": __name__, "normalized_score": result}
        result = pd.DataFrame([result])
        result["neighbors"] = neighbors

    return result


def nearest_neighbors_wrapper(
    realdata,
    fakedata,
    joint_metric,
    cat_cols=None,
    neighbors=None,
    realdata_subsample=None,
    fakedata_subsample=None,
    transformer=None,
    fit_data=None,
    random_state=1000,
):
    """Nearest Neighbors Likelihood or Closeness Approximation Wrapper

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        joint_metric (str):
            Joint metric to compute
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        neighbors (int, optional):
            Number of neigbors to calculate distance to.
            Defaults to `None`.
        realdata_subsample (int, optional):
            Number of real datapoint to evaluate. Defaults to None.
        fakedata_subsample (int, optional):
            Number of fake datapoint to evaluate. Defaults to None.
        transformer (object, optional):
            Transformer object to apply on columns. Defaults to None.
        fit_data (pd.DataFrame, optional):
            Data to fit the data transformer on. Defaults to `None`.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
    """
    if neighbors is None:
        results = joint_metric(
            realdata=realdata,
            fakedata=fakedata,
            cat_cols=cat_cols,
            return_mean=True,
            return_dataframe=True,
            realdata_subsample=realdata_subsample,
            fakedata_subsample=fakedata_subsample,
            transformer=transformer,
            fit_data=fit_data,
            kind="nearest_neighbors",
            neighbors=1,
            random_state=random_state,
        )
    else:

        results = joblib.Parallel(prefer="threads", n_jobs=N_JOBS)(
            joblib.delayed(joint_metric)(
                realdata=realdata,
                fakedata=fakedata,
                cat_cols=cat_cols,
                return_mean=True,
                return_dataframe=True,
                realdata_subsample=realdata_subsample,
                fakedata_subsample=fakedata_subsample,
                transformer=transformer,
                fit_data=fit_data,
                kind="nearest_neighbors",
                neighbors=neighbor,
                random_state=random_state,
            )
            for neighbor in neighbors
        )
        results = pd.concat(results)
        results.reset_index(drop=True, inplace=True)

    return results
