import dython.nominal as dn
import joblib
import scipy
import numpy as np
import pandas as pd

import sys

sys.path.append("..")
from utils.metrics_utils import (
    make_size_equal,
    column_transformer,
    histogram_binning,
    normalize_score,
)

__all__ = [
    "histogram_intersection",
    "jaccard_similarity",
    "column_correlation",
    "wasserstein_distance",
    "jensonshannon_distance",
    "cumulative_difference",
    "column_metric_wrapper",
]

N_JOBS = -1


def histogram_intersection(
    realdata,
    fakedata,
    column_name=None,
    categorial=False,
    bins=50,
    return_dataframe=False,
    keep_default_size=True,
    fit_data=None,
    random_state=1000,
):
    """This is a column-wise metric.

    Calculates the amount of overlap between two histograms.

    Args:
        realdata (pd.DataFrame or array-like):
            Realdata to evaluate.
        fakedata (pd.DataFrame or array-like):
            Fakedata to evaluate.
        column_name (str, optional):
            Column name to evaluate. Defaults to `None`.
            If `None`, expects `realdata` and `fakedata` to be 1D array.
        categorial (bool, optional):
            Whether or not column is categorical.
            Defaults to `False`.
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe.
            Only applicable if `column_name` is not `None`.
            Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
            If `False`, `realdata` and `fakedata` will have equal size.
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
    """

    __name__ = "histogram_intersection"

    if column_name is not None:
        real_col = realdata[column_name].values
        fake_col = fakedata[column_name].values

    else:
        assert len(realdata.squeeze().shape) == 1 and len(fakedata.squeeze().shape) == 1
        real_col = realdata.values if type(realdata) == pd.Series else realdata
        fake_col = fakedata.values if type(fakedata) == pd.Series else fakedata

    if fit_data is not None and column_name is not None:
        fit_data = fit_data[column_name].values
    elif fit_data is not None:
        assert len(fit_data.squeeze().shape) == 1
        fit_data = fit_data.values if type(fit_data) == pd.Series else fit_data

    if not keep_default_size:
        real_col, fake_col = make_size_equal(
            pd.Series(real_col), pd.Series(fake_col), random_state
        )
        real_col, fake_col = real_col.values, fake_col.values

    if categorial:
        real_col, fake_col, transformer = column_transformer(
            real_col, fake_col, kind="label", fit_data=fit_data
        )
        bins = len(transformer.classes_)
    else:
        real_col, fake_col, _ = column_transformer(
            real_col, fake_col, kind="minmax", fit_data=fit_data
        )

    hist_real, bins = histogram_binning(real_col, bins=bins, categorial=categorial)

    hist_fake, _ = histogram_binning(
        fake_col,
        bins=bins,
        categorial=categorial,
    )

    result = np.minimum(hist_real, hist_fake).sum()

    if return_dataframe and column_name is not None:
        column_type = "categorical" if categorial else "numerical"
        result = {
            "bins": len(bins) if type(bins) == np.array or list else bins,
            "column_name": column_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": result,
            "metric": __name__,
        }
        result = pd.DataFrame([result])

    return result


def jaccard_similarity(
    realdata,
    fakedata,
    column_name=None,
    categorial=False,
    bins=50,
    return_dataframe=False,
    keep_default_size=True,
    fit_data=None,
    random_state=1000,
):
    """This is a column-wise metric.

    Similar to histogram_intersection but computes minima over maxima.

    Args:
        realdata (pd.DataFrame or array-like):
            Realdata to evaluate.
        fakedata (pd.DataFrame or array-like):
            Fakedata to evaluate.
        column_name (str, optional):
            Column name to evaluate. Defaults to `None`.
            If `None`, expects `realdata` and `fakedata` to be 1D array.
        categorial (bool, optional):
            Whether or not column is categorical.
            Defaults to `False`.
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe.
            Only applicable if `column_name` is not `None`.
            Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
            If `False`, `realdata` and `fakedata` will have equal size.
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
    """

    __name__ = "jaccard_similarity"

    if column_name is not None:
        real_col = realdata[column_name].values
        fake_col = fakedata[column_name].values

    else:
        assert len(realdata.squeeze().shape) == 1 and len(fakedata.squeeze().shape) == 1
        real_col = realdata.values if type(realdata) == pd.Series else realdata
        fake_col = fakedata.values if type(fakedata) == pd.Series else fakedata

    if fit_data is not None and column_name is not None:
        fit_data = fit_data[column_name].values
    elif fit_data is not None:
        assert len(fit_data.squeeze().shape) == 1
        fit_data = fit_data.values if type(fit_data) == pd.Series else fit_data

    if not keep_default_size:
        real_col, fake_col = make_size_equal(
            pd.Series(real_col), pd.Series(fake_col), random_state
        )
        real_col, fake_col = real_col.values, fake_col.values

    if categorial:
        real_col, fake_col, transformer = column_transformer(
            real_col, fake_col, kind="label", fit_data=fit_data
        )
        bins = len(transformer.classes_)
    else:
        real_col, fake_col, _ = column_transformer(
            real_col, fake_col, kind="minmax", fit_data=fit_data
        )

    hist_real, bins = histogram_binning(real_col, bins=bins, categorial=categorial)

    hist_fake, _ = histogram_binning(
        fake_col,
        bins=bins,
        categorial=categorial,
    )

    minima = np.minimum(hist_real, hist_fake).sum()
    maxima = np.maximum(hist_real, hist_fake).sum()
    result = minima / maxima

    if return_dataframe and column_name is not None:
        column_type = "categorical" if categorial else "numerical"
        result = {
            "bins": len(bins) if type(bins) == np.array or list else bins,
            "column_name": column_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": result,
            "metric": __name__,
        }
        result = pd.DataFrame([result])

    return result


def column_correlation(
    realdata,
    fakedata,
    column_name=None,
    categorial=False,
    theils_u=False,
    return_dataframe=False,
    keep_default_size=False,
    random_state=1000,
):
    """This is a column-wise metric.

    Calculates the wasserstein distance between the two distributions.
    The probability distribution of continuous column is estimated
    using normalized histogram count.

    Args:
        realdata (pd.DataFrame or array-like):
            Realdata to evaluate.
        fakedata (pd.DataFrame or array-like):
            Fakedata to evaluate.
        column_name (str, optional):
            Column name to evaluate. Defaults to `None`.
            If `None`, expects `realdata` and `fakedata` to be 1D array.
        categorial (bool, optional):
            Whether or not column is categorical.
            Defaults to `False`.
        theils_u (bool, optional):
            Whether or not to use Theil's U statistics.
            Defaults to False.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe.
            Only applicable if `column_name` is not `None`.
            Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
            If `False`, `realdata` and `fakedata` will have equal size.
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.

        Theil's U statistics is asymmetric i.e U(realdata, fakedata) != U(fakedata, realdata).
        Cramer's V statistics is symmetric but has some limitation when dataset is small or skewed.
    """

    __name__ = "column_correlation"

    if column_name is not None:
        real_col = realdata[column_name]
        fake_col = fakedata[column_name]

    else:
        assert len(realdata.squeeze().shape) == 1 and len(fakedata.squeeze().shape) == 1
        real_col = pd.Series(realdata)
        fake_col = pd.Series(fakedata)

    if not keep_default_size:
        real_col, fake_col = make_size_equal(real_col, fake_col, random_state)

    # sort real and fake column data
    real_col = real_col.sort_values()
    fake_col = fake_col.sort_values()

    if categorial:
        if theils_u:
            corr_metric = "theils_u"
            result = dn.theils_u(real_col, fake_col)
            normalized_result = result
        else:
            corr_metric = "cramers_v"
            result = dn.cramers_v(real_col, fake_col, bias_correction=True)
            normalized_result = result
    else:
        corr_metric = "pearson"
        result, _ = scipy.stats.pearsonr(real_col, fake_col)
        normalized_result = normalize_score(
            raw_score=result, min_value=-1.0, max_value=1.0, goal="maximize"
        )

    if return_dataframe and column_name is not None:
        column_type = "categorical" if categorial else "numerical"
        result = {
            "correlation_metric": corr_metric,
            "column_name": column_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": normalized_result,
            "metric": __name__,
        }
        result = pd.DataFrame([result])

    return result


def wasserstein_distance(
    realdata,
    fakedata,
    column_name=None,
    categorial=False,
    bins=50,
    return_dataframe=False,
    keep_default_size=True,
    fit_data=None,
    random_state=1000,
):
    """This is a column-wise metric.

    Calculates the wasserstein distance between the two distributions.
    The probability distribution of continuous column is estimated
    using normalized histogram count.

    Args:
        realdata (pd.DataFrame or array-like):
            Realdata to evaluate.
        fakedata (pd.DataFrame or array-like):
            Fakedata to evaluate.
        column_name (str, optional):
            Column name to evaluate. Defaults to `None`.
            If `None`, expects `realdata` and `fakedata` to be 1D array.
        categorial (bool, optional):
            Whether or not column is categorical.
            Defaults to `False`.
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe.
            Only applicable if `column_name` is not `None`.
            Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
            If `False`, `realdata` and `fakedata` will have equal size.
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
    """

    __name__ = "wasserstein_distance"

    if column_name is not None:
        real_col = realdata[column_name].values
        fake_col = fakedata[column_name].values

    else:
        assert len(realdata.squeeze().shape) == 1 and len(fakedata.squeeze().shape) == 1
        real_col = realdata.values if type(realdata) == pd.Series else realdata
        fake_col = fakedata.values if type(fakedata) == pd.Series else fakedata

    if fit_data is not None and column_name is not None:
        fit_data = fit_data[column_name].values
    elif fit_data is not None:
        assert len(fit_data.squeeze().shape) == 1
        fit_data = fit_data.values if type(fit_data) == pd.Series else fit_data

    if not keep_default_size:
        real_col, fake_col = make_size_equal(
            pd.Series(real_col), pd.Series(fake_col), random_state
        )
        real_col, fake_col = real_col.values, fake_col.values

    if categorial:
        real_col, fake_col, transformer = column_transformer(
            real_col, fake_col, kind="label", fit_data=fit_data
        )
        bins = len(transformer.classes_)
    else:
        real_col, fake_col, _ = column_transformer(
            real_col, fake_col, kind="minmax", fit_data=fit_data
        )

    u_weights, bins = histogram_binning(real_col, bins=bins, categorial=categorial)

    v_weights, _ = histogram_binning(
        fake_col,
        bins=bins,
        categorial=categorial,
    )

    result = scipy.stats.wasserstein_distance(
        u_values=bins[:-1],
        v_values=bins[:-1],
        u_weights=u_weights,
        v_weights=v_weights,
    )

    normalized_result = normalize_score(
        raw_score=result, min_value=0, max_value=np.inf, goal="maximize"
    )

    if return_dataframe and column_name is not None:
        column_type = "categorical" if categorial else "numerical"
        result = {
            "bins": len(bins) if type(bins) == np.array or list else bins,
            "column_name": column_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": normalized_result,
            "metric": __name__,
        }
        result = pd.DataFrame([result])

    return result


def jensonshannon_distance(
    realdata,
    fakedata,
    column_name=None,
    categorial=False,
    bins=50,
    return_dataframe=False,
    keep_default_size=True,
    fit_data=None,
    random_state=1000,
):
    """This is a column-wise metric.

    Jenson Shannon Distance

    Args:
        realdata (pd.DataFrame or array-like):
            Realdata to evaluate.
        fakedata (pd.DataFrame or array-like):
            Fakedata to evaluate.
        column_name (str, optional):
            Column name to evaluate. Defaults to `None`.
            If `None`, expects `realdata` and `fakedata` to be 1D array.
        categorial (bool, optional):
            Whether or not column is categorical.
            Defaults to `False`.
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe.
            Only applicable if `column_name` is not `None`.
            Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
            If `False`, `realdata` and `fakedata` will have equal size.
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
        random_state (int, optional):
             Random state number for reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
    """

    __name__ = "jensonshannon_distance"

    if column_name is not None:
        real_col = realdata[column_name].values
        fake_col = fakedata[column_name].values

    else:
        assert len(realdata.squeeze().shape) == 1 and len(fakedata.squeeze().shape) == 1
        real_col = realdata.values if type(realdata) == pd.Series else realdata
        fake_col = fakedata.values if type(fakedata) == pd.Series else fakedata

    if fit_data is not None and column_name is not None:
        fit_data = fit_data[column_name].values
    elif fit_data is not None:
        assert len(fit_data.squeeze().shape) == 1
        fit_data = fit_data.values if type(fit_data) == pd.Series else fit_data

    if not keep_default_size:
        real_col, fake_col = make_size_equal(
            pd.Series(real_col), pd.Series(fake_col), random_state
        )
        real_col, fake_col = real_col.values, fake_col.values

    if categorial:
        real_col, fake_col, transformer = column_transformer(
            real_col, fake_col, kind="label", fit_data=fit_data
        )
        bins = len(transformer.classes_)
    else:
        real_col, fake_col, _ = column_transformer(
            real_col, fake_col, kind="minmax", fit_data=fit_data
        )

    hist_real, bins = histogram_binning(real_col, bins=bins, categorial=categorial)

    hist_fake, _ = histogram_binning(
        fake_col,
        bins=bins,
        categorial=categorial,
    )

    result = scipy.spatial.distance.jensenshannon(hist_real, hist_fake, base=2)

    if return_dataframe and column_name is not None:
        column_type = "categorical" if categorial else "numerical"
        result = {
            "bins": len(bins) if type(bins) == np.array or list else bins,
            "column_name": column_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": result,
            "metric": __name__,
        }
        result = pd.DataFrame([result])

    return result


def cumulative_difference(
    realdata,
    fakedata,
    column_name=None,
    categorial=False,
    bins=50,
    return_dataframe=False,
    keep_default_size=False,
    fit_data=None,
    random_state=1000,
):
    """This is a column-wise metric.

    Calculates the cummulative difference.

    Args:
        realdata (pd.DataFrame or array-like):
            Realdata to evaluate.
        fakedata (pd.DataFrame or array-like):
            Fakedata to evaluate.
        column_name (str, optional):
            Column name to evaluate. Defaults to `None`.
            If `None`, expects `realdata` and `fakedata` to be 1D array.
        categorial (bool, optional):
            Whether or not column is categorical.
            Defaults to `False`.
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe.
            Only applicable if `column_name` is not `None`.
            Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
            If `False`, `realdata` and `fakedata` will have equal size.
        fit_data (pd.DataFrame, optional):
            Data to fit the column transformer on. Defaults to `None`.
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.
    """

    __name__ = "cumulative_difference"

    if column_name is not None:
        real_col = realdata[column_name].values
        fake_col = fakedata[column_name].values

    else:
        assert len(realdata.squeeze().shape) == 1 and len(fakedata.squeeze().shape) == 1
        real_col = realdata.values if type(realdata) == pd.Series else realdata
        fake_col = fakedata.values if type(fakedata) == pd.Series else fakedata

    if fit_data is not None and column_name is not None:
        fit_data = fit_data[column_name].values
    elif fit_data is not None:
        assert len(fit_data.squeeze().shape) == 1
        fit_data = fit_data.values if type(fit_data) == pd.Series else fit_data

    if not keep_default_size:
        real_col, fake_col = make_size_equal(
            pd.Series(real_col), pd.Series(fake_col), random_state
        )
        real_col, fake_col = real_col.values, fake_col.values

    if categorial:
        real_col, fake_col, transformer = column_transformer(
            real_col, fake_col, kind="label", fit_data=fit_data
        )
        bins = len(transformer.classes_)
    else:
        real_col, fake_col, _ = column_transformer(
            real_col, fake_col, kind="minmax", fit_data=fit_data
        )

    hist_real, bins = histogram_binning(real_col, bins=bins, categorial=categorial)

    hist_fake, _ = histogram_binning(
        fake_col,
        bins=bins,
        categorial=categorial,
    )
    cumdist_real = np.cumsum(hist_real) * (bins[1] - bins[0])
    cumdist_fake = np.cumsum(hist_fake) * (bins[1] - bins[0])

    result = sum(cumdist_real - cumdist_fake)

    if return_dataframe and column_name is not None:
        column_type = "categorical" if categorial else "numerical"
        result = {
            "bins": len(bins) if type(bins) == np.array or list else bins,
            "column_name": column_name,
            "column_type": column_type,
            "score": result,
            "normalized_score": result,
            "metric": __name__,
        }
        result = pd.DataFrame([result])

    return result


def column_metric_wrapper(
    realdata, fakedata, column_metric, cat_cols=None, random_state=1000
):
    """Column Metric Wrapper

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        column_metric (func):
            Column metric to apply
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
            If `None`, infers categorical columns from `realdata`
        random_state (int, optional):
            Random state number for reproducibility.
            Defaults to `1000`.

    Returns:
        result_df (pd.DataFrame):
            Result of `metric` applied on each column.

    """

    sorted_real_columns = sorted(realdata.columns.tolist())
    sorted_fake_columns = sorted(fakedata.columns.tolist())

    assert sorted_real_columns == sorted_fake_columns

    realdata = realdata[sorted_real_columns]
    fakedata = fakedata[sorted_fake_columns]

    real_iter = realdata.iteritems()
    fake_iter = fakedata.iteritems()

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns

    results = joblib.Parallel(prefer="threads", n_jobs=N_JOBS)(
        joblib.delayed(column_metric)(
            realdata=real_col.to_frame(),
            fakedata=fake_col.to_frame(),
            column_name=column_name,
            categorial=True if column_name in cat_cols else False,
            random_state=random_state,
            return_dataframe=True,
        )
        for (column_name, real_col), (_, fake_col) in zip(real_iter, fake_iter)
    )
    result_df = pd.concat(results)

    return result_df.reset_index(drop=True)
