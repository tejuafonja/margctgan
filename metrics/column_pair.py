import dython.nominal as dn
import pandas as pd

import sys

sys.path.append("..")
from utils.metrics_utils import make_size_equal

__all__ = ["associations_difference"]


def associations_difference(
    realdata,
    fakedata,
    cat_cols=None,
    nom_nom_assoc="cramer",
    mean_column_difference=True,
    return_dataframe=False,
    keep_default_size=False,
    random_state=1000,
):
    """Computes the column-pair association matrix difference between `realdata` and `fakedata`.
        Correlation Metrics:
            Numerical-Numerical: `pearson correlation`
            Numerical-Categorical: `correlation ration`
            Categorical-Categorical: `cramer` or `theil`

    Args:
        realdata (pd.DataFrame):
            Realdata to evaluate
        fakedata (pd.DataFrame):
            Fakedata to evaluate
        cat_cols (array-like, optional):
            List of categorical columns. Defaults to None.
        nom_nom_assoc (str, optional):
            Categorical metric to use. Defaults to "cramer".
            Must be one of (`cramer`, `theil`).
        mean_column_difference (bool, optional):
            Whether of not to return correlation difference mean across each column.
            Defaults to `True`.
        return_dataframe (bool, optional):
            Whether or not to return result as pandas dataframe. Defaults to `False`.
        keep_default_size (bool, optional):
            Whether or not to keep default size.
                If `False`, `realdata` and `fakedata` will have equal size.
        random_state (int, optional):
            Set random state number to ensure reproducibility.
            Only applicable if `keep_default_size` is `False`.
            Defaults to `1000`.

    Returns:
        pd.DataFrame or float:
            pd.DataFrame if `mean_column_difference=True`
    """
    __name__ = "associations_difference"

    assert isinstance(realdata, pd.DataFrame)
    assert isinstance(fakedata, pd.DataFrame)

    if not keep_default_size:
        realdata, fakedata = make_size_equal(realdata, fakedata, random_state)

    if cat_cols is None:
        cat_cols = realdata.select_dtypes(exclude=["number"]).columns.to_list()

    real_corr = dn.associations(
        dataset=realdata,
        nominal_columns=cat_cols,
        mark_columns=False,
        nom_nom_assoc=nom_nom_assoc,
        annot=False,
        compute_only=True,
    )["corr"]

    fake_corr = dn.associations(
        dataset=fakedata,
        nominal_columns=cat_cols,
        mark_columns=False,
        nom_nom_assoc=nom_nom_assoc,
        annot=False,
        compute_only=True,
    )["corr"]

    if mean_column_difference:
        result = (
            abs(real_corr - fake_corr)
            .mean()
            .to_frame()
            .reset_index()
            .rename(columns={"index": "column_name", 0: "score"})
        )
        result.loc[:, "metric"] = __name__
        result.loc[:, "normalized_score"] = result["score"]
        result.loc[:, "column_type"] = [
            "categorical" if col in cat_cols else "numerical"
            for col in result["column_name"]
        ]
        return_dataframe = False

    else:
        result = abs(real_corr - fake_corr).mean().mean()

    if return_dataframe:
        result = {"score": result, "normalized_score": result, "metric": __name__}
        result = pd.DataFrame([result])

    return result
