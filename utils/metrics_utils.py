import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler, OneHotEncoder

def normalize_score(raw_score, min_value, max_value, goal="maximize"):
    """Compute the normalized value of `raw_score`

    Args:
        raw_score (float):
            The value to be computed.
        min_value (str or int):
            The mininum value that can be attained.
        max_value (_type_):
            The maximum value that can be attained.
        goal (str, optional):
            Whether to minimize or maximize. Defaults to "maximize".

    Raises:
        ValueError:
            If `raw score` > `max_value` or `min_value` > `raw_score`.
        AssertionError:
            If normalized `score` is None or `score` > 1 or `score` < 0.

    Returns:
        float:
            The normalized value of the `raw_score`
    """

    min_value = float(min_value)
    max_value = float(max_value)

    if max_value < raw_score or min_value > raw_score:
        raise ValueError("`raw_score` must be between `min_value` and `max_value`.")

    is_min_finite = min_value not in (float("-inf"), float("inf"))
    is_max_finite = max_value not in (float("-inf"), float("inf"))

    score = None
    if is_min_finite and is_max_finite:
        score = (raw_score - min_value) / (max_value - min_value)

    elif not is_min_finite and is_max_finite:
        score = np.exp(raw_score - max_value)

    elif is_min_finite and not is_max_finite:
        score = 1.0 - np.exp(min_value - raw_score)

    else:
        score = 1 / (1 + np.exp(-raw_score))

    if score is None or score < 0 or score > 1:
        raise AssertionError(
            f"This should be unreachable. The score {score} should be a value between 0 and 1."
        )

    if goal == "minimize":
        return 1.0 - score

    return score


def make_size_equal(data_a, data_b, random_state=1000):
    """Make `data_a` and `data_b` size equal

    Args:
        data_a (Union(pd.DataFrame, pd.Series)):
            data_a to evaluate.
        data_b (Union(pd.DataFrame, pd.Series)):
            data_b to evaluate.
        random_state (int):
            Set random state numer to ensure reproducibility.
            Only used if len(data_a) != len(data_b).
            Defaults to `1000`.

    Returns:
        Tuple of (data_a, data_b)
    """

    if len(data_a) < len(data_b):
        data_b = data_b.sample(len(data_a), random_state=random_state).reset_index(
            drop=True
        )
    elif len(data_a) > len(data_b):
        data_a = data_a.sample(len(data_b), random_state=random_state).reset_index(
            drop=True
        )
    else:
        assert len(data_a) == len(data_b)

    return data_a, data_b

def histogram_binning(data, bins=50, categorial=False):
    """Compute the histogram of `data`.
        This function first calculates a
        monotonically increasing array of bin edges,
        if `type(bins)==int`, then computes the histogram of `data`.

    Args:
        data (array_like):
            Data to evaluate
        bins (int or array-like, optional):
            Defines the number of equal-width bins in the given range.
            Defaults to 50. If array-like, must be monotonically increasing.
        categorial (bool, optional):
            Whether or not `data` is categorial. Defaults to `False`.
            If `categorial=False`, and `type(bins) == int`, `data` range should be between 0 and 1.

    Returns:
        hist (array):
            The normalized count of samples in each bin.
        bin_edges (array):
            Bin edges (length(hist)+1)
    """

    if type(bins) == int:

        if categorial:
            bin_edges = np.arange(0, bins + 1)
        else:
            bin_edges = np.linspace(0, 1, bins)
    else:
        bin_edges = bins

    hist, bin_edges = np.histogram(data, bins=bin_edges)

    if sum(hist) == 0:
        hist = np.zeros_like(hist)

    else:
        hist = hist / sum(hist)

    return hist, bin_edges

def column_transformer(real_col, fake_col, kind="minmax", fit_data=None):
    """Column Transformer

    Args:
        real_col (array-like):
            Real data column to evaluate
        fake_col (array-like):
            Fake data column to evaluate
        kind (str, optional):
            Kind of transformer. Defaults to "minmax".
            Must be one of (`minmax`, `label`, `onehot`)
        fit_data (array-like, optional):
            Data to fit the column transformer on. Defaults to `None`. 
            Fits the column transformer on `realdata` for numerical columns
            and `realdata+fakedata` for categorical columns.

    Returns:
        real_col (array-like):
            Transformed real data column
        fake_col (array-like):
            Transformed fake data column
        transformer (array-like):
            Transformer object
    """

    if kind == "minmax":
        transformer = MinMaxScaler()
        
        if fit_data is not None:
            transformer.fit(fit_data[:, None])
        else:
            # compute statistics on real data
            transformer.fit(real_col[:, None])

        # rescale real and fake data
        real_col = transformer.transform(real_col[:, None]).squeeze()
        fake_col = transformer.transform(fake_col[:, None]).squeeze()

    if kind == "label":
        transformer = LabelEncoder()
        
        real_col = real_col.astype("str")
        fake_col = fake_col.astype("str")
        
        if fit_data is not None:
            fit_data = fit_data.astype("str")
            transformer.fit(fit_data)
        else:
            # fit encoder on concatenated real and fake data
            real_fake = np.concatenate([real_col, fake_col])
            transformer.fit(real_fake)
        
        # encode real and fake data
        real_col = transformer.transform(real_col)
        fake_col = transformer.transform(fake_col)

    if kind == "onehot":
        transformer = OneHotEncoder(sparse=False)
        
        if fit_data is not None:
            transformer.fit(fit_data[:, None])
        else:
            # fit encoder on concatenated real and fake data
            real_fake = np.concatenate([real_col, fake_col])
            transformer.fit(real_fake[:, None])

        # encode real and fake data
        real_col = transformer.transform(real_col[:, None])
        fake_col = transformer.transform(fake_col[:, None])

    return real_col, fake_col, transformer