import logging
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def get_datasets(names_only: bool = False):
    DATASET_DICT = {
        "adult": load_adult,
        "census": load_census,
        "census_v2": load_census_v2,
        "texas": load_texas,
        "news": load_news,
        "adult_old": load_adult_old,
    }

    if names_only:
        return list(DATASET_DICT.keys())
    else:
        return DATASET_DICT


class Dataset(object):
    """Dataset Loader."""

    def __init__(
        self,
        dataset_name,
        dataset_dir="../data",
        subset=None,
        transformer=None,
        target=None,
        data_frac=None,
        random_state=1000,
        return_filtered_cols=False,
        handle_discrete_as="categorial",
    ):
        """Create a dataset loader

        Args:
            dataset_name (str):
                Dataset name. Must be one of `get_datasets(names_only=True)`
            dataset_dir (str):
                Dataset directory. Defaults to `../data`
            subset (str, optional):
                Dataset subset. Must be one of (`train`, `test`, `demo`). Defaults to None.
            transformer (callable, optional):
                Data transformer. Defaults to None.
            target (str, optional):
                Target column name. Defaults to None.
            data_frac (Union[int, float], optional):
                Fraction (size) of dataset to load. Defaults to None.
            random_state (int, optional):
                Random seed for reproducibility. Defaults to 1000.
            return_filtered_cols (bool, optional): whether or not to return `filtered_cols_list`.
                                        Defaults to False which returns all columns.
            handle_discrete_as (str, optional): specifies a methodology to combine discrete columns if it exists.
                                            Must be one of "numeric" or "categorial".
                                            Defaults to "categorial".
        """
        self._random_state = random_state
        self._transformer = transformer
        self._target = target
        self._holdout_data = None

        (
            self.data,
            self.target_name,
            self.cat_cols,
            self.num_cols,
            self.metadata,
        ) = self._load_data(
            dataset_name,
            dataset_dir=dataset_dir,
            subset=subset,
            return_filtered_cols=return_filtered_cols,
            handle_discrete_as=handle_discrete_as,
        )

        self._column_names = self.data.columns

        if transformer is not None:
            if self._transformer._target is not None and target is not None:
                assert (
                    self._transformer._target == self._target
                ), f"`target={self._target}` in {self.__class__} must be the same with `target={self._transformer._target}` in {self._transformer.__class__}!"
            elif self._transformer._target is None and self._target is not None:
                self._transformer._target = self._target
            else:
                self._target = self._transformer._target

        if data_frac is not None:
            assert (data_frac > 0 and data_frac < len(self.data)) or (
                data_frac > 0 and data_frac <= 1.0
            ), f"data_frac={data_frac} should be either positive and smaller than the number of samples {len(self.data)} or a float in the (0, 1) range"
            self._holdout_data, self.data = train_test_split(
                self.data,
                stratify=self.data[self._target]
                if self._target and self._target in self.cat_cols
                else None,
                test_size=data_frac,
                random_state=self._random_state,
            )

            self.data.reset_index(inplace=True, drop=True)
            self.metadata = get_metadata(self.data, self.cat_cols)

        if transformer is not None:
            self._transformer.fit(self.data, discrete_columns=self.cat_cols)

        self.data, self.label = self.__data_label_split(self.data)
        self.train_data = (self.data, self.label)

    def __data_label_split(self, data):
        if self._transformer is not None:
            data = self._transformer.transform(data)
            self.metadata = get_metadata(data)
            if self._target is not None:
                label = data[:, self._transformer._target_index]
                data = np.delete(data, self._transformer._target_index, axis=1)
            else:
                label = pd.DataFrame([]) if type(data) == pd.DataFrame else np.array([])
        elif self._target is not None:
            label = data.loc[:, self._target]
            data = data.drop(columns=self._target)
        else:
            label = pd.DataFrame([]) if type(data) == pd.DataFrame else np.array([])
        return data, label

    def _load_data(
        self,
        dataset_name,
        dataset_dir,
        subset,
        return_filtered_cols,
        handle_discrete_as,
    ):
        """Load the requested dataset from the lists of datasets.

        Args:
            dataset_name (str):
                Name of the dataset (``{dataset_name}.csv`` ).
             dataset_dir (str):
                Dataset directory. Defaults to `../data`
            subset (str):
                Dataset subset to load. Must be one of (``train``, ``test``, ``demo``).
            return_filtered_cols (bool, optional):
                Whether or not to return `filtered_cols_list`.
                Defaults to False which returns all columns.
            handle_discrete_as (str, optional):
                Specifies a methodology to combine discrete columns if it exists.
                Must be one of ("numeric" or "categorial"). Defaults to "categorial".

        Returns:
            tuple:
                df (pandas.DataFrame), target_name (str), cat_cols(list), num_cols(list), metadata(dict)
        """

        logging.debug(f"Loading {dataset_name} ...")

        dataset_dict = get_datasets()

        if dataset_name in dataset_dict.keys():
            func = dataset_dict[dataset_name]
            data, target_name, cat_cols, num_cols, metadata = func(
                subset=subset,
                dataset_dir=dataset_dir,
                return_filtered_cols=return_filtered_cols,
                handle_discrete_as=handle_discrete_as,
            )
        else:
            logging.error(f"{dataset_name} dataset not found.")
            raise ValueError(f"{dataset_name} dataset not found.")

        return data, target_name, cat_cols, num_cols, metadata

    def get_dim(self):
        x_dim = self.data.shape[-1]

        if len(self.label) != 0 and self._target in self.cat_cols:
            if type(self.data) == pd.DataFrame:
                y_dim = len(np.unique(self.label.to_numpy()))
            else:
                y_dim = len(np.unique(self.label))
        elif len(self.label) != 0 and self._target in self.num_cols:
            y_dim = 1
        else:
            y_dim = 0

        return x_dim, y_dim

    def train_test(self, test_size=0.2):
        if len(self.label) == 0:
            train_x, test_x = train_test_split(
                self.data, test_size=test_size, random_state=self._random_state
            )
            train_y, test_y = self.label, self.label
        else:
            train_x, test_x, train_y, test_y = train_test_split(
                self.data,
                self.label,
                test_size=test_size,
                random_state=self._random_state,
            )

        if type(self.data) == pd.DataFrame:
            train_x.reset_index(inplace=True, drop=True)
            test_x.reset_index(inplace=True, drop=True)

            if len(self.label) != 0:
                train_y.reset_index(inplace=True, drop=True)
                test_y.reset_index(inplace=True, drop=True)

        self.data = train_x
        self.label = train_y

        return train_x, train_y, test_x, test_y

    def test_subset(self, test_size=None):
        """Create test set from holdout dataset."""
        assert (
            self._holdout_data is not None
        ), "There's no holdout data. Use the train_test() method instead."

        if test_size is not None:
            self._holdout_data, test_data = train_test_split(
                self._holdout_data,
                test_size=test_size,
                random_state=self._random_state,
            )
            test_data.reset_index(inplace=True, drop=True)
        else:
            test_data = self._holdout_data
            self._holdout_data = None

        test_x, test_y = self.__data_label_split(test_data)
        return test_x, test_y

    def _inverse_transform(self, data, force=False):
        assert self._transformer is not None, "Data transformer has not been enabled."
        if self._target is not None and not force:
            assert type(data) == tuple, "Expecting data in the form of (data_x, data_y)"
            data_x, data_y = data
            data = np.insert(data_x, self._transformer._target_index, data_y, axis=1)
        return self._transformer.inverse_transform(data)

    def __getitem__(self, index):
        if len(self.label) == 0:
            label = 0
            if type(self.data) == pd.DataFrame:
                data = self.data.iloc[index]
            else:
                data = self.data[index].astype(np.float32)
        elif type(self.data) == pd.DataFrame:
            data = self.data.iloc[index]
            label = self.label.iloc[index]
        else:
            label = self.label[index]
            data = self.data[index].astype(np.float32)

        return data, label

    def __len__(self):
        return len(self.data)


def load_adult(
    dataset_dir,
    subset=None,
    return_filtered_cols=False,
    handle_discrete_as="categorial",
    **kws,
):
    """Load Adult Dataset

    Args:
        dataset_dir (str):
            Directory to load dataset from.
        subset (str, optional):
            Subset of the dataset to load. Defaults to None.
        return_filtered_cols (bool, optional):
            Whether or not to return `filtered_cols_list`.
            Defaults to False which returns all columns.
        handle_discrete_as (str, optional):
            Specifies a methodology to combine discrete columns if it exists.
            Must be one of "numeric" or "categorial". Defaults to "categorial".

    Raises:
        NotImplementedError:
            If `handle_discrete_as` not in ("numeric", "categorial")

    Returns:
        tuple: (df, target, cat_cols, num_cols, metadata)
    """
    if subset is not None:
        assert subset in (
            "train",
            "test",
            "demo",
        ), "unrecognized subset. Must be one of (``train``, ``test``, ``demo``)"
        dset_path = os.path.join(dataset_dir, f"adult/{subset}.csv")
    else:
        dset_path = os.path.join(dataset_dir, "adult/adult.csv")

    cat_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "income",
    ]
    num_cols = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    discrete_cols = []
    filtered_cols_list = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "sex",
        "race",
        "native-country",
        "age",
        "capital-gain",
        "capital-loss",
        "income",
    ]

    target = "income"

    df = pd.read_csv(dset_path, sep=",", index_col=None, **kws)

    df, cat_cols, num_cols = preprocess_dataframe(
        dataframe=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        discrete_cols=discrete_cols,
        target=target,
        filtered_cols_list=filtered_cols_list if return_filtered_cols else [],
        handle_discrete_as=handle_discrete_as,
    )
    metadata = get_metadata(df, cat_cols)
    return df, target, cat_cols, num_cols, metadata


def load_census(
    dataset_dir,
    subset=None,
    return_filtered_cols=False,
    handle_discrete_as="categorial",
    **kws,
):
    """Load Census Dataset

    Args:
        dataset_dir (str):
            Directory to load dataset from.
        subset (str, optional):
            Subset of the dataset to load. Defaults to None.
        return_filtered_cols (bool, optional):
            Whether or not to return `filtered_cols_list`.
            Defaults to False which returns all columns.
        handle_discrete_as (str, optional):
            Specifies a methodology to combine discrete columns if it exists.
            Must be one of "numeric" or "categorial". Defaults to "categorial".

    Raises:
        NotImplementedError:
            If `handle_discrete_as` not in ("numeric", "categorial")

    Returns:
        tuple: (df, target, cat_cols, num_cols, metadata)
    """

    if subset is not None:
        assert subset in (
            "train",
            "test",
            "demo",
        ), "unrecognized subset. Must be one of (``train``, ``test``, ``demo``)"
        dset_path = os.path.join(dataset_dir, f"census/{subset}.csv")
    else:
        dset_path = os.path.join(dataset_dir, "census/census.csv")

    cat_cols = [
        "sex",
        "marital_stat",
        "full_or_part_time_employment_stat",
        "hispanic_origin",
        "education",
        "race",
        "class_of_worker",
        "region_of_previous_residence",
        "migration_prev_res_in_sunbelt",
        "migration_code-move_within_reg",
        "migration_code-change_in_msa",
        "family_members_under_18",
        "migration_code-change_in_reg",
        "detailed_household_summary_in_household",
        "detailed_household_and_family_stat",
        "live_in_this_house_1_year_ago",
        "fill_inc_questionnaire_for_veteran's_admin",
        "tax_filer_stat",
        "enroll_in_edu_inst_last_wk",
        "member_of_a_labor_union",
        "state_of_previous_residence",
        "country_of_birth_self",
        "country_of_birth_father",
        "country_of_birth_mother",
        "reason_for_unemployment",
        "major_occupation_code",
        "major_industry_code",
        "citizenship",
        "income",
    ]
    num_cols = [
        "age",
        "capital_losses",
        "capital_gains",
        "wage_per_hour",
        "weeks_worked_in_year",
        "dividends_from_stocks",
        "num_persons_worked_for_employer",
    ]

    discrete_cols = [
        "detailed_industry_recode",
        "detailed_occupation_recode",
        "own_business_or_self_employed",
        "veterans_benefits",
        "year",
    ]

    filtered_cols_list = [
        "age",
        "capital_gains",
        "capital_losses",
        "class_of_worker",
        "education",
        "marital_stat",
        "major_occupation_code",
        "sex",
        "race",
        "country_of_birth_self",
        "income",
    ]

    target = "income"

    df = pd.read_csv(dset_path, sep=",", index_col=None, **kws)

    df, cat_cols, num_cols = preprocess_dataframe(
        dataframe=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        discrete_cols=discrete_cols,
        target=target,
        filtered_cols_list=filtered_cols_list if return_filtered_cols else [],
        handle_discrete_as=handle_discrete_as,
    )
    metadata = get_metadata(df, cat_cols)
    return df, target, cat_cols, num_cols, metadata


def load_census_v2(
    dataset_dir,
    subset=None,
    return_filtered_cols=False,
    handle_discrete_as="categorial",
    **kws,
):
    """Load Adult-like Census Dataset

    Args:
        dataset_dir (str):
            Directory to load dataset from.
        subset (str, optional):
            Subset of the dataset to load. Defaults to None.
        return_filtered_cols (bool, optional):
            Whether or not to return `filtered_cols_list`.
            Defaults to False which returns all columns.
        handle_discrete_as (str, optional):
            Specifies a methodology to combine discrete columns if it exists.
            Must be one of "numeric" or "categorial". Defaults to "categorial".

    Raises:
        NotImplementedError:
            If `handle_discrete_as` not in ("numeric", "categorial")

    Returns:
        tuple: (df, target, cat_cols, num_cols, metadata)
    """

    if subset is not None:
        assert subset in (
            "train",
            "test",
            "demo",
        ), "unrecognized subset. Must be one of (``train``, ``test``, ``demo``)"
        dset_path = os.path.join(dataset_dir, f"census_v2/{subset}.csv")
    else:
        dset_path = os.path.join(dataset_dir, "census_v2/census_v2.csv")

    cat_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "sex",
        "race",
        "native-country",
        "income",
    ]
    num_cols = [
        "age",
        "capital-gain",
        "capital-loss",
    ]

    discrete_cols = []

    filtered_cols_list = []

    target = "income"

    df = pd.read_csv(dset_path, sep=",", index_col=None, **kws)

    df, cat_cols, num_cols = preprocess_dataframe(
        dataframe=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        discrete_cols=discrete_cols,
        target=target,
        filtered_cols_list=filtered_cols_list if return_filtered_cols else [],
        handle_discrete_as=handle_discrete_as,
    )
    metadata = get_metadata(df, cat_cols)
    return df, target, cat_cols, num_cols, metadata


def load_texas(
    dataset_dir,
    subset=None,
    return_filtered_cols=False,
    handle_discrete_as="categorial",
    **kws,
):
    """Load Texas Hospital Discharge Dataset

    Args:
        dataset_dir (str):
            Directory to load dataset from.
        subset (str, optional):
            Subset of the dataset to load. Defaults to None.
        return_filtered_cols (bool, optional):
            Whether or not to return `filtered_cols_list`.
            Defaults to False which returns all columns.
        handle_discrete_as (str, optional):
            Specifies a methodology to combine discrete columns if it exists.
            Must be one of "numeric" or "categorial". Defaults to "categorial".

    Raises:
        NotImplementedError:
            If `handle_discrete_as` not in ("numeric", "categorial")

    Returns:
        tuple: (df, target, cat_cols, num_cols, metadata)
    """

    if subset is not None:
        assert subset in (
            "train",
            "test",
            "demo",
        ), "unrecognized subset. Must be one of (``train``, ``test``, ``demo``)"
        dset_path = os.path.join(dataset_dir, f"texas/{subset}.csv")
    else:
        dset_path = os.path.join(dataset_dir, "texas/texas.csv")

    cat_cols = [
        "DISCHARGE",
        "PAT_STATE",
        "SEX_CODE",
    ]
    num_cols = [
        "LENGTH_OF_STAY",
        "TOTAL_CHARGES",
        "TOTAL_NON_COV_CHARGES",
        "TOTAL_CHARGES_ACCOMM",
        "TOTAL_NON_COV_CHARGES_ACCOMM",
        "TOTAL_CHARGES_ANCIL",
        "TOTAL_NON_COV_CHARGES_ANCIL",
    ]

    discrete_cols = [
        "TYPE_OF_ADMISSION",
        "PAT_STATUS",
        "RACE",
        "ADMIT_WEEKDAY",
        "ETHNICITY",
        "PAT_AGE",
        "ILLNESS_SEVERITY",
        "RISK_MORTALITY",
    ]

    filtered_cols_list = [
        "DISCHARGE",
        "TYPE_OF_ADMISSION",
        "PAT_STATUS",
        "SEX_CODE",
        "RACE",
        "ADMIT_WEEKDAY",
        "ETHNICITY",
        "PAT_AGE",
        "RISK_MORTALITY",
        "LENGTH_OF_STAY",
        "TOTAL_CHARGES",
        "TOTAL_NON_COV_CHARGES",
        "TOTAL_CHARGES_ACCOMM",
        "TOTAL_NON_COV_CHARGES_ACCOMM",
        "TOTAL_CHARGES_ANCIL",
        "TOTAL_NON_COV_CHARGES_ANCIL",
    ]

    target = "RISK_MORTALITY"

    df = pd.read_csv(dset_path, sep=",", index_col=None, **kws)

    df, cat_cols, num_cols = preprocess_dataframe(
        dataframe=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        discrete_cols=discrete_cols,
        target=target,
        filtered_cols_list=filtered_cols_list if return_filtered_cols else [],
        handle_discrete_as=handle_discrete_as,
    )
    metadata = get_metadata(df, cat_cols)
    return df, target, cat_cols, num_cols, metadata


def load_news(
    dataset_dir,
    subset=None,
    return_filtered_cols=False,
    handle_discrete_as="categorial",
    **kws,
):
    """Load Online Popularity News dataset

    Args:
        dataset_dir (str):
            Directory to load dataset from.
        subset (str, optional):
            Subset of the dataset to load. Defaults to None.
        return_filtered_cols (bool, optional):
            Whether or not to return `filtered_cols_list`.
            Defaults to False which returns all columns.
        handle_discrete_as (str, optional):
            Specifies a methodology to combine discrete columns if it exists.
            Must be one of "numeric" or "categorial". Defaults to "categorial".

    Raises:
        NotImplementedError:
            If `handle_discrete_as` not in ("numeric", "categorial")

    Returns:
        tuple: (df, target, cat_cols, num_cols, metadata)
    """

    if subset is not None:
        assert subset in (
            "train",
            "test",
            "demo",
        ), "unrecognized subset. Must be one of (``train``, ``test``, ``demo``)"
        dset_path = os.path.join(dataset_dir, f"news/{subset}.csv")
    else:
        dset_path = os.path.join(dataset_dir, "news/news.csv")

    cat_cols = [
        "data_channel_is_lifestyle",
        "data_channel_is_entertainment",
        "data_channel_is_bus",
        "data_channel_is_socmed",
        "data_channel_is_tech",
        "data_channel_is_world",
        "weekday_is_monday",
        "weekday_is_tuesday",
        "weekday_is_wednesday",
        "weekday_is_thursday",
        "weekday_is_friday",
        "weekday_is_saturday",
        "weekday_is_sunday",
        "is_weekend",
        "shares_binary",
    ]
    num_cols = [
        "n_tokens_title",
        "n_tokens_content",
        "n_unique_tokens",
        "n_non_stop_words",
        "n_non_stop_unique_tokens",
        "num_hrefs",
        "num_self_hrefs",
        "num_imgs",
        "num_videos",
        "average_token_length",
        "num_keywords",
        "kw_min_min",
        "kw_max_min",
        "kw_avg_min",
        "kw_min_max",
        "kw_max_max",
        "kw_avg_max",
        "kw_min_avg",
        "kw_max_avg",
        "kw_avg_avg",
        "self_reference_min_shares",
        "self_reference_max_shares",
        "self_reference_avg_sharess",
        "LDA_00",
        "LDA_01",
        "LDA_02",
        "LDA_03",
        "LDA_04",
        "global_subjectivity",
        "global_sentiment_polarity",
        "global_rate_positive_words",
        "global_rate_negative_words",
        "rate_positive_words",
        "rate_negative_words",
        "avg_positive_polarity",
        "min_positive_polarity",
        "max_positive_polarity",
        "avg_negative_polarity",
        "min_negative_polarity",
        "max_negative_polarity",
        "title_subjectivity",
        "title_sentiment_polarity",
        "abs_title_subjectivity",
        "abs_title_sentiment_polarity",
        "shares",
    ]

    discrete_cols = []

    filtered_cols_list = [i for i in num_cols if  i !="shares" ] + cat_cols

    target = "shares_binary"

    df = pd.read_csv(dset_path, sep=",", index_col=None, **kws)

    df, cat_cols, num_cols = preprocess_dataframe(
        dataframe=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        discrete_cols=discrete_cols,
        target=target,
        filtered_cols_list=filtered_cols_list if return_filtered_cols else [],
        handle_discrete_as=handle_discrete_as,
    )
    metadata = get_metadata(df, cat_cols)
    return df, target, cat_cols, num_cols, metadata


def load_adult_old(
    dataset_dir,
    subset=None,
    return_filtered_cols=False,
    handle_discrete_as="categorial",
    **kws,
):
    """Load Adult Dataset

    Args:
        dataset_dir (str):
            Directory to load dataset from.
        subset (str, optional):
            Subset of the dataset to load. Defaults to None.
        return_filtered_cols (bool, optional):
            Whether or not to return `filtered_cols_list`.
            Defaults to False which returns all columns.
        handle_discrete_as (str, optional):
            Specifies a methodology to combine discrete columns if it exists.
            Must be one of "numeric" or "categorial". Defaults to "categorial".

    Raises:
        NotImplementedError:
            If `handle_discrete_as` not in ("numeric", "categorial")

    Returns:
        tuple: (df, target, cat_cols, num_cols, metadata)
    """
    if subset is not None:
        assert subset in (
            "train",
            "test",
            "demo",
        ), "unrecognized subset. Must be one of (``train``, ``test``, ``demo``)"
        dset_path = os.path.join(dataset_dir, f"adult_old/{subset}.csv")
    else:
        dset_path = os.path.join(dataset_dir, "adult_old/adult_old.csv")

    cat_cols = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
        "income",
    ]
    num_cols = [
        "age",
        "fnlwgt",
        "education-num",
        "capital-gain",
        "capital-loss",
        "hours-per-week",
    ]
    discrete_cols = []
    filtered_cols_list = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "sex",
        "race",
        "native-country",
        "age",
        "capital-gain",
        "capital-loss",
        "income",
    ]

    target = "income"

    df = pd.read_csv(dset_path, sep=",", index_col=None, **kws)

    df, cat_cols, num_cols = preprocess_dataframe(
        dataframe=df,
        cat_cols=cat_cols,
        num_cols=num_cols,
        discrete_cols=discrete_cols,
        target=target,
        filtered_cols_list=filtered_cols_list if return_filtered_cols else [],
        handle_discrete_as=handle_discrete_as,
    )
    metadata = get_metadata(df, cat_cols)
    return df, target, cat_cols, num_cols, metadata

def preprocess_dataframe(
    dataframe,
    cat_cols,
    num_cols,
    discrete_cols,
    target,
    filtered_cols_list,
    handle_discrete_as,
):
    """Handy function to preprocess incoming dataframe"""
    df = dataframe.copy()

    if handle_discrete_as == "numeric":
        num_cols = discrete_cols + num_cols
    elif handle_discrete_as == "categorial":
        cat_cols = discrete_cols + cat_cols
    else:
        raise NotImplementedError("Unsupported option.")

    if len(filtered_cols_list) != 0:
        cat_cols = [i for i in filtered_cols_list if i in cat_cols]
        num_cols = [i for i in filtered_cols_list if i in num_cols]

    df.loc[:, cat_cols] = df.loc[:, cat_cols].astype("object")
    df.loc[:, num_cols] = df.loc[:, num_cols].astype("float")

    for col_name in cat_cols:
        df.loc[:, col_name] = df.loc[:, col_name].apply(lambda x: str(x))

    # to ensure that the target column is always appended at the end
    if target != "":
        if target in cat_cols:
            cat_cols = [col_name for col_name in cat_cols if col_name != target]
            df = df[num_cols + cat_cols + [target]]
            cat_cols.append(target)
        else:
            num_cols = [col_name for col_name in num_cols if col_name != target]
            df = df[num_cols + cat_cols + [target]]
            num_cols.append(target)
    else:
        df = df[num_cols + cat_cols]

    return df, cat_cols, num_cols


def get_metadata(data, discrete_columns=tuple()):
    meta = []

    df = pd.DataFrame(data)
    for index in df:
        column = df[index]

        if index in discrete_columns:
            mapper = column.value_counts().index.tolist()
            meta.append(
                {
                    "name": index,
                    "type": "categorical",
                    "size": len(mapper),
                    "i2s": mapper,
                }
            )
        else:
            meta.append(
                {
                    "name": index,
                    "type": "continuous",
                    "min": column.min(),
                    "max": column.max(),
                }
            )
    return meta
