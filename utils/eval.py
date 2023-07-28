import sys
sys.path.append("..")

from metrics import efficacy_test
from utils.transformer import DataTransformer

__all__ = ["run_eval"]

def run_eval(
    fakedata, traindata, testdata, target_name, metric="f1", data_transformer=None
):
    discrete_columns = set(traindata.columns) - set(
        traindata._get_numeric_data().columns
    )

    index_names, data = [], []

    if metric == "f1":
        if data_transformer == None:
            data_transformer = DataTransformer(
                numerical_preprocess="standard",
                discrete_encode="onehot",
                target=target_name,
            )
            data_transformer.fit(traindata, discrete_columns=discrete_columns)

        # train on real (train) data, evaluate on real (train) data
        best_f1 = efficacy_test(
            fakedata=traindata,
            realdata=traindata,
            target_name=target_name,
            transformer=data_transformer,
        )

        # train on real (train) data, evaluate on real (test) data
        baseline = efficacy_test(
            fakedata=traindata,
            realdata=testdata,
            target_name=target_name,
            transformer=data_transformer,
        )

        # train on real (train) data, evaluate on fake data
        train_fake = efficacy_test(
            fakedata=traindata,
            realdata=fakedata,
            target_name=target_name,
            transformer=data_transformer,
        )

        # train on fake data, evaluate on real (train) data
        fake_train = efficacy_test(
            fakedata=fakedata,
            realdata=traindata,
            target_name=target_name,
            transformer=data_transformer,
        )

        # train on real (test) data, evaluate on fake data
        test_fake = efficacy_test(
            fakedata=testdata,
            realdata=fakedata,
            target_name=target_name,
            transformer=data_transformer,
        )

        # train on fake data, evaluate on real (test) data
        fake_test = efficacy_test(
            fakedata=fakedata,
            realdata=testdata,
            target_name=target_name,
            transformer=data_transformer,
        )

        index_names = [
            "best (train-train) f1: ",
            "baseline (train-test) f1",
            "real (train-fake) f1",
            "fake (fake-train) f1",
            "real (test-fake) f1",
            "fake (fake-test) f1",
        ]

        data = [
            best_f1,
            baseline,
            train_fake,
            fake_train,
            test_fake,
            fake_test,
        ]

    else:
        raise NotImplementedError

    return index_names, data
