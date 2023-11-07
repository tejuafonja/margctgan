import argparse
import shutil
import os
import torch

from functools import partial
import pandas as pd

from utils.misc import mkdir, reproducibility, str2bool
from utils.dataset import Dataset
from metrics import *


def main():
    ## config
    args, save_dir = check_args(parse_arguments())

    DATASET_DIR = args.dataset_dir

    ## set reproducibility
    reproducibility(args.random_state, use_cuda=torch.cuda.is_available())

    ## setup dataset class
    args.subset_size = None if int(args.subset_size) == -1 else args.subset_size
    train_dset = Dataset(
        dataset_name=args.dataset,
        dataset_dir=DATASET_DIR,
        subset="train",
        data_frac=args.subset_size,
        random_state=args.random_state,
    )
    train_data = train_dset.train_data[0]

    test_dset = Dataset(
        dataset_name=args.dataset,
        dataset_dir=DATASET_DIR,
        subset="test",
        data_frac=None,
        random_state=args.random_state,
    )
    test_data = test_dset.train_data[0]

    all_dset = Dataset(
        dataset_name=args.dataset,
        dataset_dir=DATASET_DIR,
        subset=None,
        data_frac=None,
        random_state=args.random_state,
    )
    all_data = all_dset.train_data[0]
    cat_cols = all_dset.cat_cols
    target_name = all_dset.target_name
    #####

    ## map synthesizer
    synthesizer_mapping = {
        "sdv_ctgan": "ctgan",
        "sdv_tvae": "tvae",
        "marg_ctgan": "margctgan",
        "pca_ctgan": "pmctgan",
    }

    print(f"----Metric: {args.metric_name}")

    synth_dir = (
            f"{DATASET_DIR}/synthetic_samples/{args.exp_name}/size{args.synth_size}"
        )
    # fake_dir = f"{DATASET_DIR}/fake_samples/{args.dataset}/{args.synthesizer}/{args.exp_name}/FS{args.sample_size}"
    # fake_dir = f"{DATASET_DIR}/fake_samples/{args.exp_name}/{args.dataset}/{args.synthesizer}/{args.exp_name}/FS{args.sample_size}"

    if args.metric_group == "marginal":
        if args.metric_name == "histogram_intersection":
            column_func = partial(
                histogram_intersection, bins=args.bins, fit_data=all_data
            )
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        elif args.metric_name == "jaccard_similarity":
            column_func = partial(jaccard_similarity, bins=args.bins, fit_data=all_data)
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        elif args.metric_name == "total_variation_distance":
            column_func = partial(total_variation_distance, bins=args.bins, fit_data=all_data)
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        elif args.metric_name == "chebychev_chi2":
            column_func = partial(chebychev_chi2, bins=args.bins, fit_data=all_data)
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        elif args.metric_name == "column_correlation":
            column_func = column_correlation
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        elif args.metric_name == "wasserstein_distance":
            column_func = partial(
                wasserstein_distance, bins=args.bins, fit_data=all_data
            )
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        elif args.metric_name == "jensonshannon_distance":
            column_func = partial(
                jensonshannon_distance, bins=args.bins, fit_data=all_data
            )
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        elif args.metric_name == "cumulative_difference":
            column_func = partial(
                cumulative_difference, bins=args.bins, fit_data=all_data
            )
            func = partial(
                column_metric_wrapper,
                column_metric=column_func,
                cat_cols=cat_cols,
                random_state=args.random_state,
            )
        else:
            raise NotImplementedError(
                f"Unsupported {args.metric_group} metric: `{args.metric_name}`"
            )

    if args.metric_group == "column_pair":
        if args.metric_name == "associations_difference":
            func = partial(
                associations_difference,
                mean_column_difference=True,
                cat_cols=cat_cols,
                return_dataframe=True,
                random_state=args.random_state,
            )
        else:
            raise NotImplementedError(
                f"Unsupported {args.metric_group} metric: `{args.metric_name}`"
            )

    if args.metric_group == "joint":
        if args.metric_name == "likelihood_approximation":
            func = partial(
                nearest_neighbors_wrapper,
                joint_metric=likelihood_approximation,
                cat_cols=cat_cols,
                neighbors=list(range(1, 10)),
                realdata_subsample=5000,
                fit_data=all_data,
                random_state=args.random_state,
            )
        elif args.metric_name == "closeness_approximation":
            func = partial(
                nearest_neighbors_wrapper,
                joint_metric=closeness_approximation,
                cat_cols=cat_cols,
                neighbors=list(range(1, 10)),
                realdata_subsample=5000,
                fit_data=all_data,
                random_state=args.random_state,
            )
        else:
            raise NotImplementedError(
                f"Unsupported {args.metric_group} metric: `{args.metric_name}`"
            )

    if args.metric_group == "utility":
        if args.metric_name == "efficacy_test":
            func = partial(
                efficacy_test_wrapper,
                target_name=target_name,
                cat_cols=cat_cols,
                model_names=["logistic", "tree", "mlp"],
                fit_data=all_data,
                random_state=args.random_state,
                psutil_terminate=True,
            )
        elif args.metric_name == "all_models_test":
            func = partial(
                all_models_test,
                cat_cols=cat_cols,
                return_mean=False,
                cat_model_names=["logistic", "tree", "mlp"],
                num_model_names=["linear", "tree", "mlp"],
                return_dataframe=True,
                fit_data=all_data,
                random_state=args.random_state,
                psutil_terminate=True,
            )
        else:
            raise NotImplementedError(
                f"Unsupported {args.metric_group} metric: `{args.metric_name}`"
            )

    ## train/test data
    results = []
    result = func(
        realdata=test_data,
        fakedata=train_data,
    )
    for i in range(args.eval_retries):
        result_cp = result.copy()
        result_cp.loc[:, "eval_retries"] = i
        result_cp.loc[:, "synthesizer"] = "best"
        results.append(result_cp)
    results = pd.concat(results).reset_index(drop=True)

    if not args.overwrite_results and os.path.exists(
        os.path.join(save_dir, f"{args.metric_name}_traintest.csv")
    ):
        old_results = pd.read_csv(
            os.path.join(save_dir, f"{args.metric_name}_traintest.csv")
        )
        results = pd.concat([old_results, results])

    results.to_csv(
        os.path.join(save_dir, f"{args.metric_name}_traintest.csv"), index=None
    )

    # aggregate scores
    scores_dict = {"traintest": results.normalized_score.mean()}

    ## fake/test data
    results = []
    for i in range(args.eval_retries):
        fake_data = pd.read_csv(os.path.join(synth_dir, f"synth{i+1}.csv"))
        fake_data.loc[:, cat_cols] = fake_data.loc[:, cat_cols].astype("object")
        result = func(
            realdata=test_data,
            fakedata=fake_data,
        )

        result.loc[:, "eval_retries"] = i
        result.loc[:, "synthesizer"] = synthesizer_mapping.get(
            args.synthesizer, args.synthesizer
        )
        results.append(result)

    results = pd.concat(results).reset_index(drop=True)

    # aggregate scores
    scores_dict.update({"faketest": results.normalized_score.mean()})

    if not args.overwrite_results and os.path.exists(
        os.path.join(save_dir, f"{args.metric_name}_faketest.csv")
    ):
        old_results = pd.read_csv(
            os.path.join(save_dir, f"{args.metric_name}_faketest.csv")
        )
        results = pd.concat([old_results, results])
    results.to_csv(
        os.path.join(save_dir, f"{args.metric_name}_faketest.csv"), index=None
    )

    ## fake/train data
    results = []
    for i in range(args.eval_retries):
        fake_data = pd.read_csv(os.path.join(synth_dir, f"synth{i+1}.csv"))
        fake_data.loc[:, cat_cols] = fake_data.loc[:, cat_cols].astype("object")
        result = func(
            realdata=train_data,
            fakedata=fake_data,
        )
        result.loc[:, "eval_retries"] = i
        result.loc[:, "synthesizer"] = synthesizer_mapping.get(
            args.synthesizer, args.synthesizer
        )
        results.append(result)

    results = pd.concat(results).reset_index(drop=True)

    # aggregate scores
    scores_dict.update({"faketrain": results.normalized_score.mean()})

    if not args.overwrite_results and os.path.exists(
        os.path.join(save_dir, f"{args.metric_name}_faketrain.csv")
    ):
        old_results = pd.read_csv(
            os.path.join(save_dir, f"{args.metric_name}_faketrain.csv")
        )
        results = pd.concat([old_results, results])
    results.to_csv(
        os.path.join(save_dir, f"{args.metric_name}_faketrain.csv"), index=None
    )

    ## aggregate scores
    if not args.overwrite_results and os.path.exists(
        os.path.join(save_dir, f"summary_{args.metric_name}.csv")
    ):
        with open(os.path.join(save_dir, f"summary_{args.metric_name}.csv"), "a") as f:
            for k, v in scores_dict.items():
                f.write(f"{k}, {v}\n")
    else:
        with open(os.path.join(save_dir, f"summary_{args.metric_name}.csv"), "w") as f:
            for k, v in scores_dict.items():
                f.write(f"{k}, {v}\n")

    print("=" * 20, "Done", "=" * 20)
    print(f"Metrics results directory: {save_dir}")


def check_args(args):
    """Check, store argument, and set up the save_dir

    :params args: arguments
    :return:
    """

    ## set up save_dir
    save_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        args.synthesizer,
        args.dataset,
        args.exp_name,
        f"size{args.synth_size}",
        "metrics",
        args.metric_group,
    )
    mkdir(save_dir)

    ## store the parameters
    with open(os.path.join(save_dir, "params.txt"), "w") as f:
        for k, v in vars(args).items():
            f.writelines(k + ":" + str(v) + "\n")
            print(k + ":" + str(v))

        ## store this script
        shutil.copy(os.path.realpath(__file__), save_dir)
    return args, save_dir


def parse_arguments():

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "--exp_name", "-name", type=str, required=True, help="path for storing the checkpoint") 
    parser.add_argument("--dataset", "-data", type=str, default="adult", help="dataset name")
    parser.add_argument("--dataset_dir", type=str, default="../data", help="dataset directory")
    parser.add_argument("--synthesizer", "-synth", type=str, default="sdv_ctgan", help="synthesizer name")
    parser.add_argument("--subset_size", "-subset", type=int, default=-1, help="how much data to train model with")
    parser.add_argument("--random_state", "-s", type=int, default=1000, help="random seed")
    parser.add_argument("--overwrite_results", type=str2bool, default=True, help="overwrite old results")

    parser.add_argument("--bins",  type=int, default=50, help="the number of equal-width bins in the given range.")
    
    parser.add_argument("--metric_name", type=str, default="histogram_intersection",  help="metric to compute")
    parser.add_argument("--metric_group", type=str, default="marginal", choices=["marginal", "column_pair",  "joint", "utility"], help="metric group name")
    parser.add_argument("--eval_retries", type=int, default=1, help="number of times to run the evaluation")
    parser.add_argument("--synth_size", type=int, default=-1, help="size of synthetic data")
    
   
    args = parser.parse_args()
    return args
    # fmt: on


if __name__ == "__main__":
    main()
