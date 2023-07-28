import argparse
import os
import pickle
import shutil
import sys

import numpy as np
import torch

sys.path.append("../..")
from synthesizers.tablegan import trainer
from utils.dataset import Dataset
from utils.eval import run_eval
from utils.misc import mkdir, reproducibility, str2bool, write_csv
from utils.transformer import DataTransformer, TableGANTransformer

DATASET_DIR = "../../data/"


def main():
    ## config
    args, save_dir = check_args(parse_arguments())

    ## set reproducibility
    reproducibility(args.random_state, use_cuda=torch.cuda.is_available())

    ## setup dataset class
    args.subset_size = None if args.subset_size == -1 else args.subset_size
    dset = Dataset(
        dataset_name=args.dataset,
        dataset_dir=DATASET_DIR,
        subset="train",
        data_frac=args.subset_size,
        random_state=args.random_state,
    )

    train_data = dset.train_data[0]

    if args.if_validate:
        validation_data = dset.test_subset(args.test_size)[0]
    else:
        validation_data = None

    args.sample_size = len(train_data) if args.sample_size is None else args.sample_size
    # args.batch_size = min(len(train_data), args.batch_size)

    sides = [4, 8, 16, 24, 32]
    for i in sides:
        if i * i >= train_data.shape[1]:
            side = i
            break

    data_transformer = DataTransformer(
        max_clusters=args.max_clusters,
        covariance_type=args.covariance_type,
        discrete_encode=args.discrete_preprocess,
        numerical_preprocess=args.numerical_preprocess,
        target=args.class_target,
    )
    tablegan_transformer = TableGANTransformer(side)

    # ==========
    args.side = side
    args.log_dir = save_dir
    args.metadata = dset.metadata
    args.discrete_columns = dset.cat_cols
    args.target_name = dset.target_name
    args.data_transformer = data_transformer
    args.tablegan_transformer = tablegan_transformer
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    # ==========

    if args.train:
        print("size of train dataset: %d" % train_data.shape[0])
        print("dim of train dataset: %d" % train_data.shape[1])
        print("batch size: %d" % args.batch_size)

        x_dim, y_dim = dset.get_dim()
        print("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

        write_csv(
            os.path.join(save_dir, "x_dim.csv"),
            args.subset_size,
            [x_dim],
            ["x_dim"],
        )

        model = trainer.TableGANSynthesizer(args)
        model.fit(train_data, validation_data=validation_data)
        print("training done!")

    if args.evaluate:
        ### instantiate and load model parameters
        model = trainer.TableGANSynthesizer(args)
        model.load()

        ## fit transformer
        model.data_transformer.fit(train_data, discrete_columns=args.discrete_columns)
        model.tablegan_transformer.fit(args.metadata)
        ###

        ##########
        train_all_dset = Dataset(
            dataset_name=args.dataset,
            dataset_dir=DATASET_DIR,
            subset=None,
            data_frac=None,
            random_state=args.random_state,
        )
        train_all_data = train_all_dset.train_data[0]

        test_dset = Dataset(
            dataset_name=args.dataset,
            dataset_dir=DATASET_DIR,
            subset="test",
            data_frac=None,
            random_state=args.random_state,
        )
        test_data = test_dset.train_data[0]

        discrete_columns = train_all_dset.cat_cols
        print("size of train-test dataset: %d" % len(train_all_data))

        # setup data transformer
        data_transformer = DataTransformer(
            discrete_encode="onehot",
            numerical_preprocess="standard",
            target=train_all_dset.target_name,
        )

        # fit the data transformer on both the train data and test data
        data_transformer.fit(train_all_data, discrete_columns=discrete_columns)
        ##########

        ### run evaluation
        print("running eval..")
        scores = []
        index_names = []

        for i in range(args.eval_retries):
            ### sample fake
            fake_data = model.sample(args.sample_size)
            ###
            index_names, data = run_eval(
                fakedata=fake_data,
                traindata=train_data,
                testdata=test_data,
                target_name=dset.target_name,
                data_transformer=data_transformer,
                metric="f1",
            )

            index_names = index_names
            scores.append(data)
        ###
        scores = np.array(scores).mean(axis=0)
        write_csv(
            os.path.join(save_dir, f"eval_f1.csv"),
            f"fake{len(fake_data)}",
            scores,
            index_names,
        )
        ### save eval train/test/fake data statistics
        write_csv(
            os.path.join(save_dir, "eval_stats.csv"),
            f"fake{len(fake_data)}",
            [len(train_data), len(test_data)],
            [
                "train_size",
                "test_size",
            ],
        )
        print(f"done. saved to {save_dir}")
        fake_data.sample(10).to_csv(
            os.path.join(save_dir, f"fake_sample.csv"), index=None
        )

    if args.sample:
        ### instantiate and load model parameters
        model = trainer.TableGANSynthesizer(args)
        model.load()

        ## fit transformer
        model.data_transformer.fit(train_data, discrete_columns=args.discrete_columns)
        model.tablegan_transformer.fit(args.metadata)
        ###

        ### sample fake
        fake_dir = f"{DATASET_DIR}/fake_samples/{args.dataset}/tablegan/{args.exp_name}/FS{args.sample_size}"
        mkdir(fake_dir)

        for i in range(args.eval_retries):
            fake_data = model.sample(args.sample_size)
            fake_data.sample(args.sample_size).to_csv(
                os.path.join(fake_dir, f"fakedata_{i}.csv"), index=None
            )
        ###

        print(f"fake data saved to {fake_dir}.")

    return


def check_args(args):
    """Check, store argument, and set up the save_dir

    :params args: arguments
    :return:
    """

    ## set up save_dir
    save_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "tablegan",
        args.dataset,
        args.exp_name,
    )
    mkdir(save_dir)

    ## store the parameters
    if args.train:
        with open(os.path.join(save_dir, "params.txt"), "w") as f:
            for k, v in vars(args).items():
                f.writelines(k + ":" + str(v) + "\n")
                print(k + ":" + str(v))

        with open(os.path.join(save_dir, "params.pkl"), "wb") as f:
            pickle.dump(vars(args), f, protocol=2)

        ## store this script
        shutil.copy(os.path.realpath(__file__), save_dir)
        shutil.copy(os.path.realpath(trainer.__file__), save_dir)

    return args, save_dir


def parse_arguments():

    # fmt: off
    parser = argparse.ArgumentParser()
    parser.add_argument( "--exp_name", "-name", type=str, required=True, help="path for storing the checkpoint")
    parser.add_argument("--dataset", "-data", type=str, default="adult", help="dataset name")
    parser.add_argument("--subset_size", "-subset", type=int, default=-1, help="how much data to train model with")
    parser.add_argument("--random_state", "-s", type=int, default=1000, help="random seed")
    parser.add_argument("--eval_iter", type=int, default=100)
    parser.add_argument("--save_after", type=int, default=100)
    
    parser.add_argument("--train", type=str2bool, default=False, help="train model")
    parser.add_argument("--evaluate", type=str2bool, default=False, help="evaluate trained model")
    parser.add_argument("--sample", type=str2bool, default=False, help="sample from model")
    parser.add_argument("--resume", type=str2bool, default=False, help="resume model training")
    parser.add_argument("--if_validate", type=str2bool, default=True, help="validate model during training")

    parser.add_argument("--batch_size", "-bs", type=int, default=500, help="batch size")
    parser.add_argument("--epochs", "-ep", type=int, default=300, help="number of epochs")

    parser.add_argument("--test_size", type=int, default=5000, help="size of the dataset to keep for evaluation")
    parser.add_argument("--sample_size", type=int, default=-1, help="size of synthetic data to generate")
    parser.add_argument("--eval_retries", type=int, default=1, help="number of times to run the evaluation")

    parser.add_argument( "--use_sampler", type=str2bool, default=False, help="use data sampler")
    
    ## Dataset
    parser.add_argument("--numerical_preprocess", type=str, default="none", choices=["bayesian", "standard", "minmax", "none"], help="preprocessing scheme to use for numerical columns")
    parser.add_argument("--discrete_preprocess", type=str, default="label", choices=["onehot", "label"], help="preprocessing scheme to use for discrete columns")
    parser.add_argument("--max_clusters", type=int, default=10, help="max cluster for if preprocess is bayesian")
    parser.add_argument("--covariance_type", type=str, default="full", help="covariance type if preprocess is bayesian")

    ## Encoder and Decoder
    parser.add_argument("--random_dim", type=int, default=100, help="generator noise vector size")
    parser.add_argument("--num_channels", type=float, default=64, help="number of generation/discriminator channels")
    parser.add_argument("--lr", type=float, default=1e-5, help="learning rate for optimizer")
    parser.add_argument("--class_target", "-target", type=str, default="income", help="target for classifier")

    args = parser.parse_args()
    return args
    # fmt: on


if __name__ == "__main__":
    main()
