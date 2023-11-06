import argparse
import os
import pickle
import shutil
from datetime import datetime

import torch

from synthesizers.tablegan import trainer
from utils.dataset import Dataset
from utils.logger import get_logger
from utils.misc import mkdir, reproducibility, str2bool, write_csv
from utils.transformer import DataTransformer, TableGANTransformer

MODULE_NAME = "tablegan"

# Setup logger.
LOGGER = get_logger(MODULE_NAME)

# Log module run time.
start_time = datetime.now()


def main():
    args, save_dir = check_args(parse_arguments())

    DATASET_DIR = args.dataset_dir

    # Set reproducibility.
    reproducibility(args.model_random_state, use_cuda=torch.cuda.is_available())

    # Setup dataset class.
    args.subset_size = None if args.subset_size == -1 else args.subset_size
    dset = Dataset(
        dataset_name=args.dataset,
        dataset_dir=DATASET_DIR,
        subset="train",
        data_frac=args.subset_size,
        random_state=args.dataset_random_state,
    )

    train_data = dset.train_data[0]

    args.synth_size = len(train_data) if args.synth_size == -1 else args.synth_size
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
    # ==========

    if args.train:
        LOGGER.info(f"Train size: {train_data.shape[0]}")
        LOGGER.info(f"Train feature size: {train_data.shape[1]}")
        LOGGER.info("Batch size: %d" % args.batch_size)

        x_dim, y_dim = dset.get_dim()
        LOGGER.info("x_dim: %d, y_dim: %d" % (x_dim, y_dim))

        # write_csv(
        #     os.path.join(save_dir, "x_dim.csv"),
        #     args.subset_size,
        #     [x_dim],
        #     ["x_dim"],
        # )

        model = trainer.TableGANSynthesizer(args)
        model.fit(train_data)
        LOGGER.info(f"Model saved: {save_dir}")

    if args.sample:
        # Instantiate and load model parameters.
        model = trainer.TableGANSynthesizer(args)

        print("Loading model...")
        model.load()

        # Fit transformer.
        model.data_transformer.fit(train_data, discrete_columns=args.discrete_columns)
        model.tablegan_transformer.fit(args.metadata)

        # Sample synthetic data.
        synth_dir = (
            f"{DATASET_DIR}/synthetic_samples/{args.exp_name}/size{args.synth_size}"
        )
        mkdir(synth_dir)

        for i in range(args.nsynth):
            synth_data = model.sample(args.synth_size)
            synth_data.to_csv(os.path.join(synth_dir, f"synth{i+1}.csv"), index=None)

        # Save the train subset and test subset for this random seed.
        train_data.to_csv(os.path.join(synth_dir, f"train.csv"), index=None)

        dset = Dataset(
            dataset_name=args.dataset,
            dataset_dir=DATASET_DIR,
            subset="test",
            data_frac=None,
            random_state=args.dataset_random_state,
        )

        test_data = dset.train_data[0]
        test_data.to_csv(os.path.join(synth_dir, f"test.csv"), index=None)
        LOGGER.info(f"synthetic data directory: {synth_dir}.")

    end_time = datetime.now()
    LOGGER.info(f"Time elapsed: {end_time - start_time}")

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
        MODULE_NAME,
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
    parser.add_argument("--dataset_dir", type=str, default="../../data", help="dataset directory")
    parser.add_argument("--subset_size", "-subset", type=int, default=-1, help="how much data to train model with")
    parser.add_argument("--model_random_state", "-s", type=int, default=1000, help="Model random seed for reproducibility.")
    parser.add_argument("--dataset_random_state", type=int, default=1000, help="Dataset subsampling random seed for reproducibility.")
    parser.add_argument("--eval_iter", type=int, default=100)
    parser.add_argument("--save_after", type=int, default=100)
    
    parser.add_argument("--train", type=str2bool, default=False, help="train model")
    parser.add_argument("--evaluate", type=str2bool, default=False, help="evaluate trained model")
    parser.add_argument("--sample", type=str2bool, default=False, help="sample from model")
    parser.add_argument("--resume", type=str2bool, default=False, help="resume model training")
    parser.add_argument("--if_validate", type=str2bool, default=False, help="validate model during training")

    parser.add_argument("--batch_size", "-bs", type=int, default=500, help="batch size")
    parser.add_argument("--epochs", "-ep", type=int, default=300, help="number of epochs")

    parser.add_argument("--test_size", type=int, default=5000, help="size of the dataset to keep for evaluation")
    parser.add_argument("--synth_size", type=int, default=-1, help="Size of synthetic data to generate (-1 sets it to data subset size).")
    parser.add_argument("--nsynth", type=int, default=1, help="Number of synthetic datasets to generate.")
    
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
    parser.add_argument("--device", type=str, default="cuda", help="device to use")

    args = parser.parse_args()
    return args
    # fmt: on


if __name__ == "__main__":
    main()
