import argparse
import os
import pickle
import shutil
from datetime import datetime

import pandas as pd
import torch

from utils.dataset import Dataset
from utils.logger import get_logger
from utils.metrics_utils import generate_report
from utils.misc import mkdir, reproducibility, str2bool

from sklearn.mixture import GaussianMixture
from ctgan.data_transformer import DataTransformer



MODULE_NAME = "gmm"

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
    discrete_columns = dset.cat_cols

    args.synth_size = len(train_data) if args.synth_size == -1 else args.synth_size

    if args.train:
        LOGGER.info(f"Train size: {train_data.shape[0]}")
        LOGGER.info(f"Train feature size: {train_data.shape[1]}")
        
        transformer = DataTransformer()
        transformer.fit(train_data, discrete_columns)

        train_data_ = self._transformer.transform(train_data)

        model = GaussianMixture(n_components=10, covariance_type='full')
        model.fit(train_data_)
        LOGGER.info(
            f"Train feature size (transformed): {transformer.output_dimensions}"
        )

        with open(f"{save_dir}/model.pth", "wb") as f:
            pickle.dump(model, f)
        
        with open(f"{save_dir}/transformer.pth", "wb") as f:
            pickle.dump(transformer, f)
            
        # model.save(f"{save_dir}/model.pth")
        LOGGER.info(f"Model saved: {save_dir}")

    if args.sample:
        print("Loading model...")
        
        with open(f"{save_dir}/model.pth") as f:
            model = pickle.load(f)
        
        with open(f"{save_dir}/transformer.pth") as f:
            transformer = pickle.load(f)

        # Sample synthetic data.
        synth_dir = (
            f"{DATASET_DIR}/synthetic_samples/{args.exp_name}/size{args.synth_size}"
        )
        mkdir(synth_dir)

        for i in range(args.nsynth):
            synth_data = transformer.inverse_transform(model.sample(args.synth_size))
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

    if args.evaluate:
        synth_dir = (
            f"{DATASET_DIR}/synthetic_samples/{args.exp_name}/size{args.synth_size}"
        )

        report_dir = f"{args.report_dir}/{args.exp_name}/size{args.synth_size}"
        mkdir(report_dir)
        for i in range(args.nsynth):
            synthetic_data = pd.read_csv(os.path.join(synth_dir, f"synth{i+1}.csv"))
            prop = generate_report(train_data, synthetic_data)
            prop["s"] = i
            prop.to_csv(f"{report_dir}/quality_report_{i}.csv", index=False)

    end_time = datetime.now()
    LOGGER.info(f"Time elapsed: {end_time - start_time}")


def check_args(args):
    # Set up save_dir.
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
    return args, save_dir


def parse_arguments():
    # fmt: off
    parser = argparse.ArgumentParser()
    
    parser.add_argument( "--exp_name", "-name", type=str, required=True, help="Experiment name for storing checkpoints.")
    parser.add_argument("--dataset", "-data", type=str, default="adult", help="Dataset name.")
    
    parser.add_argument("--dataset_dir", type=str, default="../../data", help="Dataset directory path.")
    parser.add_argument("--report_dir", type=str, default="../../reports", help="Quality reports directory.")
    
    parser.add_argument("--subset_size", "-subset", type=int, default=-1, help="Data subset limit for training (-1 indicate full data).")
    parser.add_argument("--model_random_state", "-s", type=int, default=1000, help="Model random seed for reproducibility.")
    parser.add_argument("--dataset_random_state", type=int, default=1000, help="Dataset subsampling random seed for reproducibility.")
    
    parser.add_argument("--train", type=str2bool, default=False, help="Flag to train the model.")
    parser.add_argument("--sample", type=str2bool, default=False, help="Flag to sample from the model.")
    parser.add_argument("--evaluate", type=str2bool, default=False, help="Flag to evaluate the quality of the model.")
    
    parser.add_argument("--synth_size", type=int, default=-1, help="Size of synthetic data to generate (-1 sets it to data subset size).")
    parser.add_argument("--nsynth", type=int, default=1, help="Number of synthetic datasets to generate.")
     
        
    args = parser.parse_args()
    return args
    # fmt: on


if __name__ == "__main__":
    main()
