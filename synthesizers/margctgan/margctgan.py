import argparse
import os
import pickle
import shutil
import sys
import numpy as np
import torch
from model import MargCTGAN

from utils.misc import mkdir, reproducibility, str2bool
from utils.dataset import Dataset


def main():
    ## config
    args, save_dir = check_args(parse_arguments())

    DATASET_DIR = args.dataset_dir

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
    discrete_columns = dset.cat_cols

    args.sample_size = len(train_data) if args.sample_size == -1 else args.sample_size
    # args.batch_size = min(len(train_data), args.batch_size)

    args.extra_param_dict = {
        "loss_weight": args.loss_weight,
        "weight_scheme": args.weight_scheme,
        "loss_type": args.loss_type,
        "variant": args.variant,
        "pca_components": args.pca_components,
    }

    if args.train:
        print("size of train dataset: %d" % train_data.shape[0])
        print("dim of train dataset: %d" % train_data.shape[1])
        print("batch size: %d" % args.batch_size)

        model = MargCTGAN(
            epochs=args.epochs,
            batch_size=args.batch_size,
            verbose=True,
            cuda=args.device,
            extra_param_dict=args.extra_param_dict,
        )
        model.fit(train_data, discrete_columns)
        model.save(f"{save_dir}/model.pth")
        print("=" * 20, "Done", "=" * 20)
        print("model saved!")

    if args.sample:
        print("--loading model")
        model = MargCTGAN(
            epochs=-1,
            batch_size=args.batch_size,
            cuda=args.device,
            extra_param_dict=args.extra_param_dict,
        ).load(f"{save_dir}/model.pth")

        ### sample fake
        n = f"{args.variant}{args.pca_components if args.variant == 'pca' else ''}"
        fake_dir = f"{DATASET_DIR}/fake_samples/{args.dataset}/{n}ctgan/{args.exp_name}/FS{args.sample_size}"
        mkdir(fake_dir)

        for i in range(args.eval_retries):
            fake_data = model.sample(args.sample_size)
            fake_data.sample(args.sample_size).to_csv(
                os.path.join(fake_dir, f"fakedata_{i}.csv"), index=None
            )
        ###
        print("=" * 20, "Done", "=" * 20)
        print(f"Fake data directory: {fake_dir}.")


def check_args(args):
    ## set up save_dir
    save_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        f"{args.variant}{args.pca_components if args.variant == 'pca' else ''}_ctgan",
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
    parser.add_argument( "--exp_name", "-name", type=str, required=True, help="path for storing the checkpoint")
    parser.add_argument("--dataset", "-data", type=str, default="adult", help="dataset name")
    parser.add_argument("--dataset_dir", type=str, default="../../data", help="dataset directory")
    parser.add_argument("--subset_size", "-subset", type=int, default=-1, help="how much data to train model with")
    parser.add_argument("--batch_size", "-bs", type=int, default=500, help="batch size")
    parser.add_argument("--random_state", "-s", type=int, default=1000, help="random seed")
    
    parser.add_argument("--train", type=str2bool, default=False, help="train model")
    parser.add_argument("--evaluate", type=str2bool, default=False, help="evaluate trained model")
    parser.add_argument("--sample", type=str2bool, default=False, help="sample from model")
    
    parser.add_argument("--epochs", "-ep", type=int, default=300, help="number of epochs")
    
    parser.add_argument("--sample_size", type=int, default=-1, help="size of synthetic data to generate")
    parser.add_argument("--eval_retries", type=int, default=1, help="number of times to run the evaluation")
    
    parser.add_argument(
        "--loss_weight", type=float, default=1.0, help="loss weight coefficient"
    )
    parser.add_argument(
        "--weight_scheme", type=str, default="weighted", help="how to weight the losses"
    )
    parser.add_argument(
        "--pca_components", type=int, default=-1, help="number of pca components"
    )
    parser.add_argument(
        "--loss_type", type=str, default="mean_and_stddev", help="type of loss to apply"
    )
    
    parser.add_argument(
        "--variant", type=str, default="marg", help="kind of marginal to apply"
    )
    
    parser.add_argument(
        "--device", type=str, default="cuda", help="device to use"
    )
    
    args = parser.parse_args()
    return args
    # fmt: on


if __name__ == "__main__":
    main()
