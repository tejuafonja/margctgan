import argparse
import os
import pickle
import shutil
import sys
import numpy as np
import torch
from ctgan import CTGAN as CTGANSynthesizer

sys.path.append("../..")

from utils.eval import run_eval
from utils.misc import mkdir, reproducibility, str2bool, write_csv
from utils.dataset import Dataset
from utils.transformer import DataTransformer

DATASET_DIR = "../../data"


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
    discrete_columns = dset.cat_cols

    args.sample_size = len(train_data) if args.sample_size == -1 else args.sample_size
    # args.batch_size = min(len(train_data), args.batch_size)

    if args.train:
        print("size of train dataset: %d" % train_data.shape[0])
        print("dim of train dataset: %d" % train_data.shape[1])
        print("batch size: %d" % args.batch_size)

        model = CTGANSynthesizer(
            epochs=args.epochs, batch_size=args.batch_size, verbose=True
        )
        model.fit(train_data, discrete_columns)
        model.save(f"{save_dir}/model.pth")
        print("model saved!")
        
    if args.evaluate:
        print("--loading model")
        model = CTGANSynthesizer(epochs=-1, batch_size=args.batch_size).load(
            f"{save_dir}/model.pth"
        )

        ##########
        train_all_dset = Dataset(
            dataset_name=args.dataset,
            dataset_dir=DATASET_DIR,
            subset=None,
            data_frac=None,
            random_state=args.random_state,
        )
        train_all_data = train_all_dset.train_data[0]

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
        
        # setup test dataset loader
        test_dset = Dataset(
            dataset_name=args.dataset,
            dataset_dir=DATASET_DIR,
            subset="test",
            data_frac=None,
            random_state=args.random_state,
        )
        test_data = test_dset.train_data[0]

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

    if args.sample:
        print("--loading model")
        model = CTGANSynthesizer(epochs=-1, batch_size=args.batch_size).load(
            f"{save_dir}/model.pth"
        )

        ### sample fake
        fake_dir = (
            f"{DATASET_DIR}/fake_samples/{args.dataset}/sdv_ctgan/{args.exp_name}/FS{args.sample_size}"
        )
        mkdir(fake_dir)

        for i in range(args.eval_retries):
            fake_data = model.sample(args.sample_size)
            fake_data.sample(args.sample_size).to_csv(
                os.path.join(fake_dir, f"fakedata_{i}.csv"), index=None
            )
        ###
        print(f"fake data saved to {fake_dir}.")


def check_args(args):
    ## set up save_dir
    save_dir = os.path.join(
        os.path.dirname(__file__),
        "results",
        "sdv_ctgan",
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
    parser.add_argument("--subset_size", "-subset", type=int, default=-1, help="how much data to train model with")
    parser.add_argument("--batch_size", "-bs", type=int, default=500, help="batch size")
    parser.add_argument("--random_state", "-s", type=int, default=1000, help="random seed")
    
    parser.add_argument("--train", type=str2bool, default=False, help="train model")
    parser.add_argument("--evaluate", type=str2bool, default=False, help="evaluate trained model")
    parser.add_argument("--sample", type=str2bool, default=False, help="sample from model")
    
    parser.add_argument("--epochs", "-ep", type=int, default=300, help="number of epochs")
    
    parser.add_argument("--sample_size", type=int, default=-1, help="size of synthetic data to generate")
    parser.add_argument("--eval_retries", type=int, default=1, help="number of times to run the evaluation")
    
    args = parser.parse_args()
    return args
    # fmt: on


if __name__ == "__main__":
    main()
