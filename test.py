"""Script to evaluate exported models in `exported_models` folder on test set.

Usage: python test.py [-h] [-m MODEL] [-d {cpu,cuda}]

optional arguments:
  -h, --help            show this help message and exit
  -m MODEL, --model MODEL
                        Choose a trained model file from the folder
                        `exported_models`. Default = 'best_efficientnet-b0.pkl'
  -d {cpu,cuda}, --device {cpu,cuda}
                        Choose to infer with 'cpu' or 'cuda'. Default is
                        'cuda' if available, else 'cpu'

Examples:
>> python test.py
>> python test.py -m best_efficientnet-b0.pkl -d cuda

"""
from pathlib import Path
import numpy as np
import pandas as pd
import time
from helper import get_car_paths, get_cars_df, get_exported_learner, get_predictions_from_df
from efficientnet_pytorch import EfficientNet
import argparse
import torch
from fastai.vision import Learner
from fastai.vision import load_learner
from fastai.vision import ImageList
from fastai.vision import DatasetType
from fastai.vision import defaults

default_checkpoint = 'best_efficientnet-b0.pkl'

def parse_args() -> str:
    """Return user input for inference model and device"""
    des = ("Script to evaluate exported models  in `exported_models` folder "
           "on test set. ")
    m_help = (f"Choose a trained model file from the folder `exported_models`. "
              f"Default = '{default_checkpoint}'")
    d_help = ("Choose to infer with 'cpu' or 'cuda'. "
              "Default is 'cuda' if available, else 'cpu'")
    bs_help = "Batch size for inference. Default = 64"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument("-m", "--model", help=m_help)
    parser.add_argument("-d", "--device", choices=['cpu', 'cuda'], help=d_help)
    parser.add_argument("-bs", "--batch_size", 
                        default=64, type=int, help=bs_help)
    args = parser.parse_args()
    model = args.model
    device = args.device
    bs = args.batch_size
    return model, device, bs

def main():
    """Main of the script to infer and evaluate on test set"""
    model_cp, device, bs = parse_args()
    if device is not None:
        defaults.device = torch.device(device)
    
    # Get test dataframe and path to test folder.
    _, test_path = get_car_paths()
    test_df = get_cars_df('cars_test_annos_withlabels.mat')

    # Load exported model
    model_folder_path = Path("./exported_models")
    if model_cp == None:
        model_checkpoint = default_checkpoint
    else:
        model_checkpoint = model_cp
    inference_learn = get_exported_learner(model_folder_path, model_checkpoint)
    inference_learn.data.batch_size = bs
    print("="*70)
    print(f"Path to test images folder\t: {str(test_path)}/")
    print(f"Number of test images\t\t: {len(test_df)}")
    print(f"Loaded inference model\t\t: {model_folder_path/model_checkpoint}")
    print(f"Inference device\t\t: {defaults.device}")
    print(f"Batch size\t\t\t: {bs}")
    print("="*70, "\n\n")
    
    # Evaluate accuracy on test set
    print("Calculating accuracy of the test set...")
    start = time.time()
    fn_col = 0
    label_col = 1
    _, y_preds, _ = get_predictions_from_df(inference_learn, test_df, test_path, cols=fn_col)
    y_true = test_df.iloc[:,label_col].map(inference_learn.data.c2i).values
    accuracy = (y_preds == y_true).mean()
    infer_time = time.time() - start
    mins, secs = divmod(infer_time, 60)
    print(f"Accuracy on the test set: {accuracy*100:.02f}%")
    print(f"Total inference time: {int(mins)} mins {int(secs)} s ")


if __name__ == "__main__":
    main()