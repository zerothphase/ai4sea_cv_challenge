# Script to evaluate on test set
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import pandas as pd
import PIL
import time
import os
from helper import get_car_paths, get_cars_df
from efficientnet_pytorch import EfficientNet
import random
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, models, transforms
from torch import nn
import torch.optim as optim

from fastai.vision import *
defaults.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def get_exported_learner(folder_path, filename):
    
    model_path = folder_path / filename
    assert model_path.exists()
    learn = load_learner(folder_path, filename)
    learn.to_fp32()
    return learn

def get_predictions(learn, test_df, test_path, cols=0):
    test_imagelist = ImageList.from_df(test_df, test_path, cols=cols)
    learn.data.add_test(test_imagelist)
    probs, _ = learn.get_preds(ds_type=DatasetType.Test)
    y_preds = torch.argmax(probs, dim=1)
    return probs.numpy(), y_preds.numpy()

def get_test_accuracy(learn, test_df, test_path, fn_col=0, label_col=1):
    _, y_preds = get_predictions(learn, test_df, test_path, cols=fn_col)
    y_true = test_df.iloc[:,label_col].map(learn.data.c2i).values
    accuracy = (y_preds == y_true).mean()
    return accuracy

def main():
    
    # get test dataframe and path to test folder.
    _, test_path = get_car_paths()
    test_df = get_cars_df('cars_test_annos_withlabels.mat')

    # Load exported model
    model_folder_path = Path("./exported_models")
    model_checkpoint = "res50_head20epochs+all40epochs.pkl"
    inference_learn = get_exported_learner(model_folder_path, model_checkpoint)

    print("="*70)
    print(f"Path to test images folder: \t{str(test_path)}/")
    print(f"Number of test images: \t\t{len(test_df)}")
    print(f"Loaded inference model: \t{model_checkpoint}")
    print(f"\n\nShowing head of the test dataframe: \n {test_df.head()}")
    print("="*70, "\n\n")
    
    # evaluate accuracy on test set
    print("Calculating accuracy of the test set...")
    accuracy = get_test_accuracy(inference_learn, test_df, test_path)
    print(f"Accuracy on the test set is: {accuracy*100:.01f}%")

if __name__ == "__main__":
    main()