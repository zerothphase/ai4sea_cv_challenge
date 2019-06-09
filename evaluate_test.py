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

# Note this import * is intentionally done, as recommended by fast.ai to make sure 
# that everything works
from fastai.vision import *
defaults.device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")



def get_exported_learner(folder_path:Path, filename:str) -> Learner:
    """
    Get the exported Learner with `filename` (e.g. `learner.pkl`) in
    the folder `folder_path`

    Parameter:
        folder_path:
            Path to the folder containing the exported Learner
        filename:
            Filename of the exported Learner
    Return:
        learn:
            Fastai Learner object
    """
    model_path = folder_path / filename
    assert model_path.exists()
    learn = load_learner(folder_path, filename)
    learn.to_fp32()
    return learn

def get_predictions(learn:Learner, test_df:pd.DataFrame, test_path:Path, 
                    cols:int=0) -> (np.ndarray, np.ndarray):
    """
    Get predictions of images in folder `test_path` with filenames listed 
    in `cols`th column of `test_df`

    Parameter:
        learn: 
            Trained fastai Learner object
        test_df: 
            DataFrame with filenames of the test images in one of the 
            columns.
        test_path: 
            Path to the folder where test images are located.
        cols:
            Column index of the images' filenames.
    
    Return:
        probs:
            Predicted probabilities of each classes
        y_preds:
            Predicted class index
    """
    test_imagelist = ImageList.from_df(test_df, test_path, cols=cols)
    learn.data.add_test(test_imagelist)
    probs, _ = learn.get_preds(ds_type=DatasetType.Test)
    y_preds = torch.argmax(probs, dim=1)
    return probs.numpy(), y_preds.numpy()

def get_test_accuracy(learn:Learner, test_df:pd.DataFrame, test_path:Path, 
                    fn_col:int=0, label_col:int=1) -> np.float:
    """
    Get test accuracy of test images  

    Parameter:
        learn: 
            Trained fastai Learner object
        test_df: 
            DataFrame with filenames and class_names of the test 
            images.
        test_path: 
            Path to the folder where the test images are located.
        fn_col:
            Column index of the images' filenames.
        label_col:
            Column index of the images' label (class_name). This 
            will be mapped to the corresponding label index to 
            be compared with the predicted label index by `learn`
    
    Return:
        probs:
            Predicted probabilities of each classes
        y_preds:
            Predicted class index
    """
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