# Script to evaluate on test set
from pathlib import Path
import numpy as np
import pandas as pd
import time
from helper import get_car_paths, get_cars_df
from efficientnet_pytorch import EfficientNet
import argparse
import torch
from fastai.vision import Learner
from fastai.vision import load_learner
from fastai.vision import ImageList
from fastai.vision import DatasetType
from fastai.vision import defaults

default_checkpoint = 'best_efficientnet-b0.pkl'


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
    assert model_path.exists(), f"{filename} not found in {str(folder_path)}."
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

def parse_args() -> str:
    des = ("Script to evaluate exported models  in `exported_models` folder "
           "on test set. ")
    m_help = (f"Choose a trained model file from the folder `exported_models`. "
              f"Default = '{default_checkpoint}'")
    d_help = ("Choose to infer with 'cpu' or 'cuda'. "
              "Default is 'cuda' if available, else 'cpu'")
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument("-m", "--model", help=m_help)
    parser.add_argument("-d", "--device", choices=['cpu', 'cuda'], help=d_help)
    args = parser.parse_args()
    model = args.model
    device = args.device
    return model, device

def main():
    
    model_cp, device = parse_args()
    if device is not None:
        defaults.device = torch.device(device)
    
    # get test dataframe and path to test folder.
    _, test_path = get_car_paths()
    test_df = get_cars_df('cars_test_annos_withlabels.mat')

    # Load exported model
    model_folder_path = Path("./exported_models")
    if model_cp == None:
        model_checkpoint = default_checkpoint
        inference_learn = get_exported_learner(model_folder_path, 
                                               model_checkpoint)
    else:
        model_checkpoint = model_cp
        inference_learn = get_exported_learner(model_folder_path, 
                                               model_checkpoint)

    print("="*70)
    print(f"Path to test images folder: \t{str(test_path)}/")
    print(f"Number of test images: \t\t{len(test_df)}")
    print(f"Loaded inference model: \t{model_checkpoint}")
    print(f"\n\nShowing head of the test dataframe: \n {test_df.head()}")
    print("="*70, "\n\n")
    
    # evaluate accuracy on test set
    print("Calculating accuracy of the test set...")
    accuracy = get_test_accuracy(inference_learn, test_df, test_path)
    print(f"Accuracy on the test set is: {accuracy*100:.02f}%")


if __name__ == "__main__":
    main()