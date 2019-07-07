from pathlib import Path
import numpy as np
import pandas as pd
import time
from helper import get_car_paths
from helper import get_cars_df
from helper import get_predictions
from helper import get_predictions_from_folder
from efficientnet_pytorch import EfficientNet
import argparse
import torch
from fastai.vision import Learner
from fastai.vision import load_learner
from fastai.vision import ImageList
from fastai.vision import DatasetType
from fastai.vision import defaults
from test import get_exported_learner, default_checkpoint


def parse_args() -> str:
    """Return user inputs"""
    des = ("Script to make predictions with a model   "
           "in `exported_models` folder on images in a folder. ")
    m_help = (f"Choose a trained model file from the folder `exported_models`. "
              f"Default = '{default_checkpoint}'")
    f_help = ("Path to the image folder to be predicted. "
              "An `output.csv` will be created with the predictions. "
              "Default is `./Data/cars_test")
    d_help = ("Choose to infer with 'cpu' or 'cuda'. "
              "Default is 'cuda' if available, else 'cpu'")
    bs_help = "Batch size for inference. Default = 64"
    parser = argparse.ArgumentParser(description=des)
    parser.add_argument("-m", "--model", help=m_help)
    parser.add_argument("-f", "--folder", help=f_help)
    parser.add_argument("-d", "--device", choices=['cpu', 'cuda'], help=d_help)
    parser.add_argument("-bs", "--batch_size", 
                        default=64, type=int, help=bs_help)
    args = parser.parse_args()
    model = args.model
    image_folder_path = args.folder
    device = args.device
    bs = args.batch_size
    return model, image_folder_path, device, bs

def main():
    model_cp, image_folder_path, device, bs = parse_args()
    if device is not None:
        defaults.device = torch.device(device)

    if image_folder_path == None:
        test_path = Path("./Data/cars_test")
    else:
        test_path = Path(image_folder_path)
    
    model_folder_path = Path("./exported_models")
    if model_cp == None:
        model_checkpoint = default_checkpoint
        inference_learn = get_exported_learner(model_folder_path, 
                                               model_checkpoint)
    else:
        model_checkpoint = model_cp
        inference_learn = get_exported_learner(model_folder_path, 
                                               model_checkpoint)
    inference_learn.data.batch_size = bs
    print("="*70)
    print(f"Path to test images folder\t: {str(test_path)}")
    print(f"Number of test images\t\t: {len(list(test_path.glob('*')))}")
    print(f"Loaded inference model\t\t: {model_folder_path/model_checkpoint}")
    print(f"Inference device\t\t: {defaults.device}")
    print(f"Batch size\t\t\t: {bs}")
    print("="*70, "\n\n")
    
    # Evaluate accuracy on test set
    print("Making predictions...")
    start = time.time()
    x, class_preds, probs = get_predictions_from_folder(inference_learn, 
                                                              test_path)
    # Export predictions to output.csv
    output_df = pd.DataFrame(np.array([x, class_preds, probs]).transpose(), 
                             columns=["image_path", "target", "probability"])
    print("Showing dataframe of the first 5 predictions")
    print(output_df.head())
    output_df.to_csv("output.csv", index=False)
    print("\n\nAll predictions are exported as `output.csv`")

    
    
    
    infer_time = time.time() - start
    mins, secs = divmod(infer_time, 60)



if __name__ == "__main__":
    main()