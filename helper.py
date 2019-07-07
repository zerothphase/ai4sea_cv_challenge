from pathlib import Path
import requests
from tqdm.auto import tqdm
import tarfile
import numpy as np
import scipy.io
import pandas as pd
import os
from fastai.vision import ImageList
from fastai.vision import crop_pad
from fastai.vision import imagenet_stats
from fastai.vision import Learner
from fastai.vision import DatasetType
from fastai.vision import load_learner
from sklearn.model_selection import train_test_split
from efficientnet_pytorch import EfficientNet
import torch
from torch import nn
import shutil

WD_PATH = Path(".")
DATA_PATH = WD_PATH / "Data"
CAR_TRAIN_URL = "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
CAR_TEST_URL = "http://imagenet.stanford.edu/internal/car196/cars_test.tgz"
DEVKIT_URL = "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
TEST_ANNOS_WITHLABELS_URL = "http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat"
# Alternative, faster url from Fast.ai Dataset: https://course.fast.ai/datasets
CAR_TRAIN_TEST_URL = "https://s3.amazonaws.com/fast-ai-imageclas/stanford-cars.tgz"

def download(url:str, dest_dir:Path, fname:str=None) -> Path:
    """Download file from a url to the target directory and return downloaded file path"""
    assert isinstance(dest_dir, Path), "dest_dir must be a Path object"
    if not dest_dir.exists(): 
        dest_dir.mkdir(parents=True, exist_ok=True)
    if fname == None: 
        filename = url.split('/')[-1]
    else: 
        filename = fname
    file_path = dest_dir / filename
    if not file_path.exists():
        print(f"{file_path} not found, downloading...")
        with open(f'{file_path}', 'wb') as f:
            response = requests.get(url, stream=True)
            total = int(response.headers.get('content-length'))
            with tqdm(total=total, unit='B', unit_scale=True, desc=filename) as pbar:
                for data in response.iter_content(chunk_size=1024*1024):
                    f.write(data)
                    pbar.update(1024*1024)
    else:
        return file_path
    return file_path

def untar_tgz(tgz_path: Path, dest_dir:Path) -> Path:
    """Unzip tgz file and return path to the unzipped folder"""
    assert isinstance(tgz_path, Path), "tgz_path must be a Path object"
    assert tgz_path.exists(), "tgz_path does not exists"
    assert tgz_path.suffix == ".tgz", "tgz_path is not a .tgz file"
    if tgz_path.stem == "car_devkit":
        final_path = dest_dir / "devkit"
    else:
        final_path = dest_dir / tgz_path.stem
    if not final_path.exists():
        with tarfile.open(tgz_path, 'r:gz') as tar:
            tar.extractall(dest_dir)
    return final_path

def download_and_untar(url:str, dest_dir:Path) -> Path:
    """Download tgz file from url, unzip and return path to the unzipped folder"""
    tar_path = download(url, dest_dir)
    final_path = untar_tgz(tar_path, dest_dir)
    return final_path

def get_car_paths() -> (Path, Path):
    """Get paths to the train and test data of Stanford Car Dataset

    Returns:
    --------
    train_path, test_path: 
        Tuple of paths to train and test set
    """
    train_path = DATA_PATH / "cars_train"
    test_path = DATA_PATH / "cars_test"
    if not train_path.exists() or not test_path.exists():
        print(f"{train_path} or {test_path} not found, downloading...")
        cars_folder_path = download_and_untar(CAR_TRAIN_TEST_URL, DATA_PATH)
        for f in cars_folder_path.glob("*"):
            shutil.move(str(f), DATA_PATH)
        cars_folder_path.rmdir()
    devkit_path = download_and_untar(DEVKIT_URL, DATA_PATH)
    download(TEST_ANNOS_WITHLABELS_URL, devkit_path)
    assert train_path.exists(), f"{train_path} does not exists!"
    assert test_path.exists(), f"{test_path} does not exists!"
    assert devkit_path.exists(), f"{devkit_path} does not exists!"
    return (train_path, test_path)

def get_idx2name_dic() -> dict:
    """Get index to name dictionary of Stanford Car dataset"""
    devkit_path = download_and_untar(DEVKIT_URL, DATA_PATH)
    cars_metas = scipy.io.loadmat(devkit_path / 'cars_meta.mat')
    cars_metas = cars_metas['class_names'].squeeze()
    df = pd.DataFrame(cars_metas, columns=['class_names'])
    df = df.applymap(np.squeeze)
    df.insert(loc=0, column='class_idx', value=np.arange(1, len(df)+1))
    idx2name_dic = {df.iloc[i, 0]:df.iloc[i, 1] for i in range(len(df))}
    return idx2name_dic

def get_cars_df(annos_matfile:str) -> pd.DataFrame:
    """Get dataframe with columns of fname, class_name and class of Stanford Car dataset"""
    assert annos_matfile in ['cars_train_annos.mat', 'cars_test_annos_withlabels.mat'], \
           "Please select 'cars_train_annos.mat' or 'cars_test_annos_withlabels.mat'"
           
    devkit_path = download_and_untar(DEVKIT_URL, DATA_PATH)
    cars_annos = scipy.io.loadmat(devkit_path / annos_matfile)
    cars_annos = cars_annos['annotations'].squeeze()
    df = pd.DataFrame(cars_annos)
    df = df.applymap(np.squeeze)
    idx2name_dic = get_idx2name_dic()
    df['class_name'] = df['class'].map(idx2name_dic)
    return df[['fname', 'class_name', 'class']]

def get_car_data(dataset:str="train", tfms=None, bs:int=32, sz=224, 
                 padding_mode:str='reflection', stratify:bool=True, 
                 seed:int=None, split_pct:float=0.2):
    """Get train or test ImageDataBunch of Stanford Car dataset
    
    Parameters:
    -----------
    dataset:
        Select "train" or "test" dataset
    tfms:
        Transformations for data augmentation of train set. 
    bs:
        Batch size
    sz: 
        Final size of image. If int is given, crop method will be used to resize.
        If a tuple of two ints is given, squish method will be used.
    padding_mode:
        Padding method. See fastai docs for more methods.
    stratify:
        If True, stratified train/valid split will be applied, else random split.
    split_pct:
        Split percentage for validation set

    Returns:
    --------
    data:
        ImageDataBunch of train or test set
    
    """
    assert dataset in ["train", "test"], "`dataset` must be `train` or `test`"
    train_path, test_path = get_car_paths()
    train_df = get_cars_df('cars_train_annos.mat')
    test_df = get_cars_df('cars_test_annos_withlabels.mat')
    if stratify:
        strat = train_df.iloc[:,1]
    else:
        strat = None
    if dataset == "train":
        _, val_idx = train_test_split(range(len(train_df)), test_size=split_pct, 
                                            random_state=seed, stratify=strat)
        data = (ImageList.from_df(train_df, train_path, cols=0)
                .split_by_idx(val_idx)
                .label_from_df(cols=1)
                .transform(tfms, size=sz, padding_mode=padding_mode)
                .databunch(bs=bs).normalize(imagenet_stats)
                )
    elif dataset == "test":
        data = (ImageList.from_df(test_df, test_path, cols=0)
                .split_none()
                .label_from_df(cols=1)
                .transform([crop_pad(), crop_pad()], size=sz, padding_mode=padding_mode)
                .databunch(bs=bs).normalize(imagenet_stats)
                )
    return data

def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.kaiming_normal_(m.weight)
    elif type(m) == nn.BatchNorm1d:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)
        
def split_effnet(m):
    blocks_length = len(m._blocks)
    return ([m._conv_stem, m._bn0] + list(m._blocks.children())[:blocks_length//2], 
            list(m._blocks.children())[blocks_length//2:] + [m._conv_head, m._bn1],
            list(m.children())[5]
            )

def get_effnet(name:str="efficientnet-b0", pretrained:bool=True, 
               n_class:int=None, dropout_p:float=0.5):
    """Get Efficientnet for multiclass classification with `n_class` 

    Parameters:
    -----------
    name: {"efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3"}
        Name of the efficientnet variant
    pretrained:
        If True, pretrained weights will be loaded
    n_class:
        Number of unique class of the targets
    dropout_p:
        Dropout probability of the FC layer of the new head.
    Returns:
    --------
    m: 
        Efficientnet nn.Module
    """
    assert n_class != None, "Please specify the number of output classes `n_class`"
    if pretrained == True:
        print(f"Getting pretrained {name}")
        m = EfficientNet.from_pretrained(name)
    else:
        print(f"Getting random initialized {name}")
        m = EfficientNet.from_name(name)
    
    n_in = m._fc.in_features
    m._fc = nn.Sequential(
        nn.Dropout(p=dropout_p), 
        nn.Linear(n_in, n_class))
    m._fc.apply(init_weights)
    return m

def get_predictions(learn:Learner, imagelist:ImageList):
    learn.data.add_test(imagelist)
    logits, _ = learn.get_preds(ds_type=DatasetType.Test)
    probs = torch.nn.functional.softmax(logits, dim=1)
    max_probs, y_preds = torch.max(probs, dim=1)
    x = imagelist.items
    return x, y_preds.numpy(), max_probs.numpy()

def get_predictions_from_folder(learn:Learner, 
                                test_path:Path) -> (np.ndarray, np.ndarray, np.ndarray):
    """Get predictions of images in a folder

    Parameters:
    -----------
    learn: 
        Inference Learner object
    test_path: 
        Path to the folder where test images are located.
    
    Returns:
    --------
    x:
        Paths of the input images
    y_preds:
        Predicted indices of the class with the highest probability
    max_probs:
        Probabilities of y_preds
    """
    test_imagelist = ImageList.from_folder(test_path)
    x, y_preds, max_probs = get_predictions(learn, test_imagelist)
    return x, y_preds, max_probs

def get_predictions_from_df(learn:Learner, test_df:pd.DataFrame, test_path:Path, 
                            cols:int=0) -> (np.ndarray, np.ndarray, np.ndarray):
    """Get predictions of images from dataframe

    Parameters:
    -----------
    learn: 
        Inference Learner object
    test_df: 
        DataFrame with filenames of the test images in one of the columns.
    test_path: 
        Path to the folder where test images are located.
    cols:
        Column index of the images' filenames.
    
    Returns:
    --------
    x:
        Paths of the input images
    y_preds:
        Predicted indices of the class with the highest probability
    max_probs:
        Probabilities of y_preds
    """
    test_imagelist = ImageList.from_df(test_df, test_path, cols=cols)
    x, y_preds, max_probs = get_predictions(learn, test_imagelist)
    return x, y_preds, max_probs

def get_exported_learner(folder_path:Path, filename:str) -> Learner:
    """Return a Learner for inference
    
    Load the exported Learner with `filename` (e.g. `learner.pkl`) in
    the folder `folder_path` and return it.

    Parameter:
    ----------
    folder_path:
        Path to the folder containing the exported Learner
    filename:
        Filename of the exported Learner, which is exported using 
        learn.export() method.
    
    Return:
    -------
    learn:
        Fastai Learner object
    """
    model_path = folder_path / filename
    assert model_path.exists(), f"{filename} not found in {str(folder_path)}."
    learn = load_learner(folder_path, filename)
    learn.to_fp32()
    return learn

def idx_to_classname(y_preds, learn:Learner):
    """Convert predicted indices to classname using mapping from learn."""
    class_preds = np.array(learn.data.single_ds.y.classes)[y_preds]
    return class_preds