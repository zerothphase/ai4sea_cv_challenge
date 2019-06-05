from pathlib import Path
import requests
from tqdm.auto import tqdm
import tarfile
import numpy as np
import scipy.io
import pandas as pd
import os

WD_PATH = Path(".")
DATA_PATH = WD_PATH / "Data"
if not DATA_PATH.exists():
    DATA_PATH.mkdir()
CAR_TRAIN_URL = "http://imagenet.stanford.edu/internal/car196/cars_train.tgz"
CAR_TEST_URL = "http://imagenet.stanford.edu/internal/car196/cars_test.tgz"
DEVKIT_URL = "https://ai.stanford.edu/~jkrause/cars/car_devkit.tgz"
TEST_ANNOS_WITHLABELS_URL = "http://imagenet.stanford.edu/internal/car196/cars_test_annos_withlabels.mat"


def download(url:str, dest_dir:Path) -> Path:
    assert isinstance(dest_dir, Path), "dest_dir must be a Path object"
    filename = url.split('/')[-1]
    file_path = dest_dir / filename
    if not file_path.exists():
        with open(f'{dest_dir}/{filename}', 'wb') as f:
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
    tar_path = download(url, dest_dir)
    final_path = untar_tgz(tar_path, dest_dir)
    return final_path

def get_car_paths() -> (Path, Path):
    train_path = download_and_untar(CAR_TRAIN_URL, DATA_PATH)
    test_path = download_and_untar(CAR_TEST_URL, DATA_PATH)
    devkit_path = download_and_untar(DEVKIT_URL, DATA_PATH)
    download(TEST_ANNOS_WITHLABELS_URL, devkit_path)
    return (train_path, test_path)

def get_idx2name_dic() -> dict:
    devkit_path = download_and_untar(DEVKIT_URL, DATA_PATH)
    cars_metas = scipy.io.loadmat(devkit_path / 'cars_meta.mat')
    cars_metas = cars_metas['class_names'].squeeze()
    df = pd.DataFrame(cars_metas, columns=['class_names'])
    df = df.applymap(np.squeeze)
    df.insert(loc=0, column='class_idx', value=np.arange(1, len(df)+1))
    idx2name_dic = {df.iloc[i, 0]:df.iloc[i, 1] for i in range(len(df))}
    return idx2name_dic

def get_cars_df(annos_matfile:str) -> pd.DataFrame:
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

def get_train_val_idx(num_examples, val_percent=0.2):
    num_train = int(num_examples*(1-val_percent))
    random_idx = np.random.permutation(num_examples)
    train_idx, val_idx = random_idx[:num_train], random_idx[num_train:]
    return train_idx, val_idx
