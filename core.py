from torch.utils.data import Dataset, DataLoader
import PIL
import torch.nn as nn
import numpy as np
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

class CarsDataset(Dataset):

    def __init__(self, data_path, df, fn_col=0, label_col=1, transform=None):

        self.data_path = data_path
        self.filenames = df.iloc[:, fn_col].values  #array

        self.labels = df.iloc[:, label_col].values  #array
        self.classes = np.unique(self.labels)
        self.num_classes = len(self.classes)
        
        self.le = LabelEncoder()
        self.encoded_labels = self.le.fit_transform(self.labels) #array
        
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        img_path = self.data_path / self.filenames[idx]
        img_label = self.encoded_labels[idx]
        img = PIL.Image.open(img_path)
        if img.mode == 'L':
            img = img.convert(mode='RGB')
        if self.transform:
            img = self.transform(img)
        return img, img_label
    
    def show_sample(self):
        tmp = self.transform
        self.transform = None
        img, img_label = self.__getitem__(np.random.randint(self.num_classes - 1))
        plt.imshow(img)
        plt.title(img_label)
        self.transform = tmp