# AI for SEA Computer Vision Challenge by Grab
This repo is my submission for the [AI for SEA Computer Vision Challenge](https://www.aiforsea.com/) organized by Grab. The dataset for the CV challenge is the [Stanford Car Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The final submission will be judged based on Code Quality, Creativity in Problem-solving, Feature Engineering and Model Performance.

This submission includes 4 main items for evaluation:
- `test.py`: Use this to evaluate the accuracy and inference time of my submitted models on test set. Two models are included in the `exported_models` folder.
- `train.py`: If needed, use this script for retraining the models with the best settings that I found.
- `predict.py`: Use this to get predictions using one of my submitted models on new images within a folder. An `output.csv` file will be exported.
- `00_Solution_Summary.ipynb`: Refer to this notebook for the summary of the training process and the evaluation of my final solution.

Please refer to [Setup](#Setup) to install the dependencies and the [Usage](#Usage) section for instructions to use `test.py`, `train.py` and `predict.py` for evaluation.

Experiment notebooks are removed because the experiments were not systematically done and not documented.

## Summary of Results
The model of my final solution is a EfficientNet-B0 of [Tan & Le, 2019](https://arxiv.org/abs/1905.11946) (PyTorch implementation by [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/README.md)) and a Resnet-50 as baseline. In this project, it was found that both models can be trained from pretrained weights to reach **92+ % test accuracy** in just **60 epochs** at a batch size of 32 (Efficientnet-B0) or 64 (Resnet-50) by using fast.ai's version of 1cycle policy [(originally by L. Smith 2018)](https://arxiv.org/abs/1803.09820) on a single GPU. In comparison, Tan & Le, 2019 used training settings from [Kornblith et al. 2018](https://arxiv.org/abs/1805.08974) for fine-tuning pretrained Efficientnet-B0 with 20,000 steps at a batch size of 256 (which corresponds to roughtly **600+ epochs**) to reach ~90.8% test accuracy (they used an image size of 224x224, I used 300x300).

*Table 1* summarizes the results of my final solution. Refer to Section 3 and 4 of `00_Solution_Summary.ipynb` where I obtained the results in the table.


*Table 1: Comparison of EfficientBet-B0 and ResNet-50 for Stanfor Car Dataset. Test accuracy and training time are averaged over 8 runs. Batch size of 1 is used to measure the CPU inference time. Training time and CPU inference time are based on a machine with Intel Core i5-8600K and Nvidia GeForce RTX 2070.*

| Model                             | #Params   | Training Time / Epoch | CPU Inference Time / Image| Test Accuracy  |
| -------------                     | :--------: | :---------:                   |:---------:           | -----:    |
| **EfficientNet-B0 (my)**          | 4.3M      | 57.4 s                | 0.11454 s  |92.43%     |
| ResNet-50 (my)                    | 25.7M     | 51.9 s                | 0.18057 s  | 92.64%  |
| EfficientNet-B0 (Tan & Le 2019)   | 5.3M*     |   -    |       -         |90.8%**     |

\* 5.3M is the #Params for ImageNet. Tan & Le, 2019 did not report the #Params for Stanford Car dataset. It should be similar.  
\*\* Image size for EfficientNet-B0 is 224x224 in Tan & Le, 2019, 300x300 in my solution.

In this project, I leveraged the [Fastai](https://docs.fast.ai/) library which is built on top of Pytorch. Fastai provides high level but also flexible APIs for fast and accurate training of neural nets using modern deep learning best practices. The following techniques are used in my final solution.

Data Augmentations:
- resize image to size of (300, 300) using squish method (not crop)
- standard fastai image transforms: random zoom, crop, horizontal flip, rotate, symmetric warp, change brightness/contrast
- extra transforms: random zoom_crop with larger scale range, cutout

Training tricks, the last two tricks were discovered in [Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/abs/1812.01187) which are also impelemted in Fastai:
- [1cycle policy](https://docs.fast.ai/callbacks.one_cycle.html) + [AdamW](https://arxiv.org/abs/1711.05101) optimizer for super fast convergence. 60 epochs to fine-tune ImageNet pretrained EfficientNet-b0 and ResNet-50 to reach 92+% accuracy (40 epochs showed a slight drop in accuracy.)
- Mixed precision training
- Label smoothing
- Mixup


# Setup
## 0. Prerequisites
- Ubuntu 18.04 / 16.04 (Other linux systems should also work, but not tested)
- Conda

## 1. Install Conda
If conda is not yet installed, run the following command in the terminal. After installation, run `conda -h` in terminal to check the installation, it should output help messages of conda.
```
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
chmod +x ~/miniconda.sh && \
~/miniconda.sh -b && \
rm ~/miniconda.sh && \
~/miniconda3/bin/conda init && \
source ~/.bashrc
```
## 2. Clone this repository
Clone this repository and change working directory to the repository.
```
git clone https://github.com/zerothphase/ai4sea_cv_challenge.git && \
cd ai4sea_cv_challenge
```

## 3. Create conda environment and activate
- Using environment.yml (Note: your current directory must be in `/ai4sea_cv_challenge`)
```
conda env create -f environment.yml
conda activate ai4sea
```
- or manually (recommended if your system is not Ubuntu 18.04 / 16.04)
```
conda create -n ai4sea python=3.7 -y
conda activate ai4sea
conda install -c pytorch -c fastai fastai=1.0.53.post2 -y
conda install scikit-learn -y
pip install efficientnet_pytorch==0.1.0
conda install -c anaconda jupyter
```
> **Note 1:**
> If you already have a conda environment named `ai4sea`, edit the name in the first line of the `environment.yml`.  
> **Note 2:** Both steps above will install the pytorch build with the latest cudatoolkit version. If you need a lower CUDA XX build (e.g. CUDA 9.0), following the instructions from [Pytorch's Website](https://pytorch.org/get-started/locally/) to install the desired pytorch build.

# Usage
Make sure you have activated the conda environment and switch your working directory to this repository. During first run of either `test.py` or `train.py`, it will automatically download the train and test data into correct folders.
## 1. Testing models
I have included two models in `exported_models` folder for evaluation, which were trained using `train.py` script: 
- `best_efficientnet-b0.pkl` (Test accuracy: 92.70%)
- `best_resnet-50.pkl` (Test accuracy: 92.79%)

Use `test.py` script with the command below to evaluate the accuracy of the models on test set. 

Basic usage:
```
python test.py -m best_efficientnet-b0.pkl
python test.py -m best_resnet-50.pkl
```
Optionally, you can specify inference device using `-d {cpu, cuda}` and batch size using `-bs int` options. 

Example usage with more options:
```
python test.py -m best_efficientnet-b0.pkl -d cpu -bs 1
python test.py -m best_resnet-50.pkl -d cpu -bs 1
```

## 2. Training models

If needed, `train.py` can be used to retrain the model from Imagenet pretrained weights using the training process of my solution. The number of runs (see options below) can also be specified and the average metrics will be shown at the end. The trained model of the last run will be exported to the `exported_models` folder with a name `exported_<model>.pkl`, which you can also test using `test.py`. You can checkout Section 3 of `00_Solution_Summary.ipynb`, where I used `train.py` to perform 8 training runs to obtain the average metrics of my solution.

Basic usage:
```
python train.py -m efficientnet-b0
python train.py -m resnet-50
```
Options:

`-m  {efficientnet-b0,efficientnet-b3,resnet-50}`  
Model with pretrained weights to be trained. Default = 'efficientnet-b0'
                    
`-n NRUNS`  
Number of training runs. Default = 1

`-e EPOCHS [EPOCHS ...]`  
Number of epochs. Note that resnet-50 will be trained in two phases, hence can accept up to two values. If only one value is specified, then the same value will be used in the second phase.  
Default for `efficientnet-bx` = 60, `resnet-50` = 20 40.

Example usage with more options:
```
python train.py -m efficientnet-b0 -e 40 -n 5
python train.py -m resnet-50 -e 10 20 -n 5
```
## 3. Getting Predictions
`predict.py` can be use to get predictions on new images located in a folder. The options are the same as `test.py`, with an additional option `-f`. Use `-f` to specify the path to the image folder. Note that there must be more than 1 images in specified folder.  
An `output.csv` will be saved in this repository with two columns of `image_path`, predicted `target` and `probability`.

Basic usage:
```
python predict.py -m best_efficientnet-b0.pkl -f ./Data/cars_test
python predict.py -m best_resnet-50.pkl -f ./Data/cars_test
```

# Acknowledgement
Thanks Grab for organizing this challenge in which I learned a lot in the process of researching and experimenting for my best solution. Many thanks to Fast.ai's library and their incredible [Practical Deep Learning for Coders, V3](https://course.fast.ai/) free MOOC. Last but not least, thanks to [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch)'s implementation of Efficientnet in Pytorch.

# References
Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenetmodels transfer better? *CVPR*, 2019.  

Tan, Mingxing and Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML* 2019
