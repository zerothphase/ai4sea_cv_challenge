# AI for SEA Computer Vision Challenge by Grab
This repo is my submission for the [AI for SEA Computer Vision Challenge](https://www.aiforsea.com/) organized by Grab. The dataset for the CV challenge is the [Stanford Car Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The final submission will be judged based on Code Quality, Creativity in Problem-solving, Feature Engineering and Model Performance.

## Summary of Work
In this project, I mainly experimented with the latest light weight [EfficientNet](https://arxiv.org/abs/1905.11946) of Tan & Le, 2019 (PyTorch implementation by [lukemelas](https://github.com/lukemelas/EfficientNet-PyTorch/blob/master/README.md)) and uses Resnet50 as comparison. *Table 1* summarizes the results of my final solution.

My Hardware:
- CPU: Intel Core i5-8600K
- GPU: Nvidia GeForce RTX 2070

*Table 1: Comparison of EfficientBet-B0 and ResNet-50. Test accuracy and training time are averaged over 8 runs. Training time and CPU inference time are based on the hardware listed above.*

| Model                             | #Params   | Training Time / Epoch | CPU Inference Time / Image| Test Accuracy  |
| -------------                     | :--------: | :---------:                   |:---------:           | -----:    |
| **EfficientNet-B0 (my)**          | 4.3M      | 57.4 s                | 0.1215 s  |92.43%     |
| ResNet-50 (my)                    | 25.7M     | 51.9 s                | 0.1926 s  | 92.64%  |
| EfficientNet-B0 (Tan & Le 2019)   | 5.3M*     |   -    |       -         |90.8%**     |

\* 5.3M is the #Params for ImageNet. Tan & Le, 2019 did not report the #Params for Stanford Car dataset.  
\*\* Image size for EfficientNet-B0 is 224x224 in Tan & Le, 2019, 300x300 in my solution.

This project mainly uses the [Fastai](https://docs.fast.ai/) library which is built on top of Pytorch. Fastai provides high level but also flexible APIs for fast training of neural nets using modern deep learning best practices. The following techniques implemented in Fastai are used in my final solution.

Data Augmentations:
- resize image to size of(300, 300) using squish/stretch method (not crop)
- standard fastai image transforms: random zoom, crop, horizontal flip, rotate, symmetric warp, change brightness/contrast
- extra transforms: random zoom_crop with larger scale range, cutout

Training tricks:
- [1cycle policy](https://docs.fast.ai/callbacks.one_cycle.html) + [AdamW](https://arxiv.org/abs/1711.05101) optimizer for super fast convergence. 60 epochs to fine-tune ImageNet pretrained EfficientNet-b0 and ResNet-50 to reach 92+% accuracy
- Label smoothing
- Mixup
- Mixed precision training


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
```
git clone https://github.com/zerothphase/ai4sea_cv_challenge.git && \
cd ai4sea_cv_challenge
```

## 3. Create conda environment and activate
- Using environment.yml (Note: your current directory must be in /ai4sea_cv_challenge)
```
conda env create -f environment.yml
conda activate ai4sea
```
- or manually
```
conda create -n ai4sea python=3.7 -y
conda activate ai4sea
conda install -c pytorch -c fastai fastai=1.0.53.post2 -y
conda install scikit-learn -y
pip install efficientnet_pytorch==0.1.0
```
> **Note:**
> If you already have a conda environment named `ai4sea`, edit the name in the first line of the `environment.yml`.

# Usage
## 1. Testing models
I have included two models in `exported_models` folder: 
- `best_efficientnet-b0.pkl` (Test accuracy: 92.70%)
- `best_resnet-50.pkl` (Test accuracy: 92.79%)

Use `test.py` script to evaluate the accuracy on test set. 
To test the model, type the name of the model after `-m` option. 

Basic usage:
```
python test.py -m best_efficientnet-b0.pkl
```
Optionally, you can specify inference device using `-d {cpu, gpu}` and batch size using `-bs int` options. 

Example usage with more options:
```
python test.py -m best_efficientnet-b0.pkl -d cpu -bs 1
```

## 2. Training models

Use `test.py` script to retrain the model from Imagenet pretrained weights. You can also specify the number of runs (see options below) and the average metrics will be shown at the end. The trained model of the last run will also be exported to the `exported_models` folder with name `exported_<model>.pkl`.  

Basic usage example:
```
python train.py -m efficientnet-b0
```
Options:

`-m  {efficientnet-b0,efficientnet-b3,resnet-50}`  
Model with pretrained weights to be trained. Default = 'efficientnet-b0'
                    
`-n NRUNS`  
Number of training runs. Default = 1

`-e EPOCHS [EPOCHS ...]`

Number of epochs. Note that resnet-50 will be trained in two phases, hence can accept up to two values. If only one value is specified, then the same value will be used in the second phase.

Default for `efficientnet-bx` = 60, `resnet-50` = 20 40.

Usage example with more options:
```
python train.py -m efficientnet-b0 -e 40 -n 5
python train.py -m resnet-50 -e 10 20 -n 5
```

# References
Kornblith, S., Shlens, J., and Le, Q. V. Do better imagenetmodels transfer better? *CVPR*, 2019.  

Tan, Mingxing and Le, Q. V. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML* 2019
