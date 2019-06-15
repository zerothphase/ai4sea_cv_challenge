# AI for SEA Computer Vision Challenge by Grab
This repo is my work for the [AI for SEA Computer Vision Challenge](https://www.aiforsea.com/) organized by Grab. The dataset for the CV challenge is the [Stanford Car Dataset](https://ai.stanford.edu/~jkrause/cars/car_dataset.html). The final submission will be judged based on Code Quality, Creativity in Problem-solving, Feature Engineering and Model Performance.

## Summary of Work

Data Augmentations:
- resize image to size of(300, 300) using squish/stretch method (not crop)
- random zoom crop, horizontal flip, rotate, symmetric warp, change brightness/contrast, cutout

Fastai supported tricks:
- 1cycle policy + AdamW optimizer for super fast convergence. 60 epochs to fine-tune ImageNet pretrained EfficientNet-b0 and ResNet-50 to reach 92+% accuracy
- Label smoothing
- Mixup
- Mixed precision training


# Setup
## Prerequisites
- Ubuntu 18.04 / 16.04 (Other linux systems should also work, but not tested)
- Conda

## Install Conda
If conda is not yet installed, run the following command block to the terminal. Run `conda -h` in terminal to check the installation, it should output help messages of conda.
```
curl -o ~/miniconda.sh -O  https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh  && \
chmod +x ~/miniconda.sh && \
~/miniconda.sh -b && \
rm ~/miniconda.sh && \
~/miniconda3/bin/conda init && \
source ~/.bashrc
```
## Clone this repository
```
git clone https://github.com/zerothphase/ai4sea_cv_challenge.git && \
cd ai4sea_cv_challenge
```

## Create conda environment using environment.yml
```
conda env create -f environment.yml
conda activate ai4sea
```

> **Note:**
> If you already have a conda environment named `ai4sea`, edit the name in the first line of the `environment.yml`.

# Usage
## Testing models
I have included two models in `exported_models` folder: `best_efficientnet-b0.pkl` and `best_resnet-50.pkl`. Use `test.py` script to evaluate the accuracy on test set.

To test my models write the name of the model after `-m` option, e.g.:
```
python test.py -m best_efficientnet-b0.pkl
```
Optionally, you can specify inference device using `-d cpu` and batch size using `-bs 64` options, e.g.:
```
python test.py -m best_efficientnet-b0.pkl -d cpu -bs 1
```

## Training

```
python train.py -m efficientnet-b0
```

# TODO
Summarize 
