import numpy as np
import pandas as pd
from helper import get_car_data
from helper import get_effnet
import time
import argparse
from fastai.vision import Learner
from fastai.vision import cnn_learner
from fastai.vision import models
from fastai.vision import imagenet_stats
from fastai.vision import get_transforms
from fastai.vision import zoom_crop
from fastai.vision import cutout
from fastai.vision import LabelSmoothingCrossEntropy
from fastai.vision import accuracy
pd.set_option('precision', 4)


def train_effnet(name, epochs=60, lr=3e-3, wd=1e-3, export_learn=False):
    """Train EfficientNet with the best setting from experiments once"""

    if isinstance(epochs, list): epochs = epochs[0]
    # Get data and learner
    xtra_tfms = (zoom_crop(scale=(0.75,1.5), do_rand=True) 
                 + [cutout(n_holes=(1,4), length=(10, 40), p=0.7)])
    tfms = get_transforms(xtra_tfms=xtra_tfms)
    train_val_data = get_car_data(dataset="train", tfms=tfms, bs=32, 
                                  sz=(300, 300), padding_mode="reflection", 
                                  stratify=True, seed=None)

    eff_net = get_effnet(name=name, pretrained=True, n_class=196)
    learn = Learner(train_val_data, eff_net, 
                    loss_func=LabelSmoothingCrossEntropy(), 
                    metrics=[accuracy], path='.').mixup(alpha=0.3)
    learn.to_fp16()
    
    # Train
    print("")
    print(f"Training {name} for {epochs} epochs...")
    start = time.time()
    learn.fit_one_cycle(epochs, max_lr=lr, wd=wd, div_factor=25, final_div=1e4)
    train_time = time.time() - start
    print("Training completed!")
    print("")   
    val_loss = learn.recorder.val_losses[-1]
    val_acc = learn.recorder.metrics[-1][0]
    print(f"Validation accuracy: \t{val_acc:.05f}")
    if export_learn:
        learn.export("exported_models/exported.pkl")
    
    # Evaluate test metrics
    learn.to_fp32()
    test_data = get_car_data(dataset="test", tfms=None, bs=32, sz=(300, 300), 
                             padding_mode="reflection")
    print("")
    print("Evaluating on test set...")
    # Test data is loaded as train_dl of `test_data`
    test_loss, test_acc = learn.validate(test_data.train_dl)
    print(f"Test accuracy: \t\t{test_acc:.05f}")
    
    stats = (val_loss, val_acc, test_loss, test_acc, train_time)

    return stats
    
def train_resnet(name, epochs=None, lr=3e-3, wd=1e-5, export_learn=False):
    """Train Resnet-50 with the best setting from experiments once"""

    # Get ImageDataBunch and Learner
    xtra_tfms = (zoom_crop(scale=(0.75,1.5), do_rand=True) 
                 + [cutout(n_holes=(1,4), length=(10, 40), p=0.5)])
    tfms = get_transforms(xtra_tfms=xtra_tfms)
    train_val_data = get_car_data(dataset="train", tfms=tfms, bs=64, 
                                  sz=(300, 300), padding_mode="reflection", 
                                  stratify=True, seed=None)

    learn = cnn_learner(train_val_data, models.resnet50, metrics=[accuracy], 
                        loss_func=LabelSmoothingCrossEntropy(), 
                        path='.').mixup(alpha=0.2)
    learn.to_fp16()
    
    if isinstance(epochs, int):
        epochs_p1, epochs_p2 = epochs, epochs
    elif len(epochs) > 1:
        epochs_p1, epochs_p2 = epochs[0], epochs[1]
    else:
        epochs_p1, epochs_p2 = 20, 40

    # Train
    print("")
    print(f"Training {name} for {epochs_p1} + {epochs_p2} epochs...")
    start = time.time()
    # Phase 1, train head only
    print("Phase 1, training head...")
    learn.fit_one_cycle(epochs_p1, max_lr=1e-2, wd=1e-5, div_factor=25, pct_start=0.3)
    # Phase 2, train whole model
    learn.unfreeze()
    print("Phase 2, unfreezed and training the whole model...")
    learn.fit_one_cycle(epochs_p2, max_lr=slice(lr/10, lr), wd=wd, div_factor=25, pct_start=0.3)
    train_time = time.time() - start
    print("Training completed!")
    print("")   
    val_loss = learn.recorder.val_losses[-1]
    val_acc = learn.recorder.metrics[-1][0]
    print(f"Validation accuracy: \t{val_acc:.05f}")
    if export_learn:
        learn.export("exported_models/exported.pkl")
    
    # Evaluate test metrics
    learn.to_fp32()
    test_data = get_car_data(dataset="test", tfms=None, bs=32, sz=(300, 300), 
                             padding_mode="reflection")
    print("")
    print("Evaluating on test set...")
    # Test data is loaded as train_dl of `test_data`
    test_loss, test_acc = learn.validate(test_data.train_dl)
    print(f"Test accuracy: \t\t{test_acc:.05f}")
    
    stats = (val_loss, val_acc, test_loss, test_acc, train_time)

    return stats

def train_n_runs(n_runs, epochs, model_name="efficientnet-b0"):
    stats_list = []
    export = False
    for i in range(n_runs):
        print("\n" + "="*70)
        print(f"Training Run #{i+1}")
        print("="*70)
        if i == n_runs-1: export=True
        if model_name in ["efficientnet-b0", "efficientnet-b3"]:
            stats_list.append(train_effnet(model_name, epochs=epochs, 
                                        lr=3e-3, wd=1e-3, export_learn=export))
        elif model_name in ["resnet-50"]:
            stats_list.append(train_resnet(model_name, epochs=epochs, export_learn=export))
    
    # Print history and average of metrics
    df = pd.DataFrame(np.array(stats_list), 
                      columns=["val_loss", "val_acc", 
                               "test_loss", "test_acc", "time(s)"])
    print("\n" + "="*70)
    print("Metrics history")
    print(df)
    print("")
    print(f"Average metrics over {n_runs} runs")
    print(pd.DataFrame(df.mean(axis=0)).T)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Script to train EfficientNet and Resnet50 for Stanford Car \
                    Dataset"
        )
    parser.add_argument("-n", "--nruns", default=1, type=int,
                        help=f"Number of runs. Default = 1")
    parser.add_argument("-e", "--epochs", default=1, nargs='+', type=int,
                        help="Number of epochs. \nDefault for `efficientnet-bx` \
                              = 60, `resnet-50` = 20 40.")
    parser.add_argument("-m", "--model", default='efficientnet-b0', 
                        choices=['efficientnet-b0', 'efficientnet-b3', 'resnet-50'],
                        help="Choose the model to be trained. \
                              Default = 'efficientnet-b0'")
    args = parser.parse_args()
    n_runs = args.nruns
    epochs = args.epochs
    model = args.model
    return n_runs, epochs, model

if __name__ == "__main__":
    n_runs, epochs, model = parse_args()
    print(n_runs, epochs, model)
    train_n_runs(n_runs, epochs, model_name=model)