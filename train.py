import numpy as np
import pandas as pd
from helper import get_car_data, get_effnet
import random
import time
from fastai.vision import *
pd.set_option('precision', 4)

model_name = "efficientnet-b0"
epochs = 1

def train_effnet(name, epochs=60, lr=3e-3, wd=1e-3, export_learn=False):
    """Train EfficientNet with the best setting from experiments once"""

    # Get ImageDataBunch and Learner
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
    print("Training...")
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
    


def train_n_runs(n_runs):
    stats_list = []
    export = False
    for i in range(n_runs):
        print("\n" + "="*70)
        print(f"Training Run #{i+1}")
        print("="*70)
        if i == n_runs-1: export=True
        stats_list.append(train_effnet(model_name, epochs=epochs, 
                                       lr=3e-3, wd=1e-3, export_learn=export))

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

train_n_runs(1)