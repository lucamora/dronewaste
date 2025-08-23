import os
import shutil
import random
import numpy as np
import yaml
import torch
from ultralytics import YOLO
import time
from datetime import datetime
import argparse
import wandb


# reference: network architecture
# https://github.com/ultralytics/ultralytics/blob/main/ultralytics/cfg/models/v8/yolov8.yaml

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--dataset_yaml", type=str, required=True)
parser.add_argument("--results_path", type=str, required=True)
parser.add_argument("--img_size", type=int, default=640)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--epochs", type=int, default=300)
parser.add_argument("--patience", type=int, default=50)
parser.add_argument("--lr_tl", type=float, default=0.001)  # 1e-3
parser.add_argument("--lr_ft", type=float, default=0.0001)  # 1e-4
parser.add_argument("--device", type=str, required=True)
parser.add_argument("--fold_id", type=str, default=None)
args = parser.parse_args()

# CLI arguments
MODEL = args.model
DATASET = args.dataset_yaml
RESULTS_PATH = args.results_path
IMG_SIZE = args.img_size
BATCH_SIZE = args.batch_size
EPOCHS = args.epochs
PATIENCE = args.patience
LR_TL = args.lr_tl
LR_FT = args.lr_ft
DEVICE = args.device
RUN = datetime.now().strftime("%Y-%m-%d_%H-%M")  # get current timestamp


# training parameters
optimizer = "AdamW"
frozen_layers_tl = 10  # freeze backbone layers
frozen_layers_ft = 2  # freeze backbone layers during fine tuning
train_cache = 'disk'  # disk cache required for deterministic training
val_during_train = True  # enable validation during training
train_save_plots = True  # save plots of training and validation metrics

# evaluation parameters
object_conf_thres = 0.001
iou_thres = 0.5
eval_save_json = True
eval_save_plots = True

# augmentation parameters
degrees = 90.0
translate = 0.2
scale = 0.1
flipud = 0.5
fliplr = 0.5
mosaic = 1.0  # combines four training images into one (default=1.0, set 0.0 to disable)
mixup = 1.0  # mixup coefficient (default=0.0, set 0.0 to disable)

hsv_h=0
hsv_s=0
hsv_v=0


def set_deterministic(seed=1337):
    # Python random module
    random.seed(seed)

    # Numpy
    np.random.seed(seed)

    # PyTorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Additional PyTorch deterministic settings
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def transfer_learning():
    print()
    print(f"[{RUN}] transfer learning ...")
    print()

    model = YOLO(f"{MODEL}.pt")

    model.train(
        data=data_path,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=wandb_project,
        name=project_name + "_tl",  # create a run folder
        freeze=frozen_layers_tl,  # freeze layers
        epochs=EPOCHS,
        patience=PATIENCE,
        optimizer=optimizer,
        verbose=False,
        seed=seed,
        deterministic=True,
        amp=False,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        lr0=LR_TL,
        cache=train_cache,
        val=val_during_train,
        plots=train_save_plots,  # plots saving is required to avoid errors
    )
    # release GPU and free RAM by deleting the model
    del model
    torch.cuda.empty_cache()


def fine_tuning():
    print()
    print(f"[{RUN}] fine tuning ...")
    print()

    model = YOLO(os.path.join(wandb_project, project_name + "_tl", "weights", "best.pt"))

    model.train(
        data=data_path,
        batch=BATCH_SIZE,
        imgsz=IMG_SIZE,
        device=DEVICE,
        project=wandb_project,
        name=project_name + "_ft",  # create a run folder
        freeze=frozen_layers_ft,  # freeze layers
        epochs=EPOCHS,
        patience=PATIENCE,
        optimizer=optimizer,
        verbose=False,
        seed=seed,
        deterministic=True,
        amp=False,
        hsv_h=hsv_h,
        hsv_s=hsv_s,
        hsv_v=hsv_v,
        degrees=degrees,
        translate=translate,
        scale=scale,
        flipud=flipud,
        fliplr=fliplr,
        mosaic=mosaic,
        mixup=mixup,
        lr0=LR_FT,
        cache=train_cache,
        val=val_during_train,
        plots=train_save_plots,  # plots saving is required to avoid errors
    )
    del model
    torch.cuda.empty_cache()
    return


def evaluate_model(model, split):
    print()
    print(f'[{RUN}] evaluating model on "{split}" set ...')
    print()

    model = YOLO(model)

    metrics = model.val(
        project=wandb_project,  # save inside the run folder
        name=f'{project_name}_{split}',  # create a split folder
        split='val' if split == 'valid' else 'test',
        conf=object_conf_thres,
        iou=iou_thres,
        save_json=eval_save_json,
        plots=eval_save_plots,
    )

    # reference: validation metrics
    # https://docs.ultralytics.com/reference/utils/metrics/#ultralytics.utils.metrics.Metric
    box = metrics.box
    return {
        "p": box.mp,  # mean precision
        "r": box.mr,  # mean recall
        "map50": box.map50,  # mean AP @ 0.5
        "map75": box.map75,  # mean AP @ 0.75
        "map": box.map,  # mean AP @ 0.5-0.95
    }


def complete_training(output_path, project_name, val, test):
    print()
    print(f"[{RUN}] saving results ...")
    print()

    # create project folder
    os.makedirs(os.path.join(output_path, project_name), exist_ok=True)

    # save metrics
    output = f"{RUN};"
    output += f'{val["p"]:.4f};{val["r"]:.4f};{val["map50"]:.4f};{val["map75"]:.4f};{val["map"]:.4f};'
    output += f'{test["p"]:.4f};{test["r"]:.4f};{test["map50"]:.4f};{test["map75"]:.4f};{test["map"]:.4f}\n'
    with open(os.path.join(output_path, "runs.txt"), "a") as f:
        f.write(output.replace(".", ","))

    # move artifacts to results folder
    shutil.move(
        src=os.path.join(wandb_project, project_name + "_tl"),
        dst=os.path.join(output_path, project_name, "tl"),
    )
    shutil.move(
        src=os.path.join(wandb_project, project_name + "_ft"),
        dst=os.path.join(output_path, project_name, "ft"),
    )
    shutil.move(
        src=os.path.join(wandb_project, project_name + "_valid"),
        dst=os.path.join(output_path, project_name, "valid"),
    )
    shutil.move(
        src=os.path.join(wandb_project, project_name + "_test"),
        dst=os.path.join(output_path, project_name, "test"),
    )
    shutil.move(
        src=os.path.join(wandb_project, f"{project_name}_params.yaml"),
        dst=os.path.join(output_path, project_name, "params.yaml"),
    )


def initialize_project(project_name):
    print()
    print(f'[{RUN}] initializing project "{project_name}" ...')
    print()

    # save kfold parameters
    with open(os.path.join(wandb_project, f"{project_name}_params.yaml"), "w") as f:
        yaml.safe_dump(
            {
                "model": MODEL,
                "dataset": DATASET,
                "img_size": IMG_SIZE,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "patience": PATIENCE,
                "lr_tl": LR_TL,
                "lr_ft": LR_FT,
                "optimizer": optimizer,
                "frozen_layers_tl": frozen_layers_tl,
                "frozen_layers_ft": frozen_layers_ft,
                "object_conf_thres": object_conf_thres,
                "iou_thres": iou_thres,
                "degrees": degrees,
                "translate": translate,
                "scale": scale,
                "flipud": flipud,
                "fliplr": fliplr,
                "mosaic": mosaic,
                "mixup": mixup,
                "hsv_h": hsv_h,
                "hsv_s": hsv_s,
                "hsv_v": hsv_v,
                "seed": seed,
            },
            f,
            sort_keys=False,
        )


seed = 1337
set_deterministic(seed=seed)

# during kfold training, use the fold id as the project name
project_name = args.fold_id

# define project details and paths
wandb_project = "uav_waste"
data_path = DATASET

# initialize wandb
wandb.login(key='') # insert your wandb API key here

# create project folder and save parameters
initialize_project(project_name)

# train: transfer learning
tl_start = time.time()
transfer_learning()
tl_end = time.time()

BATCH_SIZE = BATCH_SIZE // 2  # reduce batch size for fine tuning

# train: fine tuning
ft_start = time.time()
fine_tuning()
ft_end = time.time()

# evaluate model
model = os.path.join(wandb_project, project_name + "_ft", "weights", "best.pt")
val_metrics = evaluate_model(model, "valid")
test_metrics = evaluate_model(model, "test")

# save fold results and move artifacts
complete_training(RESULTS_PATH, project_name, val_metrics, test_metrics)

print()
print(f'transfer learning: {(tl_end-tl_start)/60:.3f} minutes')
print(f'fine tuning: {(ft_end-ft_start)/60:.3f} minutes')
print(f"[{RUN}] done!")
