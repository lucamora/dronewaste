import subprocess
import os
import json
import shutil
from datetime import datetime
import argparse
import glob
import random
import numpy as np
import torch
import argparse

from mmengine import Config


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Training script with CLI parameters')

    # Required arguments
    parser.add_argument('--model', type=str, required=True, help='Model name/type to use for training')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to the dataset')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory to save results')

    # Optional arguments with defaults
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training (default: 32)')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers for data loading (default: 4)')
    parser.add_argument('--val_interval', type=int, default=1, help='Validation interval in epochs (default: 1)')
    parser.add_argument('--gpus', type=int, required=True, help='Number of GPUs to use for training')
    parser.add_argument('--download_model', action='store_true', help='Whether to download the model config file and weight. If False, model\'s config and weights should be in config dir (default: False)')
    parser.add_argument('--include_bkg', action='store_true', help='Whether to include background images in the dataset (default: False)')
    parser.add_argument("--fold_id", type=str, default=None)

    # Conditionally required config_directory
    parser.add_argument('--config_directory', type=str, help='Directory containing model config and weights (required when download_model is False)')

    # Parse arguments
    args = parser.parse_args()

    # Custom validation to ensure config_directory is provided when download_model is False
    if not args.download_model and not args.config_directory:
        parser.error('--config_directory is required when --download_model is not set')

    return args

def set_deterministic(seed=0):
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


# utility function to run command
def run_command(command):
    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,  # Line buffering
        universal_newlines=True
    )

    for line in process.stdout:
        print(line, end='')

    for line in process.stderr:
        print(line, end='')

    process.wait()


# utility function to get dataset categories
def get_categories(dataset_path):
    # extract categories from train dataset file
    with open(dataset_path, 'r') as f:
        train_data = json.load(f)
    return [cat['name'] for cat in train_data['categories']]


def get_id_image_mapping(dataset_path):
    with open(dataset_path, 'r') as f:
        data = json.load(f)

    ids_to_images = {im['id']: im['file_name'].split('.')[0] for im in data['images']}
    return ids_to_images


def find_checkpoint(work_dir, epochs=None, best=True):
    # find model checkpoint file
    if best:
        # find best checkpoint file
        best_epoch_file = [ file for file in os.listdir(work_dir) if 'best' in file ][0]
        best_cp = os.path.join(work_dir, best_epoch_file)
    else:
        # find last checkpoint file
        best_epoch_file = [ file for file in os.listdir(work_dir) if f'epoch_{epochs}' in file ][0]
        best_cp = os.path.join(work_dir, best_epoch_file)

    return best_cp


def save_checkpoint(dir):
    print()
    print('Saving checkpoint...')
    print()

    matching_dirs = []

    for root, dirs, _ in os.walk(work_dir):
        for dir_name in dirs:

            if dir_name >= dir:
                matching_dirs.append(os.path.join(root, dir_name))

    directory = matching_dirs[0]
    print(f'Saving checkpoint to: {directory}')
    if len(matching_dirs) > 1:
        print(f'Multiple directories found: {matching_dirs}. Using first one: {directory}')

    weights = glob.glob(os.path.join(work_dir, '*.pth'))
    for weight in weights:
        shutil.move(weight, directory)

    return directory


def update_config():
    print()
    print(f'Updating configuration file for {model}...')
    print()

    # download config file
    if download_model:
        run_command(['mim', 'download', 'mmdet', '--config', model, '--dest', config_directory])

    config_file = os.path.join(config_directory, model, f'{model}.py')
    print(f'Config file: {config_file}')

    #----------------------CONFIG-------------------------#
    # Load config file
    cfg = Config.fromfile(config_file)

    # Modify dataset classes
    categories = get_categories(os.path.join(args.dataset_path, train_path))

    cfg.metainfo = {'classes': categories}

    # Modify dataset path
    cfg.data_root = args.dataset_path + '/' # dataset root
    # data augmentation pipeline
    albu_train_transforms = [
        dict(
            type='ShiftScaleRotate',
            shift_limit=shift_limit,
            scale_limit=scale_limit,
            rotate_limit=rotate_limit,
            interpolation=1,
            p=0.9),
    ]

    # train pipeline
    train_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='RandomResize',
            scale=[(1333, 640), (1333, 800)],
            keep_ratio=True),
        dict(type='Albu',
            transforms=albu_train_transforms,
            bbox_params=dict(
                type='BboxParams',
                format='pascal_voc',
                label_fields=['gt_bboxes_labels', 'gt_ignore_flags'],
                min_visibility=0.0,
                filter_lost_elements=True),
            keymap={
                'img': 'image',
                'gt_masks': 'masks',
                'gt_bboxes': 'bboxes'
            },
            skip_img_without_anno=True),
        dict(type='RandomFlip', prob=flip),
        dict(type='PackDetInputs')
    ]

    #----------------------OPTIMIZER----------------------#
    # keep default settings: momentum SGD, warmup 500 iters, MultiStepLR
    original_batch_size = cfg.auto_scale_lr.base_batch_size
    cfg.optim_wrapper.optimizer.lr = lr / (batch_size / original_batch_size) / gpus
    cfg.param_scheduler = [
        dict(begin=0,
            by_epoch=True,
            end=round(0.2*epochs),
            start_factor=0.01,
            type='LinearLR')
    ]

    # Dataloaders
    cfg.train_dataloader.dataset.dataset.ann_file = train_path
    cfg.train_dataloader.dataset.dataset.data_root = cfg.data_root
    cfg.train_dataloader.dataset.dataset.data_prefix.img = images_path
    cfg.train_dataloader.dataset.dataset.metainfo = cfg.metainfo
    cfg.train_dataloader.dataset.dataset.pipeline = train_pipeline
    if include_bkg:
        cfg.train_dataloader.dataset.dataset.filter_cfg.filter_empty_gt = False
    cfg.train_dataloader.batch_size = batch_size
    cfg.train_dataloader.num_workers = num_workers

    # test pipeline
    test_pipeline = [
        dict(type='LoadImageFromFile'),
        dict(type='Resize', scale=(1333, 800), keep_ratio=True),
        dict(type='LoadAnnotations', with_bbox=True),
        dict(type='PackDetInputs',
            meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor'))
    ]

    cfg.val_dataloader.dataset.ann_file = val_path
    cfg.val_dataloader.dataset.data_root = cfg.data_root
    cfg.val_dataloader.dataset.data_prefix.img = images_path
    cfg.val_dataloader.dataset.metainfo = cfg.metainfo
    cfg.val_dataloader.dataset.pipeline = test_pipeline
    cfg.val_dataloader.batch_size = batch_size
    cfg.val_dataloader.num_workers = num_workers

    cfg.test_dataloader = cfg.val_dataloader

    # Evaluators
    cfg.val_evaluator.ann_file = cfg.data_root + val_path

    cfg.test_evaluator = cfg.val_evaluator
    cfg.test_evaluator.format_only = True
    cfg.test_evaluator.outfile_prefix = os.path.join(work_dir, 'predictions')

    #----------------------MODEL-------------------------#
    checkpoint_file = glob.glob(os.path.join(config_directory, model, '*.pth'))[0]
    cfg.load_from = checkpoint_file

    cfg.default_hooks.checkpoint=dict(type='CheckpointHook', save_best='coco/bbox_mAP_50')

    # set number of classes in bbox head
    cfg.model.roi_head.bbox_head.num_classes = len(categories)

    #----------------------TRAINING----------------------#
    # Training config
    cfg.train_cfg = dict(
        type='EpochBasedTrainLoop',
        max_epochs=epochs,
        val_interval=val_interval
    )

    #----------------------OTHER SETTINGS----------------------#
    # Set seed thus the results are more reproducible
    setattr(cfg, 'randomness',
        dict(seed = seed,
            diff_rank_seed=True,
            deterministic=True
        )
    )

    # Tensorboard support
    cfg.visualizer.vis_backends.append({"type":'TensorboardVisBackend'})
    cfg.default_hooks.logger=dict(type='LoggerHook', interval=20)

    #-------------------------END CONFIG------------------------#

    os.makedirs(work_dir, exist_ok=True)
    config = os.path.join(work_dir, f'{model}_{gpus}xb{batch_size}-{epochs}e.py')
    with open(config, 'w') as f:
        f.write(cfg.pretty_text)

    return config


def train_model(config):
    print()
    print(f'Training {model}, config: {config}, working directory: {work_dir}, num gpus: {gpus} ...')
    print()

    time = datetime.now().strftime("%Y%m%d_%H%M")
    run_command(['mim', 'train', 'mmdet', config, '--work-dir', work_dir, '--auto-scale-lr', '--launcher', 'pytorch', '--gpus', str(gpus)])

    return time


def generate_predictions(ft_directory):
    # Find best checkpoint file
    best_cp = find_checkpoint(ft_directory)

    print()
    print('Generating predictions on validation set...')
    print(f'Best checkpoint: {best_cp}')
    print()

    for split in ['valid', 'test']:
        split_path = val_path if split == 'valid' else test_path
        print(f'Dataset path: {args.dataset_path}/{split_path}')

        cfg = Config.fromfile(config)
        cfg.test_dataloader.dataset.ann_file = split_path
        with open(config, 'w') as f:
            f.write(cfg.pretty_text)

        run_command(['mim', 'test', 'mmdet', config, '--checkpoint', best_cp, '--out', os.path.join(work_dir, f'results_{split}.pkl'), '--gpus', str(gpus)])

        preds_path = os.path.join(work_dir, 'predictions.bbox.json')
        with open(preds_path, 'r') as f:
            data = json.load(f)

        json_file = os.path.join(args.dataset_path, split_path)
        ids_to_images = get_id_image_mapping(json_file)

        for img in data:
            img['image_id'] = ids_to_images[img['image_id']]

        with open(preds_path, 'w') as f:
            json.dump(data, f, indent=4)

        os.remove(os.path.join(work_dir, f'results_{split}.pkl'))
        os.makedirs(os.path.join(work_dir, split), exist_ok=True)
        os.rename(os.path.join(work_dir, 'predictions.bbox.json'), os.path.join(work_dir, split, 'predictions.json'))


# set CUBLAS_WORKSPACE_CONFIG (needed for detrministic training)
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

# Parse command line arguments
args = parse_args()

# use the fold id as the project name
project_name = args.fold_id
# define dataset paths
images_path = os.path.join(args.dataset_path, 'images')
train_path = 'train.json'
val_path = 'val.json'
test_path = 'test.json'

# augmentation parameters
shift_limit = 0.3
scale_limit = 0.3
rotate_limit = 90
flip = 0.5
interpolation = 1

# training parameters
epochs = 300
lr = 0.001
model = args.model
batch_size = args.batch_size
num_workers = args.num_workers
val_interval = args.val_interval
gpus = args.gpus
download_model = args.download_model
config_directory = args.config_directory
include_bkg = args.include_bkg

results_dir = os.path.join(args.results_dir, project_name)
work_dir = os.path.join(os.path.dirname(config_directory), project_name)

# set seed
seed = 0
set_deterministic(seed)

print()
print('Training...')
print()
config = update_config()
ft_time = train_model(config)
ft_directory = save_checkpoint(ft_time)

# Generate predictions
print()
print('Generating predictions...')
print()

generate_predictions(ft_directory)

shutil.copytree(work_dir, results_dir)
shutil.rmtree(work_dir)
