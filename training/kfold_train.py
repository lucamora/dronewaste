import json
import sys
import argparse
import subprocess
from itertools import chain

from kfold_dataset import SitesDatasetSplit

parser = argparse.ArgumentParser()
parser.add_argument('--arch', type=str, required=True)
parser.add_argument('--site_index', type=int, required=True)
parser.add_argument('--run_id', type=str, required=True)
parser.add_argument('--dataset_path', type=str, required=True)
parser.add_argument('--tmp_dataset_path', type=str, required=True)
parser.add_argument('--results_path', type=str, required=True)
parser.add_argument('--no_empty', action='store_true', default=False)
args = parser.parse_args()


# fixed parameters
YOLO_GPUS = '0,1'  # GPU ids
FASTER_GPUS = '2'  # number of GPUs
VALID_SPLIT = 0.1


# check architecture
if args.arch not in ['faster', 'yolov8', 'yolov12']:
    raise ValueError(f'Invalid architecture: {args.arch}, must be ["faster", "yolov8", "yolov12"]')

# load sites
with open(args.dataset_path, 'r') as f:
    global_gt = json.load(f)
sites = sorted(list(set([i['site'] for i in global_gt['images']])))

# check site index
if args.site_index >= len(sites):
    raise ValueError(f'Invalid site index: {args.site_index}, must be < {len(sites)}')

# get site for current fold
SITE_NAME = sites[args.site_index]
# dataset format
DATASET_FORMAT = 'coco' if args.arch == 'faster' else 'yolo'

print(f'K-fold on {len(sites)} sites')
print(sites)
print()
print(f'Arch: {args.arch}')
print(f'Run id: {args.run_id}')
print(f'Site: {SITE_NAME} (idx: {args.site_index})')
print(f'Temporary dataset path: {args.tmp_dataset_path}')
print(f'Output path: {args.results_path}')
print()


def train_and_inference_yolov8():
    arguments = [
        ('python', 'training_yolov8.py'),
        ('--model', 'yolov8x'),
        ('--dataset_yaml', dataset.dataset_file),
        ('--results_path', args.results_path),
        ('--batch_size', '256'),
        ('--device', YOLO_GPUS),
        ('--fold_id', args.run_id),
    ]
    # flatten list of tuples
    arguments = list(chain.from_iterable(arguments))

    print(f'training yolov8: {" ".join(arguments)}')
    sys.stdout.flush()
    subprocess.run(arguments, check=True, cwd='./yolov8')


def train_and_inference_yolov12():
    arguments = [
        ('python', 'training_yolov12.py'),
        ('--model', 'yolov12x'),
        ('--dataset_yaml', dataset.dataset_file),
        ('--results_path', args.results_path),
        ('--batch_size', '256'),
        ('--device', YOLO_GPUS),
        ('--fold_id', args.run_id),
    ]
    # flatten list of tuples
    arguments = list(chain.from_iterable(arguments))

    print(f'training yolov12: {" ".join(arguments)}')
    sys.stdout.flush()
    subprocess.run(arguments, check=True, cwd='./yolov12')


def train_and_evaluate_faster_rcnn():
    arguments = [
        ('python', 'training_fasterRCNN.py'),
        ('--model', 'faster-rcnn_r50_fpn_mstrain_3x_coco'),
        ('--dataset_path', dataset.export_path),
        ('--results_dir', args.results_path),
        ('--config_directory', 'configs'),
        ('--batch_size', '64'),
        ('--gpus', FASTER_GPUS),
        ('--fold_id', args.run_id),
        ('--include_bkg'),
    ]
    # flatten list of tuples
    arguments = list(chain.from_iterable(arguments))

    print(f'training faster-rcnn: {" ".join(arguments)}')
    sys.stdout.flush()
    subprocess.run(arguments, check=True, cwd='./fasterRCNN')


print(f'Running site: {SITE_NAME} ({args.site_index+1}/{len(sites)})')
print()

# create dataset for the current site
# the current site is the test set for the current fold
dataset = SitesDatasetSplit(
    dataset_path=args.dataset_path,
    output_path=args.tmp_dataset_path,
    format=DATASET_FORMAT,
    timestamp=args.run_id,
    copy_images=False,
    no_empty=args.no_empty,
)

# get trainval and test sites
trainval_sites, _, test_sites = dataset.split(train='all', test=[SITE_NAME])

# create train, val, test sets (val is split from train)
tr, va, te = dataset.create(
    train=trainval_sites,
    val=VALID_SPLIT,
    test=test_sites,
)

print(f'fold dataset split: {SITE_NAME} ({dataset.export_path})')
print(f' - train: {tr[0]} imgs, {tr[1]} annots')
print(f' - val: {va[0]} imgs, {va[1]} annots')
print(f' - test: {te[0]} imgs, {te[1]} annots')
print()

if args.arch == 'yolov8':
    # train yolov8
    train_and_inference_yolov8()
elif args.arch == 'yolov12':
    # train yolov12
    train_and_inference_yolov12()
else:
    # train faster-rcnn
    train_and_evaluate_faster_rcnn()

# delete temporary dataset split
print()
print('deleting fold dataset split...')
print()
dataset.delete()
