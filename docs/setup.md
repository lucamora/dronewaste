# Setup instructions

## Repository cloning

The [evaluation notebook](../evaluation/evaluation.ipynb) is based on a [forked repo](https://github.com/wtiandong/tide) of the original [TIDE toolbox](https://github.com/dbolya/tide) which was adapted to report per-class APs. The modified TIDE toolbox is included in the `evaluation` folder as a submodule.

Clone both the current repository and its submodule using the following command:

```bash
git clone --recursive https://github.com/lucamora/dronewaste
```

## Virtual environments

Since YOLOv8 and YOLOv12 use the same `ultralytics` namespace, two separate virtual environments should be created to avoid conflicts. The [training script](../training/train.sh) will select the correct environment based on the selected model architecture.

### YOLOv8 and Faster-RCNN

The first virtual environment (`waste`) contains the dependencies for YOLOv8 and Faster-RCNN.
Create the virtual environment and install the dependencies both for `ultralytics` and `mmdetection` using the following command:

```bash
python setup.sh
```

### YOLOv12

The second virtual environment (`yolov12`) contains the dependencies for YOLOv12.
The installation instructions for YOLOv12 are defined in the [original GitHub repo](https://github.com/sunsmarterjie/yolov12).

Instead of creating a conda environment, as in the original repo, create a virtual environment using the following command:

```bash
python -m venv envs/yolov12
source envs/yolov12/bin/activate
```

After installing YOLOv12, install the additional dependencies using the following command:

```bash
pip install -r requirements.txt
```

*Note: If the names of the virtual environments (`waste`, `yolov12`) are changed, the [training script](../training/train.sh) must be updated accordingly.*
