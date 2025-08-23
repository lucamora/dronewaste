#!/bin/bash

echo "Setting up Python virtual environment..."

# Create virtual environment
python -m venv envs/waste
source envs/waste/bin/activate
echo "Virtual environment created at: envs/waste"

echo ""
echo "Installing PyTorch and torchvision..."
# these specific versions are required by mmdetection
pip install torch==2.4.1 torchvision==0.19.1 --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "Installing OpenMMLab packages..."
pip install -U openmim
pip install mmengine
pip install mmcv==2.1.0 -f https://download.openmmlab.com/mmcv/dist/cu121/torch2.4/index.html
pip install mmdet

echo ""
echo "Installing ultralytics..."
pip install ultralytics

echo ""
echo "Installing additional packages..."
pip install -r requirements.txt

echo ""
echo "Installation completed successfully!"

deactivate
