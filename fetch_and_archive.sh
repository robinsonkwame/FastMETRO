#!/bin/bash

# Define directories
export REPO_DIR=$PWD
mkdir -p $REPO_DIR/models  # pre-trained models
mkdir -p $REPO_DIR/models/fastmetro_checkpoint  # model checkpoints
mkdir -p $REPO_DIR/datasets  # datasets

# Set blob URL
BLOB='https://datarelease.blob.core.windows.net/metro'

# Download the ImageNet pre-trained HRNet models 
wget -nc $BLOB/models/hrnetv2_w64_imagenet_pretrained.pth -O $REPO_DIR/models/hrnet/hrnetv2_w64_imagenet_pretrained.pth
wget -nc $BLOB/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O $REPO_DIR/models/hrnet/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml

# Download FastMETRO-L-H64 Model
wget --no-check-certificate 'https://docs.google.com/uc?id=1WU6q27SV7YNGCSBLypB5IGFVWMnL26io' -O $REPO_DIR/models/fastmetro_checkpoint