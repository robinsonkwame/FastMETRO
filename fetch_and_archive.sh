#!/bin/bash

# Define directories
export REPO_DIR=$PWD
MODELS_DIR=$REPO_DIR/models/hrnet

if [ ! -d "$MODELS_DIR" ]; then
    #mkdir -p $REPO_DIR/models/hrnet  # pre-trained models
    #mkdir -p $REPO_DIR/models/fastmetro_checkpoint  # model checkpoints
    #mkdir -p $REPO_DIR/datasets  # datasets

    # Set blob URL
    BLOB='https://datarelease.blob.core.windows.net/metro'

    # Download the ImageNet pre-trained HRNet models 
    wget -nc $BLOB/models/hrnetv2_w64_imagenet_pretrained.pth -O $REPO_DIR/cog/models/hrnetv2_w64_imagenet_pretrained.pth
    wget -nc $BLOB/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml -O $REPO_DIR/cog/models/cls_hrnet_w64_sgd_lr5e-2_wd1e-4_bs32_x100.yaml

    # Download FastMETRO-L-H64 Model
    wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1WU6q27SV7YNGCSBLypB5IGFVWMnL26io' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1WU6q27SV7YNGCSBLypB5IGFVWMnL26io" -O $REPO_DIR/cog/models/FastMETRO-L-H64_h36m_state_dict.bin && rm -rf /tmp/cookies.txt

    echo "Models downloaded successfully. Run cog build -t fastmetro to build!"
else
    echo "Models directory already exists. Skipping download."
fi