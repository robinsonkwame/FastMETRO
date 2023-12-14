#!/bin/bash

# Define directories
export REPO_DIR=$PWD
MODELS_DIR=$REPO_DIR/cog/model

if [ ! -d "$MODELS_DIR" ]; then
    fileid="1Np8SAEFEou2HcfDYH7b1a4rjLI1GnwVQ"
    filename="PointHMR-HR32-Human3.6M_state_dict.bin"

    # Fetch the confirmation token by querying the download page
    confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate "https://drive.google.com/uc?export=download&id=${fileid}" -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')

    # Use the confirmation token to complete the download
    wget --load-cookies /tmp/cookies.txt "https://drive.google.com/uc?export=download&confirm=${confirm}&id=${fileid}" -O ${filename} && rm -rf /tmp/cookies.txt

    echo "Checkpoint downloaded successfully. Run cog build -t image2mesh to build!"
else
    echo "Models directory already exists. Skipping download."
fi