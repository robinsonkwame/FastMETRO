# Starting from a lightweight python image with CUDA and Conda
FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Update and install necessary packages
RUN apt-get update && apt-get install wget git -y

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# For opencv we install required libraries
ENV TZ=America/New_York
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    libgl1 \
    libgtk2.0-0

# Add conda and python to PATH
ENV PATH="/miniconda3/bin:${PATH}"

# Update conda
RUN conda update -n base -c defaults conda

# # Create a FastMETRO conda environment
COPY environment.yml environment.yml

RUN /bin/bash -c "conda env create -f environment.yml"

# Clone the FastMETRO repository and build it
# note: the prior environment.yml covers the requirements.txt
#
# Heavily modified from MIT Licensed github.com/postech-ami/FastMETRO/
RUN git clone --recursive https://github.com/robinsonkwame/FastMETRO.git && echo 306

# Set the working directory to FastMETRO
WORKDIR FastMETRO

# RUN FastMETRO setup
RUN /bin/bash -c "source activate fastmetro && python setup.py build develop"

# Install manopth (GPL)
RUN /bin/bash -c "source activate fastmetro && pip install ./manopth/."

# Build 
# docker build -t fastmetro .

# Run
# docker run --gpus all -it -v $PWD/models:/FastMETRO/models -v $PWD/images:/FastMETRO/images fastmetro bash