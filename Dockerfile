# Starting from a lightweight python image with CUDA and Conda
FROM nvidia/cuda:11.1.1-base-ubuntu20.04

# Update and install necessary packages
RUN apt-get update && apt-get install wget git -y

# Install miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    bash Miniconda3-latest-Linux-x86_64.sh -b -p /miniconda3 && \
    rm Miniconda3-latest-Linux-x86_64.sh

# Add conda and python to PATH
ENV PATH="/miniconda3/bin:${PATH}"

# Update conda
RUN conda update -n base -c defaults conda

# Create a FastMETRO conda environment
RUN conda create --name fastmetro python=3.8

# Install cudatoolkit
RUN /bin/bash -c "source activate fastmetro && conda install --yes cudatoolkit=11.1 -c pytorch -c conda-forge"

RUN /bin/bash -c "source activate fastmetro && conda install --yes  torchvision=0.9.0 -c pytorch -c conda-forge"

RUN /bin/bash -c "source activate fastmetro && conda install --yes  pytorch=1.8.0 -c pytorch -c conda-forge"

# Clone the FastMETRO repository
RUN git clone --recursive https://github.com/postech-ami/FastMETRO.git

# Set the working directory to FastMETRO
WORKDIR FastMETRO

# RUN FastMETRO setup
RUN /bin/bash -c "source activate fastmetro && python setup.py build develop"

# Install requirements
RUN /bin/bash -c "source activate fastmetro && pip install -r requirements.txt"

# Install manopth
RUN /bin/bash -c "source activate fastmetro && pip install ./manopth/."

# Build 
# docker build -t fastmetro .

# Run
# docker run -it --rm -v {$PWD}/models:/FastMETRO/models -v {$PWD}/images:/FastMETRO/images fastmetro

