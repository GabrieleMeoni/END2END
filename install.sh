#!/bin/bash

# This script installs the SARLens package and its dependencies

# Check if Conda is installed
if ! command -v conda &> /dev/null
then
    echo "Conda is not installed. Please install Conda and try again."
    echo "See https://docs.anaconda.com/anaconda/install/ for more information."
    exit 1
fi

# Create a new Conda environment and install dependencies
conda env create -f environment.yml

# Activate the AINavi environment
source $(conda info --base)/etc/profile.d/conda.sh
if conda activate end2end; then
    echo "end2end environment activated"
else
    echo "end2end environment not found"
    exit 1
fi

# check if the folder MSMatch/checkpoints exists:
if [ ! -d "MSMatch/checkpoints" ]; then
    python MSMatch/download_checkpoints.py
fi

