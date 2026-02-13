#!/bin/bash
# Download LAFAN1 Retargeting Dataset
# Using hf_download environment

# Ensure conda is available
source $(conda info --base)/etc/profile.d/conda.sh

# Activate the environment
conda activate hf_download

# Create directory if it doesn't exist (hf download usually handles this but good practice)
mkdir -p ~/datasets/LAFAN1_Retargeting_Dataset

# Download the dataset
echo "Starting download..."
hf download lvhaidong/LAFAN1_Retargeting_Dataset \
  --repo-type dataset \
  --local-dir ~/datasets/LAFAN1_Retargeting_Dataset

echo "Download complete."
