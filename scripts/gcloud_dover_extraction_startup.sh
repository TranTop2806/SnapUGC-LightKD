#!/bin/bash
# Startup script for gcloud L4 VM to extract DOVER features
set -e

# 1. System deps
sudo apt-get update -qq
sudo apt-get install -y -qq ffmpeg python3-pip git

# 2. Install Python packages
pip install -q torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install -q timm einops thop opencv-python-headless av numpy pandas scipy

# 3. Clone repos
mkdir -p /workspace && cd /workspace
if [ ! -d "DOVER" ]; then
    git clone https://github.com/VQAssessment/DOVER.git --quiet
fi
if [ ! -d "SnapUGC-LightKD" ]; then
    # User must push repo or use gsutil to copy
    echo "Please copy SnapUGC-LightKD repo to /workspace/"
fi

# 4. Download DOVER weights
python3 -c "
from huggingface_hub import hf_hub_download
hf_hub_download(repo_id='teowu/DOVER', filename='DOVER.pth', local_dir='/workspace/dover_weights')
print('Weights downloaded')
"

# 5. Extract features (example - adjust paths)
# PYTHONWARNINGS=ignore python3 /workspace/SnapUGC-LightKD/scripts/extract_dover_features.py \
#     --video-dir /workspace/videos \
#     --csv /workspace/train_subset_balanced_5000.csv \
#     --out /workspace/dover_features.npz \
#     --device cuda \
#     --ckpt /workspace/dover_weights/DOVER.pth

echo "Setup complete. Run extraction manually with adjusted paths."
