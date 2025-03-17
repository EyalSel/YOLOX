# Create and activate environment
conda create -n yolo-py310 python=3.10.16
conda activate yolo-py310

# Install PyTorch ecosystem
conda install cudatoolkit=11.3
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 -f https://download.pytorch.org/whl/torch_stable.html

# Install basic utilities
pip install numpy==1.24.4