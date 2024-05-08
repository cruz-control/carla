#!/bin/bash
BASE_PATH=~/persistent

# Install Miniconda
mkdir -p $BASE_PATH/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O $BASE_PATH/miniconda3/miniconda.sh
bash $BASE_PATH/miniconda3/miniconda.sh -b -u
rm -rf $BASE_PATH/miniconda3/miniconda.sh

$BASE_PATH/miniconda3/bin/conda init bash
$BASE_PATH/miniconda3/bin/conda init zsh

# Actiavte conda
source ~/.bashrc
source ~/.zshrc

# Create conda environment for CARLA
$BASE_PATH/miniconda3/bin/conda create -n carla python=3.8 -y

# Activate the carla conda environment
source $BASE_PATH/miniconda3/bin/activate carla

# Download and extract CARLA
wget https://carla-releases.s3.us-east-005.backblazeb2.com/Linux/CARLA_0.9.15.tar.gz -P $BASE_PATH
mkdir -p $BASE_PATH/CARLA_LATEST
echo "Extracting Carla to Desktop/CARLA_LATEST/"
tar -xvf $BASE_PATH/CARLA_0.9.15.tar.gz -C $BASE_PATH/CARLA_LATEST > /dev/null
rm $BASE_PATH/CARLA_0.9.15.tar.gz

# Install Python CARLA Library
pip3 install carla

# Navigate to PythonAPI/examples directory
cd $BASE_PATH/CARLA_LATEST/PythonAPI/examples

# Install python requirements
python3 -m pip install -r requirements.txt

# Finished!
echo "------------------------------------------------------------"
echo "Installation complete! Please close and re-open your console"
echo "Remember to activate your conda carla environment using \033[1mconda activate carla\033[0m every time you reopen a new console tab"

