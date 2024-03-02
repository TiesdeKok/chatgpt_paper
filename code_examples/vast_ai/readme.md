## Vast.AI basics

Vast.ai is a platform to rent cloud instances with various types of GPUs. 

## Instance setup

In order to run the notebooks you need to provision an instance with the following requirements:

- An NVIDIA GPU with at least 24gb vram (e.g., an RTX 4090).  
- At least 50gb of storage space and ideally 32gb ram or higher.  
- I use the following docker image: `nvcr.io/nvidia/pytorch:22.12-py3`

After starting the instance and connecting to the instance through SSH you need to run the following setup commands:

```bash

## Install HTOP for resource monitoring 

apt install htop

## Change directory to the home directory

cd ..
cd home

## Install jupyterlab 

pip install jupyterlab[all] -U

## Install the required Python packages

pip install vllm --use-deprecated=legacy-resolver
pip install joblib==1.3.2

## Bug fix to avoid errors (you can safely ignore this part if it throws errors)

pip uninstall -y transformer-engine
pip uninstall -y apex

## Start Jupyter lab (note this requires you to create an SSH tunnel to access it locally)

jupyter lab --port=8990 --no-browser --allow-root

```