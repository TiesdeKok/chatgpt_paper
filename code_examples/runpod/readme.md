## Runpod basics

[Runpod.io](https://www.runpod.io/) is a platform to rent cloud instances with various types of GPUs.  

It can be more expensive than Vast.ai, but it is more secure, easier, and has more powerful GPUs available. 

## Fine-tuning basics

There are multiple ways to "fine-tune" a generative LLM:

- `full` --> continue the training and make all the parameters tunable.
- `partial` --> methods that only tune a subset of the parameters, such as by freezing layers or by using a technique like LoRa.

The `partial` variants require significantly less GPU VRAM (or system memory if you off load using Deepspeed), but they might not work as well as a `full` fine-tune. Here is a copy of the memory requirements diagram by `LlaMA-Factory`:

| Method | Bits |   7B  |  13B  |  30B  |   65B  |   8x7B |
| ------ | ---- | ----- | ----- | ----- | ------ | ------ |
| Full   |  16  | 160GB | 320GB | 600GB | 1200GB |  900GB |
| Freeze |  16  |  20GB |  40GB | 120GB |  240GB |  200GB |
| LoRA   |  16  |  16GB |  32GB |  80GB |  160GB |  120GB |
| QLoRA  |   8  |  10GB |  16GB |  40GB |   80GB |   80GB |
| QLoRA  |   4  |   6GB |  12GB |  24GB |   48GB |   32GB |

## Fine-tuning libraries

It is possible to fine-tune GLLMs directly using PyTorch or Transformers. However, doing so is (very) hard.    
To make it easier, several frameworks are available, two notable ones being: [`Axolotl`](https://github.com/OpenAccess-AI-Collective/axolotl) and [`LLaMA-Factory`](https://github.com/hiyouga/LLaMA-Factory). 

For this demonstration we will use `LLaMA-Factory` to fine-tune a [Mistral 7b](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.1) and a [Phi-2 2.7b](https://huggingface.co/microsoft/phi-2) model.

In order to spread the fine-tuning accross multiple GPUs we will use the [Microsoft Deepspeed](https://github.com/microsoft/DeepSpeed) library.

## Instance setup

Go to runpod.io and start an instance with the following requirements:

- GPUs: 5x Nvidia RTX 6000 ADA (~$5.70 per hour) or 3x H100 (~$12 to ~$14 per hour)
    - Other GPU configurations will work as well if you tweak the Deepspeed config.
- You can also select the geographic region which can matter for download speed, especially with big models.
- Select the "RunPod Pytorch 2.1" template (i.e., Docker image)
    - `runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04`

You can use the automatic Jupyter Lab option to access your instance (or SSH if you are comfortable with that).
See the options by clicking "Connect" after starting the pod

After starting the instance, use the Terminal panel in Jypyter Lab (or SSH) to run the following setup commands:

```bash

## Packages for fine-tuning

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt
pip install deepspeed

## When using RTX3090 or RTX4090 cards, you also need to:

#export NCCL_P2P_DISABLE="1"
#export NCCL_IB_DISABLE="1" 

```

Then upload the Jupyter Notebook (`finetune.ipynb`) from within Jupyter Lab (or SFTP) and follow the instructions.