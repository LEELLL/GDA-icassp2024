# GDA

## Environment
**We follow the set of https://github.com/zf223669/DiffMotion.git: python=3.10.5 pytorch=1.13 pytorch-lighting=1.6.5 hydra-core=1.2.0 CUDAtoolkit=11.7  
** We use [Lightning-Hydra-Template](https://github.com/ashleve/lightning-hydra-template) for combining the [pytorchLightning](https://www.pytorchlightning.ai/) and [Hydra](https://hydra.cc/).  
** Our model architecture is inspirited by the [Pytorch-ts](https://github.com/zalandoresearch/pytorch-ts)  

### Clone and download the code
'git clone https://github.com/LEELLL/GDA-icassp2024.git' 

### Setting conda environment

1: open the project in PyCharm IDE.  
2: Setting the project env to GDA.
3: Install the packages listed in requirements.txt.
4: Set pymo folder as sources root. The folder icon will become blue.

## Data Prepare
1: Follow https://pantomatrix.github.io/BEAT/ to prepare the data.  


## Training

`python train_gesture_generation.py`  
