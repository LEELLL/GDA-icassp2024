# GESTURE GENERATION via DIFFUSION MODEL with ATTENTION MECHANISM

This is the code for the [GDA] paper, which has been accepted by ICASSP2024.

## Install Dependencies
1. Python requirements:  python=3.10.5 pytorch=1.13 pytorch-lighting=1.6.5 hydra-core=1.2.0 CUDAtoolkit=11.7. 
2. Install dependencies:
    ```
    conda env create -f environment.yml
    ```

## Preparing Dataset
Follow https://pantomatrix.github.io/BEAT/ to prepare the data.

## Training
`python train_gesture_generation.py`  

## Acknowledgement
The implementation relies on resources from [Diffmotion](https://github.com/zf223669/DiffMotion.git) and [BEAT](https://pantomatrix.github.io/BEAT/).

## Reference
If you find this code useful, please consider citing.

```
coming soon.
```
