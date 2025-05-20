
# MambaStereo 
This is the implementation of the paper: "MambaStereo: Enhancing Stereo Matching Accuracy in Ill-Posed Regions via Mamba-Based Cost Volume Construction"

## Introduction



![image](image/net.png)

# How to use

## Environment
The code is tested on:
* Python 3.9
* Pytorch 2.2.0
* torchvision 0.17.0
* mamba-ssm 2.2.0
* CUDA 12.2
(CUDA should be greater than 11.6)


### Create a virtual environment and activate it.

```
conda create -n name python=3.9
conda activate name
```
### install Dependencies

```
conda install pytorch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install numpy
pip install Pillow
pip install tensorboard
pip install matplotlib 
...
```
### Install Mamba module
```pip install causal-conv1d>=1.4.0```: an efficient implementation of a simple causal Conv1d layer used inside the Mamba block.

```pip install mamba-ssm```: the core Mamba package.

## Data Preparation
Download [Scene Flow Datasets](https://lmb.informatik.uni-freiburg.de/resources/datasets/SceneFlowDatasets.en.html), [KITTI 2012](http://www.cvlibs.net/datasets/kitti/eval_stereo_flow.php?benchmark=stereo), [KITTI 2015](http://www.cvlibs.net/datasets/kitti/eval_scene_flow.php?benchmark=stereo)

## Train
Use the following command to train MambaStereo on Scene Flow
The args are detailed in the paper
```
python main.py
```

## Finetuning on KITTI
Use the following command to train MambaStereo on Scene Flow
The args are detailed in the paper
```
python main.py
```
## Test
Load the weight and the file path,you can get the output images
```
python prediction.py
```



### Pretrained Model

[Scene Flow](https://drive.google.com/file/d/1uipxPgePS8pjk0F-xW0y4iLDkl8Fv39i/view?usp=drive_link)

## KITTI 2015 finetuning module
[KITTI2012](https://drive.google.com/file/d/1Nzv4XbNq06wH6XNZx05nuLjBKfGneDjk/view?usp=drive_link)

[KITTI2015](https://drive.google.com/file/d/1tIwJyUqSPP2RdWmJoruEcZXcTGhIlI5f/view?usp=drive_link)



# Citation


# Acknowledgements

Our work is inspired by this work and part of codes are migrated from [GwcNet](https://github.com/xy-guo/GwcNet).
