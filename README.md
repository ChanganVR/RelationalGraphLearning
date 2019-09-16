# CrowdNavExt
Codebase for CorL 2019 submission "Relational Graph Learning for Crowd Navigation". 

## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting started
This repository are organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


## Replicate the results
1. LM-SARL-Linear
```
python train.py --config configs/icra_benchmark/sarl.py --output_dir data/sarl
```
2. RGL-Linear
```
python train.py --config configs/icra_benchmark/mp_linear.py --output_dir data/rgl_linear
```
3. MP-RGL-Onestep
```
python train.py --config configs/icra_benchmark/mp_rgl_onestep.py --output_dir data/mp_rgl_onestep
```
4. MP-RGL-Multistep
```
python train.py --config configs/icra_benchmark/mp_rgl_multistep.py --output_dir data/mp_rgl_multistep
```
The the search depth and width can be modified manually in the config file.


## Visualization
Visualize one test case with trained model
```
python test.py --model_dir data/mp_rgl_multistep --visualize
```
