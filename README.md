# RelationalGraphLearning
This repository contains the codes for our paper, which is in submission to ICRA 2020. 
For more details, please refer to the [preprint](https://github.com/ChanganVR/RelationalGraphLearning)


## Abstract
We present a relational graph learning approach for robotic crowd navigation using model-based deep reinforcement 
learning that plans actions by looking into the future.
Our approach reasons about the relations between all agents based on their latent features and uses a Graph Convolutional 
Network to encode higher-order interactions in each agent's state representation, which is subsequently leveraged for 
state prediction and value estimation.
The ability to predict human motion allows us to perform multi-step lookahead planning, taking into account the temporal 
evolution of human crowds.
We evaluate our approach against a state-of-the-art baseline for crowd navigation and ablations of our model to 
demonstrate that navigation with our approach is more efficient, results in fewer collisions, and avoids failure cases 
involving oscillatory and freezing behaviors.



## Method Overview
<img src="https://i.imgur.com/8unQNIv.png" width="1000" />


## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository are organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy rgl
```
2. Test policies with 500 test cases.
```
python test.py --policy rgl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy rgl --model_dir data/output --phase test --visualize --test_case 0
```
4. Plot training curve
```
python utils/plot.py data/output/output.log
```



## Video Demo
[<img src="https://i.imgur.com/SnqVhit.png" width="70%">](https://youtu.be/U3quW30Eu3A)


## Citation
If you find the codes or paper useful for your research, please cite the following papers:
```
@misc{chen2019relational,
    title={Relational Graph Learning for Crowd Navigation},
    author={Changan Chen and Sha Hu and Payam Nikdel and Greg Mori and Manolis Savva},
    year={2019},
    eprint={1909.13165},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
@misc{chen2018crowdrobot,
    title={Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning},
    author={Changan Chen and Yuejiang Liu and Sven Kreiss and Alexandre Alahi},
    year={2018},
    eprint={1809.08835},
    archivePrefix={arXiv},
    primaryClass={cs.RO}
}
```