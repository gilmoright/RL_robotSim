#!/bin/bash

#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/Continuous_ppo_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/Continuous_ppo_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 8
#SBATCH --gres=gpu:k80:4

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

python MyTrain.py --config-file Configs/FollowerContinuous/PPO.yml