#!/bin/bash

#SBATCH -J ppo_env28
#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/Continuous_%x_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/Continuous_%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 4
#SBATCH --gres=gpu:k80:1
#SBATCH --time=72:00:00

export HOME=/s/ls4/users/grartem
source activate rl_robots

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst_2.conf --experiments ppo_e29_b1_f14VPC_mv4 ppo_e29_b2_2_f14VPC_ar ppo_e29_b2_f14VPC_mv4
