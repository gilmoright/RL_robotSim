#!/bin/bash

#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/MakeVideo/Cont_PPO_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/MakeVideo/Cont_PPO_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 1
#SBATCH --gres=gpu:k80:1
#SBATCH -t 01:00:00
export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

python MyEvaluate.py /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/PPO/tests/v0v0/PPO_continuous-grid_4d06f_00000_0_2022-02-23_01-46-41/checkpoint_000940/checkpoint-940 \
--run PPO \
--episodes 4 \
--video-dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/PPO/tests/v0v0/PPO_continuous-grid_4d06f_00000_0_2022-02-23_01-46-41/checkpoint_000940/