#!/bin/bash

#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/MakeVideo/Cont_PPO_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/MakeVideo/Cont_PPO_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 2
#SBATCH --gres=gpu:k80:1

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

python MyEvaluate.py /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/PPO/tests/v0/PPO_continuous-grid_3f086_00000_0_2022-02-15_19-06-34/checkpoint_000500/checkpoint-500 \
--run PPO \
--episodes 1 \
--video-dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/PPO/tests/v0/PPO_continuous-grid_3f086_00000_0_2022-02-15_19-06-34/checkpoint_000500/

python MyEvaluate.py /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/PPO/tests/v0v1/PPO_continuous-grid_3f086_00001_1_2022-02-15_19-06-51/checkpoint_000500/checkpoint-500 \
--run PPO \
--episodes 1 \
--video-dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/PPO/tests/v0v1/PPO_continuous-grid_3f086_00001_1_2022-02-15_19-06-51/checkpoint_000500/