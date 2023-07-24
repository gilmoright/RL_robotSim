#!/bin/bash

#SBATCH -J ppo_e29_b2_f14VPC_mv4_mtransv4v5_tv5v16
#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.err
#SBATCH -p hpc4-el7-gpu-3d
#SBATCH -n 2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=48:00:00

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:


RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_test_sensors/env29/PPO/trans/feats_v14VPC/bear2/ppo_e29_b2_f14VPC_mv4_mtransv4v5_tv5v16/PPO_continuous-grid_e56c7_00001_1_2023-07-20_18-06-05

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR_1 \
--checkpoint_number 250

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR_1 \
--checkpoint_number 275

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR_1 \
--checkpoint_number 300