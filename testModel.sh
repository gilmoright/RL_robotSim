#!/bin/bash

#SBATCH -J testppoE4F12
#SBATCH -D /s/ls4/users/slava1195/rl_rob/RL_robotSim
#SBATCH -o /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/%x_%j.out
#SBATCH -e /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=36:00:00

export HOME=/s/ls4/users/slava1195
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4/PPO/env4feats12_train/ppo_env4feats12_train5v2/PPO_continuous-grid_448ca_00001_1_2022-06-14_16-10-15
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4/PPO/env4feats12_train/ppo_env4feats12_train5v6/PPO_continuous-grid_448ca_00002_2_2022-06-14_16-10-32
RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env8/PPO/feats_v12_train/ppo_env8feats_v12_train5v7/PPO_continuous-grid_52abf_00000_0_2022-11-17_16-54-15



python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 350
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 360
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 340
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 40
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 50
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 60
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 70

