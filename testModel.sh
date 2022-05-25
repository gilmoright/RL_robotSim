#!/bin/bash

#SBATCH -J test_ppo
#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=24:00:00

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:
RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env7/PPO/ppo_env7/PPO_continuous-grid_43e23_00000_0_2022-05-19_12-44-24
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env7/PPO/feats10v7_train/ppo_env7feats10v7_train5v2/PPO_continuous-grid_43e23_00002_2_2022-05-19_12-45-17
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 110
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 120
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 130
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 140
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 150
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 160
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 170
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 180
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 190
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 200