#!/bin/bash

#SBATCH -J testppoE4F12
#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=36:00:00

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4/PPO/env4feats12_train/ppo_env4feats12_train5v2/PPO_continuous-grid_448ca_00001_1_2022-06-14_16-10-15
RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4/PPO/env4feats12_train/ppo_env4feats12_train5v6/PPO_continuous-grid_448ca_00002_2_2022-06-14_16-10-32


python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 710
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 720
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 730
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 740
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 750
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 760
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 770
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 780
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 790
python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 800