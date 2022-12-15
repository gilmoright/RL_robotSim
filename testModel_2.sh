#!/bin/bash

#SBATCH -J testppoE10F12
#SBATCH -D /s/ls4/users/slava1195/rl_rob/RL_robotSim
#SBATCH -o /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/%x_%j.out
#SBATCH -e /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=48:00:00

export HOME=/s/ls4/users/slava1195
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

CONFIG_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env19/PPO/feats_v12_train_35/ppo_env19v2_feats_v12_train5v7/PPO_continuous-grid_5c8cb_00000_0_2022-12-14_12-19-08
RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env19/PPO/feats_v12_train_70/ppo_env19v4_feats_v12_train5v7/PPO_continuous-grid_5c8cb_00002_2_2022-12-14_12-19-46

python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR \
--config_dir $CONFIG_DIR \
--checkpoint_number 310

python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR \
--config_dir $CONFIG_DIR \
--checkpoint_number 260

python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR \
--config_dir $CONFIG_DIR \
--checkpoint_number 180

python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR \
--config_dir $CONFIG_DIR \
--checkpoint_number 340
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 220
#№№№
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 70
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 310
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 300
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 220
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 320
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 210
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 260
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 120
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 280
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 330
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 170
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 110
