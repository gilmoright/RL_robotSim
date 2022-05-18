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

python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 10
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 20
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 30
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 40
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 50
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 60
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 70
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 80
python TestModel.py --rlalgo PPO \
--run_dir /s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_0a7bf_00000_0_2022-04-12_15-49-50/ \
--checkpoint_number 90
