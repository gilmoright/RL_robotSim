#!/bin/bash

#SBATCH -J ppo_env10
#SBATCH -D /s/ls4/users/slava1195/rl_rob/RL_robotSim
#SBATCH -o /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/Continuous_%x_%j.out
#SBATCH -e /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/Continuous_%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 12
#SBATCH --gres=gpu:k80:3
#SBATCH --time=72:00:00

export HOME=/s/ls4/users/slava1195
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

#python MyTrain.py --config-file Configs/FollowerContinuous/TD3.yml
#python MyTrain.py --config-file Configs/FollowerContinuous/TD3_obst.conf --experiments td3_algov3 td3_algov5 td3_algov6
#python MyTrain.py --config-file Configs/FollowerContinuous/A3C_obstDiscr.conf --experiments a3c_arch6 a3c_feats2 a3c_feats1
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_obst.conf --experiments ppo_env4_feats12 ppo_env4feats12_train5v2 ppo_env4feats12_train5v6
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env8feats_v12_train5v2
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env8v2feats_v12_train5v7
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env8v2feats_v12_v1_train5v7

#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env9v1feats_v12_train5v7
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env9v2feats_v12_train5v7 ppo_env9v3feats_v12_train5v7
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env9v3feats_v12_train5v7

#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env9v1_d2_feats_v12_train5v7 ppo_env9v4_d1_feats_v12_train5v7

#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env9v5_feats_v12_train5v7 ppo_env10v1feats_v12_train5v2_use_lstm ppo_env10v1feats_v12_train5v7_use_lstm

python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env9v6_feats_v12_train5v7 ppo_env10v2feats_v12_train5v2_use_lstm ppo_env10v2feats_v12_train5v7_use_lstm



# ppo_env4_feats11
