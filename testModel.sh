#!/bin/bash

#SBATCH -J ppo_e29_b2_f14VPC_mv4_backup
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


#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_test_sensors/env29/PPO/default_model/feats_v14VPC/bear2/ppo_e29_b2_f14VPC_mv4/PPO_continuous-grid_d58fc_00000_0_2023-07-29_01-41-04
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_test_sensors/env29v2/PPO/default_model/feats_v14VPC/bear2/ppo_e29v2_b2_f14VPC_mv4/PPO_continuous-grid_e1e7b_00000_0_2023-07-25_16-40-57
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_test_sensors/env29v2/PPO/trans/feats_v14VPC/bear2/ppo_e29v2_b2_f14VPC_mv4_mtransv4v5_tv5v19/PPO_continuous-grid_d5927_00000_0_2023-07-29_01-41-04
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_test_sensors/env29/PPO/trans/feats_v14VPC/bear2/ppo_e29_b2_f14VPC_mv4_mtransv4v5_tv5v16/PPO_continuous-grid_d58fc_00001_1_2023-07-29_01-41-27
RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_test_sensors/env29/PPO/default_model/feats_v14VPC/bear2/ppo_e29_b2_f14VPC_mv4_backup/PPO_continuous-grid_399ae_00000_0_2023-07-21_14-24-58/

#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR_1 \
#--checkpoint_number 200

#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR_1 \
#--checkpoint_number 250

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR_1 \
--checkpoint_number 275

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR_1 \
--checkpoint_number 300