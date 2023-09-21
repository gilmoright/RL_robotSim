#!/bin/bash

#SBATCH -J test_ppo_e30_f1v2_t5v19_mtransv4v5
#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/%x_%j.err
#SBATCH -p hpc4-el7-gpu-3d
#SBATCH -n 2
#SBATCH --gres=gpu:k80:1
#SBATCH --time=48:00:00
#SBATCH --exclude=g[001,018]

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_Prev/e30v2/PPO/ppo_e30v2_t5v7/PPO_continuous-grid_3a1e4_00002_2_2023-09-13_05-57-59
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_Prev/e30v1/PPO/ppo_e30v1_t5v7/PPO_continuous-grid_3a1e4_00001_1_2023-09-13_05-57-41
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_Prev/e30/PPO/ppo_e30_t5v7/PPO_continuous-grid_3a1e4_00000_0_2023-09-13_05-57-08
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_Prev/e30/PPO/ppo_e30_f1v2_t5v7/PPO_continuous-grid_3c1bc_00000_0_2023-09-12_21-43-16
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_Prev/e30/PPO/ppo_e30_f1v3_t5v7/PPO_continuous-grid_3c1bc_00001_1_2023-09-12_21-43-46
#RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_Prev/e30/PPO/ppo_e30_f1v2_t5v19_mtransv4v5/PPO_continuous-grid_41e56_00000_0_2023-09-12_21-43-25
RUN_DIR_1=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous_Prev/e30/PPO/ppo_e30_f1v3_t5v19_mtransv4v5/PPO_continuous-grid_41e56_00001_1_2023-09-12_21-43-55

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR_1 \
--checkpoint_number 60