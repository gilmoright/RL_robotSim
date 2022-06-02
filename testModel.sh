#!/bin/bash

#SBATCH -J test_ppo_env7_9v3
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
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4/PPO/feats2v2_train/ppo_env4feats2v2_train5v2/PPO_continuous-grid_f1709_00000_0_2022-05-31_15-26-49/
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/feats/ppo_env4feats9v3_train5v2/PPO_continuous-grid_f1709_00001_1_2022-05-31_15-27-05/
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/env4feats10v7_train/ppo_env4feats10v7_train5v2/PPO_continuous-grid_f1709_00002_2_2022-05-31_15-27-22/

# feat 10v7
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4v2/PPO/env4v2feats10v7_train/ppo_env4v2feats10v7_train5v2/PPO_continuous-grid_f2066_00002_2_2022-05-31_15-27-23/
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env7/PPO/feats10v7_train/ppo_env7feats10v7_train5v2/PPO_continuous-grid_dd8f1_00002_2_2022-05-31_15-26-54/

# feat 2v2
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env7/PPO/feats2v2_train/ppo_env7feats2v2_train5v2/PPO_continuous-grid_dd8f1_00000_0_2022-05-31_15-26-15/
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4v2/PPO/feats2v2_train/ppo_env4v2feats2v2_train5v2/PPO_continuous-grid_f2066_00000_0_2022-05-31_15-26-50/

# feat 9v3
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4v2/PPO/feats2v2_train/ppo_env4v2feats9v3_train5v2/PPO_continuous-grid_f2066_00001_1_2022-05-31_15-27-07/
RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env7/PPO/feats2v2_train/ppo_env7feats9v3_train5v2/PPO_continuous-grid_dd8f1_00001_1_2022-05-31_15-26-34/

python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 710
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 720
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 730
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 740
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 750
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 760
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 770
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 780
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 790
python TestModel_onEnv4f9v3.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 800