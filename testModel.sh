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
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4/PPO/env4feats12_train/ppo_env4feats12_train5v2/PPO_continuous-grid_448ca_00001_1_2022-06-14_16-10-15
#RUN_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/env4/PPO/env4feats12_train/ppo_env4feats12_train5v6/PPO_continuous-grid_448ca_00002_2_2022-06-14_16-10-32
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env8/PPO/feats_v12_train/ppo_env8feats_v12_train5v7/PPO_continuous-grid_52abf_00000_0_2022-11-17_16-54-15
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env8v2/PPO/feats_v12_train/ppo_env8v2feats_v12_train5v7/PPO_continuous-grid_509a3_00000_0_2022-11-18_18-18-54
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env8v2/PPO/feats_v12_train/ppo_env8v2feats_v12_v1_train5v7/PPO_continuous-grid_50c30_00000_0_2022-11-20_00-58-35
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env9/PPO/feats_v12_train/ppo_env9v1feats_v12_train5v7/PPO_continuous-grid_acd96_00000_0_2022-11-22_16-50-50
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env9/PPO/feats_v12_train/ppo_env9v2feats_v12_train5v7/PPO_continuous-grid_7f349_00000_0_2022-11-23_23-07-46
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env9/PPO/feats_v12_train/ppo_env9v3feats_v12_train5v7/PPO_continuous-grid_7f349_00001_1_2022-11-23_23-08-28
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env9/PPO/feats_v12_train/ppo_env9v1_d1_feats_v12_train5v7/PPO_continuous-grid_b539a_00000_0_2022-11-28_12-04-47

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env9/PPO/feats_v12_train/ppo_env9v1_d2_feats_v12_train5v7/PPO_continuous-grid_21183_00000_0_2022-11-30_19-57-53

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env9/PPO/feats_v12_train/ppo_env9v2feats_v12_train5v7/PPO_continuous-grid_7f349_00000_0_2022-11-23_23-07-46

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env9/PPO/feats_v12_train/ppo_env9v5_feats_v12_train5v7/PPO_continuous-grid_57f05_00000_0_2022-12-01_18-10-51

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env10/PPO/feats_v12_train/ppo_env10v1feats_v12_train5v2_use_lstm/PPO_continuous-grid_57f05_00001_1_2022-12-01_18-11-18

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env13/PPO/feats_v12_train/ppo_env13v1feats_v12_train5v7/PPO_continuous-grid_314b2_00001_1_2022-12-04_18-20-49

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env13/PPO/feats_v12_train/ppo_env13v1_feats_v12_train5v7/PPO_continuous-grid_314b2_00001_1_2022-12-04_18-20-49

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env13/PPO/feats_v12_train/ppo_env13v1_feats_v12_feat_v1_train5v7/PPO_continuous-grid_f286a_00001_1_2022-12-05_13-24-26

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env14/PPO/feats_v12_train_feats_v1/ppo_env14v1_feats_v12_feats_v1_train5v7/PPO_continuous-grid_4d6da_00001_1_2022-12-06_20-35-14
################ после косяков

#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env16/PPO/feats_v12_train/ppo_env16v2_feats_v12_train5v7/PPO_continuous-grid_a2f58_00001_1_2022-12-08_16-24-44
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env14/PPO/feats_v12_train/ppo_env14v1_feats_v12_train5v7/PPO_continuous-grid_4d6da_00000_0_2022-12-06_20-34-57
#RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env16/PPO/feats_v12_train/ppo_env16v1_feats_v12_train5v7/PPO_continuous-grid_a2f58_00000_0_2022-12-08_16-24-26

RUN_DIR=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous/env16/PPO/feats_v12_train/ppo_env16v2_feats_v12_train5v7/PPO_continuous-grid_a2f58_00001_1_2022-12-08_16-24-44

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 370

python TestModel.py --rlalgo PPO \
--run_dir $RUN_DIR \
--checkpoint_number 290
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 140
#
#python TestModel.py --rlalgo PPO \
#--run_dir $RUN_DIR \
#--checkpoint_number 230
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
