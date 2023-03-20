#!/bin/bash

#SBATCH -J ppo_env24_2
#SBATCH -D /s/ls4/users/slava1195/rl_rob/RL_robotSim
#SBATCH -o /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/Continuous_%x_%j.out
#SBATCH -e /s/ls4/users/slava1195/rl_rob/RL_robotSim/Logs/Continuous_%x_%j.err
#SBATCH -p hpc4-el7-gpu-3d
#SBATCH -n 48
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

#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env9v6_feats_v12_train5v7 ppo_env10v2_feats_v12_train5v2_use_lstm ppo_env10v3feats_v12_train5v2_use_lstm

#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments

#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env12v1_feats_v12_train5v7 ppo_env13v1_feats_v12_train5v7 ppo_env13v2_feats_v12_train5v7

#### 05.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env13v3_feats_v12_train5v7 ppo_env13v1_feats_v12_feat_v1_train5v7 ppo_env13v3_feats_v12_feat_v1_train5v7

##### 06.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env14v1_feats_v12_train5v7 ppo_env14v1_feats_v12_feats_v1_train5v7 ppo_env14v1_feats_v12_feats_v1_train5v7_use_lstm

##### 08.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env15v1_feats_v13_train5v7 ppo_env15v2_feats_v13_train5v7

#### 08.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env16v1_feats_v12_train5v7 ppo_env16v2_feats_v12_train5v7 ppo_env16v2_feats_v12_feats_v1_train5v7

##### 09.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env17v1_feats_v13_train5v7 ppo_env17v2_feats_v13_train5v7 ppo_env17v2_feats_v14_train5v7

### 10.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env18v1_feats_v12_train5v2_lstmv1 ppo_env18v1_feats_v12_train5v2_lstmv2 ppo_env18v1_feats_v12_train5v2_lstm

#### 10.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env18v1_feats_v12_train5_lstmv2 ppo_env18v1_feats_v12_train5_lstm ppo_env18v1_feats_v12_train5_lstm8v1

##### 12.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env18v1_feats_v12_train5_lstm8v2 ppo_env18v1_feats_v12_train5v2_arch8 ppo_env18v1_feats_v12_train5v2_archv8v1

##### 14.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env19v2_feats_v12_train5v7 ppo_env19v3_feats_v12_train5v7 ppo_env19v4_feats_v12_train5v7

##### 15.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env19v1_feats_v12_train5v7

##### 16.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env20v2_feats_v12_train5v7 ppo_env20v4_feats_v12_train5v7 ppo_env20v2_feats_v12_feats_v1_train5v7

##### 21.09.2022
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env20v1_feats_v12_train5v7 ppo_env20v1_1_feats_v12_train5v7 ppo_env20v1_feats_v12_train5v7_1500

#### 08.02.2023
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env21v1_feats_v15_lstmv7v2 ppo_env21v2_feats_v15_lstmv7v2 ppo_env21v1_feats_v15_v5v7_sqd

##### 10.02.2023
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_env22v1_feats_v15_fi_lstmv7v2 ppo_env22v2_feats_v15_fi_lstmv7v2 ppo_env22v1_feats_v15_fi_v5v7_sqd

##### 12.02.2023
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_dv2_env22v1_feats_v15_fi_lstmv7v2 ppo_dv2_env22v2_feats_v15_fi_lstmv7v2 ppo_dv2_env22v1_feats_v15_fi_v5v7_sqd

##### 13.02.2023
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_sv1_env23v1_feats_v15_fi_lstm_v10v1 ppo_sv1_env23v2_feats_v15_fi_lstm_v10_v2 ppo_sv1_env23v1_feats_v15_fi_v5v8_sqd

##### 15.02.2023
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_sv1_env24v1_feats_v17_fi_lstm_v10v1 ppo_sv1_env24v1_feats_v17_fi_lstm_v10_v2 ppo_sv1_env24v1_feats_v17_fi_v5v8_sqd

##### 17.02.2023
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_sv1_env24v1_feats_v16_fi_v5v8_sqd ppo_sv2_env24v1_feats_v17_fi_v5v2_sqd ppo_sv1_env24v1_feats_v17_fi_v5v8_sqd

##### 17.03.2023
#python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_sv2_env24v1_1_feats_v16_fi_v5v8_sqd_arch_arch_d_v1 ppo_sv2_env24v1_1_feats_v16_fi_v5v8_sqd_arch_arch_d_v2 ppo_sv2_env24v1_1_feats_v16_fi_v5v8_sqd_arch_arch_d_v1_bear_v2


#### 17.03.2023
python MyTrain.py --config-file Configs/FollowerContinuous/PPO_dyn_obst.conf --experiments ppo_sv2_env24v1_1_feats_v16_fi_v5v8_sqd_arch_arch_d_v2_bear_v2v2 ppo_sv2_env24v1_1_feats_v16_fi_v5v8_sqd_arch_arch_d_v2_bear_v2_th
