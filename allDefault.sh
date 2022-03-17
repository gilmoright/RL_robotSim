#!/bin/bash
#SBATCH -J mbmpo
#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/Continuous_%x_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/Continuous_%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 4
#SBATCH --gres=gpu:k80:1
#SBATCH --time=24:00:00

export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:

#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments a2c_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments a3c_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments ars_v2  # slow AF?
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments bc_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments cql_v2  # doesn't work?
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments es_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments ddpg_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments td3_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments apexddpg_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments dreamer_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments impala_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments maml_v2  # doesn't work?
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments marwil_v2
python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments mbmpo_v2  # doesn't work?
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments pg_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments ppo_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments appo_v2
#python MyTrain.py --config-file Configs/FollowerContinuous/allDefault_obst.conf --experiments sac_v2