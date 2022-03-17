#!/bin/bash

#SBATCH -D /s/ls4/users/grartem/RL_robots/RL_robotSim
#SBATCH -o /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/MakeVideo/Cont_TD3_%j.out
#SBATCH -e /s/ls4/users/grartem/RL_robots/RL_robotSim/Logs/MakeVideo/Cont_TD3_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 1
#SBATCH --gres=gpu:k80:1
#SBATCH -t 01:00:00
export HOME=/s/ls4/users/grartem
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:
export EXPERIMENT_DIR=/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst/TD3/td3_v0/TD3_continuous-grid_7b319_00000_0_2022-03-09_22-45-32/

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000010/checkpoint-10 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000010

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000100/checkpoint-100 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000100

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000200/checkpoint-200 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000200

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000300/checkpoint-300 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000300

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000400/checkpoint-400 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000400

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000500/checkpoint-500 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000500

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000600/checkpoint-600 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000600

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000700/checkpoint-700 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000700

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000800/checkpoint-800 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000800

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_000900/checkpoint-900 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_000900

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001000/checkpoint-1000 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001000

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001100/checkpoint-1100 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001100
python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001200/checkpoint-1200 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001200
python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001300/checkpoint-1300 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001300
python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001400/checkpoint-1400 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001400
python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001500/checkpoint-1500 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001500
python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001600/checkpoint-1600 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001600
python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001700/checkpoint-1700 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001700

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001800/checkpoint-1800 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001800

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_001900/checkpoint-1900 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_001900

python MyEvaluate.py $EXPERIMENT_DIR/checkpoint_002000/checkpoint-2000 \
--run TD3 \
--episodes 1 \
--video-dir $EXPERIMENT_DIR/checkpoint_002000