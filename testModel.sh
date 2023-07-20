#!/bin/bash

#SBATCH -J testppoPaper
#SBATCH -D /s/ls4/users/slava1195/rl_rob_2/RL_robotSim
#SBATCH -o /s/ls4/users/slava1195/rl_rob_2/RL_robotSim/Logs/%x_%j.out
#SBATCH -e /s/ls4/users/slava1195/rl_rob_2/RL_robotSim/Logs/%x_%j.err
#SBATCH -p hpc5-el7-gpu-3d
#SBATCH -n 4
#SBATCH --gres=gpu:k80:2
#SBATCH --time=48:00:00

export HOME=/s/ls4/users/slava1195
export PATH=$HOME/anaconda3/envs/rl_robots/bin:$PATH

export LD_LIBRARY_PATH=/s/ls4/sw/cuda/10.1/lib64:/s/ls4/sw/cuda/10.1/nvvm/lib64:$HOME/installation_dists/cudnn-9.0-linux-x64-v7.1.ga/lib64:



RUN_DIR_1=/s/ls4/users/slava1195/rl_rob_2/RL_robotSim/results/FollowerContinuous_test_sensors/env29/PPO/default_model/feats_v14VPC/bear2/ppo_e29_b1_f14VPC_mv4/PPO_continuous-grid_fded9_00000_0_2023-06-24_16-35-32



RUN_DIR_2=/s/ls4/users/slava1195/rl_rob_2/RL_robotSim/results/FollowerContinuous_test_sensors/env29/PPO/default_model/feats_v14VPC/bear2/ppo_e29_b2_2_f14VPC_ar/PPO_continuous-grid_fded9_00001_1_2023-06-24_16-35-51



RUN_DIR_3=/s/ls4/users/slava1195/rl_rob_2/RL_robotSim/results/FollowerContinuous_test_sensors/env29/PPO/default_model/feats_v14VPC/bear2/ppo_e29_b2_f14VPC_mv4/PPO_continuous-grid_fded9_00002_2_2023-06-24_16-36-10


# RUN_DIR_4=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous_paper/env29/PPO/default_model/feats_v14/bear1/ppo_e29_b1_f14_ar/PPO_continuous-grid_9549b_00000_0_2023-06-20_12-05-20

# RUN_DIR_5=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous_paper/env29/PPO/default_model/feats_v14/bear1/ppo_e29_b1_f14_mv4/PPO_continuous-grid_9549b_00001_1_2023-06-20_12-05-49

# RUN_DIR_6=/s/ls4/users/slava1195/rl_rob/RL_robotSim/results/FollowerContinuous_paper/env29/PPO/default_model/feats_v14/bear2/ppo_e29_b2_f14_ar/PPO_continuous-grid_9549b_00002_2_2023-06-20_12-06-15



# feats14v2 bear 1 mv4
python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR_1 \
--checkpoint_number 130



# feats12v2 1 bear ar
python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR_2 \
--checkpoint_number 95

python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR_2 \
--checkpoint_number 135

python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR_2 \
--checkpoint_number 115



# feats12v2 1 bear mv4
python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR_3 \
--checkpoint_number 130

python TestModel_2.py --rlalgo PPO \
--run_dir $RUN_DIR_3 \
--checkpoint_number 175

# # feats14 1 bear ar
# python TestModel_2.py --rlalgo PPO \
# --run_dir $RUN_DIR_4 \
# --checkpoint_number 160

# # feats14 1 bear mv4
# python TestModel_2.py --rlalgo PPO \
# --run_dir $RUN_DIR_5 \
# --checkpoint_number 155

# # feats14 2 bear ar
# python TestModel_2.py --rlalgo PPO \
# --run_dir $RUN_DIR_6 \
# --checkpoint_number 155

# python TestModel_2.py --rlalgo PPO \
# --run_dir $RUN_DIR_6 \
# --checkpoint_number 210
