#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("/s/ls4/users/grartem/RL_robots/continuous-grid-arctic/")
# sys.path.append("/home/sheins/rl_test/continuous-grid-arctic/")
#sys.path.append("/s/ls4/users/slava1195/rl_rob/continuous-grid-arctic/")
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
import json
import time
from collections import OrderedDict
import gym
import pyhocon
import numpy as np
import pandas as pd
import ray
from ray.rllib.agents.ppo.ppo import PPOTrainer
from ray.rllib.agents.ddpg.td3 import TD3Trainer

import MyMisc
import MyTfModel
import MyNewModels

#TEST_SEEDS = range(100)
#TEST_SEEDS = [1, 4, 5, 7 ,9, 10, 11, 14, 16, 17, 21, 24, 28, 29, 30, 32, 36, 38, 40, 43, 44, 45, 46, 49,
             #53, 58, 59, 60, 62, 63, 64, 65, 66, 70, 72, 73, 76, 82, 84, 85, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100,
             #103, 105, 107, 108, 109, 116,117, 119, 121,124, 127, 129, 131, 132, 138, 139, 140, 141, 146, 149, 150,
             #152, 153, 154, 155, 156, 159, 160, 161, 166, 170, 172, 173, 174, 175, 176, 177, 179, 181, 182, 183, 185, 186,
             #187, 189, 192, 194, 195, 196, 197, 198, 199]  # env 7
# TEST_SEEDS = [77]
# TEST_SEEDS = [7,12,13,14,16,17,18,20,27,30,32,35,47,48,49,50,
#               52,58,64,65,73,74,75,76,77,78,80,81,82,85,86,88,90,93,94,95,96,97,98,99,
#              101,103,104,105,109,112,113,115,116,117,118,119,120,121,122,123,125,126,127,128,129,130,131,132,135,138,141,143,144,145,147,149,150,
#              152,153,154,155,157,159,160,161,162,163,166,168,170,172,175,176,179,182,183,188,192,194,195,196,197,198,199]  # env 4

#env 9
# TEST_SEEDS = [4, 7,13,14,16,22,23,25,27,30,32,35, 37,44, 46, 47,48,49,50,55,57,59,60,
#               65,74,75,76,77,78,81,82,83,85,86,88,89,90,93,95,97,98,
#               99,101,102, 103,104,105,110,111,112,113,114,
#               116,117,118,123, 124,125,126,127,128, 130, 134,135,
#               139, 141, 145,146,147, 148, 149,150, 152,154,155,157,160,161,162,163,164,
#               166,168,170,174,175,176,185,187, 188,189,190,192,193,194,195,196,197,198,199]

# env 9v1 - > env10
# TEST_SEEDS = [1, 4, 5, 6, 7, 8, 9, 11, 12, 14, 16, 17, 18, 19, 20, 21, 22, 24, 25,
#               27, 28, 29, 33, 34, 36, 38, 39, 40, 41, 44, 45, 46, 47, 48, 51, 52,
#               53, 55, 58, 59, 60, 62, 63, 64, 65, 67, 68, 70, 71, 72, 73, 74, 76,
#               77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 89, 91, 92, 94, 95, 96,
#               97, 98, 102, 104, 105, 106, 107, 108, 109, 110, 111, 113, 114, 116,
#               117, 118, 119, 120, 122, 123, 124, 125, 126, 127, 128, 129, 130, 132,
#               133, 134, 135, 139, 140, 141, 142, 144, 146, 148, 149, 152, 153, 154,
#               155, 156, 157, 159, 160, 161, 162, 163, 164, 166, 167, 168, 170, 171,
#               172, 173, 176, 177, 178, 179, 180, 182, 183, 185, 186, 187, 188, 189,
#               190, 194, 195, 196, 197, 198]
# TEST_SEEDS = [4, 7, 9, 11, 12, 18, 19, 20, 22, 25, 27, 28, 29, 33, 36, 38, 39, 41, 44, 45, 46, 51, 53, 55, 58,
#               59, 60, 62, 63, 65, 68, 70, 72, 73, 74, 76, 77, 79, 81, 82, 83, 86, 87, 89, 94, 95, 96, 102, 104,
#               106, 107, 108, 109, 110, 113, 114, 116, 117, 119, 120, 122, 124, 125, 126, 128, 130, 133, 134, 135,
#               140, 141, 142, 144, 146, 148, 149, 154, 157, 159, 160, 164, 166, 167, 170, 171, 172, 173, 178, 179,
#               180, 182, 183, 186, 187, 188, 189, 190, 194, 195, 196]

# env 14
# seeds
# [1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25,
#  27, 28, 29, 30, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 43, 44, 45, 46, 47, 48,
#  49, 50, 51, 52, 53, 55, 57, 58, 59, 60, 62, 63, 64, 65, 66, 67, 68, 70, 72, 73, 74,
#  75, 76, 77, 78, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96,
#  97, 98, 99, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111, 112, 113, 114, 116,
#  117, 118, 119, 120, 121, 122, 123, 124, 125, 126, 127, 128, 129, 130, 131, 132, 133, 134,
#  135, 138, 139, 140, 141, 142, 144, 145, 146, 147, 148, 149, 150, 152, 153, 154, 155, 156,
#  157, 159, 160, 161, 162, 163, 164, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 178,
#  179, 180, 183, 185, 186, 187, 188, 189, 190, 192, 193, 194, 195, 196, 197, 198, 199]

#####################################
# TEST_SEEDS = [4, 6, 7, 8, 11, 12, 13, 14, 19, 20, 21, 22, 24, 25, 28, 32, 33, 34, 35, 37, 40, 41, 44, 47, 51, 53,
#               63, 65, 66, 67, 68, 70, 72, 74, 75, 76, 78, 80, 82, 83, 84, 85, 86, 88, 89, 92, 94, 95, 98, 103, 105,
#               109, 111, 112, 114, 120, 121, 122, 124, 125, 126, 128, 129, 130, 131, 132, 133, 139, 140, 141, 142, 144,
#               147, 148, 149, 150, 152, 155, 156, 157, 162, 163, 166, 167, 168, 170, 172, 173, 174, 175, 176, 180, 183,
#               185, 187, 188, 190, 195, 196, 197]
#
#
# RECORD_SEEDS = [4, 6, 7, 8, 11, 12, 13, 14, 19, 20, 21]
ppo_sv2_env24v1_1_feats_v16_fi_v5v8_sqd_arch_arch_d_v1 = [4, 5, 7, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 25, 27, 30, 32, 35, 38, 40, 44, 46, 47, 48, 49,
                 50, 52, 55, 59, 60, 64, 65, 68, 73, 74, 75, 76, 77, 81, 82, 83, 85, 86, 88, 90, 93, 95, 97, 98, 99,
                 101, 103, 105, 112, 113, 116, 117, 118, 120, 121, 122, 123, 125, 126, 128, 129, 132, 135,
                 138, 141, 142, 144, 145, 147, 149, 150, 152, 153, 154, 155, 157, 161, 163, 168, 170, 175,
                 176, 182, 183, 185, 188, 192, 193, 194, 195, 196, 197, 198, 199]
seeds_env19v1_100 = [4, 5, 7, 9, 10, 12, 13, 14, 16, 17, 18, 21, 22, 23, 25, 27, 30, 32, 35, 38, 40, 44, 46, 47, 48, 49,
                 50, 52, 55, 59, 60, 64, 65, 68, 73, 74, 75, 76, 77, 81, 82, 83, 85, 86, 88, 90, 93, 95, 97, 98, 99,
                 101, 103, 105, 112, 113, 116, 117, 118, 120, 121, 122, 123, 125, 126, 128, 129, 132, 135,
                 138, 141, 142, 144, 145, 147, 149, 150, 152, 153, 154, 155, 157, 161, 163, 168, 170, 175,
                 176, 182, 183, 185, 188, 192, 193, 194, 195, 196, 197, 198, 199]

# подобрал для 29 среды, но seeds_env19v1_100 тоже подходят. И по seeds_env19v1_100 оценивал слава.
seeds_env29v1_100_artem = [7, 12, 13, 14, 16, 17, 18, 20, 22, 23, 25, 27, 30, 32, 35, 36, 37, 38, 43, 44,
 46, 47, 48, 49, 50, 55, 56, 58, 62, 64, 65, 73, 74, 75, 76, 77, 78, 80, 86, 88, 90, 93, 94, 95,97, 99, 
101, 103, 105, 112, 113, 116, 117, 118, 119, 121, 122, 123, 126, 127, 128, 129, 132,135,
138, 141, 142, 147, 149, 150, 152, 153, 154, 155, 157, 159, 161, 162, 163, 166, 168,
170, 172, 174,175,176, 179, 181, 182, 183, 185, 188, 190, 192, 193, 194, 196, 197, 198, 199]

seeds_env29v1_100_updated = [4, 5, 7, 10, 12, 13, 14, 16, 17, 18, 20, 21, 22, 23, 25, 27, 30, 32, 35, 36, 38, 40, 43, 44, 46, 48, 49,
                 50, 55, 56, 58, 59, 60, 62, 64, 65, 68, 73, 74, 75, 76, 77, 78, 80, 82, 83, 85, 86, 88, 90, 94, 95, 97, 98, 99,
                 101, 103, 105, 112, 113, 116, 117, 119, 120, 121, 122, 123, 127, 128, 129, 132, 135,
                 138, 141, 142, 144, 145, 147, 149, 150, 152, 153, 154, 155, 159, 162, 168, 170, 175, 179,
                 181, 183, 185, 188, 192, 193, 195, 196, 197, 198, 199]

seeds_env30 = [0, 2, 3, 5, 6, 7, 9, 11, 12, 14, 16, 17, 18, 19,20, 21, 24, 25, 29, 31, 33, 35, 36, 37, 38, 39, 42, 44, 48, 50,
               51, 53, 54, 56, 58, 68, 74, 76, 78, 80, 84, 91, 92, 93, 95, 96, 98, 101, 103, 106, 112, 113, 114, 115, 116,
               117,118, 120, 122, 124, 126, 127, 130, 131, 133, 134, 135, 139, 140, 142, 143, 152, 153, 154, 155, 156, 157,
               159, 160, 161, 165, 166, 168, 169, 170, 172, 175, 176, 178, 179,181, 182, 183, 188, 189, 191, 193, 195, 197, 199]
#скучные    : 2, 11, 17, 23, 27, 28, 31, 41, 47, 57, 70, 88,106, 133, 136,, 145, 169, 186, 187
#нормаль    : 3, 4, 7, 9, 12, 15, 18, 19, 20, 40, 42, 44, 45, 49, 53, 55, 56, 63, 69, 73, 76, 79, 83, 87, 95, 107, 114, 121, 123, 124, 131, 134, 138, 141, 147, 149, 150, 155, 157, 162, 164, 173, 176, 177, 182, 183, 190, 194, 195, 199
#долгие    : 5, 8, 10, 13, 21, 22, 25, 26, 34, 36, 37, 38, 43, 46, 48, 50, 51, 54, 58, 59, 60, 61, 64, 65, 68, 74, 75, 77, 80, 84, 86, 90, 91, 94, 96, 98, 99, 101, 102,104,108,109, 112, 115, 116, 117, 125, 127 130, 132, 139, 140, 142, 146, 148, 152, 153, 163, 171, 174, 179, 184, 188, 193, 198

#TEST_SEEDS = seeds_env30
TEST_SEEDS = seeds_env30
#RECORD_SEEDS = [4, 5, 7, 10, 12, 13, 193, 195, 196, 197, 198, 199]
RECORD_SEEDS = [14, 17, 33, 74, 76, 78, 130, 143, 197]



if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate RL model on fized seeds")
    # See also the base parser definition in ray/tune/config_parser.py
    parser.add_argument(
        "--rlalgo",
        default=None,
        type=str,
        help="RL algorythm (PPO, TD3, A3C, etc")
    parser.add_argument(
        "--run_dir",
        default=None,
        type=str,
        help="path experiment dir with config and checkpoints")
    parser.add_argument(
        "--checkpoint_number",
        default=None,
        type=str,
        help="number of the checkpoint to evaluate")
    args = parser.parse_args()
    EXPERIMENT_DIR = args.run_dir
    with open(EXPERIMENT_DIR + "/params.json", "r") as f:
        CONFIG = json.load(f)
    #CONFIG = configs[args.experiment_name].as_plain_ordered_dict()
    CONFIG["num_workers"]=1

    checkpoint_path = EXPERIMENT_DIR + "/checkpoint_{}/checkpoint-{}".format(args.checkpoint_number.zfill(6), args.checkpoint_number)
    table_save_path = EXPERIMENT_DIR + "/checkpoint-{}_test.csv".format(args.checkpoint_number)

    trainer = ray.rllib.agents.registry.get_trainer_class(args.rlalgo)(CONFIG)
    trainer.restore(checkpoint_path)
    env = MyMisc.continuous_env_maker(config=CONFIG["env_config"])

    history = []
    for seed_i in TEST_SEEDS:
        print(seed_i)
        env.seed(seed_i)
        obs = env.reset()
        done = False
        total_reward = 0.0
        steps_count = 0
        # Play one episode.
        start_time = time.time()
        while not done:
            # Compute a single action, given the current observation
            # from the environment.
            action = trainer.compute_single_action(obs, explore=False)
            # Apply the computed action in the environment.
            obs, reward, done, info = env.step(action)
            # Sum up rewards for reporting purposes.
            total_reward += reward
            steps_count += 1
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(elapsed_time)
        history.append({
            "seed": seed_i,
            "reward": total_reward,
            "done": done,
            "time": elapsed_time,
            "steps_count": steps_count
        })
        history[-1].update(info)
        pd.DataFrame(history).to_csv(table_save_path, index=False, sep=";", encoding="cp1251")
        # Report results.

    # # record some videos

    monitor_env = gym.wrappers.Monitor(env=env,
                                    directory= os.path.dirname(checkpoint_path)+"/videos",
                                   video_callable=lambda _:True,
                                   force=False,
                                  uid="seeds_test", mode="evaluation")
    #for seed_i in [9, 35, 36, 38, 44, 47, 59, 89, 91, 97]:
    for seed_i in RECORD_SEEDS:
        monitor_env.seed(seed_i)
        obs = monitor_env.reset()
        done = False
        total_reward = 0.0
        # Play one episode.
        start_time = time.time()
        while not done:
            # Compute a single action, given the current observation
            # from the environment.
            action = trainer.compute_single_action(obs, explore=False)
            # Apply the computed action in the environment.
            obs, reward, done, info = monitor_env.step(action)
            # Sum up rewards for reporting purposes.
            total_reward += reward
    obs = monitor_env.reset()
