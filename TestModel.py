#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("/s/ls4/users/grartem/RL_robots/continuous-grid-arctic/")
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

#TEST_SEEDS = range(100)
#TEST_SEEDS = [1, 4, 5, 7 ,9, 10, 11, 14, 16, 17, 21, 24, 28, 29, 30, 32, 36, 38, 40, 43, 44, 45, 46, 49,
             #53, 58, 59, 60, 62, 63, 64, 65, 66, 70, 72, 73, 76, 82, 84, 85, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100,
             #103, 105, 107, 108, 109, 116,117, 119, 121,124, 127, 129, 131, 132, 138, 139, 140, 141, 146, 149, 150,
             #152, 153, 154, 155, 156, 159, 160, 161, 166, 170, 172, 173, 174, 175, 176, 177, 179, 181, 182, 183, 185, 186,
             #187, 189, 192, 194, 195, 196, 197, 198, 199]  # env 7
TEST_SEEDS = [7,12,13,14,16,17,18,20,27,30,32,35,47,48,49,50,
              52,58,64,65,73,74,75,76,77,78,80,81,82,85,86,88,90,93,94,95,96,97,98,99,
             101,103,104,105,109,112,113,115,116,117,118,119,120,121,122,123,125,126,127,128,129,130,131,132,135,138,141,143,144,145,147,149,150,
             152,153,154,155,157,159,160,161,162,163,166,168,170,172,175,176,179,182,183,188,192,194,195,196,197,198,199]  # env 4
RECORD_SEEDS = []
#RECORD_SEEDS = [1, 4, 5, 7 ,9, 10, 11, 14, 16, 17, 21, 24, 28, 29, 30, 32, 36, 38, 40, 43, 44, 45, 46, 49,
             #53, 58, 59, 60, 62, 63, 64, 65, 66, 70, 72, 73, 76, 82, 84, 85, 87, 88, 89, 90, 91, 92, 94, 95, 96, 97, 98, 99, 100,
             #103, 105, 107, 108, 109, 116,117, 119, 121,124, 127, 129, 131, 132, 138, 139, 140, 141, 146, 149, 150,
             #152, 153, 154, 155, 156, 159, 160, 161, 166, 170, 172, 173, 174, 175, 176, 177, 179, 181, 182, 183, 185, 186,
             #187, 189, 192, 194, 195, 196, 197, 198, 199] # env 7


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
        elapsed_time = time.time() - start_time
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        print(elapsed_time)
        history.append({
            "seed": seed_i,
            "reward": total_reward,
            "done": done,
            "time": elapsed_time
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
