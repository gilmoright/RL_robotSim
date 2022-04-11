#!/usr/bin/env python
# coding: utf-8
import sys
sys.path.append("/s/ls4/users/grartem/RL_robots/continuous-grid-arctic/")
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
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

RECORD_SEEDS = []

configs = pyhocon.ConfigFactory.parse_file("/s/ls4/users/grartem/RL_robots/RL_robotSim/Configs/FollowerContinuous/PPO_obst.conf")
#CONFIG = configs["ppo_archv3"].as_plain_ordered_dict()
CONFIG = configs["ppo_env4_feats10v7"].as_plain_ordered_dict()
CONFIG["config"]["num_workers"]=1


EXPERIMENT_DIR = "/s/ls4/users/grartem/RL_robots/RL_robotSim/results/FollowerContinuous/obst_dynLSpd_dynFPS/PPO/feats/ppo_env4_feats10v7/PPO_continuous-grid_d1217_00000_0_2022-04-11_13-26-15/"
checkpoint_path = EXPERIMENT_DIR + "/checkpoint_000080/checkpoint-80"
table_save_path = EXPERIMENT_DIR + "/checkpoint-80_test.csv"

trainer = ray.rllib.agents.registry.get_trainer_class(CONFIG["run"])(CONFIG["config"])
trainer.restore(checkpoint_path)
env = MyMisc.continuous_env_maker(config=CONFIG["config"]["env_config"])

history = []
for seed_i in range(100):
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
                                directory= os.path.dirname(checkpoint_path)+"/videos2",
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
