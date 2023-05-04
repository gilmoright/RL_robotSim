#!/usr/bin/env python
import sys
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.append("/s/ls4/users/grartem/RL_robots/continuous_grid_arctic")
import continuous_grid_arctic.follow_the_leader_continuous_env
from continuous_grid_arctic.utils.wrappers import MyFrameStack, ContinuousObserveModifier_v0, ContinuousObserveModifier_lidarMap2d, LeaderTrajectory_v0

import gym
from collections import deque
import numpy as np
from ray.rllib.utils.numpy import one_hot
from ray.tune import register_env
from gym import ObservationWrapper
from gym.spaces import Box

import gym_minigrid.envs

class OneHotWrapper(ObservationWrapper):
    def __init__(self, env, vector_index, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.single_frame_dim = 49 * (11 + 6 + 3) + 4
        self.init_x = None
        self.init_y = None
        self.x_positions = []
        self.y_positions = []
        self.x_y_delta_buffer = deque(maxlen=100)
        self.vector_index = vector_index
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.single_frame_dim, )))

        self.observation_space = gym.spaces.Box(
            0.0,
            1.0,
            shape=(self.single_frame_dim * self.framestack, ),
            dtype=np.float32)

    def observation(self, obs):
        # Debug output: max-x/y positions to watch exploration progress.
        if self.step_count == 0:
            for _ in range(self.framestack):
                self.frame_buffer.append(np.zeros((self.single_frame_dim, )))
            if self.vector_index == 0:
                if self.x_positions:
                    max_diff = max(
                        np.sqrt((np.array(self.x_positions) - self.init_x)**2 +
                                (np.array(self.y_positions) - self.init_y)**2))
                    self.x_y_delta_buffer.append(max_diff)
                    print("100-average dist travelled={}".format(
                        np.mean(self.x_y_delta_buffer)))
                    self.x_positions = []
                    self.y_positions = []
                self.init_x = self.agent_pos[0]
                self.init_y = self.agent_pos[1]

        # Are we carrying the key?
        # if self.carrying is not None:
        #    print("Carrying KEY!!")

        self.x_positions.append(self.agent_pos[0])
        self.y_positions.append(self.agent_pos[1])

        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)
        # Is the door we see open?
        # for x in range(7):
        #    for y in range(7):
        #        if objects[x, y, 4] == 1.0 and states[x, y, 0] == 1.0:
        #            print("Door OPEN!!")

        all_ = np.concatenate([objects, colors, states], -1)
        all_flat = np.reshape(all_, (-1, ))
        direction = one_hot(
            np.array(self.agent_dir), depth=4).astype(np.float32)
        single_frame = np.concatenate([all_flat, direction])
        self.frame_buffer.append(single_frame)
        return np.concatenate(self.frame_buffer)

class MyOneHotWrapper(ObservationWrapper):
    def __init__(self, env, framestack):
        super().__init__(env)
        self.framestack = framestack
        # 49=7x7 field of vision; 11=object types; 6=colors; 3=state types.
        # +4: Direction.
        self.frame_width, self.frame_height, _ = self.observation_space.low.shape
        self.frame_buffer = deque(maxlen=self.framestack)
        for _ in range(self.framestack):
            self.frame_buffer.append(np.zeros((self.frame_width, self.frame_height, 11+6+3))) # 11 objects + 6 colors + 3 states
        self.observation_space = gym.spaces.Box(
            0.0,
            1.0,
            shape=(self.frame_width * framestack, self.frame_height, 11+6+3),  # 11 objects + 6 colors + 3 states
            dtype=np.float32)

    def observation(self, obs):
        # One-hot the last dim into 11, 6, 3 one-hot vectors, then flatten.
        objects = one_hot(obs[:, :, 0], depth=11)
        colors = one_hot(obs[:, :, 1], depth=6)
        states = one_hot(obs[:, :, 2], depth=3)

        single_frame = np.concatenate([objects, colors, states], -1)
        self.frame_buffer.append(single_frame)
        assert len(self.frame_buffer) == self.framestack, (len(self.frame_buffer), self.framestack)
        return np.concatenate(self.frame_buffer)

    #def step(self, action):
    #    observation, reward, done, info = self.env.step(action)
    #    self.frame_buffer.append(observation)
    #    return self.observation(observation), reward, done, info

    #def reset(self, **kwargs):
    #    observation = self.env.reset(**kwargs)
    #    for _ in range(self.framestack):
    #        self.frame_buffer.append(np.zeros((self.frame_width, self.frame_height, 11+6+3))) # 11 objects + 6 colors + 3 states
    #    return self.observation(observation)


def minigrid_env_maker(config):
    name = config["name"]  # .get("name", "MiniGrid-Empty-5x5-v0")
    framestack = config.get("framestack", 4)
    env = gym.make(name)
    # Only use image portion of observation (discard goal and direction).
    env = gym_minigrid.wrappers.ImgObsWrapper(env)
    #env = gym.wrappers.frame_stack.FrameStack(env, num_stack=framestack)
    #env = MyFrameStack(env, num_stack=framestack)
    env = MyOneHotWrapper(env, framestack=framestack)
    #env = OneHotWrapper(
    #    env,
    #    config.vector_index if hasattr(config, "vector_index") else 0,
    #    framestack=framestack)
    return env

register_env("mini-grid", minigrid_env_maker)


def continuous_env_maker(config):
    name = config["name"]  # .get("name", "MiniGrid-Empty-5x5-v0")
    framestack = config.get("framestack", 4)
    action_values_range = config.get("action_values_range", None)
    env = gym.make(name, **config["base_env_config"])
    assert not (ContinuousObserveModifier_v0 in config["wrappers"] and LeaderTrajectory_v0 in config["wrappers"])
    if 'ContinuousObserveModifier_v0' in config["wrappers"]:
        env = ContinuousObserveModifier_v0(env, action_values_range)
    if 'ContinuousObserveModifier_lidarMap2d' in config["wrappers"]:
        env = ContinuousObserveModifier_lidarMap2d(env, action_values_range, map_wrapper_forgetting_rate=config.get("map_wrapper_forgetting_rate", None), add_safezone_on_map=config.get("add_safezone_on_map", False))
    elif "LeaderTrajectory_v0" in config["wrappers"]:
        env = LeaderTrajectory_v0(env, framestack, config.get('radar_sectors_number', 180))
    if 'MyFrameStack' in config['wrappers']:
        env = MyFrameStack(env, framestack)
    
    return env

register_env("continuous-grid", continuous_env_maker)