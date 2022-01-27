#!/usr/bin/env python

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

class MyFrameStack(ObservationWrapper):
    r"""Observation wrapper that stacks the observations in a rolling manner.
    For example, if the number of stacks is 4, then the returned observation contains
    the most recent 4 observations. For environment 'Pendulum-v1', the original observation
    is an array with shape [3], so if we stack 4 observations, the processed observation
    has shape [4, 3].
    .. note::
        To be memory efficient, the stacked observations are wrapped by :class:`LazyFrame`.
    .. note::
        The observation space must be `Box` type. If one uses `Dict`
        as observation space, it should apply `FlattenDictWrapper` at first.
    Example::
        >>> import gym
        >>> env = gym.make('PongNoFrameskip-v0')
        >>> env = FrameStack(env, 4)
        >>> env.observation_space
        Box(4, 210, 160, 3)
    Args:
        env (Env): environment object
        num_stack (int): number of stacks
        lz4_compress (bool): use lz4 to compress the frames internally
    """

    def __init__(self, env, num_stack, lz4_compress=False):
        super().__init__(env)
        self.num_stack = num_stack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=num_stack)

        low = np.repeat(self.observation_space.low[...], num_stack, axis=0)
        high = np.repeat(
            self.observation_space.high[...], num_stack, axis=0
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self):
        assert len(self.frames) == self.num_stack, (len(self.frames), self.num_stack)
        observes = np.concatenate(self.frames)
        #observes[:,:,0] = observes[:,:,0] / 11
        #observes[:,:,1] = observes[:,:,1] / 6
        #observes[:,:,2] = observes[:,:,2] / 3
        return observes
        #return gym.wrappers.frame_stack.LazyFrames(observes, self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.num_stack)]
        return self.observation()


def env_maker(config):
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

register_env("mini-grid", env_maker)
