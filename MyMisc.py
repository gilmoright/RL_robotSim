#!/usr/bin/env python
import sys
import os
os.environ["SDL_VIDEODRIVER"] = "dummy"
sys.path.append("/s/ls4/users/grartem/RL_robots/continuous-grid-arctic/")
import follow_the_leader_continuous_env

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

    def __init__(self, env, framestack, lz4_compress=False):
        super().__init__(env)
        self.framestack = framestack
        self.lz4_compress = lz4_compress

        self.frames = deque(maxlen=framestack)

        low = np.tile(self.observation_space.low[...], framestack)
        high = np.tile(
            self.observation_space.high[...], framestack
        )
        self.observation_space = Box(
            low=low, high=high, dtype=self.observation_space.dtype
        )

    def observation(self):
        assert len(self.frames) == self.framestack, (len(self.frames), self.framestack)
        observes = np.concatenate(self.frames)
        #observes[:,:,0] = observes[:,:,0] / 11
        #observes[:,:,1] = observes[:,:,1] / 6
        #observes[:,:,2] = observes[:,:,2] / 3
        assert not np.isnan(observes).any()
        return observes
        #return gym.wrappers.frame_stack.LazyFrames(observes, self.lz4_compress)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.frames.append(observation)
        return self.observation(), reward, done, info

    def reset(self, **kwargs):
        observation = self.env.reset(**kwargs)
        [self.frames.append(observation) for _ in range(self.framestack)]
        return self.observation()


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

class ContinuousObserveModifier_v0(ObservationWrapper):
    
    def __init__(self, env, lz4_compress=False):
        super().__init__(env)
        self.prev_obs = None
        self.max_diag = np.sqrt(np.power(self.DISPLAY_WIDTH,2)+np.power(self.DISPLAY_HEIGHT, 2))
        """
        self.observation_space = Box(np.array([-self.DISPLAY_WIDTH,-self.DISPLAY_HEIGHT,
                                      -self.leader.max_speed,
                                      -360,
                                      -self.leader.max_rotation_speed,
                                      -self.DISPLAY_WIDTH,-self.DISPLAY_HEIGHT,
                                      -self.follower.max_speed,
                                      -360,
                                      -self.follower.max_rotation_speed,
                                      -self.DISPLAY_WIDTH,-self.DISPLAY_HEIGHT,
                                      -self.leader.max_speed -self.follower.max_speed,
                                      -720,
                                      -self.DISPLAY_WIDTH-self.DISPLAY_HEIGHT,
                                      -self.max_diag,-self.max_diag
                                      ], dtype=np.float32),
                             np.array([self.DISPLAY_WIDTH,self.DISPLAY_HEIGHT,
                                      self.leader.max_speed,
                                      360,
                                      self.leader.max_rotation_speed,
                                      self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT,
                                      self.follower.max_speed,
                                      360,
                                      self.follower.max_rotation_speed,
                                      self.DISPLAY_WIDTH, self.DISPLAY_HEIGHT,
                                      self.leader.max_speed +self.follower.max_speed,
                                      720,
                                      +self.DISPLAY_WIDTH+self.DISPLAY_HEIGHT,
                                      self.max_diag, self.max_diag
                                      ], dtype=np.float32
                                      ))
        """
        self.observation_space = Box(np.array([-1,-1,
                                      -1,
                                      -1,
                                      -1,
                                      -1,-1,
                                      -1,
                                      -1,
                                      -1,
                                      -1,-1,
                                      -1,
                                      -1,
                                      -1,
                                      -1,-1
                                      ], dtype=np.float32),
                             np.array([1,1,
                                      1,
                                      1,
                                      1,
                                      1, 1,
                                      1,
                                      1,
                                      1,
                                      1, 1,
                                      1,
                                      1,
                                      1,
                                      1, 1
                                      ], dtype=np.float32
                                      ))

    def observation(self, obs):
        # TODO:
        # не обязательно делить на максимальное значение, можно на максимально допустимое.
        # так как я разницу между позициями даю, лидер или агент не могут на целый экран прыгнуть. 
        obs[0] /= self.DISPLAY_WIDTH
        obs[1] /= self.DISPLAY_HEIGHT
        obs[2] /= self.leader.max_speed
        obs[3] /= 360
        obs[4] /= self.leader.max_rotation_speed
        obs[5] /= self.DISPLAY_WIDTH
        obs[6] /= self.DISPLAY_HEIGHT
        obs[7] /= self.follower.max_speed
        obs[8] /= 360
        obs[9] /= self.follower.max_rotation_speed

        # change leader absolute pos, speed, direction to relative
        relativePositions = obs[0:4] - obs[5:9]
        distance = np.linalg.norm(relativePositions[:2])
        distanceFromBorders = [distance-(obs[-3]/self.max_diag ), (obs[-2]/self.max_diag) - distance]
        obs = obs[:-3]
        
        
        if self.prev_obs is None:
            self.prev_obs = obs
        
        
        obs = np.concatenate([obs - self.prev_obs, relativePositions, [distance], distanceFromBorders])
        #print("OBSS", obs)
        return np.clip(obs, -1, 1)

def continuous_env_maker(config):
    name = config["name"]  # .get("name", "MiniGrid-Empty-5x5-v0")
    framestack = config.get("framestack", 4)
    env = gym.make(name)
    env = ContinuousObserveModifier_v0(env)
    env = MyFrameStack(env, framestack)
    return env

register_env("continuous-grid", continuous_env_maker)