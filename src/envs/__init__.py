from functools import partial
import pretrained
from collections.abc import Iterable
from smac.env import MultiAgentEnv, StarCraft2Env
import sys
import os
import gym
from gym import ObservationWrapper, spaces
from gym.spaces import flatdim, Discrete, Box, Tuple
import numpy as np
from gym.wrappers import TimeLimit as GymTimeLimit
from griddly import gd
import pathlib


def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)


if sys.platform == "linux":
    os.environ.setdefault(
        "SC2PATH", os.path.join(os.getcwd(), "3rdparty", "StarCraftII")
    )


class TimeLimit(GymTimeLimit):
    def __init__(self, env, max_episode_steps=None):
        super().__init__(env)
        if max_episode_steps is None and self.env.spec is not None:
            max_episode_steps = env.spec.max_episode_steps
        # if self.env.spec is not None:
        #     self.env.spec.max_episode_steps = max_episode_steps
        self._max_episode_steps = max_episode_steps
        self._elapsed_steps = None

    def step(self, action):
        assert (
            self._elapsed_steps is not None
        ), "Cannot call env.step() before calling reset()"
        observation, reward, done, info = self.env.step(action)
        self._elapsed_steps += 1
        if self._elapsed_steps >= self._max_episode_steps:
            info["TimeLimit.truncated"] = not all(done) if isinstance(done, Iterable) else not done
            done = len(observation) * [True]
        return observation, reward, done, info


class FlattenObservation(ObservationWrapper):
    r"""Observation wrapper that flattens the observation of individual agents."""

    def __init__(self, env):
        super(FlattenObservation, self).__init__(env)

        ma_spaces = []

        for sa_obs in env.observation_space:
            flatdim = spaces.flatdim(sa_obs)
            ma_spaces += [
                spaces.Box(
                    low=-float("inf"),
                    high=float("inf"),
                    shape=(flatdim,),
                    dtype=np.float32,
                )
            ]

        self.observation_space = spaces.Tuple(tuple(ma_spaces))

    def observation(self, observation):
        return tuple(
            [
                spaces.flatten(obs_space, obs)
                for obs_space, obs in zip(self.env.observation_space, observation)
            ]
        )


class _GymmaWrapper(MultiAgentEnv):
    def __init__(self, key, time_limit, pretrained_wrapper, **kwargs):
        self.episode_limit = time_limit
        self._env = TimeLimit(gym.make(f"{key}"), max_episode_steps=time_limit)
        self._env = FlattenObservation(self._env)

        if pretrained_wrapper:
            self._env = getattr(pretrained, pretrained_wrapper)(self._env)

        self.n_agents = self._env.n_agents
        self._obs = None

        self.longest_action_space = max(self._env.action_space, key=lambda x: x.n)
        self.longest_observation_space = max(
            self._env.observation_space, key=lambda x: x.shape
        )

        self._seed = kwargs["seed"]
        self._env.seed(self._seed)

    def step(self, actions):
        """ Returns reward, terminated, info """
        actions = [int(a) for a in actions]
        self._obs, reward, done, info = self._env.step(actions)
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        return float(sum(reward)), all(done), {}

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self._obs

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_obs_agent(self, agent_id):
        """ Returns observation for agent_id """
        raise self._obs[agent_id]

    def get_obs_size(self):
        """ Returns the shape of the observation """
        return flatdim(self.longest_observation_space)

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.n_agents * flatdim(self.longest_observation_space)

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(self.n_agents):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        """ Returns the available actions for agent_id """
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.longest_action_space.n - len(valid))
        return valid + invalid

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        # TODO: This is only suitable for a discrete 1 dimensional action space for each agent
        return flatdim(self.longest_action_space)

    def reset(self):
        """ Returns initial observations and states"""
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render()

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed

    def save_replay(self):
        pass

    def get_stats(self):
        return {}


class _GriddlyEnv(MultiAgentEnv):
    WIN_REWARD = 200
    ACTION_SPACE_MAPPING = {
        '0': np.array([0, 0]),
        '1': np.array([1, 0]),
        '2': np.array([0, 1]),
        '3': np.array([1, 1]),
        '4': np.array([0, 2]),
        '5': np.array([1, 2]),
        '6': np.array([0, 3]),
        '7': np.array([1, 3]),
        '8': np.array([0, 4]),
        '9': np.array([1, 4])
    }

    def __init__(self, n_agents=5, max_episode_steps=100, env_name='GDY-5m-vs-6m-lite-v0', render_game=True, **kwargs):
        root_dir = pathlib.Path(__file__).parent.resolve().parent.resolve().parent.resolve()
        img_dir = os.path.join(root_dir, 'smac_lite/resources')
        config_dir = os.path.join(root_dir, 'smac_lite/gdy')
        env = TimeLimit(gym.make(env_name,
                                 gdy_path=config_dir,
                                 image_path=img_dir,
                                 # Currently only handles VECTOR observation space
                                 player_observer_type=gd.ObserverType.VECTOR,
                                 global_observer_type=gd.ObserverType.VECTOR),
                                 max_episode_steps=max_episode_steps)

        env.action_space = [Discrete(len(_GriddlyEnv.ACTION_SPACE_MAPPING)) for _ in range(n_agents)]
        # TODO: Make this not hardcoded
        obs_action_space = [
            Box(low=int(o.low), high=int(o.high), shape=(300,), dtype=np.float32)
            for o in env.observation_space
        ]
        env.observation_space = Tuple(obs_action_space)

        self._env = FlattenObservation(env)
        # TODO: Make this also not hardcoded
        self.longest_action_space = Discrete(len(_GriddlyEnv.ACTION_SPACE_MAPPING))
        self.longest_observation_space = max(self._env.observation_space, key=lambda x: x.shape)
        # TODO: Integrate n_agents with GDY file so that its not effectively hard-coded
        self.n_agents = n_agents
        self._obs = None
        self._seed = kwargs["seed"]
        self._env.seed(self._seed)
        self.episode_limit = max_episode_steps
        self.battles_won = 0
        self.battles_lost = 0
        self.battles_game = 0
        self.render_game = _GriddlyEnv.parse_bool(os.environ.get('RENDER_GAME', render_game))
        print(f'Render option set to [{self.render_game}]')

    @staticmethod
    def parse_bool(raw_bool):
        if isinstance(raw_bool, bool):
            return raw_bool
        try:
            return {
                'false': False,
                'true': True
            }[raw_bool.lower()]
        except KeyError:
            raise ValueError(f'Could not parse {raw_bool} into bool.')

    def step(self, actions):
        _actions = [_GriddlyEnv.ACTION_SPACE_MAPPING[str(int(id))] for id in actions]
        self.obs, reward, done, info = self._env.step(_actions)
        if self.render_game is True:
            self.render()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]

        if done is True:
            self.battles_game += 1
            if sum(reward) >= _GriddlyEnv.WIN_REWARD:
                self.battles_won += 1
            else:
                self.battles_lost += 1

        return float(sum(reward)), done, {
            'num_battles': self.battles_game,
            'num_lost': self.battles_lost,
            'num_won': self.battles_won,
            'win_rate': self.battles_won/self.battles_game if self.battles_game > 0 else 0
        }

    def get_obs(self):
        return self._obs

    def get_obs_agent(self, agent_id):
        return self._obs[agent_id]

    def get_obs_size(self):
        return int(flatdim(self.longest_observation_space))

    def get_state(self):
        return np.concatenate(self._obs, axis=0).astype(np.float32)

    def get_state_size(self):
        return int(self.n_agents * flatdim(self.longest_observation_space))

    def get_avail_actions(self):
        avail_actions = []
        for agent_id in range(len(self._env.action_space)):
            avail_agent = self.get_avail_agent_actions(agent_id)
            avail_actions.append(avail_agent)
        return avail_actions

    def get_avail_agent_actions(self, agent_id):
        valid = flatdim(self._env.action_space[agent_id]) * [1]
        invalid = [0] * (self.get_total_actions() - len(valid))
        return valid + invalid

    def get_total_actions(self):
        # return 10
        return int(flatdim(self.longest_action_space))

    def reset(self):
        self._obs = self._env.reset()
        self._obs = [
            np.pad(
                o,
                (0, self.longest_observation_space.shape[0] - len(o)),
                "constant",
                constant_values=0,
            )
            for o in self._obs
        ]
        return self.get_obs(), self.get_state()

    def render(self):
        self._env.render(observer="global")
        for p in range(self.n_agents):
            self._env.render(observer=p)

    def close(self):
        self._env.close()

    def seed(self):
        return self._env.seed()

    def save_replay(self):
        pass

    def get_stats(self):
       pass


REGISTRY = {
    "sc2": partial(env_fn, env=StarCraft2Env),
    "gymma": partial(env_fn, env=_GymmaWrapper),
    "griddly": partial(env_fn, env=_GriddlyEnv)
}
