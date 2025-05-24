from typing import Optional

from etils import epath

from brax.envs.base import Env, State, Wrapper
from brax.envs.wrappers import training

from brax.envs.wrappers import gym as gym_wrapper

from brax.envs import fast
from brax.envs import half_cheetah
from brax.envs import hopper
from brax.envs import humanoid
from brax.envs import humanoidstandup
from brax.envs import inverted_double_pendulum
from brax.envs import inverted_pendulum
from brax.envs import pusher
from brax.envs import reacher
from brax.envs import swimmer
from brax.envs import walker2d

import jax
from jax import numpy as jp

from . import ant
from . import ant_v5

_envs = {
    'ant': ant.Ant,
    'Ant-v5': ant_v5.Ant,
    # 'fast': fast.Fast,
    # 'halfcheetah': half_cheetah.Halfcheetah,
    # 'hopper': hopper.Hopper,
    # 'humanoid': humanoid.Humanoid,
    # 'humanoidstandup': humanoidstandup.HumanoidStandup,
    # 'inverted_pendulum': inverted_pendulum.InvertedPendulum,
    # 'inverted_double_pendulum': inverted_double_pendulum.InvertedDoublePendulum,
    # 'pusher': pusher.Pusher,
    # 'reacher': reacher.Reacher,
    # 'swimmer': swimmer.Swimmer,
    # 'walker2d': walker2d.Walker2d,
}


class EpisodeWrapper(Wrapper):
  """Maintains episode step count and sets done at episode end."""

  def __init__(self, env: Env, episode_length: int, action_repeat: int):
    super().__init__(env)
    self.episode_length = episode_length
    self.action_repeat = action_repeat

  def reset(self, rng: jax.Array) -> State:
    state = self.env.reset(rng)
    state.info['steps'] = jp.zeros(rng.shape[:-1])
    state.info['rewards'] = jp.zeros(rng.shape[:-1])
    state.info['truncation'] = jp.zeros(rng.shape[:-1])
    return state

  def step(self, state: State, action: jax.Array) -> State:

    def f(state, _):
      nstate = self.env.step(state, action)
      return nstate, nstate.reward

    state, rewards = jax.lax.scan(f, state, (), self.action_repeat)
    state = state.replace(reward=jp.sum(rewards, axis=0))
    steps = state.info['steps'] + self.action_repeat
    rewards = state.info['rewards'] + state.reward
    one = jp.ones_like(state.done)
    zero = jp.zeros_like(state.done)
    episode_length = jp.array(self.episode_length, dtype=jp.int32)
    done = jp.where(steps >= episode_length, one, state.done)
    state.info['truncation'] = jp.where(steps >= episode_length, 1 - state.done,
                                        zero)
    state.info['steps'] = steps
    state.info['rewards'] = rewards
    return state.replace(done=done)


def create(
    env_name: str,
    episode_length: int = 1000,
    action_repeat: int = 1,
    auto_reset: bool = True,
    batch_size: Optional[int] = None,
    seed: int = 0,
    gym_backend: Optional[str] = None,
    **kwargs,
) -> Env:
  """Creates an environment from the registry.

  Args:
    env_name: environment name string
    episode_length: length of episode
    action_repeat: how many repeated actions to take per environment step
    auto_reset: whether to auto reset the environment after an episode is done
    batch_size: the number of environments to batch together
    **kwargs: keyword argments that get passed to the Env class constructor

  Returns:
    env: an environment
  """
  env = _envs[env_name](**kwargs)

  if episode_length is not None:
    env = EpisodeWrapper(env, episode_length, action_repeat)
  if batch_size:
    env = training.VmapWrapper(env, batch_size)
  if auto_reset:
    env = training.AutoResetWrapper(env)

  # gym_wrapper
  if batch_size:
    env = gym_wrapper.VectorGymWrapper(env, seed=seed, backend=gym_backend)
  else:
    env = gym_wrapper.GymWrapper(env, seed=seed, backend=gym_backend)

  return env


def get_resource_filepath(env_name: str) -> str:
  return _envs[env_name].get_resource_filepath()
