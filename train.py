import dotenv

dotenv.load_dotenv()

import os
from collections import defaultdict
from functools import partial
import xml.etree.ElementTree as ET
import time
import tempfile

import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import omegaconf
import tqdm
import wandb
from flax.metrics import tensorboard
from flax.training.train_state import TrainState

from tdmpc2_jax import TDMPC2, WorldModel
from tdmpc2_jax.common.activations import mish, simnorm
from tdmpc2_jax.data import SequentialReplayBuffer
from tdmpc2_jax.networks import NormedLinear

from scipy.stats import loguniform

import envs


def zero_grads() -> optax.GradientTransformation:
  """Zero gradients transformation.

  Returns:
      optax.GradientTransformation: Gradient transformation.
  """

  # from https://github.com/deepmind/optax/issues/159#issuecomment-896459491
  def init_fn(_):
    return ()

  def update_fn(updates, state, params=None):
    return jax.tree_map(jnp.zeros_like, updates), ()

  return optax.GradientTransformation(init_fn, update_fn)


def freeze(params: TrainState) -> TrainState:
  """Freeze parameters.

  Args:
      params (TrainState): Parameters.

  Returns:
      TrainState: Frozen parameters.
  """
  return params.replace(tx=zero_grads())


@hydra.main(config_name='config', config_path='.', version_base=None)
def main(cfg: dict):
  """Main function for training TD-MPC2 on Brax environments.

  Args:
      cfg (dict): Configuration dictionary.
  """
  env_config = cfg['env']
  encoder_config = cfg['encoder']
  model_config = cfg['world_model']
  tdmpc_config = cfg['tdmpc2']
  exp_config = cfg['experiments']

  ##############################
  # Wandb setup
  ##############################
  wandb.init(
      project=os.getenv('WANDB_PROJECT'),
      config=omegaconf.OmegaConf.to_container(
          cfg,
          resolve=True,
          throw_on_missing=True,
      ),
      name=cfg.run_name or None,
  )

  ##############################
  # Logger setup
  ##############################
  output_dir = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
  writer = tensorboard.SummaryWriter(os.path.join(output_dir, 'tensorboard'))
  writer.hparams(cfg)

  ##############################
  # Environment setup
  ##############################
  def reset_env(
      env_config,
      gym_backend=None,
      seed=None,
      friction=None,
  ):
    if friction:
      friction_str = ' '.join(map(lambda r: f'{r:.4f}', friction))
      xml_path = envs.get_resource_filepath(env_config.env_id)
      tree = ET.parse(xml_path)
      root = tree.getroot()

      for elem in root.iter():
        if 'friction' in elem.attrib:
          elem.set('friction', friction_str)

      with tempfile.NamedTemporaryFile(
          mode='w+',
          suffix='.xml',
          delete=True,
      ) as temp_file:
        tree.write(temp_file.name)
        env = envs.create(
            env_config.env_id,
            env_path=temp_file.name,
            batch_size=env_config.num_envs,
            action_repeat=env_config.action_repeat,
            episode_length=env_config.episode_length,
            auto_reset=True,
            backend=env_config.backend,
            gym_backend=gym_backend,
            seed=int(seed),
        )
    else:
      env = envs.create(
          env_config.env_id,
          batch_size=env_config.num_envs,
          action_repeat=env_config.action_repeat,
          episode_length=env_config.episode_length,
          auto_reset=True,
          backend=env_config.backend,
          gym_backend=gym_backend,
          seed=int(seed),
      )

    return env

  _reset_env = partial(
      reset_env,
      env_config,
      gym_backend=cfg.get('backend'),
      seed=cfg.seed,
  )
  env = _reset_env()

  np.random.seed(cfg.seed)
  rng = jax.random.PRNGKey(cfg.seed)

  ##############################
  # Agent setup
  ##############################
  dtype = jnp.dtype(model_config.dtype)
  rng, model_key, encoder_key = jax.random.split(rng, 3)
  encoder_module = nn.Sequential([
      NormedLinear(encoder_config.encoder_dim, activation=mish, dtype=dtype)
      for _ in range(encoder_config.num_encoder_layers - 1)
  ] + [
      NormedLinear(
          model_config.latent_dim,
          activation=partial(
              simnorm,
              simplex_dim=model_config.simnorm_dim,
          ),
          dtype=dtype,
      )
  ])

  if encoder_config.tabulate:
    print("Encoder")
    print("--------------")
    print(
        encoder_module.tabulate(
            jax.random.key(0),
            env.observation_space.sample(),
            compute_flops=True,
        ))

  ##############################
  # Replay buffer setup
  ##############################
  dummy_obs = env.reset()
  dummy_action = env.action_space.sample()
  dummy_next_obs, dummy_reward, dummy_done, dummy_info = \
      env.step(dummy_action)
  replay_buffer = SequentialReplayBuffer(
      capacity=cfg.max_steps // env_config.num_envs,
      num_envs=env_config.num_envs,
      seed=cfg.seed,
      dummy_input=dict(
          observation=dummy_obs,
          action=dummy_action,
          reward=dummy_reward,
          next_observation=dummy_next_obs,
          terminated=dummy_done,
          truncated=dummy_info['truncation'],
      ),
  )

  encoder = TrainState.create(
      apply_fn=encoder_module.apply,
      params=encoder_module.init(encoder_key, dummy_obs)['params'],
      tx=optax.chain(
          optax.zero_nans(),
          optax.clip_by_global_norm(model_config.max_grad_norm),
          optax.adam(encoder_config.learning_rate),
      ),
  )

  model = WorldModel.create(
      action_dim=np.prod(env.action_space.shape[1:]),
      encoder=encoder,
      **model_config,
      key=model_key,
  )
  if model.action_dim >= 20:
    tdmpc_config.mppi_iterations += 2

  agent = TDMPC2.create(world_model=model, **tdmpc_config)
  global_step = 0

  options = ocp.CheckpointManagerOptions(
      max_to_keep=1,
      save_interval_steps=cfg['save_interval_steps'],
  )
  checkpoint_path = os.path.join(output_dir, 'checkpoint')
  with ocp.CheckpointManager(
      checkpoint_path,
      options=options,
      item_names=('agent', 'global_step', 'buffer_state'),
  ) as mngr:
    if mngr.latest_step() is not None:
      print('Checkpoint folder found, restoring from', mngr.latest_step())
      abstract_buffer_state = jax.tree.map(
          ocp.utils.to_shape_dtype_struct,
          replay_buffer.get_state(),
      )
      restored = mngr.restore(
          mngr.latest_step(),
          args=ocp.args.Composite(
              agent=ocp.args.StandardRestore(agent),
              global_step=ocp.args.JsonRestore(),
              buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
          ),
      )
      agent, global_step = restored.agent, restored.global_step
      replay_buffer.restore(restored.buffer_state)
    else:
      print('No checkpoint folder found, starting from scratch')
      mngr.save(
          global_step,
          args=ocp.args.Composite(
              agent=ocp.args.StandardSave(agent),
              global_step=ocp.args.JsonSave(global_step),
              buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
          ),
      )
      mngr.wait_until_finished()

    ##############################
    # Friction
    ##############################
    frictions = [
        [1, 0.5, 0.5],
        *[[
            loguniform.rvs(*env_config.friction.static_range),
            loguniform.rvs(*env_config.friction.dynamic_range),
            loguniform.rvs(*env_config.friction.roll_range),
        ] for _ in range(env_config.friction.num_cycles)],
    ]
    if env_config.friction.permute:
      idx = np.random.choice(
          len(frictions),
          size=cfg.max_steps // env_config.friction.cycle,
      )
      friction_sequence = [frictions[i] for i in idx]
    else:
      friction_sequence = frictions * (cfg.max_steps //
                                       env_config.friction.cycle)

    current_friction = friction_sequence.pop(0)
    wandb.log(
        {'episode/friction_norm': np.linalg.norm(current_friction)},
        step=global_step,
    )
    writer.scalar(
        'episode/friction_norm',
        np.linalg.norm(current_friction),
        global_step,
    )

    ##############################
    # Training loop
    ##############################
    cycle_count = 2
    prev_logged_step = global_step
    prev_plan = (
        jnp.zeros((
            env_config.num_envs,
            agent.horizon,
            agent.model.action_dim,
        )),
        jnp.full(
            (env_config.num_envs, agent.horizon, agent.model.action_dim),
            agent.max_plan_std,
        ),
    )
    observation = env.reset()

    T = 500
    seed_steps = int(
        max(5 * T, 1000) * env_config.num_envs * env_config.utd_ratio)
    seed_steps = min(seed_steps, 15_000)
    print('Seed steps:', seed_steps)
    pbar = tqdm.tqdm(initial=global_step, total=cfg.max_steps)

    # train loop
    for global_step in range(global_step, cfg.max_steps, env_config.num_envs):

      if exp_config.freeze_dynamics_after and global_step >= exp_config.freeze_dynamics_after:
        print('Freezing dynamics model')
        agent = agent.replace(model=agent.model.replace(
            dynamics_model=freeze(agent.model.dynamics_model)))
        exp_config.freeze_dynamics_after = None

      # Change friction when needed
      if cycle_count * env_config.friction.cycle <= global_step:
        cycle_count += 1
        current_friction = friction_sequence.pop(0)
        env = _reset_env(friction=current_friction)
        observation = env.reset()
        writer.scalar(
            'episode/friction_norm',
            np.linalg.norm(current_friction),
            global_step,
        )
        wandb.log(
            {'episode/friction_norm': np.linalg.norm(current_friction)},
            step=global_step,
        )

      # Sample action
      if global_step <= seed_steps:
        action = env.action_space.sample()
      else:
        rng, action_key = jax.random.split(rng)
        prev_plan = (
            prev_plan[0],
            jnp.full_like(
                prev_plan[1],
                agent.max_plan_std,
            ),
        )
        action, prev_plan = agent.act(
            observation,
            prev_plan=prev_plan,
            train=True,
            key=action_key,
        )

      # step
      next_observation, reward, done, info = env.step(action)

      # Store transition
      replay_buffer.insert(
          dict(
              observation=observation,
              action=action,
              reward=reward,
              next_observation=next_observation,
              terminated=done,
              truncated=info['truncation'],
          ),)
      observation = next_observation

      # Handle terminations/truncations
      done = jnp.logical_or(info['truncation'], env._state.done)
      if jnp.any(done):
        prev_plan = (
            prev_plan[0].at[done].set(0),
            prev_plan[1].at[done].set(agent.max_plan_std),
        )
        avg_rewards = jnp.mean(info['rewards'][done])
        avg_lengths = jnp.mean(info['steps'][done])
        pbar.set_description(
            f'Average reward: {avg_rewards:.2f} | Average length: {avg_lengths:.1f}'
        )

        writer.scalar(f'episode/return', avg_rewards, global_step)
        writer.scalar(f'episode/length', avg_lengths, global_step)

        wandb.log(
            {
                'episode/return': avg_rewards,
                'episode/length': avg_lengths,
            },
            step=global_step,
        )

      if global_step >= seed_steps:
        if global_step == seed_steps:
          print('Pre-training on seed data...')
          num_updates = seed_steps
        else:
          num_updates = max(1, int(env_config.num_envs * env_config.utd_ratio))

        rng, *update_keys = jax.random.split(rng, num_updates + 1)
        log_this_step = global_step >= prev_logged_step + \
            cfg['log_interval_steps']
        if log_this_step:
          all_train_info = defaultdict(list)
          prev_logged_step = global_step

        for iupdate in range(num_updates):
          batch = replay_buffer.sample(agent.batch_size, agent.horizon)
          agent, train_info = agent.update(
              observations=batch['observation'],
              actions=batch['action'],
              rewards=batch['reward'],
              next_observations=batch['next_observation'],
              terminated=batch['terminated'],
              truncated=batch['truncated'],
              key=update_keys[iupdate],
          )

          if log_this_step:
            for k, v in train_info.items():
              all_train_info[k].append(np.array(v))

        if log_this_step:
          for k, v in all_train_info.items():
            writer.scalar(f'train/{k}_mean', np.mean(v), global_step)
            writer.scalar(f'train/{k}_std', np.std(v), global_step)

            wandb.log(
                {
                    f'train/{k}_mean': np.mean(v),
                    f'train/{k}_std': np.std(v),
                },
                step=global_step,
            )

        mngr.save(
            global_step,
            args=ocp.args.Composite(
                agent=ocp.args.StandardSave(agent),
                global_step=ocp.args.JsonSave(global_step),
                buffer_state=ocp.args.StandardSave(replay_buffer.get_state()),
            ),
        )

      # wandb log train/global_step, time
      wandb.log(
          {'train/time': time.time() - wandb.run.start_time},
          step=global_step,
      )
      pbar.update(env_config.num_envs)
    pbar.close()


if __name__ == '__main__':
  main()
