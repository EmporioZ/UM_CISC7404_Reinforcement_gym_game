import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import gymnax
from distreqx import distributions
# from gymnax.environments import cartpole
from typing import Tuple, NamedTuple
import numpy as np
from dataclasses import dataclass


@dataclass
class Config:
    # 环境参数
    env_name: str = "CartPole-v1"
    seed: int = 42
    obs_dim: int = 4  # CartPole observation dimension
    action_dim: int = 2  # CartPole action dimension (0 or 1)

    # SAC超参数
    hidden_dim: int = 256
    learning_rate: float = 1e-4
    buffer_size: int = 10000
    batch_size: int = 1024
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.1
    auto_entropy_tuning: bool = True
    target_entropy: float = -2.0  # -dim(A)

    # 训练参数
    num_episodes: int = 10000
    eval_interval: int = 10
    warmup_steps: int = 10000


import jax
import jax.numpy as jnp
import equinox as eqx
from typing import NamedTuple, Optional
import numpy as np


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray


class ReplayBuffer(eqx.Module):
    buffer_size: int
    obs_dim: int
    action_dim: int
    ptr: jnp.ndarray
    size: jnp.ndarray

    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray

    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int, key: jax.random.PRNGKey):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Use JAX arrays for counters to make them JIT-compatible
        self.ptr = jnp.array(0, dtype=jnp.int32)
        self.size = jnp.array(0, dtype=jnp.int32)

        # Initialize storage with JAX arrays
        self.obs = jnp.zeros((buffer_size, obs_dim), dtype=jnp.float32)
        self.actions = jnp.zeros((buffer_size, 1), dtype=jnp.int32)
        self.rewards = jnp.zeros((buffer_size, 1), dtype=jnp.float32)
        self.next_obs = jnp.zeros((buffer_size, obs_dim), dtype=jnp.float32)
        self.dones = jnp.zeros((buffer_size, 1), dtype=jnp.float32)

    def add(self, obs, action, reward, next_obs, done):
        # Convert inputs to JAX arrays if they aren't already
        obs = jnp.asarray(obs, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.int32)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        next_obs = jnp.asarray(next_obs, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.float32)

        # Update the buffer at current pointer position
        new_obs = self.obs.at[self.ptr].set(obs)
        new_actions = self.actions.at[self.ptr].set(action)
        new_rewards = self.rewards.at[self.ptr].set(reward)
        new_next_obs = self.next_obs.at[self.ptr].set(next_obs)
        new_dones = self.dones.at[self.ptr].set(done)

        # Update pointer and size
        new_ptr = (self.ptr + 1) % self.buffer_size
        new_size = jnp.minimum(self.size + 1, self.buffer_size)

        # Return a new buffer with updated state (functional update)
        return eqx.tree_at(
            lambda x: (x.obs, x.actions, x.rewards, x.next_obs, x.dones, x.ptr, x.size),
            self,
            (new_obs, new_actions, new_rewards, new_next_obs, new_dones, new_ptr, new_size)
        )

    def sample(self, key, batch_size: int):
        # Sample random indices
        indices = jax.random.randint(
            key,
            shape=(batch_size,),
            minval=0,
            maxval=self.size,
            dtype=jnp.int32
        )

        # Gather the batch
        batch = Transition(
            obs=self.obs[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_obs=self.next_obs[indices],
            done=self.dones[indices]
        )

        return batch

    @property
    def is_full(self):
        return self.size == self.buffer_size






# class ReplayBuffer:
#     def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
#         self.buffer_size = buffer_size
#         self.obs_dim = obs_dim
#         self.action_dim = action_dim
#         self.ptr = 0
#         self.size = 0
#
#         self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
#         self.actions = np.zeros((buffer_size, 1), dtype=np.int32)  # Changed to int32 for discrete actions
#         self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
#         self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
#         self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
#
#     def add(self, obs, action, reward, next_obs, done):
#         self.obs[self.ptr] = obs
#         self.actions[self.ptr] = action
#         self.rewards[self.ptr] = reward
#         self.next_obs[self.ptr] = next_obs
#         self.dones[self.ptr] = done
#
#         self.ptr = (self.ptr + 1) % self.buffer_size
#         self.size = min(self.size + 1, self.buffer_size)
#
#     def sample(self, batch_size: int):
#         indices = np.random.randint(0, self.size, size=batch_size)
#         return (
#             self.obs[indices],
#             self.actions[indices],
#             self.rewards[indices],
#             self.next_obs[indices],
#             self.dones[indices]
#         )


class Actor(eqx.Module):
    # hidden: eqx.nn.MLP
    # logits: eqx.nn.Linear
    trunk: eqx.nn.Sequential

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key):
        keys = jax.random.split(key, 2)

        self.trunk = eqx.nn.Sequential(
            [
                eqx.nn.MLP(in_size=obs_dim, out_size=hidden_dim, width_size=hidden_dim, depth=2, key=keys[0]),
                eqx.nn.Lambda(jax.nn.relu),
                eqx.nn.Linear(hidden_dim, action_dim, key=keys[1])
            ]
        )

    def __call__(self, x, key=None):
        # Handle both single and batch inputs
        logits = self.trunk(x)
        dist = distributions.Categorical(logits=logits)
        return dist


class DoubleCritic(eqx.Module):
    q1: eqx.nn.MLP
    q2: eqx.nn.MLP
    action_dim: int

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key):
        keys = jax.random.split(key, 2)
        self.action_dim = action_dim

        self.q1 = eqx.nn.MLP(
            in_size=obs_dim + action_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=2,
            key=keys[0]
        )
        self.q2 = eqx.nn.MLP(
            in_size=obs_dim + action_dim,
            out_size=1,
            width_size=hidden_dim,
            depth=2,
            key=keys[1]
        )

    def __call__(self, obs, action):
        # Handle both single and batch inputs
        action = jax.nn.one_hot(action, self.action_dim)
        if action.ndim == 2:
            action = action.squeeze(0)
        input = jnp.concatenate([obs, action])
        q1 = self.q1(input).squeeze(-1)
        q2 = self.q2(input).squeeze(-1)
        return q1, q2


class Alpha(eqx.Module):
    value: jax.Array

    def __init__(self, init_value=0.1):
        self.value = jnp.array([jnp.log(init_value)])

    def __call__(self):
        return jnp.exp(self.value)


class State(eqx.Module):
    """Base class"""

    def replace(self, **kwargs):
        """Replaces existing fields.

        E.g., s = State(bork=1, dork=2)
        s.replace(dork=3)
        print(s)
            >> State(bork=1, dork=3)
        """
        fields = self.__dataclass_fields__
        assert set(kwargs.keys()).issubset(fields)
        new_pytree = {}
        for k in fields:
            if k in kwargs:
                new_pytree[k] = kwargs[k]
            else:
                new_pytree[k] = getattr(self, k)
        return type(self)(**new_pytree)


class TrainState(State):
    actor: Actor
    critic: DoubleCritic
    alpha: Alpha
    target_critic: DoubleCritic
    actor_opt: optax.GradientTransformation
    actor_opt_state: optax.OptState
    critic_opt: optax.GradientTransformation
    critic_opt_state: optax.OptState
    alpha_opt: optax.GradientTransformation
    alpha_opt_state: optax.OptState


@eqx.filter_jit
def _update_critic(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])
    # 计算目标Q值
    next_actions_dist = eqx.filter_vmap(train_state.actor)(next_obs)
    next_actions = next_actions_dist.sample(key=key)
    log_prob = next_actions_dist.log_prob(next_actions).sum(-1)
    target_q1, target_q2 = eqx.filter_vmap(train_state.target_critic)(next_obs, next_actions)
    target_V = jnp.minimum(target_q1, target_q2) - train_state.alpha() * log_prob
    target_Q = rewards + (1 - dones) * config.gamma * target_V

    # 更新critic
    def critic_loss(model, target, obs, actions):
        q1, q2 = eqx.filter_vmap(model)(obs, actions)
        loss = jnp.mean((q1 - target_Q) ** 2) + jnp.mean((q2 - target_Q) ** 2)
        return loss

    grads = eqx.filter_grad(critic_loss)(train_state.critic, target_Q, obs, actions)
    updates, new_critic_opt_state = train_state.critic_opt.update(grads, train_state.critic_opt_state,
                                                                  eqx.filter(train_state.critic, eqx.is_array))

    new_critic = eqx.apply_updates(train_state.critic, updates)
    new_train_state = train_state.replace(critic=new_critic, critic_opt_state=new_critic_opt_state)
    # jax.debug.print("critic change:{}", eqx.tree_equal(new_train_state.critic, train_state.critic))
    return new_train_state


@eqx.filter_jit
def _update_actor(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])

    def actor_loss(actor):
        dist = eqx.filter_vmap(actor)(obs, key_array)
        new_actions = dist.sample(key=key)
        q1, q2 = eqx.filter_vmap(train_state.critic)(obs, new_actions)
        q = jnp.minimum(q1, q2)

        log_prob = dist.log_prob(new_actions)

        loss = jnp.mean(train_state.alpha() * log_prob.sum(-1) - q)
        return loss

    grads = eqx.filter_grad(actor_loss)(train_state.actor)
    updates, new_actor_opt_state = train_state.actor_opt.update(grads, train_state.actor_opt_state,
                                                                eqx.filter(train_state.actor, eqx.is_array))
    new_actor = eqx.apply_updates(train_state.actor, updates)
    new_train_state = train_state.replace(actor=new_actor, actor_opt_state=new_actor_opt_state)
    # jax.debug.print("actor change:{}", eqx.tree_equal(new_train_state.actor, train_state.actor))

    return new_train_state


@eqx.filter_jit
def _update_alpha(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])

    def alpha_loss(log_alpha):
        dist = eqx.filter_vmap(train_state.actor)(obs, key_array)
        new_actions = dist.sample(key=key)
        log_prob = dist.log_prob(new_actions)

        loss = jnp.mean(log_alpha() * (-log_prob - config.target_entropy))
        return loss

    grads = jax.grad(alpha_loss)(train_state.alpha)
    updates, new_alpha_opt_state = train_state.alpha_opt.update(grads, train_state.alpha_opt_state)
    new_alpha = eqx.apply_updates(train_state.alpha, updates)
    new_train_state = train_state.replace(alpha=new_alpha, alpha_opt_state=new_alpha_opt_state)
    # jax.debug.print("alpha change:{}", eqx.tree_equal(train_state.alpha, new_train_state.alpha))

    return new_train_state


def train(train_state, config, env, env_params, buffer):
    key = jax.random.PRNGKey(config.seed)
    obs, state = env.reset(key, env_params)
    for episode in range(config.num_episodes):

        episode_reward = 0
        done = False

        while not done:
            # 收集经验
            key, action_key, step_key = jax.random.split(key, 3)
            if buffer.size < config.warmup_steps:
                action = jax.random.randint(action_key, shape=(), minval=0, maxval=2)  # Random action 0 or 1

            else:
                action = train_state.actor(obs).sample(key=action_key)

            next_obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
            buffer.add(obs, action, reward, next_obs, done)

            # 更新网络
            if buffer.size >= config.warmup_steps:
                batch = buffer.sample(batch_size=config.batch_size, key=key)
                keys = jax.random.split(key, 3)
                train_state = _update_critic(train_state, batch, keys[0])
                train_state = _update_actor(train_state, batch, keys[1])
                train_state = _update_alpha(train_state, batch, keys[2])

                critic_params, critic_arch = eqx.partition(train_state.critic, eqx.is_array)
                target_critic_params, target_critic_arch = eqx.partition(train_state.target_critic, eqx.is_array)

                new_target_critic_params = jax.tree.map(
                    lambda o, t: o * config.tau + t * (1 - config.tau),
                    critic_params,
                    target_critic_params
                )

                new_target_critic = eqx.combine(new_target_critic_params, target_critic_arch)
                critic = eqx.combine(critic_params, critic_arch)

                train_state = train_state.replace(target_critic=new_target_critic, critic=critic)

            obs = next_obs
            episode_reward += reward
            key, _ = jax.random.split(key)

        if episode % config.eval_interval == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")


if __name__ == "__main__":
    config = Config()
    key = jax.random.PRNGKey(config.seed)
    actor = Actor(config.obs_dim, config.action_dim, config.hidden_dim, key)
    critic = DoubleCritic(config.obs_dim, config.action_dim, config.hidden_dim, key)
    alpha = Alpha()
    target_critic = DoubleCritic(config.obs_dim, config.action_dim, config.hidden_dim, key)
    actor_opt = optax.adam(config.learning_rate)
    actor_opt_state = actor_opt.init(eqx.filter(actor, eqx.is_array))
    critic_opt = optax.adam(config.learning_rate)
    critic_opt_state = critic_opt.init(eqx.filter(critic, eqx.is_array))
    alpha_opt = optax.adam(config.learning_rate)
    alpha_opt_state = alpha_opt.init(eqx.filter(alpha, eqx.is_array))

    train_state = TrainState(
        actor=actor,
        critic=critic,
        alpha=alpha,
        target_critic=target_critic,
        actor_opt=actor_opt,
        actor_opt_state=actor_opt_state,
        critic_opt=critic_opt,
        critic_opt_state=critic_opt_state,
        alpha_opt=alpha_opt,
        alpha_opt_state=alpha_opt_state,
    )

    env, env_params = gymnax.make(config.env_name)
    buffer = ReplayBuffer(config.buffer_size, config.obs_dim, config.action_dim, key=key)
    train(train_state, config, env, env_params, buffer)
