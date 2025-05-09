import jax
import jax.numpy as jnp
import equinox as eqx
import optax
from skittles import SkittlesEasy, SkittlesMedium, SkittlesHard
from distreqx import distributions
from typing import Tuple, NamedTuple
import numpy as np
import wandb
from dataclasses import dataclass

@dataclass
class Config:
    env_name: str = "CartPole-v1"
    seed: int = 42
    obs_dim: int = 64
    action_dim: int = 5
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    buffer_size: int = 10000
    batch_size: int = 64
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.1
    auto_entropy_tuning: bool = True
    target_entropy: float = -np.log((1 / action_dim)) * 0.98
    num_episodes: int = 10000
    eval_interval: int = 10
    warmup_steps: int = 1000

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
        self.ptr = jnp.array(0, dtype=jnp.int32)
        self.size = jnp.array(0, dtype=jnp.int32)
        self.obs = jnp.zeros((buffer_size, obs_dim // 8, obs_dim // 8), dtype=jnp.float32)
        self.actions = jnp.zeros((buffer_size, 1), dtype=jnp.int32)
        self.rewards = jnp.zeros((buffer_size, 1), dtype=jnp.float32)
        self.next_obs = jnp.zeros((buffer_size, obs_dim // 8, obs_dim // 8), dtype=jnp.float32)
        self.dones = jnp.zeros((buffer_size, 1), dtype=jnp.float32)

    def add(self, obs, action, reward, next_obs, done):
        obs = jnp.asarray(obs, dtype=jnp.float32)
        action = jnp.asarray(action, dtype=jnp.int32)
        reward = jnp.asarray(reward, dtype=jnp.float32)
        next_obs = jnp.asarray(next_obs, dtype=jnp.float32)
        done = jnp.asarray(done, dtype=jnp.float32)
        new_obs = self.obs.at[self.ptr].set(obs)
        new_actions = self.actions.at[self.ptr].set(action)
        new_rewards = self.rewards.at[self.ptr].set(reward)
        new_next_obs = self.next_obs.at[self.ptr].set(next_obs)
        new_dones = self.dones.at[self.ptr].set(done)
        new_ptr = (self.ptr + 1) % self.buffer_size
        new_size = jnp.minimum(self.size + 1, self.buffer_size)
        return eqx.tree_at(
            lambda x: (x.obs, x.actions, x.rewards, x.next_obs, x.dones, x.ptr, x.size),
            self,
            (new_obs, new_actions, new_rewards, new_next_obs, new_dones, new_ptr, new_size)
        )

    def sample(self, key, batch_size: int):
        indices = jax.random.randint(key, shape=(batch_size,), minval=0, maxval=self.size, dtype=jnp.int32)
        return Transition(
            obs=self.obs[indices],
            action=self.actions[indices],
            reward=self.rewards[indices],
            next_obs=self.next_obs[indices],
            done=self.dones[indices]
        )

class Actor(eqx.Module):
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
        x = jnp.reshape(x, -1)
        logits = self.trunk(x)
        action_prob = jax.nn.softmax(logits, axis=-1)
        max_logits_action = jnp.argmax(action_prob, axis=-1)
        action_dist = distributions.Categorical(logits=logits)
        action = action_dist.sample(key=key)
        z = logits == 0.0
        z = jnp.float32(z) * 1e-8
        log_action_prob = jnp.log(action_prob + z)
        return action, (action_prob, log_action_prob), max_logits_action

class DoubleCritic(eqx.Module):
    q1: eqx.nn.MLP
    q2: eqx.nn.MLP
    action_dim: int

    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key):
        keys = jax.random.split(key, 2)
        self.action_dim = action_dim
        self.q1 = eqx.nn.MLP(in_size=obs_dim, out_size=action_dim, width_size=hidden_dim, depth=2, key=keys[0])
        self.q2 = eqx.nn.MLP(in_size=obs_dim, out_size=action_dim, width_size=hidden_dim, depth=2, key=keys[1])

    def __call__(self, obs):
        obs = jnp.reshape(obs, -1)
        q1 = self.q1(obs)
        q2 = self.q2(obs)
        return q1, q2

class Alpha(eqx.Module):
    value: jax.Array

    def __init__(self, init_value=0.2):
        self.value = jnp.array([jnp.log(init_value)])

    def __call__(self):
        return jnp.exp(self.value)

class State(eqx.Module):
    def replace(self, **kwargs):
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
    next_actions, (action_prob, log_action_prob), _ = eqx.filter_vmap(train_state.actor)(next_obs, key_array)
    next_target_q1, next_target_q2 = eqx.filter_vmap(train_state.target_critic)(next_obs)
    min_next_target_q = jnp.sum(action_prob * (jnp.minimum(next_target_q1, next_target_q2) - train_state.alpha() * log_action_prob), axis=-1, keepdims=True)
    target_q = rewards + (1 - dones) * config.gamma * min_next_target_q
    # jax.debug.print("target_q:{}", target_q.shape)

    def critic_loss(model):
        q1, q2 = eqx.filter_vmap(model)(obs)
        actions_int = actions.squeeze().astype(jnp.int32)
        q1_a = q1[jnp.arange(obs.shape[0]), actions_int]
        q2_a = q2[jnp.arange(obs.shape[0]), actions_int]
        loss = jnp.mean((q1_a - target_q.squeeze()) ** 2) + jnp.mean((q2_a - target_q.squeeze()) ** 2)
        return loss

    loss, grads = eqx.filter_value_and_grad(critic_loss)(train_state.critic)
    updates, new_critic_opt_state = train_state.critic_opt.update(grads, train_state.critic_opt_state, eqx.filter(train_state.critic, eqx.is_array))
    new_critic = eqx.apply_updates(train_state.critic, updates)
    new_train_state = train_state.replace(critic=new_critic, critic_opt_state=new_critic_opt_state)
    return new_train_state, loss

@eqx.filter_jit
def _update_actor(train_state, batch, key):
    obs, _, _, _, _ = batch
    key_array = jax.random.split(key, obs.shape[0])

    def actor_loss(actor):
        _, (action_prob, log_action_prob), _ = eqx.filter_vmap(actor)(obs, key_array)
        q1, q2 = eqx.filter_vmap(train_state.critic)(obs)
        q = jnp.minimum(q1, q2)
        inside_term = q - train_state.alpha() * log_action_prob
        loss = -jnp.sum(action_prob * inside_term, axis=-1).mean()
        return loss

    loss, grads = eqx.filter_value_and_grad(actor_loss)(train_state.actor)
    updates, new_actor_opt_state = train_state.actor_opt.update(grads, train_state.actor_opt_state, eqx.filter(train_state.actor, eqx.is_array))
    new_actor = eqx.apply_updates(train_state.actor, updates)
    new_train_state = train_state.replace(actor=new_actor, actor_opt_state=new_actor_opt_state)
    return new_train_state, loss

@eqx.filter_jit
def _update_alpha(train_state, batch, key):
    obs, _, _, _, _ = batch
    key_array = jax.random.split(key, obs.shape[0])

    def alpha_loss(log_alpha):
        _, (action_prob, log_action_prob), _ = eqx.filter_vmap(train_state.actor)(obs, key_array)
        entropy = -jnp.sum(action_prob * log_action_prob, axis=-1)
        alpha = jnp.exp(log_alpha())
        loss = jnp.mean(-alpha * (entropy + config.target_entropy))
        return loss

    loss, grads = eqx.filter_value_and_grad(alpha_loss)(train_state.alpha)
    updates, new_alpha_opt_state = train_state.alpha_opt.update(grads, train_state.alpha_opt_state)
    new_alpha = eqx.apply_updates(train_state.alpha, updates)
    new_train_state = train_state.replace(alpha=new_alpha, alpha_opt_state=new_alpha_opt_state)
    # updates, new_alpha_opt_state = train_state.alpha_opt.update(grads, train_state.alpha_opt_state)
    # new_alpha = eqx.tree_at(lambda a: a, train_state.alpha(), train_state.alpha() + updates)
    # new_train_state = train_state.replace(alpha=new_alpha, alpha_opt_state=new_alpha_opt_state)
    return new_train_state, loss

def train(train_state, config, env, env_params, buffer):
    key = jax.random.PRNGKey(config.seed)
    obs, state = env.reset(key, env_params)
    for episode in range(config.num_episodes):
        episode_reward = 0
        done = False
        cum_critic_loss = 0
        cum_actor_loss = 0
        cum_alpha_loss = 0
        num_steps = 0

        while not done:
            key, action_key, step_key = jax.random.split(key, 3)
            if buffer.size < config.warmup_steps:
                action = jax.random.randint(action_key, shape=(), minval=0, maxval=2)
            else:
                action, (_, _), _ = train_state.actor(obs, action_key)

            next_obs, state, reward, done, _ = env.step(step_key, state, action, env_params)
            buffer = buffer.add(obs, action, reward, next_obs, done)

            if buffer.size >= config.warmup_steps:
                batch = buffer.sample(batch_size=config.batch_size, key=key)
                keys = jax.random.split(key, 3)
                train_state, critic_loss = _update_critic(train_state, batch, keys[0])
                train_state, actor_loss = _update_actor(train_state, batch, keys[1])
                train_state, alpha_loss = _update_alpha(train_state, batch, keys[2])

                critic_params, critic_arch = eqx.partition(train_state.critic, eqx.is_array)
                target_critic_params, target_critic_arch = eqx.partition(train_state.target_critic, eqx.is_array)
                new_target_critic_params = jax.tree.map(
                    lambda o, t: o * config.tau + t * (1 - config.tau),
                    critic_params,
                    target_critic_params
                )
                new_target_critic = eqx.combine(new_target_critic_params, target_critic_arch)
                train_state = train_state.replace(target_critic=new_target_critic)

                cum_critic_loss += critic_loss
                cum_actor_loss += actor_loss
                cum_alpha_loss += alpha_loss
                num_steps += 1

            obs = next_obs
            episode_reward += reward
            key, _ = jax.random.split(key)

        avg_critic_loss = cum_critic_loss / num_steps if num_steps > 0 else 0
        avg_actor_loss = cum_actor_loss / num_steps if num_steps > 0 else 0
        avg_alpha_loss = cum_alpha_loss / num_steps if num_steps > 0 else 0

        wandb.log({
            "episode": episode,
            "episode_reward": episode_reward,
        })
        print(f"Episode {episode}, Reward: {episode_reward}, "
              f"Avg Critic Loss: {avg_critic_loss:.4f}, "
              f"Avg Actor Loss: {avg_actor_loss:.4f}, "
              f"Avg Alpha Loss: {avg_alpha_loss:.4f}")

def evaluate(mode, env):
    key = jax.random.split(key, 3)

if __name__ == "__main__":

    config = Config()
    wandb.init(
        entity="",
        project="sac",
        name=f'sac_for_{config.seed}',
        mode="online",
    )
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

    env = SkittlesEasy()
    # env = SkittlesMedium()
    buffer = ReplayBuffer(config.buffer_size, config.obs_dim, config.action_dim, key=key)
    train(train_state, config, env, env.default_params, buffer)