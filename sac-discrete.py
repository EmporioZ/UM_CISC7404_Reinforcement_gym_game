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
    obs_dim = 4  # CartPole observation dimension
    action_dim = 2  # CartPole action dimension (0 or 1)

    # SAC超参数
    hidden_dim: int = 256
    learning_rate: float = 3e-4
    buffer_size: int = 100000
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = 0.2
    auto_entropy_tuning: bool = True
    target_entropy: float = -4.0  # -dim(A)
    
    # 训练参数
    num_episodes: int = 1000
    eval_interval: int = 10
    warmup_steps: int = 1000

class ReplayBuffer:
    def __init__(self, buffer_size: int, obs_dim: int, action_dim: int):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.ptr = 0
        self.size = 0
        
        self.obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.actions = np.zeros((buffer_size, 1), dtype=np.int32)  # Changed to int32 for discrete actions
        self.rewards = np.zeros((buffer_size, 1), dtype=np.float32)
        self.next_obs = np.zeros((buffer_size, obs_dim), dtype=np.float32)
        self.dones = np.zeros((buffer_size, 1), dtype=np.float32)
    
    def add(self, obs, action, reward, next_obs, done):
        self.obs[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_obs[self.ptr] = next_obs
        self.dones[self.ptr] = done
        
        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)
    
    def sample(self, batch_size: int):
        indices = np.random.randint(0, self.size, size=batch_size)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices]
        )

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
        # self.hidden = eqx.nn.MLP(
        #     in_size=obs_dim,
        #     out_size=hidden_dim,
        #     width_size=hidden_dim,
        #     depth=2,
        #     key=keys[0]
        # )
        # self.logits = eqx.nn.Linear(hidden_dim, action_dim, key=keys[1])
    
    def __call__(self, x, key=None):
        # Handle both single and batch inputs
        if x.ndim == 1:
            logits = self.trunk(x)
        else:
            logits = eqx.filter_vmap(self.trunk)(x)
        dist = distributions.Categorical(logits=logits)
        return dist
    

class Critic(eqx.Module):
    q1: eqx.nn.MLP
    q2: eqx.nn.MLP
    
    def __init__(self, obs_dim: int, action_dim: int, hidden_dim: int, key):
        keys = jax.random.split(key, 2)
        self.q1 = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=action_dim,
            width_size=hidden_dim,
            depth=2,
            key=keys[0]
        )
        self.q2 = eqx.nn.MLP(
            in_size=obs_dim,
            out_size=action_dim,
            width_size=hidden_dim,
            depth=2,
            key=keys[1]
        )
    
    def __call__(self, obs, action=None):
        # Handle both single and batch inputs
        if obs.ndim == 1:
            obs = obs[None, :]
            q1 = self.q1(obs)
            q2 = self.q2(obs)

        else:
            q1 = eqx.filter_vmap(self.q1)(obs)
            q2 = eqx.filter_vmap(self.q2)(obs)
        return q1, q2


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
    critic: Critic
    target_critic: Critic
    actor_opt: optax.GradientTransformation
    actor_opt_state: optax.OptState
    critic_opt: optax.GradientTransformation
    critic_opt_state : optax.OptState
    alpha_opt: optax.GradientTransformation
    alpha_opt_state : optax.OptState
    log_alpha: jax.Array


# class SAC:
#     def __init__(self, config: Config):
#         self.config = config

        
#         # 初始化网络
#         key = jax.random.PRNGKey(config.seed)
#         keys = jax.random.split(key, 3)
        

        
#         self.actor = Actor(obs_dim, action_dim, config.hidden_dim, keys[0])
#         self.critic = Critic(obs_dim, action_dim, config.hidden_dim, keys[1])
#         self.target_critic = Critic(obs_dim, action_dim, config.hidden_dim, keys[2])
        

#         # 初始化buffer
#         self.buffer = ReplayBuffer(config.buffer_size, obs_dim, 1)  # action_dim=1 for discrete actions
        
#         # 初始化alpha
#         self.log_alpha = jnp.array(np.log(config.alpha))
#         self.target_entropy = config.target_entropy
        
    
@eqx.filter_jit
def _update_critic(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])
    # 计算目标Q值
    next_actions = train_state.actor(next_obs, key_array).sample(key=key)
    next_q1, next_q2 = train_state.target_critic(next_obs)
    next_q = jnp.minimum(next_q1, next_q2)
    next_q = jnp.take_along_axis(next_q, next_actions[..., None], axis=-1)
    target_q = rewards + (1 - dones) * config.gamma * next_q
    
    # 更新critic
    def critic_loss(model):
        q1, q2 = model(obs, actions)
        loss = jnp.mean((q1 - target_q) ** 2 + (q2 - target_q) ** 2)
        return loss
    
    grads = eqx.filter_grad(critic_loss)(train_state.critic)
    updates, new_critic_opt_state = train_state.critic_opt.update(grads, train_state.critic_opt_state, eqx.filter(train_state.critic, eqx.is_array))
    new_critic = eqx.apply_updates(train_state.critic, updates)
    new_train_state = train_state.replace(critic=new_critic, critic_opt_state=new_critic_opt_state)
    return new_train_state
    
@eqx.filter_jit
def _update_actor(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])
    def actor_loss(actor, critic):
        dist = train_state.actor(obs, key_array)
        new_actions = dist.sample(key=key)
        q1, q2 = train_state.critic(obs)
        q = jnp.minimum(q1, q2)
        q = jnp.take_along_axis(q, new_actions[..., None], axis=-1)
        
        log_prob = dist.log_prob(new_actions)
        
        loss = jnp.mean(train_state.log_alpha * log_prob - q)
        return loss
    
    grads = eqx.filter_grad(actor_loss)(train_state.actor, train_state.critic)
    updates, new_actor_opt_state = train_state.actor_opt.update(grads, train_state.actor_opt_state, eqx.filter(train_state.actor, eqx.is_array))
    new_actor = eqx.apply_updates(train_state.actor, updates)
    new_train_state = train_state.replace(actor=new_actor, actor_opt_state=new_actor_opt_state)
    return new_train_state
    
@eqx.filter_jit
def _update_alpha(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])
    def alpha_loss(log_alpha):
        dist = train_state.actor(obs, key_array)
        new_actions = dist.sample(key=key)
        log_prob = dist.log_prob(new_actions)
        
        loss = -jnp.mean(log_alpha * (log_prob + config.target_entropy))
        return loss
    
    grads = jax.grad(alpha_loss)(train_state.log_alpha)
    updates, new_alpha_opt_state = train_state.alpha_opt.update(grads, train_state.alpha_opt_state)
    new_log_alpha = train_state.log_alpha - updates
    new_train_state = train_state.replace(log_alpha=new_log_alpha, alpha_opt_state=new_alpha_opt_state)
    return new_train_state
    
def train(train_state, config, env, env_params, buffer):
    key = jax.random.PRNGKey(config.seed)
    
    for episode in range(config.num_episodes):
        obs, state = env.reset(key)
        episode_reward = 0
        done = False
        
        while not done:
            # 收集经验
            if buffer.size < config.warmup_steps:
                action = jax.random.randint(key, shape=(), minval=0, maxval=2)  # Random action 0 or 1
            else:
                action = train_state.actor(obs, key).sample(key=key)
            
            next_obs, state, reward, done, _ = env.step(key, state, action, env_params)
            buffer.add(obs, action, reward, next_obs, done)
            
            # 更新网络
            if buffer.size >= config.warmup_steps:
                batch = buffer.sample(config.batch_size)
                keys = jax.random.split(key, 3)
                train_state = _update_critic(train_state, batch, keys[0])
                train_state = _update_actor(train_state, batch, keys[1])
                train_state = _update_alpha(train_state, batch, keys[2])
                
                critic_params, critic = eqx.partition(train_state.critic, eqx.is_array)
                target_critic_params, target_critic = eqx.partition(train_state.target_critic, eqx.is_array)

                new_target_critic_params = jax.tree.map(
                    lambda o, t: o * config.tau + t * (1 - config.tau),
                    critic_params,
                    target_critic_params
                )

                new_target_critic = eqx.combine(new_target_critic_params, target_critic)
                critic = eqx.combine(critic_params, critic)

                train_state = train_state.replace(target_critic=new_target_critic)
            
            obs = next_obs
            episode_reward += reward
            key, _ = jax.random.split(key)
        
        if episode % config.eval_interval == 0:
            print(f"Episode {episode}, Reward: {episode_reward}")

if __name__ == "__main__":

    config = Config()
    key = jax.random.PRNGKey(config.seed)
    actor = Actor(config.obs_dim, config.action_dim, config.hidden_dim, key)
    critic = Critic(config.obs_dim, config.action_dim, config.hidden_dim, key)
    target_critic = Critic(config.obs_dim, config.action_dim, config.hidden_dim, key)
    log_alpha = jnp.array(jnp.log(config.alpha))
    actor_opt = optax.adam(config.learning_rate)
    actor_opt_state = actor_opt.init(eqx.filter(actor, eqx.is_array))
    critic_opt = optax.adam(config.learning_rate)
    critic_opt_state = critic_opt.init(eqx.filter(critic, eqx.is_array))
    alpha_opt = optax.adam(config.learning_rate)
    alpha_opt_state = alpha_opt.init(eqx.filter(log_alpha, eqx.is_array))

    train_state = TrainState(
        actor=actor,
        critic=critic,
        target_critic=target_critic,
        actor_opt=actor_opt,
        actor_opt_state=actor_opt_state,
        critic_opt=critic_opt,
        critic_opt_state=critic_opt_state,
        alpha_opt=alpha_opt,
        alpha_opt_state=alpha_opt_state,
        log_alpha=log_alpha,
    )

    env, env_params = gymnax.make(config.env_name)
    buffer = ReplayBuffer(config.buffer_size, config.obs_dim, config.action_dim)
    train(train_state, config, env, env_params, buffer)
