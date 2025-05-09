import jax
import jax.numpy as jnp
import equinox as eqx
import popgym_arcade
from equinox import nn
import optax
import wandb
from distreqx import distributions
from typing import Tuple, NamedTuple
import numpy as np
from optax import linear_schedule

# 将Config类改为字典形式
config = {
    # 环境参数
    "wandb_mode": "online",
    "entity": "",
    "project": "sac",
    "alg_name": "soft actor-critic",
    "env_name": "SkittlesEasy",
    "seed": 42,
    "obs_dim": (128, 128, 3),  # CartPole observation dimension
    "action_dim": 5,  # CartPole action dimension (0 or 1)

    # SAC超参数
    "learning_rate": 3e-4,
    "clip_eps": 0.2,
    "buffer_size": 10000,
    "batch_size": 256,
    "gamma": 0.99,
    "tau": 0.005,
    "alpha": 0.1,
    "auto_entropy_tuning": True,
    "target_entropy": -np.log((1 / 5)) * 0.98,  # 注意这里直接使用了action_dim的值5


    # 训练参数
    "num_episodes": 5000,
    "eval_interval": 10,
    "warmup_steps": 5000
}


class Transition(NamedTuple):
    obs: jnp.ndarray
    action: jnp.ndarray
    reward: jnp.ndarray
    next_obs: jnp.ndarray
    done: jnp.ndarray

class ReplayBuffer(eqx.Module):
    buffer_size: int
    obs_dim: Tuple
    action_dim: int
    ptr: jnp.ndarray
    size: jnp.ndarray

    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    next_obs: jnp.ndarray
    dones: jnp.ndarray

    def __init__(self, buffer_size: int, obs_dim: Tuple, action_dim: int, key: jax.random.PRNGKey):
        self.buffer_size = buffer_size
        self.obs_dim = obs_dim
        self.action_dim = action_dim

        # Use JAX arrays for counters to make them JIT-compatible
        self.ptr = jnp.array(0, dtype=jnp.int32)
        self.size = jnp.array(0, dtype=jnp.int32)

        # Initialize storage with JAX arrays
        self.obs = jnp.zeros((buffer_size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=jnp.float32)
        self.actions = jnp.zeros((buffer_size, 1), dtype=jnp.int32)
        self.rewards = jnp.zeros((buffer_size, 1), dtype=jnp.float32)
        self.next_obs = jnp.zeros((buffer_size, obs_dim[0], obs_dim[1], obs_dim[2]), dtype=jnp.float32)
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

    def add_batch(self, obs_batch, action_batch, reward_batch, next_obs_batch, done_batch):
        batch_size = obs_batch.shape[0]
        indices = (jnp.arange(batch_size) + self.ptr) % self.buffer_size
        new_obs = self.obs.at[indices].set(obs_batch)
        new_actions = self.actions.at[indices].set(action_batch)
        new_rewards = self.rewards.at[indices].set(reward_batch)
        new_next_obs = self.next_obs.at[indices].set(next_obs_batch)
        new_dones = self.dones.at[indices].set(done_batch)
        new_ptr = (self.ptr + batch_size) % self.buffer_size
        new_size = jnp.minimum(self.size + batch_size, self.buffer_size)
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

class Actor(eqx.Module):
    action_dim: int
    cnn: nn.Sequential
    trunk: nn.Sequential

    def __init__(self, action_dim: int, key):
        key_array = jax.random.split(key, 14)
        self.action_dim = action_dim
        self.cnn = nn.Sequential(
            [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, key=key_array[0]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, key=key_array[1]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, key=key_array[2]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, key=key_array[3]),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )

        self.trunk = nn.Sequential(
            [
                nn.Linear(in_features=512, out_features=256, key=key_array[4]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=256, key=key_array[5]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=self.action_dim, key=key_array[6])
            ]
        )

    def __call__(self, x, key=None):
        x = jnp.transpose(x, (2, 0, 1))
        x = self.cnn(x)
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
    cnn_1: nn.Sequential
    cnn_2: nn.Sequential
    trunk_1: nn.Sequential
    trunk_2: nn.Sequential
    action_dim: int

    def __init__(self, action_dim: int, key):
        key_array = jax.random.split(key, 14)
        self.action_dim = action_dim

        self.cnn_1 = nn.Sequential(
            [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, key=key_array[0]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, key=key_array[1]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, key=key_array[2]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, key=key_array[3]),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )
        self.cnn_2 = nn.Sequential(
            [
                nn.Conv2d(in_channels=3, out_channels=64, kernel_size=5, stride=2, key=key_array[4]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2, key=key_array[5]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, key=key_array[6]),
                nn.Lambda(jax.nn.leaky_relu),
                nn.MaxPool2d(kernel_size=3, stride=1),
                nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1, stride=1, key=key_array[7]),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )
        self.trunk_1 = nn.Sequential(
            [
                nn.Linear(in_features=512, out_features=256, key=key_array[8]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=256, key=key_array[9]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=self.action_dim, key=key_array[10])
            ]
        )
        self.trunk_2 = nn.Sequential(
            [
                nn.Linear(in_features=512, out_features=256, key=key_array[11]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=256, key=key_array[12]),
                nn.LayerNorm(shape=256),
                nn.Lambda(jax.nn.leaky_relu),
                nn.Linear(in_features=256, out_features=self.action_dim, key=key_array[13])
            ]
        )

    def __call__(self, obs):
        obs = jnp.transpose(obs, (2, 0, 1))
        feature1 = self.cnn_1(obs)
        feature2 = self.cnn_2(obs)
        feature1 = jnp.reshape(feature1, -1)
        feature2 = jnp.reshape(feature2, -1)
        q1 = self.trunk_1(feature1)
        q2 = self.trunk_2(feature2)
        return q1, q2

class Alpha(eqx.Module):
    value: jax.Array

    def __init__(self, init_value=0.2):
        self.value = jnp.array([jnp.log(init_value)])

    def __call__(self):
        return jnp.exp(self.value)

class State(eqx.Module):
    """Base class"""

    def replace(self, **kwargs):
        """Replaces existing fields."""
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
    next_actions, (action_prob, log_action_prob), max_logits_action = eqx.filter_vmap(train_state.actor)(next_obs, key_array)
    next_target_q1, next_target_q2 = eqx.filter_vmap(train_state.target_critic)(next_obs)
    min_next_target_q = jnp.sum(action_prob * (jnp.minimum(next_target_q1, next_target_q2) - train_state.alpha() * log_action_prob), axis=-1, keepdims=True)
    target_q = rewards + (1 - dones) * config["gamma"] * min_next_target_q

    def critic_loss(model):
        q1, q2 = eqx.filter_vmap(model)(obs)
        actions_int = actions.squeeze().astype(jnp.int32)
        q1_a = q1[jnp.arange(obs.shape[0]), actions_int]
        q2_a = q2[jnp.arange(obs.shape[0]), actions_int]
        loss = jnp.mean((q1_a - target_q.squeeze()) ** 2) + jnp.mean((q2_a - target_q.squeeze()) ** 2)
        return loss

    loss, grads = eqx.filter_value_and_grad(critic_loss)(train_state.critic)
    updates, new_critic_opt_state = train_state.critic_opt.update(grads, train_state.critic_opt_state,
                                                              eqx.filter(train_state.critic, eqx.is_array))
    new_critic = eqx.apply_updates(train_state.critic, updates)
    new_train_state = train_state.replace(critic=new_critic, critic_opt_state=new_critic_opt_state)
    return new_train_state, loss

@eqx.filter_jit
def _update_actor(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])

    def actor_loss(actor):
        next_action, (action_prob, log_action_prob), max_logits_action = eqx.filter_vmap(actor)(obs, key_array)
        q1, q2 = eqx.filter_vmap(train_state.critic)(obs)
        q = jnp.minimum(q1, q2)
        inside_term = q - train_state.alpha() * log_action_prob
        loss = -(action_prob * inside_term).sum(axis=-1).mean()
        return loss

    loss, grads = eqx.filter_value_and_grad(actor_loss)(train_state.actor)
    updates, new_actor_opt_state = train_state.actor_opt.update(grads, train_state.actor_opt_state,
                                                              eqx.filter(train_state.actor, eqx.is_array))
    new_actor = eqx.apply_updates(train_state.actor, updates)
    new_train_state = train_state.replace(actor=new_actor, actor_opt_state=new_actor_opt_state)
    return new_train_state, loss

@eqx.filter_jit
def _update_alpha(train_state, batch, key):
    obs, actions, rewards, next_obs, dones = batch
    key_array = jax.random.split(key, obs.shape[0])

    def alpha_loss(log_alpha):
        next_action, (action_prob, log_action_prob), max_logits_action = eqx.filter_vmap(train_state.actor)(obs, key_array)
        entropy = -jnp.sum(action_prob * log_action_prob, axis=-1)
        alpha = log_alpha()
        loss = jnp.mean(-alpha * (entropy + config["target_entropy"]))
        return loss

    loss, grads = eqx.filter_value_and_grad(alpha_loss)(train_state.alpha)
    updates, new_alpha_opt_state = train_state.alpha_opt.update(grads, train_state.alpha_opt_state)
    new_alpha = eqx.apply_updates(train_state.alpha, updates)
    new_train_state = train_state.replace(alpha=new_alpha, alpha_opt_state=new_alpha_opt_state)
    return new_train_state, loss

def train(train_state, config, env_reset, env_step, env_params, buffer):

    key = jax.random.PRNGKey(config["seed"])
    # reset_key = jax.random.split(key, config["n_env"])
    obs, state = env_reset(key, env_params)


    for episode in range(config["num_episodes"]):

        episode_reward = 0
        done = False
        cum_critic_loss = 0
        cum_actor_loss = 0
        cum_alpha_loss = 0
        num_steps = 0

        while not done:
            key, action_key, step_key = jax.random.split(key, 3)

            if buffer.size < config["warmup_steps"]:
                action = jax.random.randint(action_key, shape=(), minval=0, maxval=5)
            else:
                action, (_, _), _ = train_state.actor(obs, action_key)

            next_obs, state, reward, done, _ = env_step(step_key, state, action, env_params)

            buffer = buffer.add(obs, action, reward, next_obs, done)

            if buffer.size >= config["warmup_steps"]:
                batch = buffer.sample(batch_size=config["batch_size"], key=key)
                keys = jax.random.split(key, 3)
                train_state, critic_loss = _update_critic(train_state, batch, keys[0])
                train_state, actor_loss = _update_actor(train_state, batch, keys[1])
                train_state, alpha_loss = _update_alpha(train_state, batch, keys[2])

                critic_params, critic_arch = eqx.partition(train_state.critic, eqx.is_array)
                target_critic_params, target_critic_arch = eqx.partition(train_state.target_critic, eqx.is_array)

                new_target_critic_params = jax.tree.map(
                    lambda o, t: o * config["tau"] + t * (1 - config["tau"]),
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
            num_updates +=

        avg_critic_loss = cum_critic_loss / num_steps if num_steps > 0 else 0
        avg_actor_loss = cum_actor_loss / num_steps if num_steps > 0 else 0
        avg_alpha_loss = cum_alpha_loss / num_steps if num_steps > 0 else 0

        dict = {
            "episode": episode,
            "episode_return": episode_reward,
            "avg_critic_loss": avg_critic_loss,
            "avg_actor_loss": avg_actor_loss,
            "avg_alpha_loss": avg_alpha_loss,
        }

        if buffer.size >= config["warmup_steps"]:
            wandb.log(dict)

        print(f"Episode {episode}, Reward: {episode_reward}, "
              f"Avg Critic Loss: {avg_critic_loss:.4f}, "
              f"Avg Actor Loss: {avg_actor_loss:.4f}, "
              f"Avg Alpha Loss: {avg_alpha_loss:.4f}")
    return train_state

def evaluate(model, config):
    seed = jax.random.PRNGKey(10)
    seed, _rng = jax.random.split(seed)
    env, env_params = popgym_arcade.make(config["env_name"], obs_size=config["OBS_SIZE"])
    vmap_reset = lambda n_envs: lambda rng: jax.vmap(env.reset, in_axes=(0, None))(
        jax.random.split(rng, n_envs), env_params
    )
    vmap_step = lambda n_envs: lambda rng, env_state, action: jax.vmap(
        env.step, in_axes=(0, 0, 0, None)
    )(jax.random.split(rng, n_envs), env_state, action, env_params)

    obs, state = vmap_reset(2)(_rng)

    wandb.init(project=f'{config["PROJECT"]}')
    frames = []
    for i in range(600):
        rng, rng_act, rng_step, _rng = jax.random.split(_rng, 4)
        action, (_, _), _ = eqx.filter_vmap(model)(obs)
        obs, new_state, reward, term, _ = vmap_step(2)(rng_step, state, action)
        state = new_state
        frame = np.asarray(obs[0])
        frame = (frame * 255).astype(np.uint8)
        frames.append(frame)
    frames = np.array(frames, dtype=np.uint8)
    frames = frames.transpose((0, 3, 1, 2))
    wandb.log({"{}_{}_SEED={}".format(
        config["alg_name"],
              config["env_name"],
              config["seed"]): wandb.Video(frames, fps=2)
              })

if __name__ == "__main__":
    wandb.init(
        entity=config["entity"],
        project=config["project"],
        tags=[
            config["alg_name"].upper(),
            config["env_name"].upper(),
            f"jax_{jax.__version__}",
        ],
        name=f'{config["alg_name"]}_{config["env_name"]}_{"SEED="}{config["seed"]}',
        config=config,
        mode=config["wandb_mode"],
    )

    wandb.run.log_code(".")

    # schedule = optax.linear_schedule(
    #     init_value=config["learning_rate"],
    #     end_value=1e-6,
    #     transition_steps=10000,
    # )

    key = jax.random.PRNGKey(config["seed"])
    actor = Actor(config["action_dim"], key)
    critic = DoubleCritic(config["action_dim"], key)
    alpha = Alpha()
    target_critic = DoubleCritic(config["action_dim"], key)
    actor_opt = optax.chain(
        optax.clip_by_global_norm(config["clip_eps"]),
        optax.adam(learning_rate=config["learning_rate"], eps=1e-5)
    )
    actor_opt_state = actor_opt.init(eqx.filter(actor, eqx.is_array))
    critic_opt = optax.chain(
        optax.clip_by_global_norm(config["clip_eps"]),
        optax.adam(learning_rate=config["learning_rate"], eps=1e-5)
    )
    critic_opt_state = critic_opt.init(eqx.filter(critic, eqx.is_array))
    alpha_opt = optax.adam(learning_rate=config["learning_rate"], eps=1e-5)

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

    env, env_params = popgym_arcade.make(config["env_name"], obs_size=128)
    env_reset = eqx.filter_jit(env.reset)
    env_step = eqx.filter_jit(env.step)
    buffer = ReplayBuffer(config["buffer_size"], config["obs_dim"], config["action_dim"], key=key)
    train_state = train(train_state, config, env_reset, env_step, env_params, buffer)
    evaluate(train_state.actor, config)