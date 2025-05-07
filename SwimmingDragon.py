from typing import Any, Dict, Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax import lax
import jax.numpy as jnp
from gymnax.environments import environment, spaces
from popgym_arcade.environments.draw_utils import (draw_heart,
                                            draw_spade,
                                            draw_club,
                                            draw_diamond,
                                            draw_number,
                                            draw_str,
                                            draw_sub_canvas)

@struct.dataclass
class EnvState(environment.EnvState):
    matrix_state: chex.Array
    x: int
    xp: int 
    time: int


@struct.dataclass
class EnvParams(environment.EnvParams):
    pass


class SwimmingDragon(environment.Environment[EnvState, EnvParams]):
    """
    Jax compilable environment for the Swimming Dragon.
    
    ### Description
    In Swimming Dragon, the agent is tasked with avoiding enemies that are moving down the screen. 
    The agent can move left or right to dodge the enemies. The goal is to survive as long as possible without being hit by an enemy.
    There are three difficulties: easy, medium, and hard. Each difficulty has a different grid size and maximum steps in an episode.
    Easy: 8x8 grid, 200 steps
    Medium: 10x10 grid, 400 steps
    Hard: 12x12 grid, 600 steps
    The episode ends when the agent is hit by an enemy or the maximum number of steps is reached.

    ### Board Elements
    - 0: Empty
    - 1: Enemy
    The player can only move within the last row of the matrix, and their position is indicated by the column index.

    ### Action Space
    | Action | Description                         |
    |--------|-------------------------------------|
    | 0      | Up (No-op)                          |
    | 1      | Down (No-op)                        |
    | 2      | Left                                |
    | 3      | Right                               |
    | 4      | Fire (No-op)                        |
    
    ### Observation Space

    ### Reward
    - Reward Scale: 1.0 / max_steps_in_episode

    ### Termination & Truncation
    The episode ends when the agent is hit by an enemy or 
     the maximum number of steps is reached.

    ### Args
    - max_steps_in_episode: Maximum number of steps in an episode.
    - grid_size: Size of the grid (number of rows and columns).
    
    TODO:
        - obs_size: Size of the observation space.
        - partial_obs: Whether to use partial observation or not.

    """
    def __init__(
            self, 
            max_steps_in_episode: int, 
            grid_size: int,
            obs_size: int = 128,
            partial_obs = False,
            ):
        super().__init__()
        self.obs_size = obs_size
        self.max_steps_in_episode = max_steps_in_episode
        self.reward_scale = (1.0 / max_steps_in_episode)
        self.grid_size = grid_size
        self.partial_obs = partial_obs

    @property
    def default_params(self) -> EnvParams:
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,
        state: EnvState,
        action: int,
        params: EnvParams,
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """Perform a step in the environment."""
        key, newkey = jax.random.split(key)

        x = state.x
        x = jnp.clip(jnp.where(action == 2, x-1, x), 0, self.grid_size-1)
        x = jnp.clip(jnp.where(action == 3, x+1, x), 0, self.grid_size-1)
        
        matrix_state = state.matrix_state

        # all rows move down one (leaving the 0th row for new enemies) 
        matrix_state = matrix_state.at[1:self.grid_size, :].set(matrix_state[0:self.grid_size-1, :])

        newkey, enemy_key = jax.random.split(newkey)
        enemy_new = self.random_enemy(enemy_key)
        enemy_new = jnp.squeeze(enemy_new)

        matrix_state = matrix_state.at[0, :].set(enemy_new)
        
        state = EnvState(
            matrix_state = matrix_state,
            x = x,
            xp = matrix_state[self.grid_size-1, x],
            time = state.time + 1,
        )
        
        done = self.is_terminal(state, params)

        return (
            lax.stop_gradient(self.get_obs(state)),
            lax.stop_gradient(state),
            jnp.array(self.reward_scale),
            done,
            {"discount": self.discount(state, params)},
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """Reset the environment to an initial state."""
        key, subkey1 = jax.random.split(key)
        matrix_state = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32)
        x = jax.random.randint(subkey1, shape=(), minval=0, maxval=self.grid_size).astype(jnp.int32) 

        state = EnvState(
            x = x,
            matrix_state = matrix_state,
            xp = matrix_state[self.grid_size-1, x],
            time = 0,
        )
        return self.get_obs(state), state

    def random_enemy(self, key) -> jnp.ndarray:
        """Generate a random enemy row."""
        key, subkey2 = jax.random.split(key)
        enemy_row = jnp.zeros(self.grid_size, dtype=jnp.int32)
        indices = jax.random.choice(subkey2, jnp.arange(self.grid_size), shape=(2,), replace=False)
        enemy_row = enemy_row.at[indices].set(1)
        enemy_row = enemy_row.reshape(1, -1)
        return enemy_row

    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        # return self.render(state)
        return state.matrix_state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """Check if the episode is done."""
        done_crash = state.xp
        done_steps = state.time >= self.max_steps_in_episode
        done = jnp.logical_or(done_crash, done_steps)
        return done

    def render(self, state: EnvState, params: EnvParams):
        """Render the current state of the environment."""
        pass

    @property
    def name(self) -> str:
        return "SwimmingDragon"

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """Action space of the environment."""
        return spaces.Discrete(5)

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """Observation space of the environment."""
        return spaces.Box(jnp.zeros((0,)), jnp.ones((1,)), (self.obs_size, self.obs_size, 3), dtype=jnp.float32)


class SwimmingDragonEasy(SwimmingDragon):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 200, grid_size = 8, **kwargs)


class SwimmingDragonMedium(SwimmingDragon):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 400, grid_size = 10, **kwargs)


class SwimmingDragonHard(SwimmingDragon):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 600, grid_size = 12, **kwargs)