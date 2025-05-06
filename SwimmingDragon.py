# 导入类型注解和常用库
from typing import Any, Dict, Optional, Tuple, Union

# JAX相关的数值计算库
import chex
from flax import struct  # Flax的数据结构工具
import jax
from jax import lax  # JAX的低级操作库
import jax.numpy as jnp  # JAX的数值计算库

# Gymnax环境基类和空间定义
from gymnax.environments import environment
from gymnax.environments import spaces


# 定义环境状态结构体（继承自EnvState）
@struct.dataclass
class EnvState(environment.EnvState):
    matrix_state: jnp.array
    x: int    #飞船坐标值（两种状态：0或1）
    xp: int 
    over: int     
    time: int    # 当前步数计数器


# 定义环境参数结构体
@struct.dataclass
class EnvParams(environment.EnvParams):
    
    pass   # 最大步数限制（比v0版本更多）


class SwimmingDragon(environment.Environment[EnvState, EnvParams]):

    def __init__(self, max_steps_in_episode: int, grid_size: int):
        self.max_steps_in_episode = max_steps_in_episode
        self.reward = (1.0 / max_steps_in_episode)
        self.grid_size = grid_size  # 8/10/12
        self.obs_shape = ()  # 观测空间维度（4维状态向量）

    @property
    def default_params(self) -> EnvParams:
        
        return EnvParams()

    def step_env(
        self,
        key: chex.PRNGKey,        # 随机数种子
        state: EnvState,          # 当前环境状态
        action: int,              # 动作输入(0,1,2,3,4)
        params: EnvParams,        # 环境参数
    ) -> Tuple[chex.Array, EnvState, jnp.ndarray, jnp.ndarray, Dict[Any, Any]]:
        """执行环境步进逻辑"""
        # # 判断前一步是否已终止
        # prev_terminal = self.is_terminal(state, params)
        key, newkey = jax.random.split(key)
        xp = state.xp
        over = state.over
        x = state.x
        x = jnp.clip(jnp.where(action == 2, x-1, x), 0, self.grid_size-1)
        x = jnp.clip(jnp.where(action == 3, x+1, x), 0, self.grid_size-1)
        
        # 更新状态并判断当前是否终止
        # 更新矩阵状态（敌人移动）
        matrix_state = state.matrix_state
        xp = matrix_state[self.grid_size-1, x]
        over = xp # 检查终止条件
        # 所有行向下移动一行（腾出第0行给新敌人）
        matrix_state = matrix_state.at[1:self.grid_size, :].set(matrix_state[0:self.grid_size-1, :])
        # 在最底部生成新的敌人行
        # enemy_new = self.random_enemy(key)
        # self.matrix_state[self.grid_size, :] = enemy_new
        newkey, enemy_key = jax.random.split(newkey)
        enemy_new = self.random_enemy(enemy_key)
        enemy_new = jnp.squeeze(enemy_new)
        jax.debug.print("matrix_state: {}", enemy_new)
        matrix_state = matrix_state.at[0, :].set(enemy_new)
        xp = matrix_state[self.grid_size-1, x]
        
        # 奖励函数：
        reward = self.reward
        state = EnvState(
            matrix_state = matrix_state, # 更新状态矩阵(整个棋盘状态)
            x = x,
            xp = matrix_state[self.grid_size-1, x],
            over = over,
            time = state.time + 1,
        )
        
        done = self.is_terminal(state, params)  # 检查终止条件

        # 返回观测值、新状态、奖励、终止标志和折扣因子
        return (
            lax.stop_gradient(self.get_obs(state)),  # 停止梯度传播
            lax.stop_gradient(state),
            jnp.array(reward),
            done,
            {"discount": self.discount(state, params)},  # 折扣因子
        )

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> Tuple[chex.Array, EnvState]:
        """重置环境到初始状态"""
        key, subkey1 = jax.random.split(key)
        matrix_state = jnp.zeros((self.grid_size, self.grid_size), dtype=jnp.int32) #生成棋盘网格大小的全0状态矩阵
        x = jax.random.randint(subkey1, shape=(), minval=0, maxval=self.grid_size).astype(jnp.int32) #随机出生位置       
        # 构建初始状态对象
        state = EnvState(
            x = x,
            matrix_state = matrix_state,
            over = 0,
            xp = matrix_state[self.grid_size-1, x], #飞船随机出生
            time = 0,  # 步数清零
        )
        return self.get_obs(state), state  # 返回初始观测和状态
    
    def random_enemy(self, key) -> jnp.ndarray:
        """生成一行随机两个敌人(值为1)的数组"""
        
        # key = jax.random.PRNGKey(42)

        # # 1. 生成 [1, 1, 0, ..., 0]
        # base = jnp.array([1, 1] + [0] * (self.grid_size - 2), dtype=jnp.int32)  # shape (8,)

        # # 2. 打乱顺序
        # shuffled = jax.random.permutation(key, base)

        # # 3. 调整为 1 行 8 列
        # enemy_row = shuffled.reshape(1, 8)

        # print(enemy_row)
        key, subkey2 = jax.random.split(key)
        enemy_row = jnp.zeros(self.grid_size, dtype=jnp.int32)
        indices = jax.random.choice(subkey2, jnp.arange(self.grid_size), shape=(2,), replace=False)
        enemy_row = enemy_row.at[indices].set(1)
        enemy_row = enemy_row.reshape(1, -1)

        return enemy_row
     
    def get_obs(self, state: EnvState, params=None, key=None) -> chex.Array:
        """将状态转换为观测值"""
        # 返回完整的状态向量作为观测值
        return state.matrix_state

    def is_terminal(self, state: EnvState, params: EnvParams) -> jnp.ndarray:
        """判断当前状态是否终止"""
        # 检查是否碰撞
        
        done_crash = state.xp + state.over
        # 检查是否达到最大步数
        done_steps = state.time >= self.max_steps_in_episode
        # 组合所有终止条件
        done = jnp.logical_or(done_crash, done_steps)
        return done

    def render(self, state: EnvState, params: EnvParams):
        pass

    @property
    def name(self) -> str:
        """返回环境名称"""
        return "SwimmingDragon"

    @property
    def num_actions(self) -> int:
        """返回动作空间大小（5种动作）"""
        return 5

    def action_space(self, params: Optional[EnvParams] = None) -> spaces.Discrete:
        """定义动作空间（离散空间）"""
        return spaces.Discrete(5)  # 5

    def observation_space(self, params: EnvParams) -> spaces.Box:
        """定义观测空间（连续空间）"""
        # 观测包含4个值：[x, over, matrix_state, xp]
        # 但matrix_state是二维数组，需要展平或选择关键信息
        # 这里简化处理，只使用关键标量信息
    
        # 定义各维度的上下界
        high = jnp.array([
            self.grid_size - 1,  # x的最大值
            1,                   # over的最大值
            1,                   # matrix_state的最大值（假设）
            1                    # xp的最大值
        ], dtype=jnp.float32)
    
        return spaces.Box(low=0, high=high, shape=(4,), dtype=jnp.float32)

    def state_space(self, params: EnvParams) -> spaces.Dict:
        """定义完整状态空间（包含所有状态变量）"""
        return spaces.Dict({
            #"over": spaces.Discrete(2),  # 0或1，表示是否结束
            "matrix_state": spaces.Box(
                low=0,
                high=1,
                shape=(self.grid_size, self.grid_size),
                dtype=jnp.float32
            ),  # 网格状态矩阵，值在0-1之间
            "x": spaces.Discrete(self.grid_size),  # 飞船x坐标，范围0到grid_size-1
            "xp": spaces.Discrete(2),  # 飞船状态，0或1
            "time": spaces.Discrete(params.max_steps_in_episode + 1)  # 时间步数
        })
    
class SwimmingDragonEasy(SwimmingDragon):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 200, grid_size = 8, **kwargs)


class SwimmingDragonMedium(SwimmingDragon):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 400, grid_size = 10, **kwargs)


class SwimmingDragonHard(SwimmingDragon):
    def __init__(self, **kwargs):
        super().__init__(max_steps_in_episode = 600, grid_size = 12, **kwargs)