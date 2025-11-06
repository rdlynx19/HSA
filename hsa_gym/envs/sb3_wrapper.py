import gymnasium as gym
import numpy as np

class SB3Wrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

        # Flatten the action space for SB3 compatibility
        motors_low = self.env.action_space['motors'].low
        motors_high = self.env.action_space['motors'].high

        self.has_locks = "locks" in self.env.action_space.spaces
        if self.has_locks:
            locks_shape = self.env.action_space['locks'].shape
            locks_low = np.zeros(locks_shape, dtype=np.float32)
            locks_high = np.ones(locks_shape, dtype=np.float32)

            low = np.concatenate([motors_low, locks_low])
            high = np.concatenate([motors_high, locks_high])
        else:
            low = motors_low
            high = motors_high

        self.action_space = gym.spaces.Box(
            low=low,
            high=high,
            dtype=np.float32
        )

        # Cache size for action splitting
        self.motors_size = motors_low.shape[0]
        self.locks_size = locks_low.shape[0] if self.has_locks else 0

    def action(self, action):
        # Split flat action back into dict format
        motors_len = self.motors_size
        motors_act = action[:motors_len]
        if self.has_locks:
            locks_act = np.round(action[motors_len:]).astype(np.int8)
            return {
                'motors': motors_act,
                'locks': locks_act
            }
        else:
            return {
                'motors': motors_act
            }
