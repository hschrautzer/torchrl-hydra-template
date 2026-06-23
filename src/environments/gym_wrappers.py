from __future__ import annotations

import gymnasium as gym
import numpy as np


class RawEpisodeStatistics(gym.Wrapper):
    """Expose raw full-game return before reward clipping / EpisodicLifeEnv."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.raw_episode_return = 0.0
        self.raw_episode_length = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.raw_episode_return = 0.0
        self.raw_episode_length = 0
        info["raw_episode_return"] = np.array(0.0, dtype=np.float32)
        info["raw_episode_length"] = np.array(0, dtype=np.int64)
        info["raw_episode_done"] = np.array(False, dtype=np.bool_)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        self.raw_episode_return += float(reward)
        self.raw_episode_length += 1
        done = bool(terminated or truncated)

        info["raw_episode_return"] = np.array(
            self.raw_episode_return if done else 0.0,
            dtype=np.float32,
        )
        info["raw_episode_length"] = np.array(
            self.raw_episode_length if done else 0,
            dtype=np.int64,
        )
        info["raw_episode_done"] = np.array(done, dtype=np.bool_)

        return obs, reward, terminated, truncated, info

class NoPositiveRewardTimeout(gym.Wrapper):
    """Truncate if no positive reward is observed for too many env steps."""

    def __init__(self, env: gym.Env, max_no_reward_steps: int = 1000):
        super().__init__(env)
        self.max_no_reward_steps = max_no_reward_steps
        self.steps_since_positive_reward = 0

    def reset(self, **kwargs):
        self.steps_since_positive_reward = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        if reward > 0:
            self.steps_since_positive_reward = 0
        else:
            self.steps_since_positive_reward += 1

        if self.steps_since_positive_reward >= self.max_no_reward_steps:
            truncated = True
            info["no_positive_reward_timeout"] = True
        else:
            info["no_positive_reward_timeout"] = False

        return obs, reward, terminated, truncated, info

class FireAfterLifeLoss(gym.Wrapper):
    """Press FIRE after losing a life without ending the real game episode."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        meanings = env.unwrapped.get_action_meanings()
        if "FIRE" not in meanings:
            raise ValueError("FireAfterLifeLoss requires an environment with FIRE action.")
        self.fire_action = meanings.index("FIRE")
        self.lives = 0

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.lives = self.env.unwrapped.ale.lives()
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)

        current_lives = self.env.unwrapped.ale.lives()
        lost_life = current_lives < self.lives and current_lives > 0
        self.lives = current_lives

        if lost_life and not terminated and not truncated:
            fire_obs, fire_reward, terminated, truncated, fire_info = self.env.step(
                self.fire_action
            )
            obs = fire_obs
            reward += fire_reward
            info.update(fire_info)
            info["fire_after_life_loss"] = True
        else:
            info["fire_after_life_loss"] = False

        return obs, reward, terminated, truncated, info