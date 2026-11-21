from __future__ import annotations

import json
import os
from typing import Any, Union

import gymnasium as gym
import h5py
import numpy as np
import numpy.typing as npt
import torch
from mani_skill.utils.structs.types import Array

from utils.datasets import Dataset

NDArray = npt.NDArray[Any]


def recursive_to_numpy(
    data: Union[torch.Tensor, dict[str, Any], list[Any], tuple[Any, ...], NDArray],
) -> Union[NDArray, dict[str, Any], list[Any], tuple[Any, ...]]:
    if isinstance(data, torch.Tensor):
        data = data.cpu().numpy()
    elif isinstance(data, dict):
        return {k: recursive_to_numpy(v) for k, v in data.items()}
    elif isinstance(data, (list, tuple)):
        return type(data)(recursive_to_numpy(v) for v in data)

    if isinstance(data, np.ndarray):
        return data.item() if data.size == 1 else data.squeeze()

    return data


class NumpyFlattenWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super().__init__(env)
    
    def step(self, action: Array | dict[str, Array]) -> tuple[Array, float, bool, bool, dict[str, Any]]:
        observation, reward, termination, truncation, info = self.env.step(action)

        return (
            recursive_to_numpy(observation),
            recursive_to_numpy(reward),
            recursive_to_numpy(termination),
            recursive_to_numpy(truncation),
            recursive_to_numpy(info),
        )

    def reset(
        self,
        seed: int | list[int] | None = None,
        options: dict[str, Any] | None = None,
    ):
        observation, info = self.env.reset(seed=seed, options=options)

        return (
            recursive_to_numpy(observation),
            recursive_to_numpy(info),
        )


def make_env(env_name: str):
    """Make ManiSkill3 environment."""
    env = gym.make(
        env_name,
        num_envs=1,
        reward_mode="normalized_dense",
        control_mode="pd_ee_delta_pose",
        render_mode="rgb_array",
        human_render_camera_configs={
            "width": 224,
            "height": 224,
        },
    )
    env = NumpyFlattenWrapper(env)
    return env


def get_dataset(
    demo_dataset_path: str,
    num_demos: int = -1,
    shuffle: bool = False,
    skip_failed=True,
    rng: np.random.Generator | None = None,
):
    """Make ManiSkill3 dataset.

    Args:
        env: Environment instance.
        env_name: Name of the environment.
    """
    demo_dataset_path = os.path.expanduser(demo_dataset_path)
    demo_dataset = h5py.File(demo_dataset_path)
    with open(demo_dataset_path.replace(".h5", ".json"), "r") as f:
        demo_dataset_meta = json.load(f)
    if num_demos == -1:
        num_demos = len(demo_dataset_meta["episodes"])
    if shuffle:
        rng = np.random.default_rng(rng)
        rng.shuffle(demo_dataset_meta["episodes"])
    
    all_observations = []
    all_actions = []
    all_next_observations = []
    all_terminals = []
    all_rewards = []
    all_masks = []

    load_count = 0

    for episode in demo_dataset_meta["episodes"]:
        # NOTE (stao): MS3 dataset stores success elsewhere
        if not episode["success"] and skip_failed:
            continue

        demo_id = episode["episode_id"]
        demo = demo_dataset[f"traj_{demo_id}"]
        reset_kwargs = episode["reset_kwargs"]
        seed = None
        if "episode_seed" in episode:
            seed = episode["episode_seed"]
        elif "seed" in reset_kwargs:
            seed = reset_kwargs["seed"]

        observations = demo["obs"][:-1]
        actions = demo["actions"][:]
        rewards = demo["rewards"][:]
        next_observations = demo["obs"][1:]
        # terminateds = demo["terminated"][:]
        terminateds = np.zeros_like(rewards, dtype=bool)  # ignore_terminations=True
        truncateds = demo["truncated"][:]
        terminals = (terminateds | truncateds).astype(np.float32)
        masks = np.ones_like(rewards)

        all_observations.append(observations)
        all_actions.append(actions)
        all_next_observations.append(next_observations)
        all_terminals.append(terminals)
        all_rewards.append(rewards)
        all_masks.append(masks)

        load_count += 1
        if load_count >= num_demos:
            break
    
    print(f"{load_count} demonstrations loaded.")

    return Dataset.create(
        observations=np.concatenate(all_observations, dtype=np.float32),
        actions=np.concatenate(all_actions, dtype=np.float32),
        next_observations=np.concatenate(all_next_observations, dtype=np.float32),
        terminals=np.concatenate(all_terminals, dtype=np.float32),
        rewards=np.concatenate(all_rewards, dtype=np.float32),
        masks=np.concatenate(all_masks, dtype=np.float32),
    )
