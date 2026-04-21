from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Literal

import gym_pusht  # noqa: F401
import gymnasium as gym
import numpy as np
import pygame
import tyro
import zarr

from hw1_imitation.data import (
    Normalizer,
    PushtChunkDataset,
    download_pusht,
    load_pusht_zarr,
)


@dataclass
class PreviewConfig:
    data_dir: Path = Path("data")
    mode: Literal["preview", "play", "play-images"] = "preview"
    num_examples: int = 5
    chunk_size: int = 8
    episode_idx: int = 0
    fps: float = 20.0
    max_steps: int | None = None
    render_mode: Literal["human", "rgb_array"] = "human"


def print_array_preview(name: str, array: np.ndarray, num_examples: int) -> None:
    print(f"{name}.shape = {array.shape}")
    preview = array[:num_examples]
    print(
        np.array2string(
            preview,
            precision=3,
            suppress_small=True,
            max_line_width=120,
        )
    )
    print()


def play_episode(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    episode_idx: int,
    fps: float,
    max_steps: int | None,
    render_mode: Literal["human", "rgb_array"],
) -> None:
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise ValueError(
            f"episode_idx must be in [0, {len(episode_ends) - 1}], got {episode_idx}"
        )

    start = int(episode_starts[episode_idx])
    end = int(episode_ends[episode_idx])
    if max_steps is not None:
        end = min(end, start + max_steps)

    env = gym.make("gym_pusht/PushT-v0", obs_type="state", render_mode=render_mode)
    step_delay = 1.0 / fps if fps > 0 else 0.0

    try:
        env.reset(options={"reset_to_state": states[start]})
        env.render()
        if step_delay > 0:
            time.sleep(step_delay)

        for t in range(start + 1, end):
            env.unwrapped._set_state(states[t])
            env.render()
            action = actions[t - 1]
            print(
                f"step {t - start:03d} | "
                f"state={np.array2string(states[t], precision=2, suppress_small=True)} | "
                f"action={np.array2string(action, precision=2, suppress_small=True)}"
            )
            if step_delay > 0:
                time.sleep(step_delay)
    finally:
        env.close()


def play_image_episode(
    zarr_path: Path,
    episode_ends: np.ndarray,
    episode_idx: int,
    fps: float,
    max_steps: int | None,
) -> None:
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    if episode_idx < 0 or episode_idx >= len(episode_ends):
        raise ValueError(
            f"episode_idx must be in [0, {len(episode_ends) - 1}], got {episode_idx}"
        )

    start = int(episode_starts[episode_idx])
    end = int(episode_ends[episode_idx])
    if max_steps is not None:
        end = min(end, start + max_steps)

    root = zarr.open(zarr_path, mode="r")
    frames = np.asarray(root["data"]["img"][start:end], dtype=np.uint8)
    if len(frames) == 0:
        raise ValueError("No image frames found for the selected episode.")

    pygame.init()
    height, width = frames[0].shape[:2]
    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption(f"Push-T images episode {episode_idx}")
    clock = pygame.time.Clock()

    try:
        for idx, frame in enumerate(frames):
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

            surface = pygame.surfarray.make_surface(np.transpose(frame, (1, 0, 2)))
            screen.blit(surface, (0, 0))
            pygame.display.flip()

            print(f"image frame {idx:03d} / {len(frames) - 1:03d}")
            if fps > 0:
                clock.tick(fps)
    finally:
        pygame.quit()


def preview_dataset(
    states: np.ndarray,
    actions: np.ndarray,
    episode_ends: np.ndarray,
    chunk_size: int,
    num_examples: int,
    zarr_path: Path,
) -> None:
    normalizer = Normalizer.from_data(states, actions)
    dataset = PushtChunkDataset(
        states,
        actions,
        episode_ends,
        chunk_size=chunk_size,
        normalizer=normalizer,
    )
    episode_starts = np.concatenate(([0], episode_ends[:-1]))
    episode_lengths = episode_ends - episode_starts

    print(f"dataset path: {zarr_path}")
    print(f"num episodes: {len(episode_ends)}")
    print(f"state dim: {states.shape[1]}")
    print(f"action dim: {actions.shape[1]}")
    print(f"chunk size: {chunk_size}")
    print(f"num chunk samples: {len(dataset)}")
    print(f"first episode lengths: {episode_lengths[:num_examples]}")
    print()

    print_array_preview("states[:num_examples]", states, num_examples)
    print_array_preview("actions[:num_examples]", actions, num_examples)

    sample_state, sample_action_chunk = dataset[0]
    print("dataset[0] preview")
    print(f"normalized state shape: {tuple(sample_state.shape)}")
    print(f"normalized action chunk shape: {tuple(sample_action_chunk.shape)}")
    print(
        np.array2string(
            sample_state.numpy(),
            precision=3,
            suppress_small=True,
            max_line_width=120,
        )
    )
    print(
        np.array2string(
            sample_action_chunk.numpy(),
            precision=3,
            suppress_small=True,
            max_line_width=120,
        )
    )
    print()

    print("normalizer stats")
    print(
        f"state mean (first 5): {np.array2string(normalizer.state_mean[:5], precision=3, suppress_small=True)}"
    )
    print(
        f"action mean: {np.array2string(normalizer.action_mean, precision=3, suppress_small=True)}"
    )


def main() -> None:
    config = tyro.cli(
        PreviewConfig,
        description="Preview the Push-T dataset or play back an expert trajectory.",
    )

    zarr_path = download_pusht(config.data_dir)
    states, actions, episode_ends = load_pusht_zarr(zarr_path)

    if config.mode == "play-images":
        play_image_episode(
            zarr_path,
            episode_ends,
            episode_idx=config.episode_idx,
            fps=config.fps,
            max_steps=config.max_steps,
        )
        return

    if config.mode == "play":
        play_episode(
            states,
            actions,
            episode_ends,
            episode_idx=config.episode_idx,
            fps=config.fps,
            max_steps=config.max_steps,
            render_mode=config.render_mode,
        )
        return

    preview_dataset(
        states,
        actions,
        episode_ends,
        chunk_size=config.chunk_size,
        num_examples=config.num_examples,
        zarr_path=zarr_path,
    )


if __name__ == "__main__":
    main()
