"""Convert the generated LIBERO-Core-LT RLDS dataset to OpenPI's LeRobot schema."""

from __future__ import annotations

import pathlib
import shutil

from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset
import tensorflow_datasets as tfds
import tyro


def main(
    data_dir: pathlib.Path,
    repo_id: str = "ecl/libero_core_lt",
    dataset_name: str = "libero_core_lt",
    fps: int = 10,
    overwrite: bool = False,
) -> None:
    output_path = HF_LEROBOT_HOME / repo_id
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"{output_path} exists; pass --overwrite to replace it")
        shutil.rmtree(output_path)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        robot_type="panda",
        fps=fps,
        features={
            "image": {"dtype": "image", "shape": (256, 256, 3), "names": ["height", "width", "channel"]},
            "wrist_image": {
                "dtype": "image",
                "shape": (256, 256, 3),
                "names": ["height", "width", "channel"],
            },
            "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
            "actions": {"dtype": "float32", "shape": (7,), "names": ["actions"]},
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )
    source = tfds.load(dataset_name, data_dir=str(data_dir), split="train")
    episode_count = 0
    for episode in source:
        for step in episode["steps"].as_numpy_iterator():
            dataset.add_frame(
                {
                    "image": step["observation"]["image"],
                    "wrist_image": step["observation"]["wrist_image"],
                    "state": step["observation"]["state"],
                    "actions": step["action"],
                    "task": step["language_instruction"].decode(),
                }
            )
        dataset.save_episode()
        episode_count += 1
    if episode_count != 154:
        raise RuntimeError(f"Expected 154 LIBERO-Core-LT episodes, converted {episode_count}")
    print(f"Converted {episode_count} episodes to {output_path}")


if __name__ == "__main__":
    tyro.cli(main)

