#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
import subprocess

import tyro


@dataclass
class Args:
    host_input_folder: str
    host_output_folder: str
    image_tag: str = "tsd-sr:cu128"


def main(args: Args) -> None:
    host_input_dir = Path(args.host_input_folder).expanduser().resolve()
    host_output_dir = Path(args.host_output_folder).expanduser().resolve()

    if not host_input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {host_input_dir}")

    host_output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        "docker",
        "run",
        "--gpus",
        "all",
        "--rm",
        "-i",
        "-v",
        f"{Path.cwd()}:/workspace/TSD-SR",
        "-v",
        f"{host_input_dir}:/input:ro",
        "-v",
        f"{host_output_dir}:/output",
        "-w",
        "/workspace/TSD-SR",
        args.image_tag,
        "python",
        "test/test_tsdsr.py",
        "--pretrained_model_name_or_path",
        "checkpoint/sd3-medium",
        "--lora_dir",
        "checkpoint/tsdsr",
        "--embedding_dir",
        "dataset/default",
        "--input_dir",
        "/input",
        "--output_dir",
        "/output",
        "--upscale",
        "1",
        "--align_method",
        "adain",
        "--mixed_precision",
        "fp16",
        "--recursive",
        "--preserve_dir_structure",
    ]

    subprocess.run(command, check=True)


if __name__ == "__main__":
    main(tyro.cli(Args))
