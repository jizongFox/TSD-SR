#!/usr/bin/env python3
from dataclasses import dataclass
import subprocess

import tyro


@dataclass
class Args:
    input_path: str
    output_dir: str = "outputs/deblur"


def main(args: Args) -> None:
    command = [
        "python",
        "test/test_tsdsr.py",
        "--pretrained_model_name_or_path",
        "checkpoint/sd3-medium",
        "--lora_dir",
        "checkpoint/tsdsr",
        "--embedding_dir",
        "dataset/default",
        "--input_dir",
        args.input_path,
        "--output_dir",
        args.output_dir,
        "--upscale",
        "1",
        "--align_method",
        "adain",
        "--mixed_precision",
        "fp16",
    ]
    subprocess.run(command, check=True)
    print(f"Deblur finished: {args.output_dir}")


if __name__ == "__main__":
    main(tyro.cli(Args))
