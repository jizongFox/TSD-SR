#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
import subprocess
import signal

import tyro


@dataclass
class Args:
    host_input_folder: str
    host_output_folder: str
    image_tag: str = "tsd-sr:cu128"
    num_gpus: int = 1
    gpu_ids: str = ""


def parse_gpu_ids(args: Args):
    if args.gpu_ids.strip():
        ids = [int(item.strip()) for item in args.gpu_ids.split(",") if item.strip()]
        if not ids:
            raise ValueError("gpu_ids is set but empty after parsing")
        return ids

    if args.num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {args.num_gpus}")
    return list(range(args.num_gpus))


def main(args: Args) -> None:
    host_input_dir = Path(args.host_input_folder).expanduser().resolve()
    host_output_dir = Path(args.host_output_folder).expanduser().resolve()

    if not host_input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {host_input_dir}")

    host_output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_gpu_ids(args)
    num_shards = len(gpu_ids)
    processes = []

    for shard_id, gpu_id in enumerate(gpu_ids):
        command = [
            "docker",
            "run",
            "--gpus",
            f"device={gpu_id}",
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
            "--num_shards",
            str(num_shards),
            "--shard_id",
            str(shard_id),
        ]

        print(f"[launch] gpu={gpu_id} shard={shard_id}/{num_shards}")
        processes.append((gpu_id, subprocess.Popen(command)))

    failed = False
    for gpu_id, process in processes:
        return_code = process.wait()
        if return_code != 0:
            failed = True
            print(f"[error] worker on gpu {gpu_id} failed with exit code {return_code}")

    if failed:
        for _, process in processes:
            if process.poll() is None:
                process.send_signal(signal.SIGTERM)
        raise SystemExit(1)

    print("All shards completed.")


if __name__ == "__main__":
    main(tyro.cli(Args))
