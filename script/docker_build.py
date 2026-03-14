#!/usr/bin/env python3
from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

import tyro


@dataclass
class Args:
    image_tag: str = "tsd-sr:cu128"
    log_file: str = "docker-build.log"


def main(args: Args) -> None:
    log_path = Path(args.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    command = ["docker", "build", "--progress=plain", "-t", args.image_tag, "."]

    with log_path.open("w", encoding="utf-8") as log_fp:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            log_fp.write(line)

        return_code = process.wait()

    if return_code != 0:
        raise SystemExit(return_code)


if __name__ == "__main__":
    main(tyro.cli(Args))
