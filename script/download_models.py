#!/usr/bin/env python3
from warnings import warn
import argparse
import importlib
import os
import shutil
import sys
from pathlib import Path

import warnings
warnings.filterwarnings("ignore")



def copy_first_match(src_root: Path, filename: str, dst_path: Path) -> bool:
    matches = list(src_root.rglob(filename))
    if not matches:
        return False
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(matches[0], dst_path)
    return True


def validate_required_assets(project_root: Path) -> None:
    required = [
        project_root / "checkpoint" / "tsdsr" / "transformer.safetensors",
        project_root / "checkpoint" / "tsdsr" / "vae.safetensors",
        project_root / "dataset" / "default" / "prompt_embeds.pt",
        project_root / "dataset" / "default" / "pool_embeds.pt",
        project_root / "checkpoint" / "sd3-medium" / "transformer",
        project_root / "checkpoint" / "sd3-medium" / "vae",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        print("\n[ERROR] Missing required files/directories:")
        for path in missing:
            print(f"  - {path}")
        print(
            "\nTSD-SR assets (LoRA + prompt embeddings) are hosted on Google Drive/OneDrive in the official repo.\n"
            "If automatic download misses files, download them manually from the README links and place:\n"
            "  - transformer.safetensors + vae.safetensors -> checkpoint/tsdsr/\n"
            "  - prompt_embeds.pt + pool_embeds.pt -> dataset/default/\n"
        )
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download SD3 + TSD-SR inference assets"
    )
    parser.add_argument(
        "--project_root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the TSD-SR project root",
    )
    parser.add_argument(
        "--sd3_repo",
        type=str,
        default="stabilityai/stable-diffusion-3-medium-diffusers",
        help="Hugging Face model repo id",
    )
    parser.add_argument(
        "--tsdsr_drive_url",
        type=str,
        default="https://drive.google.com/drive/folders/1XJY9Qxhz0mqjTtgDXr07oFy9eJr8jphI",
        help="Google Drive folder URL from the official README",
    )
    parser.add_argument(
        "--skip_sd3",
        action="store_true",
        help="Skip downloading SD3 from Hugging Face",
    )
    parser.add_argument(
        "--skip_tsdsr",
        action="store_true",
        help="Skip downloading TSD-SR assets from Google Drive",
    )
    args = parser.parse_args()

    project_root = args.project_root.resolve()
    checkpoint_dir = project_root / "checkpoint"
    sd3_dir = checkpoint_dir / "sd3-medium"
    tsdsr_dir = checkpoint_dir / "tsdsr"
    emb_dir = project_root / "dataset" / "default"
    tmp_drive_dir = project_root / ".downloads" / "tsdsr_gdrive"

    sd3_dir.mkdir(parents=True, exist_ok=True)
    tsdsr_dir.mkdir(parents=True, exist_ok=True)
    emb_dir.mkdir(parents=True, exist_ok=True)
    tmp_drive_dir.mkdir(parents=True, exist_ok=True)

    hf_token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACE_HUB_TOKEN")
    )
    print(f"[1/3] Downloading SD3 components from {args.sd3_repo} ...")
    print("      (requires Hugging Face access to SD3; set HF_TOKEN if needed)")
    if not args.skip_sd3:
        try:
            from huggingface_hub import snapshot_download
            # snapshot_download = importlib.import_module(
                # "huggingface_hub"
            # ).snapshot_download

            snapshot_download(
                repo_id=args.sd3_repo,
                local_dir=str(sd3_dir),
                token=hf_token,
                allow_patterns=[
                    "transformer/*",
                    "vae/*",
                    "model_index.json",
                ],
                local_dir_use_symlinks=False,
                resume_download=True,
            )
        except Exception as exc:
            print(f"[WARN] SD3 download failed: {exc}")
    else:
        print("      - skipped by --skip_sd3")

    print("[2/3] Downloading official TSD-SR assets from Google Drive ...")
    if not args.skip_tsdsr:
        try:
            gdown = importlib.import_module("gdown")

            gdown.download_folder(
                url=args.tsdsr_drive_url, output=str(tmp_drive_dir), quiet=False
            )
        except Exception as exc:
            print(f"[WARN] Google Drive automatic download failed: {exc}")
    else:
        print("      - skipped by --skip_tsdsr")

    print("[3/3] Locating required LoRA and embedding files ...")
    copied = {
        "transformer.safetensors": copy_first_match(
            tmp_drive_dir,
            "transformer.safetensors",
            tsdsr_dir / "transformer.safetensors",
        ),
        "vae.safetensors": copy_first_match(
            tmp_drive_dir, "vae.safetensors", tsdsr_dir / "vae.safetensors"
        ),
        "prompt_embeds.pt": copy_first_match(
            tmp_drive_dir, "prompt_embeds.pt", emb_dir / "prompt_embeds.pt"
        ),
        "pool_embeds.pt": copy_first_match(
            tmp_drive_dir, "pool_embeds.pt", emb_dir / "pool_embeds.pt"
        ),
    }
    for name, ok in copied.items():
        print(f"      - {name}: {'ok' if ok else 'not found'}")

    validate_required_assets(project_root)
    print("\nDone. All required files are ready for inference.")


if __name__ == "__main__":
    main()
