#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <input_image_or_dir> [output_dir]"
  exit 1
fi

INPUT_PATH="$1"
OUTPUT_DIR="${2:-outputs/deblur}"

python test/test_tsdsr.py \
  --pretrained_model_name_or_path checkpoint/sd3-medium \
  --lora_dir checkpoint/tsdsr \
  --embedding_dir dataset/default \
  --input_dir "$INPUT_PATH" \
  --output_dir "$OUTPUT_DIR" \
  --upscale 1 \
  --align_method adain \
  --mixed_precision fp16

echo "Deblur finished: $OUTPUT_DIR"
