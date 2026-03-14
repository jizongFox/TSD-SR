# Docker deployment (single-image deblur)

This setup uses `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` and runs TSD-SR on your own image with **no upscaling** (`--upscale 1`).

## 1) Build image

```bash
python script/docker_build.py --image-tag tsd-sr:cu128 --log-file docker-build.log
```

This prints full build logs to terminal and also saves them to `docker-build.log`.

## 2) (Optional) Start interactive container

```bash
docker run --gpus all --rm -it \
  -v "$PWD":/workspace/TSD-SR \
  -w /workspace/TSD-SR \
  tsd-sr:cu128
```

If SD3 access on Hugging Face is gated for your account, pass token:

```bash
docker run --gpus all --rm -it \
  -e HF_TOKEN=hf_xxx \
  -v "$PWD":/workspace/TSD-SR \
  -w /workspace/TSD-SR \
  tsd-sr:cu128
```

## 3) Download model assets

You do not need to enter the container. Run downloader directly from host:

```bash
docker run --rm \
  -e HUGGINGFACE_HUB_TOKEN=hf_xxx \
  -v "$PWD":/workspace/TSD-SR \
  -w /workspace/TSD-SR \
  tsd-sr:cu128 \
  python script/download_models.py
```

If you are in mainland China (proxy example):

```bash
docker run --rm --network host \
  -e HUGGINGFACE_HUB_TOKEN=hf_xxx \
  -e http_proxy=http://127.0.0.1:10808 \
  -e https_proxy=http://127.0.0.1:10808 \
  -e HTTP_PROXY=http://127.0.0.1:10808 \
  -e HTTPS_PROXY=http://127.0.0.1:10808 \
  -v "$PWD":/workspace/TSD-SR \
  -w /workspace/TSD-SR \
  tsd-sr:cu128 \
  python script/download_models.py
```

Alternative practical approach:

1) Download SD3 from Hugging Face mirror (requires your HF token and SD3 permission):

```bash
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download stabilityai/stable-diffusion-3-medium-diffusers \
  --local-dir checkpoint/sd3-medium \
  --include "transformer/*" "vae/*" "model_index.json"
```

2) Download TSD-SR LoRA + embeddings from **OneDrive** link in `README.md` and place files manually:

- `transformer.safetensors` -> `checkpoint/tsdsr/transformer.safetensors`
- `vae.safetensors` -> `checkpoint/tsdsr/vae.safetensors`
- `prompt_embeds.pt` -> `dataset/default/prompt_embeds.pt`
- `pool_embeds.pt` -> `dataset/default/pool_embeds.pt`

3) Verify files from host (no container login):

```bash
docker run --rm \
  -v "$PWD":/workspace/TSD-SR \
  -w /workspace/TSD-SR \
  tsd-sr:cu128 \
  python -c "from pathlib import Path; req=['checkpoint/sd3-medium/transformer','checkpoint/sd3-medium/vae','checkpoint/tsdsr/transformer.safetensors','checkpoint/tsdsr/vae.safetensors','dataset/default/prompt_embeds.pt','dataset/default/pool_embeds.pt']; missing=[p for p in req if not Path(p).exists()]; print('missing:', missing if missing else 'none')"
```

Expected files after download:

- `checkpoint/sd3-medium/transformer/*`
- `checkpoint/sd3-medium/vae/*`
- `checkpoint/tsdsr/transformer.safetensors`
- `checkpoint/tsdsr/vae.safetensors`
- `dataset/default/prompt_embeds.pt`
- `dataset/default/pool_embeds.pt`

## 4) Deblur your own image (same output size)

Put image in repo, for example: `imgs/test/my_blurry.png`

```bash
python script/deblur.py --input-path imgs/test/my_blurry.png --output-dir outputs/deblur
```

Output is written to `outputs/deblur/` with same filename and same spatial size.

## 5) Folder-to-folder inference (preserve relative paths)

This maps one host folder as input and one host folder as output. The output keeps the same subfolder structure as input.

The same interface supports 1 GPU (default) and N GPUs via sharding.

```bash
python script/docker_infer_folder.py \
  --host-input-folder /path/on/host/input_images \
  --host-output-folder /path/on/host/output_images \
  --image-tag tsd-sr:cu128 \
  --num-gpus 1
```

Example:

```bash
python script/docker_infer_folder.py \
  --host-input-folder /data/blur_set \
  --host-output-folder /data/deblur_set \
  --num-gpus 4
```

You can also choose specific GPU ids:

```bash
python script/docker_infer_folder.py \
  --host-input-folder /data/blur_set \
  --host-output-folder /data/deblur_set \
  --gpu-ids 0,2,3
```
