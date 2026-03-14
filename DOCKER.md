# Docker deployment (single-image deblur)

This setup uses `pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel` and runs TSD-SR on your own image with **no upscaling** (`--upscale 1`).

## 1) Build image

```bash
bash script/docker_build.sh tsd-sr:cu128 docker-build.log
```

This prints full build logs to terminal and also saves them to `docker-build.log`.

## 2) Start container (GPU)

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

Inside container:

```bash
python script/download_models.py
```

If you are in mainland China, use this practical approach:

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

3) Verify files:

```bash
python script/download_models.py --skip_sd3 --skip_tsdsr
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
bash script/deblur.sh imgs/test/my_blurry.png outputs/deblur
```

Output is written to `outputs/deblur/` with same filename and same spatial size.
