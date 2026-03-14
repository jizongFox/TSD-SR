FROM pytorch/pytorch:2.7.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /workspace/TSD-SR

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    curl \
    wget \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

RUN python -m pip install --upgrade pip setuptools wheel && \
    python -m pip install \
    diffusers==0.29.1 \
    transformers==4.49.0 \
    peft==0.15.0 \
    pillow==10.3.0 \
    accelerate \
    numpy==1.26.4 \
    loralib==0.1.2 \
    opencv-python-headless==4.10.0.84 \
    scipy \
    safetensors \
    tqdm \
    gdown \
    sentencepiece \
    huggingface_hub[hf_transfer] \
    tyro \
    einops

COPY . /workspace/TSD-SR

RUN mkdir -p /workspace/TSD-SR/checkpoint/tsdsr \
    /workspace/TSD-SR/checkpoint/sd3-medium \
    /workspace/TSD-SR/dataset/default \
    /workspace/TSD-SR/imgs/test \
    /workspace/TSD-SR/outputs

ENV HF_HOME=/workspace/.cache/huggingface \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["bash"]
