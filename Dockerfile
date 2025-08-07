FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux lsof && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

RUN ln -s /usr/bin/python3.10 /usr/bin/python
# install uv pip
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv pip install --system -U pip wheel setuptools
RUN uv pip install --system torch==2.7.1+cu126 torchvision==0.22.1+cu126 torchaudio==2.7.1+cu126 --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install --system transformers accelerate datasets evaluate trl peft deepspeed liger-kernel lomo-optim apollo-torch transformer_engine[pytorch]\
    bitsandbytes scipy sentencepiece pillow fastapi uvicorn unsloth==2025.7.3 unsloth-zoo==2025.7.4 xformers==0.0.29.post2 opensloth==0.1.8 \
    ruff natsort setproctitle glances[gpu] wandb cmake

RUN uv pip install --system https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl --no-build-isolation
