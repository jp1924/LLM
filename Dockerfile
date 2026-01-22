FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

WORKDIR /root
USER root

# 이게 없어서 uv 설치할 때 애러가 발생하더라.
RUN mkdir -p /tmp && chmod 1777 /tmp 

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV UV_SYSTEM_PYTHON="1"
COPY ./.tmux.conf /root/.tmux.conf
COPY ./.bashrc /root/.bashrc
COPY ./.viminfo /root/.viminfo
COPY ./.vimrc /root/.vimrc

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux lsof && \
    ln -s /usr/bin/python3.10 /usr/bin/python

# install uv pip
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

# install python packages
RUN uv pip install -U pip wheel setuptools debugpy pydevd
RUN uv pip install transformers accelerate datasets liger-kernel trl peft deepspeed lomo-optim apollo-torch transformer_engine[pytorch] evaluate \
    bitsandbytes scipy sentencepiece pillow fastapi uvicorn ruff natsort setproctitle glances[gpu] wandb cmake latex2sympy2_extended math_verify pytest

RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN uv pip install flash_attn --no-build-isolation

RUN git clone -b v0.4.9.2 https://github.com/EleutherAI/lm-evaluation-harness.git && \
    cd lm-evaluation-harness && \
    uv pip install -e . 