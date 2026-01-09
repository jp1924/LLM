FROM nvcr.io/nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH="/usr/local/cuda/bin:$PATH"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:$LD_LIBRARY_PATH"
ENV UV_SYSTEM_PYTHON="1"

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux lsof && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

RUN ln -s /usr/bin/python3.10 /usr/bin/python
COPY .tmux.conf /root/.tmux.conf

RUN echo 'alias tmux="tmux -f ~/workspace/.tmux.conf"' >> /root/.bashrc
RUN echo 'PROMPT_COMMAND='history -a'' >> /root/.bashrc

# install uv pip
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh
ENV PATH="/root/.local/bin/:$PATH"

RUN uv pip install -U pip wheel setuptools
RUN uv pip install transformers accelerate datasets liger-kernel trl peft deepspeed lomo-optim transformer_engine[pytorch] evaluate \
    bitsandbytes scipy sentencepiece pillow fastapi uvicorn ruff natsort setproctitle glances[gpu] wandb cmake latex2sympy2_extended math_verify

RUN uv pip install vllm
RUN uv pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

RUN uv pip install flash_attn