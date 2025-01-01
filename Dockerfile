FROM nvcr.io/nvidia/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /root
USER root

ENV PATH /usr/local/cuda/bin:$PATH
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64:$LD_LIBRARY_PATH

# 서버 관련 유틸
RUN apt-get update && \
    apt-get install -y --no-install-recommends apt-utils && \
    apt-get install -y ffmpeg wget net-tools build-essential git curl vim nmon tmux lsof && \
    apt-get install -y python3.10 python3.10-dev python3.10-venv python3-pip

RUN ln -s /usr/bin/python3.10 /usr/bin/python

# 파이썬 관련 유틸
RUN pip install -U pip wheel setuptools && \
    pip install git+https://github.com/huggingface/transformers.git@187439c3fa139b2102a874483e9f8f0cfa8e5557 && \
    pip install accelerate==1.1.1 datasets==3.1.0 evaluate==0.4.3 trl==0.12.1 peft==0.13.2 deepspeed==0.15.4 && \
    pip install bitsandbytes scipy sentencepiece pillow liger-kernel && \
    pip install ruff natsort setproctitle glances[gpu] wandb comet-ml cmake

RUN pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121 && \
    pip install flash-attn==2.7.0.post2
