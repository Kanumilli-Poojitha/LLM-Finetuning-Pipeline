# Use an official NVIDIA PyTorch base image.
# This ships with a matched CUDA runtime + cuDNN, which is the single
# biggest source of "works on my machine" pain with bitsandbytes/torch.
# Pinned to 26.03-py3, which ships torch 2.11.0 — the same torch version
# Google Colab's GPU runtime currently provides, so requirements.txt only
# has to be compatible with one torch version across both workflows.
FROM nvcr.io/nvidia/pytorch:26.03-py3

# Avoid interactive prompts during apt installs
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/workspace/.cache/huggingface \
    TRANSFORMERS_CACHE=/workspace/.cache/huggingface

WORKDIR /workspace

# System deps (git needed for some HF hub operations / private repos)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (better layer caching: only rebuilds
# this layer when requirements.txt actually changes).
# requirements.txt deliberately does NOT pin torch — this base image
# already ships a torch build matched to its CUDA/cuDNN version, and
# re-pinning it here would silently swap in a mismatched generic wheel.
COPY requirements.txt /workspace/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /workspace/requirements.txt && \
    python -c "import torch, transformers, peft, trl, bitsandbytes, accelerate; from trl import SFTTrainer; print('Import check OK — torch', torch.__version__)"
# Note: GPU/CUDA availability can't be checked here — Docker builds have no
# GPU passthrough. It's verified at container RUN time instead (see README).

# Copy the rest of the project. In practice docker-compose bind-mounts
# the working directory over this, so this COPY mainly matters for
# building a standalone/production image.
COPY . /workspace

CMD ["bash"]
