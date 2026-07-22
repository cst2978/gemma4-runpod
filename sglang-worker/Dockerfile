# Custom RunPod SGLang serverless worker rebased on a Gemma-4-capable SGLang.
#
# The stock runpod/worker-sglang pins SGLang v0.5.2 (base lmsysorg/sglang:v0.5.2),
# which predates Gemma 4 -> the worker crash-loops on this model. We instead start
# from a recent lmsysorg/sglang NIGHTLY (SGLang main, Gemma 4 supported) and layer
# RunPod's worker handler on top. The handler just shells out to
# `python -m sglang.launch_server`, so it's SGLang-version-agnostic.
#
# Base pinned to a dated nightly for reproducibility (2026-07-21, CUDA 12).
FROM lmsysorg/sglang:nightly-dev-cu12-20260721-8905cbd4

# RunPod worker handler dependencies (from worker-sglang/requirements.txt).
RUN pip install --no-cache-dir \
    ray pandas pyarrow "runpod>=1.8,<2.0" pydantic-settings hf_transfer

# Pull the RunPod SGLang worker handler source onto the newer SGLang base.
RUN git clone --depth 1 https://github.com/runpod-workers/worker-sglang.git /sgl-worker
WORKDIR /sgl-worker

# Weights cache on the RunPod volume (or ephemeral if no volume attached).
ENV BASE_PATH=/runpod-volume \
    HF_HOME=/runpod-volume/huggingface-cache/hub \
    HUGGINGFACE_HUB_CACHE=/runpod-volume/huggingface-cache/hub \
    HF_HUB_ENABLE_HF_TRANSFER=1

CMD ["python3", "handler.py"]
