# RunPod POD image serving SemIf (open-Jev) decision scoring over HTTP.
#
# SemIf ships as a Python library + CLI, no server; server.py wraps it (FastAPI on
# :8000) and start.sh runs sshd + a supervised uvicorn. The model (Qwen3.5-4B,
# ~8.1 GB bf16) is NOT baked in — it downloads to HF_HOME on the network volume
# on first start and is reused after that.
#
# We own this image, so instead of the --no-deps workaround (for images that
# already ship torch) we install SemIf with its own exact pins, torch==2.10.0 included.
FROM python:3.12-slim

# SemIf commit to install. The default branch is `master` (`main` 404s); pinned
# to a SHA so rebuilds are reproducible. Bump deliberately.
ARG SEMIF_REF=23cf1f39fc9534fe81437200959b6dfc7106e45a

# sshd for interactive access; git for the pip git+ install; gcc because Triton
# (used by flash-linear-attention) compiles small launcher stubs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd

# torch first, from the CUDA 12.8 index, so SemIf's torch==2.10.0 pin is already
# satisfied by a CUDA build (the wheel bundles the CUDA runtime; host needs only the driver).
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# SemIf with its pinned deps (transformers==5.17.0 is required for native
# Qwen3_5ForCausalLM); flash-linear-attention is required by Qwen3.5 itself.
RUN pip install --no-cache-dir \
        "semif-phase1 @ git+https://github.com/TheoLeeCJ/SemIf-OpenJev@${SEMIF_REF}" \
        flash-linear-attention==0.5.2 \
        fastapi==0.123.10 "uvicorn==0.38.0"

# Fail the build (not the pod) if transformers can't load Qwen3.5 natively.
RUN python -c "import transformers, semif_phase1.shared; \
assert hasattr(transformers, 'Qwen3_5ForCausalLM'), 'transformers lacks Qwen3_5ForCausalLM'"

# Weights cache on the network volume when attached (/workspace), else ephemeral.
ENV HF_HOME=/workspace/hf \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY server.py start.sh /app/
RUN chmod +x /app/start.sh

EXPOSE 8000 22
CMD ["/app/start.sh"]
