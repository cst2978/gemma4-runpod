# RunPod POD image serving SemIf (open-Jev) decision scoring over HTTP.
#
# SemIf ships as a Python library + CLI, no server; server.py wraps it (FastAPI on
# :8000) and start.sh runs sshd + a supervised uvicorn. The model (Qwen3.5-4B,
# ~8.1 GB bf16) is NOT baked in — it downloads to HF_HOME on the network volume
# on first start and is reused after that.
#
# Install steps follow semif_openjev_deploy_spec.md ("Install").
FROM python:3.12-slim

# sshd for interactive access; git for the pip git+ install; gcc because Triton
# (used by flash-linear-attention) compiles small launcher stubs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server git gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd

# The spec's --no-deps install assumes the image already ships torch. This base
# has none, so first take SemIf's pinned torch==2.10.0 (CUDA 12.8 build; the
# wheel bundles the CUDA runtime, the host only needs the driver).
RUN pip install --no-cache-dir torch==2.10.0 --index-url https://download.pytorch.org/whl/cu128

# --- spec "Install", verbatim ---
# transformers==5.17.0 is not optional (native Qwen3_5ForCausalLM);
# flash-linear-attention==0.5.2 is required by Qwen3.5 itself.
RUN pip install -q transformers==5.17.0 accelerate safetensors huggingface-hub \
                tokenizers sentencepiece protobuf flash-linear-attention==0.5.2
# Branch is master, not main (main 404s).
RUN pip install -q --no-deps git+https://github.com/TheoLeeCJ/SemIf-OpenJev@master

# HTTP server for server.py.
RUN pip install --no-cache-dir fastapi==0.123.10 uvicorn==0.38.0

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
