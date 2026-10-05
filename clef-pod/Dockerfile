# RunPod POD image serving Cloudflare Clef / Clef-flash over the Jev/SystemOne API.
#
# server.py wraps Cloudflare's own joint_schema_model.py (downloaded with the weights from
# the pinned Hugging Face revision) behind FastAPI on :8000; start.sh runs sshd + a
# supervised uvicorn. Weights are NOT baked in: Clef-flash (~18 GB) or Clef (~54 GB)
# downloads to HF_HOME on first start.
FROM python:3.12-slim

# sshd for interactive access; gcc because Triton (flash-linear-attention, used by the
# Qwen3.5 backbone's linear-attention layers) compiles small launcher stubs at runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
        openssh-server gcc libc6-dev \
    && rm -rf /var/lib/apt/lists/* && mkdir -p /run/sshd

# Versions from the Clef model card: "Tested with torch 2.11 and transformers 5.10.2".
# torchvision too: Clef's processor includes Qwen3VLVideoProcessor, which imports it even
# for text-only requests (the first pod failed to load without it).
RUN pip install --no-cache-dir torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir transformers==5.10.2 accelerate safetensors huggingface-hub \
        tokenizers pillow flash-linear-attention==0.5.2 \
        fastapi==0.123.10 uvicorn==0.38.0

# Fail the build (not the pod) if transformers lacks the Qwen3.5 backbone class Clef loads.
RUN python -c "import transformers, torchvision; \
assert hasattr(transformers, 'Qwen3_5ForConditionalGeneration'), 'transformers lacks Qwen3_5ForConditionalGeneration'"

ENV HF_HOME=/workspace/hf \
    PYTHONUNBUFFERED=1

WORKDIR /app
COPY server.py start.sh /app/
RUN chmod +x /app/start.sh

EXPOSE 8000 22
CMD ["/app/start.sh"]
