# Serverless vLLM worker for google/gemma-4-26B-A4B-it on RunPod.
#
# We base on vLLM's OFFICIAL Gemma-4-capable image (`vllm/vllm-openai:gemma4`).
# This is the critical choice: the stock RunPod `worker-vllm` image ships an
# older vLLM that CANNOT load the Gemma 4 MoE architecture. The `:gemma4` tag is
# the build the vLLM team blessed as working for this model family.
#
# NOTE: `:latest` does NOT include Gemma 4 support — you must use a gemma4 tag.
# Pinned to a specific, reproducible build for H100 (x86_64 + CUDA 13.0) so the
# image doesn't drift between builds. The moving `:gemma4` tag also works.
FROM vllm/vllm-openai:gemma4-0505-x86_64-cu130

# The base image's ENTRYPOINT launches the OpenAI server directly. We clear it so
# our RunPod serverless handler is the process RunPod invokes instead.
ENTRYPOINT []

# RunPod serverless SDK (provides the job protocol the platform calls into).
RUN pip install --no-cache-dir runpod

# Where vLLM looks for cached weights. Mounted from a RunPod network volume at
# runtime (see README) so the ~50GB of weights are downloaded once, not per boot.
ENV HF_HOME=/runpod-volume/hf_cache

COPY handler.py /handler.py

CMD ["python", "-u", "/handler.py"]
