# Gemma 4 (26B-A4B) on RunPod Serverless

Deploy `google/gemma-4-26B-A4B-it` as an autoscaling, scale-to-zero inference
endpoint on RunPod, served with vLLM.

| Setting | Value |
|---|---|
| Model | `google/gemma-4-26B-A4B-it` — MoE, 25.2B total / 3.8B active, 256K max ctx |
| Precision | BF16 |
| GPU | 1× **H100 80GB**, tensor-parallel = 1 (Gemma 4 is TP=1 only for now) |
| Context served | `MAX_MODEL_LEN=32768` (full 256K would blow up KV cache VRAM) |
| Scaling | Scale-to-zero: 0 active / 3 max workers, request-count strategy, FlashBoot on |
| Weights storage | Network volume (~100GB), so weights download once |

## Files

- `Dockerfile` — serverless worker on a **Gemma-4-capable vLLM** base image.
- `handler.py` — RunPod handler wrapping vLLM's `AsyncLLMEngine` (streaming).
- `.env.example` — the env vars to set on the endpoint; copy to `.env` for local tests.
- `test_request.py` — send a test prompt to the live endpoint.

---

## Step 1 — Build & push the image

> ⚠️ **The vLLM version is the make-or-break detail.** The stock RunPod
> `worker-vllm` image ships an old vLLM that cannot load Gemma 4. The Dockerfile
> bases on `vllm/vllm-openai:gemma4`. Before deploying, confirm that tag exists
> (`docker pull vllm/vllm-openai:gemma4`); if not, use `vllm/vllm-openai:latest`
> and verify it loads the model.

Build on a Linux/GPU host or any machine with Docker (the image is large, multi-GB):

```bash
docker build -t <your-dockerhub-user>/gemma4-runpod:latest .
docker push <your-dockerhub-user>/gemma4-runpod:latest
```

## Step 2 — Create a network volume

RunPod console → **Storage → Network Volumes → New**:
- Size: **100 GB** (weights ~50GB + headroom)
- Region: pick one that has **H100 80GB** availability
- Cost: ~$0.07/GB/mo ≈ **$7/mo** for 100GB

The volume mounts at `/runpod-volume` inside the worker; `HF_HOME` points there,
so weights persist across cold starts instead of re-downloading each time.

## Step 3 — Create the serverless endpoint

RunPod console → **Serverless → New Endpoint**:

1. **Container image:** `<your-dockerhub-user>/gemma4-runpod:latest`
2. **GPU:** H100 80GB
3. **Network volume:** attach the one from Step 2
4. **Workers:** Active (min) = **0**, Max = **3**
5. **Scaling strategy:** Request count (best for LLMs)
6. **Idle timeout:** 5s (raise to keep workers warm longer between requests)
7. **FlashBoot:** enabled
8. **Environment variables** (from `.env.example`):
   ```
   MODEL_NAME=google/gemma-4-26B-A4B-it
   MAX_MODEL_LEN=32768
   DTYPE=bfloat16
   GPU_MEMORY_UTILIZATION=0.95
   TENSOR_PARALLEL_SIZE=1
   ```
   Add `HUGGING_FACE_HUB_TOKEN` only if the model is gated.

Create the endpoint, wait for it to initialize, and copy the **Endpoint ID**.

## Step 4 — Test it

```bash
cp .env.example .env          # fill in RUNPOD_API_KEY + RUNPOD_ENDPOINT_ID
pip install requests python-dotenv
python test_request.py "Explain mixture-of-experts models simply."
```

First call triggers a cold start (model load) — up to a minute or two the very
first time. Later calls are fast; when idle past the timeout, workers scale to 0
and you stop paying.

---

## Cost summary (H100 80GB @ $4.55/hr, billed per second)

- **Idle:** ~$0 (scale-to-zero)
- **Per request:** ~$0.009 (≈2s compute + ~5s idle tail)
- **Network volume:** ~$7/mo (100GB)
- Flip **1 active worker** on later (~$3,280/mo) only once you need zero cold starts.

## Notes / gotchas

- **TP=1 only:** don't set `TENSOR_PARALLEL_SIZE > 1` for this model yet.
- **Context:** 32K is a safe default. Raising `MAX_MODEL_LEN` costs KV-cache VRAM.
- **Retraining/fine-tuning** (your later goal) is a *Pod*, not serverless — a
  persistent GPU box. Different setup; ask when you're ready.
