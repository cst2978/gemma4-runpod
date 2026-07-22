"""RunPod serverless handler for google/gemma-4-26B-A4B-it served with vLLM.

RunPod calls `handler(job)` for each request. We keep a single vLLM
AsyncLLMEngine alive across invocations (loaded once at cold start), and stream
generated tokens back to RunPod, which relays them to the caller.

Request format (job["input"]):
    {
        "prompt": "Explain MoE models in one paragraph.",   # OR use "messages"
        "messages": [{"role": "user", "content": "Hi"}],     # chat-style (preferred)
        "sampling_params": {"temperature": 0.7, "max_tokens": 512, "top_p": 0.95}
    }

Config comes from environment variables (set on the RunPod endpoint):
    MODEL_NAME              default: google/gemma-4-26B-A4B-it
    MAX_MODEL_LEN           default: 32768
    DTYPE                   default: bfloat16
    GPU_MEMORY_UTILIZATION  default: 0.95
    TENSOR_PARALLEL_SIZE    default: 1   (Gemma 4 currently supports TP=1 only)
    HUGGING_FACE_HUB_TOKEN  optional (only if the model becomes gated)
"""

import os

import runpod
from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
from vllm.utils import random_uuid

MODEL_NAME = os.getenv("MODEL_NAME", "google/gemma-4-26B-A4B-it")

# Build the engine once, at import time (cold start). Subsequent warm requests
# reuse it with zero reload cost.
engine_args = AsyncEngineArgs(
    model=MODEL_NAME,
    dtype=os.getenv("DTYPE", "bfloat16"),
    max_model_len=int(os.getenv("MAX_MODEL_LEN", "32768")),
    gpu_memory_utilization=float(os.getenv("GPU_MEMORY_UTILIZATION", "0.95")),
    tensor_parallel_size=int(os.getenv("TENSOR_PARALLEL_SIZE", "1")),
    trust_remote_code=True,
    enforce_eager=False,  # CUDA graphs on -> faster decode
)
engine = AsyncLLMEngine.from_engine_args(engine_args)

# Cache the tokenizer's chat template so we can accept `messages` input.
_tokenizer = None


async def _get_tokenizer():
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = await engine.get_tokenizer()
    return _tokenizer


async def _build_prompt(job_input):
    """Turn either `prompt` or chat `messages` into a single prompt string."""
    if job_input.get("prompt"):
        return job_input["prompt"]

    messages = job_input.get("messages")
    if not messages:
        raise ValueError("Request must include either 'prompt' or 'messages'.")

    tokenizer = await _get_tokenizer()
    return tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )


async def handler(job):
    """Async generator -> RunPod streams each yielded chunk to the client."""
    job_input = job["input"]
    prompt = await _build_prompt(job_input)

    sp = job_input.get("sampling_params", {})
    sampling_params = SamplingParams(
        temperature=sp.get("temperature", 0.7),
        top_p=sp.get("top_p", 0.95),
        max_tokens=sp.get("max_tokens", 512),
        stop=sp.get("stop"),
    )

    request_id = random_uuid()
    previous_text = ""
    async for output in engine.generate(prompt, sampling_params, request_id):
        text = output.outputs[0].text
        # vLLM yields the full text so far; emit only the new delta.
        delta = text[len(previous_text):]
        previous_text = text
        if delta:
            yield delta


# `return_aggregate_stream=True` also exposes the concatenated result to callers
# that use /runsync instead of the streaming /run endpoint.
runpod.serverless.start(
    {"handler": handler, "return_aggregate_stream": True}
)
