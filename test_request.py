"""Send a test request to the deployed RunPod Gemma 4 serverless endpoint.

Usage:
    pip install requests python-dotenv
    python test_request.py "Explain mixture-of-experts models simply."

Reads RUNPOD_API_KEY and RUNPOD_ENDPOINT_ID from the .env file.
"""

import os
import sys

import requests

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass  # dotenv optional; fall back to real environment vars

API_KEY = os.environ["RUNPOD_API_KEY"]
ENDPOINT_ID = os.environ["RUNPOD_ENDPOINT_ID"]

prompt = sys.argv[1] if len(sys.argv) > 1 else "Say hello and name your model."

# /runsync blocks until the job finishes and returns the full result — simplest
# for testing. Use /run for async + /stream for token streaming in production.
url = f"https://api.runpod.ai/v2/{ENDPOINT_ID}/runsync"

payload = {
    "input": {
        "messages": [{"role": "user", "content": prompt}],
        "sampling_params": {"temperature": 0.7, "max_tokens": 512},
    }
}

print(f"→ Sending to {ENDPOINT_ID} (cold start may take a minute)...\n")
resp = requests.post(
    url,
    json=payload,
    headers={"Authorization": f"Bearer {API_KEY}"},
    timeout=300,
)
resp.raise_for_status()
data = resp.json()

print("Status:", data.get("status"))
# Aggregated stream output lands under "output"; shape depends on the worker.
print("Output:\n", data.get("output"))
