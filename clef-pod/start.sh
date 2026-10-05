#!/bin/bash
# Pod entrypoint: set up SSH, then supervise the Clef SystemOne server (server.py).
# Overridable via env: PORT, API_KEY, CLEF_VARIANT, MAX_LENGTH, MAX_BATCH_TOKENS,
# MAX_REQUESTS_PER_BATCH.
set -e

# --- SSH (RunPod injects the account public key as $PUBLIC_KEY) ---
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
    mkdir -p /run/sshd && /usr/sbin/sshd
    echo "[start.sh] sshd started"
fi

PORT="${PORT:-8000}"

if [ -n "$API_KEY" ]; then
    echo "[start.sh] API key auth: ENABLED"
else
    echo "[start.sh] API key auth: DISABLED — set API_KEY env to secure it"
fi

mkdir -p "${HF_HOME:-/workspace/hf}"

# Supervise (don't exec): a server crash shouldn't kill the container / sshd.
# One worker only — the model lives in that process's GPU memory.
set +e
cd /app
while true; do
    echo "[start.sh] launching Clef server (variant=${CLEF_VARIANT:-flash}) on :$PORT"
    python3 -m uvicorn server:app --host 0.0.0.0 --port "$PORT" --workers 1
    rc=$?
    echo "[start.sh] server exited with code $rc — retrying in 30s (SSH in to debug; stop the pod to end)"
    sleep 30
done
