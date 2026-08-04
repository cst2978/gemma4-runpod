#!/bin/bash
# Pod entrypoint: set up SSH, then supervise an OpenAI-compatible SGLang server.
# Overridable via env: MODEL_PATH, SERVED_MODEL_NAME, PORT, API_KEY,
# ENABLE_GEMMA4_PARSERS, SGLANG_EXTRA_ARGS.
set -e

# --- SSH (RunPod injects the account public key as $PUBLIC_KEY) ---
if [ -n "$PUBLIC_KEY" ]; then
    mkdir -p /root/.ssh && chmod 700 /root/.ssh
    echo "$PUBLIC_KEY" >> /root/.ssh/authorized_keys && chmod 600 /root/.ssh/authorized_keys
    mkdir -p /run/sshd && /usr/sbin/sshd
    echo "[start.sh] sshd started"
fi

MODEL_PATH="${MODEL_PATH:-google/gemma-4-26B-A4B-it}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-gemma4}"
PORT="${PORT:-8000}"

# Bearer-token auth for the public :8000 proxy. Set API_KEY to secure it.
API_KEY_ARGS=""
if [ -n "$API_KEY" ]; then
    API_KEY_ARGS="--api-key $API_KEY"
    echo "[start.sh] API key auth: ENABLED"
else
    echo "[start.sh] API key auth: DISABLED — set API_KEY env to secure it"
fi

# The gemma4 reasoning/tool-call parsers are the SUSPECTED cause of the <pad>-token
# failure on the old build, so they are OFF by default now. Set ENABLE_GEMMA4_PARSERS=1
# to re-enable them once confirmed working.
PARSER_ARGS=""
if [ "$ENABLE_GEMMA4_PARSERS" = "1" ]; then
    PARSER_ARGS="--reasoning-parser gemma4 --tool-call-parser gemma4"
    echo "[start.sh] gemma4 parsers: ENABLED"
fi

mkdir -p "${HF_HOME:-/workspace/hf}"

# Supervise (don't exec): a server crash shouldn't kill the container / sshd.
set +e
while true; do
    echo "[start.sh] launching SGLang: model=$MODEL_PATH served-as=$SERVED_MODEL_NAME port=$PORT"
    python3 -m sglang.launch_server \
        --model-path "$MODEL_PATH" \
        --served-model-name "$SERVED_MODEL_NAME" \
        $PARSER_ARGS \
        --host 0.0.0.0 --port "$PORT" \
        $API_KEY_ARGS \
        $SGLANG_EXTRA_ARGS
    rc=$?
    echo "[start.sh] SGLang exited with code $rc — retrying in 60s (SSH in to debug; stop the pod to end)"
    sleep 60
done
