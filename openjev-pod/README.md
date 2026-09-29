# openjev-pod — SemIf (open-Jev) on a RunPod pod

HTTP wrapper around [SemIf-OpenJev](https://github.com/TheoLeeCJ/SemIf-OpenJev): a frozen
`Qwen/Qwen3.5-4B` that returns one probability per declared option in a single forward pass
(no generation, no parsing).

| File | What it does |
|---|---|
| `Dockerfile` | python 3.12 + torch 2.10.0 (CUDA 12.8) + SemIf pinned to commit `23cf1f3` |
| `server.py` | FastAPI: `GET /health`, `POST /score`; loads the model once at startup |
| `start.sh` | sshd + supervised uvicorn on :8000 |
| `pod.yaml` | pod config for `ti/create_pod.py` |
| `smoke_test.py` | waits for `/health`, scores 4 sample decisions |
| `../.github/workflows/build-openjev-pod.yml` | builds → `ghcr.io/cst2978/openjev-pod` |

## Deploy

```bash
# 1. build the image: push a change under openjev-pod/ (or run the workflow manually)
# 2. create the A100 pod (uses runpod_key from .env)
python ti/create_pod.py --config openjev-pod/pod.yaml       # add --no-volume if no A100 in US-NE-1
# 3. first start downloads 8.1 GB of weights to /workspace/hf; later starts ~40 s load
POD_ID=<id> python openjev-pod/smoke_test.py
python ti/create_pod.py --terminate <id>                    # stop billing
```

## API

```bash
curl -s https://<pod>-8000.proxy.runpod.net/score \
  -H "Content-Type: application/json" \
  -d '{"rows": [{"id": "q1", "state": "…text…", "question": "Does the dealer dispute a charge?",
                 "options": [{"id": "yes", "description": "Yes."}, {"id": "no", "description": "No."}]}]}'
```

Response: `{"results": [{"id", "option_ids", "probabilities", "option_logits", "input_tokens",
"prompt_sha256", ...}], "model": {...}, "seconds": 1.6, "errors": 0}`. Results are in request order;
options in the order sent.

- **Send all questions for the same text in one request.** Rows with a byte-identical `state`
  share one prefill (chunks of `SHARED_BATCH=32`) — that is the whole speed story.
  State length dominates cost: budget by characters, not by decision count.
- `state` may be a string, JSON object or array. 2–16 options per row. Row ids unique per request.
- A row over `max_tokens` (default 16384) is **rejected, not truncated** — it comes back with an
  `error` field; the rest of the request still scores.
- Probabilities are **uncalibrated** and lean high vs other models; re-derive thresholds on your
  own labelled data. Explicit exclusions in the question ("X is NOT this") move scores a lot.

## Env (set in `pod.yaml`)

`API_KEY` (optional bearer token; unset = open, the default here), `MAX_TOKENS`, `SHARED_BATCH`, `MAX_ROWS_PER_REQUEST` (512),
`MODEL_ID` / `MODEL_REVISION` (must be a 40-char SHA for remote models).

## Notes on the spec vs the code (SemIf `23cf1f3`)

- `score_shared` returns `(results, timing)`, not a plain list.
- Over-length prompts raise `ValueError` ("no truncation allowed"); they don't truncate silently.
- One bad row fails a whole `score_shared` call, so the server retries that chunk row-by-row
  with `direct.score` to isolate it.
