# clef-pod — Cloudflare Clef on a RunPod pod (Jev/SystemOne API)

Self-hosted [Clef](https://blog.cloudflare.com/clef-decision-models/) decision model. Speaks the
same `POST /v1/systemone` request/response as Jev (TypeSafe), so a Jev request body can be sent
here unchanged.

| File | What it does |
|---|---|
| `Dockerfile` | python 3.12 + torch 2.11 (CUDA 12.8) + transformers 5.10.2 (the versions on the model card) |
| `server.py` | FastAPI wrapper around Cloudflare's `joint_schema_model.py` |
| `start.sh` | sshd + supervised uvicorn on :8000 |
| `pod.yaml` | pod config for `ti/create_pod.py` (L40S / A40 first, A100 fallback) |
| `smoke_test.py` | waits for `/health`, sends synthetic SystemOne requests |
| `../.github/workflows/build-clef-pod.yml` | builds → `ghcr.io/cst2978/clef-pod:latest` |

Weights come from pinned Hugging Face revisions (set by `CLEF_VARIANT`):

| Variant | Repo @ revision | Size | GPU |
|---|---|---|---|
| `flash` (default) | `Cloudflare/clef-flash` @ `17f0b0a` | 9B, ~18 GB bf16 | 24–48 GB |
| `full` | `Cloudflare/clef` @ `2f3de3d` | 27B, ~54 GB bf16 | 80 GB (set `container_disk_gb: 100`) |

## Deploy

```bash
python ti/create_pod.py --config clef-pod/pod.yaml --no-volume
POD_ID=<id> python clef-pod/smoke_test.py     # first start downloads the weights
python ti/create_pod.py --stop <id>           # pause GPU billing, keep the disk
```

## API

```bash
curl -s https://<pod>-8000.proxy.runpod.net/v1/systemone -H "Content-Type: application/json" -d '{
  "model": "clef", "state": "…text…",
  "questions": {"threatened_cancel": {"type": "noul", "instructions": "The dealer says they will cancel."}}}'
```

- Response: `{"model", "answers": {id: {...}}, "usage": {"input_tokens", "output_tokens": 0,
  "truncated", "state_tokens_dropped", "seconds"}}`. `noul` answers carry the probability in `noul`;
  `choice` has `choice`, `confidence`, `probabilities`; `score` has `score`, `confidence`, `legend`,
  `probabilities`.
- `POST /v1/systemone/batch` with `{"requests": [...]}` runs many requests in length-sorted GPU
  batches (up to `MAX_BATCH_TOKENS` padded tokens per forward); results are in request order and a
  bad request only fails its own slot.
- Clef silently cuts the state at `MAX_LENGTH` (16,384) tokens; this server reports it in
  `usage.truncated` / `usage.state_tokens_dropped`.
- `joint_schema_model.py` is Cloudflare's code loaded from the pinned revision (reviewed: imports only
  json/math/dataclasses/pathlib/torch, no network or subprocess calls).
