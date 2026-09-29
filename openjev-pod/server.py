"""HTTP server for SemIf (open-Jev) decision scoring on a RunPod pod.

Loads the frozen Qwen3.5-4B once at startup (in a background thread, so /health
answers while weights load), then scores decision rows:

    POST /score   {"rows": [{"id", "state", "question", "options": [{"id", "description"}]}],
                   "max_tokens": 16384}          # optional per-request override
    GET  /health  503 while loading, 200 when ready, 500 if the load failed

Rows are grouped by exact `state` and each group is scored with `score_shared`
in chunks of <= SHARED_BATCH (one state prefill, questions branch off its KV
cache). Results come back in request order. A row that can't be scored (e.g.
over max_tokens) gets an `error` field instead of failing the whole request.

Env: MODEL_ID, MODEL_REVISION, MAX_TOKENS, SHARED_BATCH, MAX_ROWS_PER_REQUEST, API_KEY.
"""
import hmac
import json
import logging
import os
import threading
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-4B")
# load_causal_model refuses a remote model without a pinned 40-char commit SHA.
MODEL_REVISION = os.environ.get("MODEL_REVISION", "851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))   # upstream default 4096 is too small
SHARED_BATCH = int(os.environ.get("SHARED_BATCH", "32"))  # state cache is replicated per row
MAX_ROWS = int(os.environ.get("MAX_ROWS_PER_REQUEST", "512"))
API_KEY = os.environ.get("API_KEY", "")

log = logging.getLogger("openjev")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_model: dict = {}          # model, tokenizer, metadata, load_seconds once loaded
_load_error: str | None = None
_gpu = threading.Lock()    # one model on one GPU: score one request at a time


def _load():
    global _load_error
    try:
        from semif_phase1.core import load_causal_model

        t0 = time.time()
        log.info("loading %s @ %s", MODEL_ID, MODEL_REVISION)
        model, tokenizer, metadata = load_causal_model(
            MODEL_ID, MODEL_REVISION, device="cuda", dtype="bfloat16")
        _model.update(model=model, tokenizer=tokenizer, metadata=metadata,
                      load_seconds=round(time.time() - t0, 1))
        log.info("model ready in %.1fs", _model["load_seconds"])
    except Exception as e:  # noqa: BLE001 - surfaced on /health instead of killing the server
        _load_error = f"{type(e).__name__}: {e}"
        log.exception("model load failed")


app = FastAPI(title="SemIf open-Jev")


@app.on_event("startup")
def _startup():
    if not API_KEY:
        log.warning("API_KEY is not set — /score is open to anyone with the proxy URL")
    threading.Thread(target=_load, daemon=True).start()


def _auth(authorization: str = Header(default="")):
    if API_KEY and not hmac.compare_digest(authorization, f"Bearer {API_KEY}"):
        raise HTTPException(401, "invalid or missing bearer token")


class Option(BaseModel):
    id: str
    description: str


class Row(BaseModel):
    id: str = Field(min_length=1)
    state: str | dict | list
    question: str = Field(min_length=1)
    options: list[Option] = Field(min_length=2, max_length=16)


class ScoreRequest(BaseModel):
    rows: list[Row] = Field(min_length=1)
    max_tokens: int | None = Field(default=None, ge=256, le=65536)


@app.get("/health")
def health():
    if _load_error:
        return JSONResponse({"status": "error", "error": _load_error}, status_code=500)
    if not _model:
        return JSONResponse({"status": "loading"}, status_code=503)
    return {"status": "ok", "model": MODEL_ID, "revision": MODEL_REVISION,
            "load_seconds": _model["load_seconds"]}


def _groups(rows):
    """Group rows by exact state. Keyed on the serialized state (key order kept),
    because score_shared rebuilds the prefix from the first row's serialization."""
    groups: dict[str, list] = {}
    for row in rows:
        groups.setdefault(json.dumps(row["state"], ensure_ascii=False), []).append(row)
    return list(groups.values())


def _score_rows(rows, max_tokens):
    import torch
    from semif_phase1.direct import score as score_direct
    from semif_phase1.shared import score_shared

    m = _model
    out = {}
    for group in _groups(rows):
        for i in range(0, len(group), SHARED_BATCH):
            chunk = group[i:i + SHARED_BATCH]
            if len(chunk) > 1:
                try:
                    results, _timing = score_shared(
                        m["model"], m["tokenizer"], chunk, m["metadata"], max_tokens=max_tokens)
                    out.update((r["id"], r) for r in results)
                    continue
                except (ValueError, torch.OutOfMemoryError) as e:
                    # One bad row (or OOM) fails the whole shared call; score the
                    # rows one by one so only the offending row reports an error.
                    log.warning("shared call failed (%s); scoring %d rows individually", e, len(chunk))
                    torch.cuda.empty_cache()
            for row in chunk:
                try:
                    out[row["id"]] = score_direct(
                        m["model"], m["tokenizer"], row, m["metadata"], max_tokens=max_tokens)
                except (ValueError, torch.OutOfMemoryError) as e:
                    torch.cuda.empty_cache()
                    out[row["id"]] = {"id": row["id"], "error": f"{type(e).__name__}: {e}"}
    return out


@app.post("/score", dependencies=[Depends(_auth)])
def score(req: ScoreRequest):
    if _load_error:
        raise HTTPException(500, f"model failed to load: {_load_error}")
    if not _model:
        raise HTTPException(503, "model is still loading — poll /health")
    if len(req.rows) > MAX_ROWS:
        raise HTTPException(413, f"{len(req.rows)} rows > MAX_ROWS_PER_REQUEST={MAX_ROWS}")

    from semif_phase1.core import validate_row

    rows = [r.model_dump() for r in req.rows]
    if len({r["id"] for r in rows}) != len(rows):
        raise HTTPException(400, "row ids must be unique within a request")
    for r in rows:
        try:
            validate_row(r)
        except ValueError as e:
            raise HTTPException(400, f"row {r['id']}: {e}") from e

    t0 = time.time()
    with _gpu:
        scored = _score_rows(rows, req.max_tokens or MAX_TOKENS)
    results = []
    for r in rows:
        res = dict(scored[r["id"]])
        res.pop("model", None)  # identical per row; returned once below
        results.append(res)
    return {"results": results,
            "model": _model["metadata"],
            "seconds": round(time.time() - t0, 3),
            "errors": sum("error" in r for r in results)}
