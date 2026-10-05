"""HTTP server for Cloudflare Clef / Clef-flash on a RunPod pod, speaking the Jev/SystemOne API.

    POST /v1/systemone         one Jev/SystemOne request body -> SystemOne response body
    POST /v1/systemone/batch   {"requests": [body, ...]} -> {"responses": [...]}, batched on the GPU
    GET  /health               503 while loading, 200 when ready, 500 if the load failed

Request body (same as Jev's client.system_one):
    {"model": "clef-flash", "state": "...", "questions": {"q": {"type": "noul|choice|score",
     "instructions": "...", "criteria": {...} | [...]}}}

Clef's encode_record silently cuts the state at MAX_LENGTH tokens; this server reports that as
usage.truncated and usage.state_tokens_dropped instead of hiding it.

Env: CLEF_VARIANT (flash|full), MAX_LENGTH, MAX_BATCH_TOKENS, MAX_REQUESTS_PER_BATCH, API_KEY.
"""
import hmac
import logging
import os
import sys
import threading
import time

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

# Pinned Hugging Face revisions, so a re-upload can't change scores under the baseline.
VARIANTS = {
    "flash": ("Cloudflare/clef-flash", "17f0b0ad64efb65d273590632833508766b2aae6"),
    "full": ("Cloudflare/clef", "2f3de3dd85f379784083b0814d997ab627200f0c"),
}
VARIANT = os.environ.get("CLEF_VARIANT", "flash")
REPO, REVISION = VARIANTS[VARIANT]
MAX_LENGTH = int(os.environ.get("MAX_LENGTH", "16384"))             # Clef's own default
MAX_BATCH_TOKENS = int(os.environ.get("MAX_BATCH_TOKENS", "32768"))  # padded tokens per forward
MAX_REQUESTS = int(os.environ.get("MAX_REQUESTS_PER_BATCH", "256"))
API_KEY = os.environ.get("API_KEY", "")

log = logging.getLogger("clef")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

_m: dict = {}              # model, processor, jsm (joint_schema_model module), load_seconds
_load_error: str | None = None
_gpu = threading.Lock()    # one model on one GPU: one forward at a time


def _load():
    global _load_error
    try:
        from huggingface_hub import snapshot_download

        t0 = time.time()
        log.info("downloading %s @ %s", REPO, REVISION)
        path = snapshot_download(REPO, revision=REVISION)
        sys.path.insert(0, path)
        import joint_schema_model as jsm  # Cloudflare's inference code from the pinned revision

        model, processor = jsm.load_release_model(path, device="cuda")
        _m.update(model=model, processor=processor, jsm=jsm, load_seconds=round(time.time() - t0, 1))
        log.info("clef-%s ready in %.1fs", VARIANT, _m["load_seconds"])
    except Exception as e:  # noqa: BLE001 - surfaced on /health instead of killing the server
        _load_error = f"{type(e).__name__}: {e}"
        log.exception("model load failed")


app = FastAPI(title=f"Clef ({VARIANT})")


@app.on_event("startup")
def _startup():
    if not API_KEY:
        log.warning("API_KEY is not set — the endpoint is open to anyone with the proxy URL")
    threading.Thread(target=_load, daemon=True).start()


def _auth(authorization: str = Header(default="")):
    if API_KEY and not hmac.compare_digest(authorization, f"Bearer {API_KEY}"):
        raise HTTPException(401, "invalid or missing bearer token")


def _ready():
    if _load_error:
        raise HTTPException(500, f"model failed to load: {_load_error}")
    if not _m:
        raise HTTPException(503, "model is still loading — poll /health")


@app.get("/health")
def health():
    if _load_error:
        return JSONResponse({"status": "error", "error": _load_error}, status_code=500)
    if not _m:
        return JSONResponse({"status": "loading", "model": f"clef-{VARIANT}"}, status_code=503)
    return {"status": "ok", "model": f"clef-{VARIANT}", "repo": REPO, "revision": REVISION,
            "load_seconds": _m["load_seconds"], "max_length": MAX_LENGTH}


def _validate(req):
    """Same checks as joint_schema_model.systemone, so batch and single requests agree."""
    questions = req.get("questions") if isinstance(req, dict) else None
    if not isinstance(req, dict) or "state" not in req:
        raise ValueError("state is required")
    if not isinstance(questions, dict) or not questions:
        raise ValueError("at least one question is required")
    for qid, q in questions.items():
        if not isinstance(q, dict) or q.get("type") not in _m["jsm"].QUESTION_TYPES:
            raise ValueError(f"{qid}: type must be noul, choice, or score")
        if q["type"] != "noul" and not q.get("criteria"):
            raise ValueError(f"{qid}: criteria must not be empty")


def _encode(req):
    jsm, tok, proc = _m["jsm"], _m["processor"].tokenizer, _m["processor"]
    enc = jsm.encode_record(tok, req, max_length=MAX_LENGTH, processor=proc)
    full = len(jsm.encode_record(tok, req, max_length=10**9, processor=proc).input_ids)
    return enc, max(0, full - len(enc.input_ids))


def _answer(req, enc, logits, dropped, seconds):
    jsm = _m["jsm"]
    answers = {
        q.question_id: jsm.systemone_answer(
            req["questions"][q.question_id],
            dict(zip(q.option_ids, ql.float().softmax(-1).tolist())))
        for q, ql in zip(enc.questions, logits)
    }
    return {"model": req.get("model") or f"clef-{VARIANT}", "answers": answers,
            "usage": {"input_tokens": len(enc.input_ids), "output_tokens": 0,
                      "truncated": dropped > 0, "state_tokens_dropped": dropped,
                      "seconds": round(seconds, 4)}}


def _score(reqs):
    """Encode every request, run them in length-sorted GPU batches, return results in order."""
    import torch

    jsm, model, pad = _m["jsm"], _m["model"], _m["processor"].tokenizer.pad_token_id
    device = next(model.parameters()).device
    out = [None] * len(reqs)
    encoded = []
    for i, req in enumerate(reqs):
        try:
            _validate(req)
            enc, dropped = _encode(req)
            encoded.append((i, enc, dropped))
        except ValueError as e:
            out[i] = {"error": str(e)}
    encoded.sort(key=lambda x: len(x[1].input_ids))

    batch = []

    def flush():
        if not batch:
            return
        t0 = time.time()
        try:
            with torch.inference_mode():
                logits = model(jsm.collate_records([e for _, e, _ in batch], pad, device))
            per = (time.time() - t0) / len(batch)
            for (i, enc, dropped), lg in zip(batch, logits):
                out[i] = _answer(reqs[i], enc, lg, dropped, per)
        except torch.OutOfMemoryError:
            torch.cuda.empty_cache()
            if len(batch) == 1:
                i = batch[0][0]
                out[i] = {"error": "OutOfMemoryError: request too large for this GPU"}
            else:  # split the batch and retry each half
                items = list(batch)
                batch.clear()
                for part in (items[: len(items) // 2], items[len(items) // 2:]):
                    batch.extend(part)
                    flush()
        batch.clear()

    for item in encoded:
        if batch and (len(batch) + 1) * len(item[1].input_ids) > MAX_BATCH_TOKENS:
            flush()
        batch.append(item)
    flush()
    return out


@app.post("/v1/systemone", dependencies=[Depends(_auth)])
def systemone(req: dict):
    _ready()
    with _gpu:
        res = _score([req])[0]
    if "error" in res:
        raise HTTPException(400, res["error"])
    return res


@app.post("/v1/systemone/batch", dependencies=[Depends(_auth)])
def systemone_batch(body: dict):
    _ready()
    reqs = body.get("requests")
    if not isinstance(reqs, list) or not reqs:
        raise HTTPException(400, "requests must be a non-empty list")
    if len(reqs) > MAX_REQUESTS:
        raise HTTPException(413, f"{len(reqs)} requests > MAX_REQUESTS_PER_BATCH={MAX_REQUESTS}")
    t0 = time.time()
    with _gpu:
        responses = _score(reqs)
    return {"responses": responses, "seconds": round(time.time() - t0, 3),
            "errors": sum("error" in r for r in responses)}
