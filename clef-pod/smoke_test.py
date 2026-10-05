#!/usr/bin/env python3
"""Smoke-test a running Clef pod: wait for /health, then send Jev/SystemOne requests.

    export POD_ID=<pod id>                 # or CLEF_URL=https://<pod>-8000.proxy.runpod.net
    # API key: CLEF_API_KEY env var, else clef_api_key in ti/.env
    python clef-pod/smoke_test.py
"""
import json
import os
import pathlib
import sys
import time

import requests

POD_ID = os.environ.get("POD_ID", "")
BASE = os.environ.get("CLEF_URL") or (f"https://{POD_ID}-8000.proxy.runpod.net" if POD_ID else "")
READY_TIMEOUT_S = 1800  # first start downloads 18 GB (flash) or 54 GB (full)

# Synthetic text only — no real dealer correspondence.
STATE = ("One episode of pricing correspondence between a vendor and a dealer.\n"
         "--- 2026-03-02 | VENDOR -> dealer | subject: Your 2026 package rate\n"
         "Starting April 1 your monthly package moves from $899 to $1,049.\n"
         "--- 2026-03-05 | DEALER -> vendor | subject: RE: Your 2026 package rate\n"
         "This is the first we've heard of this. $150 more a month is too much - "
         "if it goes through we'll cancel at renewal.\n")
QUESTIONS = {
    "vendor_sent_notice": {"type": "noul",
                           "instructions": "The vendor told the dealer about the price change."},
    "threatened_cancel": {"type": "noul",
                          "instructions": "The dealer says they will cancel or leave over the price."},
    "outcome": {"type": "choice", "instructions": "How the conversation ended",
                "criteria": {"accepted": "The dealer accepted the new price",
                             "objected_open": "The dealer objected and it is unresolved",
                             "negotiated": "A different price was agreed"}},
    "question_level": {"type": "score", "instructions": "How strongly the dealer pushes back",
                       "criteria": ["No pushback", "Mild questions", "Strong objection",
                                    "Threatens to leave"]},
}


def api_key():
    k = os.environ.get("CLEF_API_KEY")
    if k:
        return k.strip()
    here = pathlib.Path(__file__).resolve().parent
    for f in [d / ".env" for d in [here, *here.parents]] + [here.parent / "ti" / ".env"]:
        if f.exists():
            for line in f.read_text(encoding="utf-8").splitlines():
                name, sep, value = line.partition("=")
                if sep and name.strip().lower() == "clef_api_key":
                    return value.strip().strip("'\"")
    sys.exit("no CLEF_API_KEY env var or clef_api_key in .env")


def main():
    if not BASE:
        sys.exit("set POD_ID or CLEF_URL")
    t0 = time.time()
    while True:
        try:
            r = requests.get(f"{BASE}/health", timeout=20)
            print(f"[{time.time() - t0:5.0f}s] /health -> {r.status_code} {r.text[:200]}")
            if r.status_code == 200:
                break
            if r.status_code == 500:
                sys.exit("model failed to load (see above)")
        except requests.RequestException as e:
            print(f"[{time.time() - t0:5.0f}s] /health -> {type(e).__name__}")
        if time.time() - t0 > READY_TIMEOUT_S:
            sys.exit("server did not become ready in time")
        time.sleep(15)

    headers = {"Authorization": f"Bearer {api_key()}"}
    r = requests.post(f"{BASE}/v1/systemone", json={"model": "clef", "state": "x",
                      "questions": {"q": {"type": "noul"}}}, timeout=60)
    print(f"\nwithout key -> HTTP {r.status_code} (expect 401)")
    if r.status_code != 401:
        sys.exit("SECURITY: the pod answered a request without the API key")

    body = {"model": "clef", "state": STATE, "questions": QUESTIONS}
    for attempt in ("first (includes GPU warm-up)", "second (warm)"):
        t = time.time()
        r = requests.post(f"{BASE}/v1/systemone", json=body, headers=headers, timeout=600)
        r.raise_for_status()
        print(f"\n/v1/systemone {attempt}: {time.time() - t:.2f}s round trip")
    print(json.dumps(r.json(), indent=2))

    batch = {"requests": [body] * 8}
    t = time.time()
    r = requests.post(f"{BASE}/v1/systemone/batch", json=batch, headers=headers, timeout=600)
    r.raise_for_status()
    b = r.json()
    print(f"\n/v1/systemone/batch: 8 requests in {b['seconds']}s on the GPU "
          f"({time.time() - t:.2f}s round trip), errors={b['errors']}")


if __name__ == "__main__":
    main()
