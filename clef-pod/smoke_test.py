#!/usr/bin/env python3
"""Smoke-test a running Clef pod: wait for /health, then send Jev/SystemOne requests.

    export POD_ID=<pod id>                 # or CLEF_URL=https://<pod>-8000.proxy.runpod.net
    python clef-pod/smoke_test.py
"""
import json
import os
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

    body = {"model": "clef", "state": STATE, "questions": QUESTIONS}
    for attempt in ("first (includes GPU warm-up)", "second (warm)"):
        t = time.time()
        r = requests.post(f"{BASE}/v1/systemone", json=body, timeout=600)
        r.raise_for_status()
        print(f"\n/v1/systemone {attempt}: {time.time() - t:.2f}s round trip")
    print(json.dumps(r.json(), indent=2))

    batch = {"requests": [body] * 8}
    t = time.time()
    r = requests.post(f"{BASE}/v1/systemone/batch", json=batch, timeout=600)
    r.raise_for_status()
    b = r.json()
    print(f"\n/v1/systemone/batch: 8 requests in {b['seconds']}s on the GPU "
          f"({time.time() - t:.2f}s round trip), errors={b['errors']}")


if __name__ == "__main__":
    main()
