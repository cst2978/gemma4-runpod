#!/usr/bin/env python3
"""Smoke-test a running SemIf pod: wait for /health, then score a few decisions.

    export POD_ID=<pod id>                 # or OPENJEV_URL=https://<pod>-8000.proxy.runpod.net
    python openjev-pod/smoke_test.py
"""
import json
import os
import sys
import time

import requests

POD_ID = os.environ.get("POD_ID", "")
BASE = os.environ.get("OPENJEV_URL") or (f"https://{POD_ID}-8000.proxy.runpod.net" if POD_ID else "")
READY_TIMEOUT_S = 900
YES_NO = [{"id": "yes", "description": "This is true of the text."},
          {"id": "no", "description": "This is not true of the text."}]
STATE = ("Hi team — our March invoice shows 14 listings billed but we only had 11 live. "
         "Also the lead form on our RV pages has been broken since Tuesday, customers "
         "say it spins forever. Otherwise traffic has been great this quarter, thanks!")
ROWS = [
    # Three questions on one state -> one shared prefill.
    {"id": "billing_dispute", "state": STATE, "question": "Does the dealer dispute a charge?",
     "options": YES_NO},
    {"id": "product_bug", "state": STATE, "question": "Does the dealer report a broken feature?",
     "options": YES_NO},
    {"id": "sentiment", "state": STATE, "question": "What is the dealer's overall tone?",
     "options": [{"id": "positive", "description": "Mostly positive."},
                 {"id": "mixed", "description": "A mix of positive and negative."},
                 {"id": "negative", "description": "Mostly negative."}]},
    # Different state -> its own group (scored alone).
    {"id": "churn_risk", "state": "Please cancel our subscription at the end of the month.",
     "question": "Is the dealer asking to cancel?", "options": YES_NO},
]


def main():
    if not BASE:
        sys.exit("set POD_ID or OPENJEV_URL")
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
        time.sleep(10)

    r = requests.post(f"{BASE}/score", json={"rows": ROWS}, timeout=300)
    r.raise_for_status()
    body = r.json()
    print(f"\nscored {len(body['results'])} rows in {body['seconds']}s, errors={body['errors']}")
    for res in body["results"]:
        if "error" in res:
            print(f"  {res['id']:16} ERROR {res['error']}")
        else:
            probs = ", ".join(f"{o}={p:.3f}" for o, p in zip(res["option_ids"], res["probabilities"]))
            print(f"  {res['id']:16} {probs}  ({res['input_tokens']} tok)")
    print("\nmodel:", json.dumps(body["model"]))


if __name__ == "__main__":
    main()
