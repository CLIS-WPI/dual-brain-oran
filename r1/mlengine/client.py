"""Submit a training job to the ML-engine server and wait for the report (used by the experiment runners).

usage: client.py --url http://127.0.0.1:8095 --train train.csv --test test.csv --spec spec.json --out <dir>
Writes <dir>/report.json, model.onnx, pred.csv (+ preprocess.json for NeuralSmith); prints the report.
"""
import argparse
import json
import os
import time

import httpx


def submit(url, train, test, spec, out, poll_s=5, timeout_s=7200):
    with open(train, 'rb') as a, open(test, 'rb') as b:
        r = httpx.post(f'{url}/train', timeout=120, files={'dataset.csv': ('dataset.csv', a, 'text/csv'),
                                                           'test.csv': ('test.csv', b, 'text/csv'),
                                                           'spec.json': ('spec.json', json.dumps(spec), 'application/json')})
    r.raise_for_status()
    job_id = r.json()['job_id']
    t0 = time.time()
    while True:
        j = httpx.get(f'{url}/jobs/{job_id}', timeout=30).json()
        if j['status'] in ('done', 'failed') or time.time() - t0 > timeout_s:
            break
        time.sleep(poll_s)
    if j['status'] != 'done':
        raise RuntimeError(f"job {job_id} {j['status']}: {j.get('error')}")
    os.makedirs(out, exist_ok=True)
    for f in ('model.onnx', 'pred.csv', 'report.json', 'preprocess.json'):
        r = httpx.get(f'{url}/jobs/{job_id}/{f}', timeout=60)
        if r.status_code == 200:
            open(os.path.join(out, f), 'wb').write(r.content)
    return job_id, j['report']


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    for k in ('url', 'train', 'test', 'spec', 'out'):
        ap.add_argument(f'--{k}', required=True)
    a = ap.parse_args()
    jid, rep = submit(a.url, a.train, a.test, json.load(open(a.spec)), a.out)
    print(json.dumps({'job_id': jid, **{k: v for k, v in rep.items() if k not in ('spec', 'best_config', 'latency_runs')}}, indent=1))
