"""ML-engine service: the paper's training contract, served by NeuralSmith or FLAML.

  r1/.venv-mlengine/bin/python r1/mlengine/server.py --backend neuralsmith|flaml [--port 8095]

POST /train   multipart: dataset.csv (train split, features + label), spec.json {task, target_column,
              latency_budget_ms, max_size_kb, mode, seed[, time_budget_s]}, optional test.csv (our
              time-ordered test split; evaluation uses it, never a random split)      → 202 {job_id}
GET  /jobs/{job_id}  → {status, model_url, report: {algorithm, accuracy, f1, precision, recall, fpr,
              confusion: [[tn, fp], [fn, tp]], onnx_latency_ms_p50, onnx_latency_ms_p99, size_kb, train_time_s,
              features_used, ...backend extras}}
GET  /jobs/{job_id}/{file}  model.onnx | preprocess.json | pred.csv | report.json
Budget loop: if p50 > latency_budget_ms or size_kb > max_size_kb, retrain once with a restricted search
(NeuralSmith --mode fast / FLAML lgbm only) and set report.retried = true. One job at a time.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import threading
import time
import traceback
import uuid

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from fastapi import FastAPI, File, HTTPException, UploadFile  # noqa: E402
from fastapi.responses import FileResponse  # noqa: E402

RUNS = os.getenv('MLENGINE_RUNS', os.path.join(os.path.dirname(os.path.abspath(__file__)), 'runs'))
FILES = {'model.onnx', 'preprocess.json', 'pred.csv', 'report.json', 'flaml.log'}
REQUIRED = ('task', 'target_column', 'latency_budget_ms', 'max_size_kb', 'mode', 'seed')
app = FastAPI(title='R1 ML engine')
JOBS: dict[str, dict] = {}
LOCK = threading.Lock()
BACKEND = None  # module, set in main()


def run_job(job_id: str, spec: dict) -> None:
    d = os.path.join(RUNS, job_id)
    job = JOBS[job_id]
    with LOCK:
        job['status'] = 'running'
        try:
            rep = BACKEND.train(os.path.join(d, 'dataset.csv'), os.path.join(d, 'test.csv'), spec, d)
            retried = False
            if rep['onnx_latency_ms_p50'] > spec['latency_budget_ms'] or rep['size_kb'] > spec['max_size_kb']:
                first = {k: rep[k] for k in ('algorithm', 'onnx_latency_ms_p50', 'size_kb', 'train_time_s')}
                rep = BACKEND.train(os.path.join(d, 'dataset.csv'), os.path.join(d, 'test.csv'), spec, d, restricted=True)
                rep['first_attempt'] = first
                retried = True
            rep.update(retried=retried, latency_budget_ms=spec['latency_budget_ms'], max_size_kb=spec['max_size_kb'],
                       within_budget=rep['onnx_latency_ms_p50'] <= spec['latency_budget_ms']
                       and rep['size_kb'] <= spec['max_size_kb'], spec=spec)
            json.dump(rep, open(os.path.join(d, 'report.json'), 'w'), indent=1)
            job.update(status='done', report=rep, model_url=f'/jobs/{job_id}/model.onnx')
        except Exception as exc:  # report, never hang the job
            job.update(status='failed', error=f'{type(exc).__name__}: {exc}', trace=traceback.format_exc()[-2000:])
        job['finished'] = time.time()


@app.post('/train', status_code=202)
async def train(dataset: UploadFile = File(..., alias='dataset.csv'), spec: UploadFile = File(..., alias='spec.json'),
                test: UploadFile | None = File(None, alias='test.csv')):
    try:
        s = json.loads(await spec.read())
    except ValueError as exc:
        raise HTTPException(400, f'spec.json: {exc}') from exc
    missing = [k for k in REQUIRED if k not in s]
    if missing:
        raise HTTPException(400, f'spec.json missing {missing}')
    if s['target_column'] != 'label' or s['task'] != 'classification':
        raise HTTPException(400, "supported: task 'classification', target_column 'label'")
    if test is None:
        raise HTTPException(400, 'test.csv is required: evaluation uses the time-ordered test split')
    job_id = time.strftime('%Y%m%d-%H%M%S-') + uuid.uuid4().hex[:6]
    d = os.path.join(RUNS, job_id)
    os.makedirs(d)
    with open(os.path.join(d, 'dataset.csv'), 'wb') as fh:
        shutil.copyfileobj(dataset.file, fh)
    with open(os.path.join(d, 'test.csv'), 'wb') as fh:
        shutil.copyfileobj(test.file, fh)
    json.dump(s, open(os.path.join(d, 'spec.json'), 'w'), indent=1)
    JOBS[job_id] = {'status': 'queued', 'created': time.time(), 'backend': BACKEND.__name__.replace('_backend', '')}
    threading.Thread(target=run_job, args=(job_id, s), daemon=True).start()
    return {'job_id': job_id}


@app.get('/jobs/{job_id}')
def job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(404, 'unknown job')
    j = JOBS[job_id]
    return {'status': j['status'], 'model_url': j.get('model_url'), 'report': j.get('report'), 'error': j.get('error'),
            'backend': j['backend']}


@app.get('/jobs/{job_id}/{name}')
def job_file(job_id: str, name: str):
    p = os.path.join(RUNS, job_id, name)
    if name not in FILES or job_id not in JOBS or not os.path.isfile(p):
        raise HTTPException(404, 'not found')
    return FileResponse(p)


@app.get('/health')
def health():
    return {'ok': True, 'backend': BACKEND.__name__.replace('_backend', '')}


def main():
    global BACKEND
    ap = argparse.ArgumentParser()
    ap.add_argument('--backend', choices=('neuralsmith', 'flaml'), default='neuralsmith')
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=8095)
    a = ap.parse_args()
    import importlib
    BACKEND = importlib.import_module(f'{a.backend}_backend')
    os.makedirs(RUNS, exist_ok=True)
    import uvicorn
    uvicorn.run(app, host=a.host, port=a.port, log_level='info')


if __name__ == '__main__':
    main()
