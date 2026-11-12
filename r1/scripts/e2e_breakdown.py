"""Exp-04 end-to-end breakdown (intent_received -> first_e2_control_sent) per official run, read-only over the run dirs.

usage: python3 r1/scripts/e2e_breakdown.py -> prints per-group (20-/10-min) tables, writes r1/results/exp04/e2e_breakdown.json
Sources: timing.jsonl (runner marks, monotonic clock), verify_events.jsonl (xApp start + first reserve, wall clock, mapped
onto the monotonic clock via register_end), run_meta.json verify.t0 (verify traffic start), the collector meta.json
(traffic t0 + duration). Asserts that the segments sum to the recorded end-to-end time.
"""
import datetime as dt, json, os, statistics as st
R = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'exp04')
RUNS = ['exp04-20261105d-s1-20m', 'exp04-20261105d-s3-20m', 'exp04-20261105d-s5-20m',
        'exp04-20261105d-s2-10m', 'exp04-20261105d-s4-10m', 'exp04-20261105d-s6-10m-r2']
def wall(iso): return dt.datetime.fromisoformat(iso).timestamp()
out = {}
for rid in RUNS:
    d = os.path.join(R, rid)
    ev = {}
    for l in open(os.path.join(d, 'timing.jsonl')):
        e = json.loads(l); ev.setdefault(e['event'], e)
    # monotonic clock for all runner marks (first_e2_control_sent is placed on it from the xApp's wall time)
    T = {k: v['monotonic_ns'] / 1e9 for k, v in ev.items()}
    off = T['register_end'] - wall(ev['register_end']['wall_iso'])  # monotonic - wall (ms-precision iso)
    vev = [json.loads(l) for l in open(os.path.join(d, 'verify_events.jsonl'))]
    xstart = next(e['t_us'] for e in vev if e['type'] == 'start') / 1e6 + off
    first = next(e['t_us'] for e in vev if e.get('action') == 'reserve') / 1e6 + off
    m = json.load(open(os.path.join(d, 'run_meta.json')))
    t0v = m['verify']['t0'] + off
    cm = json.load(open(os.path.join(d, 'raw', rid + '-collect', 'meta.json')))
    traffic_t0 = cm['t0_wall_s'] + off
    traffic_end = traffic_t0 + cm['duration_s']
    seg = [
        ('1 orchestration: intent -> validated spec (LLM)', T['intent_received'], T['spec_ready']),
        ('2 gate (deterministic + read-back)', T['spec_ready'], T['gate_end']),
        ('3a collection: start gate (idle check + loaded probe) + traffic setup', T['gate_end'], traffic_t0),
        ('3b collection: traffic (frozen profile)', traffic_t0, traffic_end),
        ('3c collection: stop, fetch logs, write telemetry', traffic_end, T['collection_end']),
        ('4 lease + dataset build -> train request', T['collection_end'], T['train_post']),
        ('5 training (NeuralSmith)', T['train_post'], T['model_received']),
        ('6 RAN mapping snapshot (protected UE ref)', T['model_received'], T['render_start']),
        ('7 render', T['render_start'], T['render_end']),
        ('8 model-parity gate', T['render_end'], T['gate_model_parity_end']),
        ('9 pre-deploy: pod snapshot, isolation, traffic-gen restart, stop KPM monitor', T['gate_model_parity_end'], T['register_start']),
        ('10a registration: stop old xApp, write + sha256-check files, start xApp', T['register_start'], xstart),
        ('10b registration: xApp start -> KPM indications flowing', xstart, T['register_end']),
        ('11 verify setup: UE routing, UE3 flow + ping, 10 s settle', T['register_end'], t0v),
        ('12 ramp-up until first predicted congestion -> first reserve', t0v, first),
    ]
    s = {k: round(b - a, 2) for k, a, b in seg}
    s['TOTAL (sum)'] = round(first - T['intent_received'], 2)
    assert abs(sum(b - a for _, a, b in seg) - (T['first_e2_control_sent'] - T['intent_received'])) < 0.05, rid
    s['end_to_end_s (timing_summary)'] = round(T['first_e2_control_sent'] - T['intent_received'], 2)
    s['verify cycle-1 reserve_s_after_ramp_start'] = m['verify']['cycles'][0]['reserve_s_after_ramp_start']
    s['collection: skipped_starts'] = len(cm.get('skipped_starts') or [])
    s['collection: traffic duration_s'] = cm['duration_s']
    out[rid] = s
keys = list(next(iter(out.values())))
for grp, rr in (('20-min', RUNS[:3]), ('10-min', RUNS[3:])):
    tot = {r: out[r]['TOTAL (sum)'] for r in rr}
    med_run = sorted(rr, key=lambda r: tot[r])[1]
    print(f'\n=== {grp}: runs {[r[-6:] for r in rr]}  median run = {med_run}')
    print(f"{'segment':78s} {'median':>8s} {'min':>8s} {'max':>8s} {'med.run':>8s}")
    for k in keys:
        v = [out[r][k] for r in rr]
        print(f'{k:78s} {st.median(v):8.2f} {min(v):8.2f} {max(v):8.2f} {out[med_run][k]:8.2f}')
json.dump(out, open(os.path.join(R, 'e2e_breakdown.json'), 'w'), indent=1)
