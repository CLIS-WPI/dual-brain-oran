"""Robust 60→80 % rise per cycle from a calib_ramp output folder (offline, read-only).

usage: python3 -I ramp_rise.py <ramp-folder>
After idle, OAI DL link adaptation restarts near MCS 0 and climbs to 28 within ~2 s, so every ON onset
shows a 1–2 s PRB spike (≈90 % at 5 Mbps/UE). Rule: t80 = first second PRB ≥ 80 % sustained ≥ SUSTAIN_S
consecutive samples; t60 = last sample < 60 % before t80; rise = t80 − t60. Cycles from burst.schedule
(seeded, deterministic) with the run's params from calib_ramp.json.
"""
import csv, json, os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'traffic'))
import burst

SUSTAIN_S = 3


def rise(series, sustain=SUSTAIN_S):
    """series [(t, prb)] of one cycle (ON window) → (t60, t80, rise) or (None, None, None)."""
    for i in range(len(series) - sustain + 1):
        if all(v >= 80 for _, v in series[i:i + sustain]):
            t80 = series[i][0]
            below = [t for t, v in series[:i] if v < 60]
            return (below[-1], t80, round(t80 - below[-1], 1)) if below else (None, t80, None)
    return None, None, None


assert rise([(0, 90), (1, 50), (2, 65), (3, 75), (4, 82), (5, 85), (6, 88)]) == (1, 4, 3)  # onset spike skipped
assert rise([(0, 50), (1, 90), (2, 50)]) == (None, None, None)  # no sustained 80


def main(d):
    meta = json.load(open(f'{d}/calib_ramp.json'))
    out = {}
    for p in meta['plans']:
        prm = p['params']
        s = burst.schedule('ramp', cycles=3, seed=meta['seed'],  # every calib_ramp profile runs 3 cycles
                           **{k: v for k, v in prm.items() if k in burst.PROFILES['ramp']})
        rows = [(float(r['t_wall_s']), float(r['cell_prb_pct']))
                for r in csv.DictReader(open(f"{d}/prb_{p['name'].replace('@', '_')}.csv"))]
        cyc = []
        for c in s['cycles']:
            a, b = c['phases']['on']
            cyc.append(rise([(t, v) for t, v in rows if a <= t <= b]))
        rises = [r[2] for r in cyc]
        out[p['name']] = {'cycles': [{'t60': r[0], 't80': r[1], 'rise_s': r[2]} for r in cyc],
                          'rise_ge3_most': sum(1 for r in rises if r is not None and r >= 3) > len(rises) / 2}
        print(f"{p['name']:14} rise 60→80 s per cycle {rises}  (t60,t80: {[(r[0], r[1]) for r in cyc]}) "
              f"≥3 s in most: {out[p['name']]['rise_ge3_most']}")
    json.dump(out, open(f'{d}/rise_robust.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1])
