"""Compare ramp profiles during hold from a saved calib_ramp gNB log (offline, read-only).

usage: python3 -I ramp_hold_compare.py <ramp-folder> [gnb log, default <folder>/gnb_ramp_run.log]
Per profile, pooled over the hold phases of all cycles: per-UE DL PRB share (AURA-PRB tx_rb / budget), DL MCS
and BLER (MAC stats), DL scheduled-slot fraction (Δ dlsch transmissions / Δ DL slots, DL slots = budget / 51),
MAC LCID 4 (DRB) TX Mbps per wall-clock second. The run's T0 is recovered by matching the cell-PRB series in
prb_<first profile>.csv to the AURA-PRB lines. The gNB log has no RLC buffer occupancy; LCID TX bytes are the
only buffer-side indicator.
"""
import csv, datetime as dt, json, os, re, statistics as st, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'traffic'))
import burst

PRB_PER_SLOT = 51
_UE = re.compile(r"rnti=0x([0-9a-f]+):slice=-?\d+:tx_rb=(\d+)")
_DL = re.compile(r"UE ([0-9a-f]{4}): dlsch_rounds (\d+)/(\d+)/(\d+)/(\d+),.*?BLER (\d+(?:\.\d+)?) MCS \(\d+\) (\d+)")
_LC = re.compile(r"UE ([0-9a-f]{4}): LCID 4: TX\s+(\d+)")


def ts(line):
    return dt.datetime.fromisoformat(line.split(" ", 1)[0].replace("Z", "")[:26]).replace(tzinfo=dt.timezone.utc).timestamp()


def parse(log):
    prb, dl, lc = [], [], []
    for line in log.splitlines():
        if "[AURA-PRB] 1s" in line:
            b = int(re.search(r"budget=(\d+)", line).group(1))
            if b > 0:
                prb.append((ts(line), b, {r: int(x) for r, x in _UE.findall(line)}))
        elif m := _DL.search(line):
            dl.append((ts(line), m.group(1), sum(int(m.group(i)) for i in range(2, 6)), float(m.group(6)), int(m.group(7))))
        elif m := _LC.search(line):
            lc.append((ts(line), m.group(1), int(m.group(2))))
    return prb, dl, lc


def find_t0(prb, csv_rows, offset):
    """T0 such that AURA cell PRB at T0 + offset + t_rel equals the CSV series (first 20 rows)."""
    cell = [(t, 100 * sum(u.values()) / b) for t, b, u in prb]
    for i, (t, v) in enumerate(cell):
        t0 = t - offset - csv_rows[0][0]
        if all(any(abs(t2 - (t0 + offset + tr)) < 0.05 and abs(v2 - vr) < 0.051 for t2, v2 in cell[i:i + 40])
               for tr, vr in csv_rows[:20]):
            return t0
    raise SystemExit('could not align the CSV with the gNB log')


def delta(series, a, b):
    """Counter increase between the last sample at/before a and the last sample at/before b."""
    s0 = [v for t, v in series if t <= a]
    s1 = [v for t, v in series if t <= b]
    return (s1[-1] - s0[-1]) if s0 and s1 else None


def main(d, log_path):
    meta = json.load(open(f'{d}/calib_ramp.json'))
    prb, dl, lc = parse(open(log_path).read())
    first = meta['plans'][0]
    rows0 = [(float(r['t_wall_s']), float(r['cell_prb_pct']))
             for r in csv.DictReader(open(f"{d}/prb_{first['name'].replace('@', '_')}.csv"))]
    t0 = find_t0(prb, rows0, first['offset'])
    rnti = meta['rnti']
    out = {}
    for p in meta['plans']:
        prm = {k: v for k, v in p['params'].items() if k in burst.PROFILES['ramp']}
        s = burst.schedule('ramp', cycles=3, seed=meta['seed'], **prm)  # every calib_ramp profile: 3 cycles
        holds = [(t0 + p['offset'] + c['phases']['hold'][0] + 1, t0 + p['offset'] + c['phases']['hold'][1]) for c in s['cycles']]
        inh = lambda t: any(a <= t <= b for a, b in holds)
        r = {}
        for ue, rn in rnti.items():
            share = [100 * u.get(rn, 0) / b for t, b, u in prb if inh(t)]
            mac = [(m, bl) for t, r_, _, bl, m in dl if r_ == rn and inh(t)]
            slots = sum(b / PRB_PER_SLOT for t, b, _ in prb if inh(t))
            tx = [delta([(t, n) for t, r_, n, _, _ in dl if r_ == rn], a, b) for a, b in holds]
            byt = [delta([(t, n) for t, r_, n in lc if r_ == rn], a, b) for a, b in holds]
            secs = sum(b - a for a, b in holds)
            r[ue] = {'prb_share_pct': round(st.mean(share), 1) if share else None,
                     'dl_mcs': round(st.mean(m for m, _ in mac), 1) if mac else None,
                     'dl_bler': round(st.mean(bl for _, bl in mac), 3) if mac else None,
                     'sched_slot_frac': round(sum(x for x in tx if x) / slots, 3) if slots and any(tx) else None,
                     'lcid4_tx_mbps_wall': round(sum(x for x in byt if x) * 8 / 1e6 / secs, 2) if any(byt) else None}
        r['cell_prb_pct'] = round(st.mean(100 * sum(u.values()) / b for t, b, u in prb if inh(t)), 1)
        r['sim_speed'] = round(st.mean(b / 61200 for t, b, _ in prb if inh(t)), 2)
        out[p['name']] = r
        print(f"{p['name']:14} cell {r['cell_prb_pct']}% speed {r['sim_speed']} | " + ' | '.join(
            f"{ue}: share {v['prb_share_pct']}% MCS {v['dl_mcs']} BLER {v['dl_bler']} slots {v['sched_slot_frac']} "
            f"LCID4 {v['lcid4_tx_mbps_wall']} Mbps" for ue, v in r.items() if isinstance(v, dict)))
    json.dump({'t0': t0, 'hold_compare': out}, open(f'{d}/hold_compare.json', 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else f'{sys.argv[1]}/gnb_ramp_run.log')
