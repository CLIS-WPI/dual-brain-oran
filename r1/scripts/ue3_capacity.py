"""UE3 (protected) DL capacity with the current traffic generator: UE1/UE2 idle, UE3 offered a saturating UDP DL load.

usage: ue3_capacity.py [--offered 40] [--seconds 60]
Isolation first (no leftover flows, idle cell), fresh traffic generator, then UE3 alone at --offered Mbps for
--seconds; the first 10 s are skipped (link adaptation / ramp). Reports UE3 iperf receiver goodput (mean, p10),
UE3 PRB share and cell PRB (gNB [AURA-PRB] 1 s), UE3 DL MCS/BLER (MAC stats), RTT (10 Hz ping, raw log kept),
sim speed. Output: results/calibration/ue3cap-<time>/summary.json + ue3_ping.log (gitignored).
"""
import json
import os
import re
import statistics as st
import sys
import time

import r1env as E
from r1env import k, pdu_ip, pod

import burst
from services.dual_brain import channel_guard as g, gnb_truth

arg = lambda key, d: type(d)(sys.argv[sys.argv.index(key) + 1]) if key in sys.argv else d
OFFERED, SECONDS, SKIP = arg('--offered', 40.0), arg('--seconds', 60), 10


def main():
    out = os.path.join(E.RESULTS, 'calibration', 'ue3cap-' + time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(out)
    prov = E.provenance(__file__)
    iso = E.isolate()
    trf_name, trf_ip = E.restart_trafficgen()
    ue3 = pod('virtual-ue', 'ue3')[0]
    E.route_to(ue3, trf_ip)
    import atexit
    atexit.register(E.stop_traffic)
    t0 = time.time()
    k('exec', ue3, '-c', 'ue', '--', *burst.remote_command({'cycles': [{'flows': [(0.0, OFFERED, SECONDS)]}]},
                                                          server_ip=trf_ip, bind_ip=pdu_ip(ue3)))
    E.start_ping(ue3, trf_ip, SECONDS)
    time.sleep(SECONDS + 3)
    glog = k('logs', pod('gnb-rfsim')[0], '-c', 'gnb', '--timestamps', f'--since={int(time.time() - t0) + 5}s')
    ping = k('exec', ue3, '-c', 'ue', '--', 'cat', E.PING_LOG)
    open(os.path.join(out, 'ue3_ping.log'), 'w').write(ping)
    recv = burst.receiver_mbps(k('exec', ue3, '-c', 'ue', '--', 'cat', burst.RECV_LOG))
    E.stop_traffic()
    a, b = t0 + SKIP, t0 + SECONDS
    good = sorted(v for t, v in recv if a <= t <= b)
    rows = [r for r in gnb_truth.parse(glog)[0] if a <= r['t'] <= b]
    rnti = g.ue_rnti(k('logs', ue3, '-c', 'ue', '--since=90s'))
    share = [100 * r['ues'].get(rnti, (0, 0))[1] / r['budget'] for r in rows]
    cell = [100 * sum(x for _, x in r['ues'].values()) / r['budget'] for r in rows]
    mac = [(int(m.group(3)), float(m.group(2))) for ln in glog.splitlines()
           if (m := g._GNB_DL.search(ln)) and m.group(1) == rnti and a <= gnb_truth._ts(ln) <= b]
    rtt = sorted(float(y) for x, y in re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', ping) if a <= float(x) <= b)
    res = {'offered_mbps': OFFERED, 'window_s': [SKIP, SECONDS], 'isolation': iso,
           'ue3_goodput_mbps': {'mean': round(st.mean(good), 2), 'p10': good[len(good) // 10], 'n': len(good)} if good else None,
           'ue3_prb_share_pct': round(st.mean(share), 1) if share else None, 'cell_prb_pct': round(st.mean(cell), 1) if cell else None,
           'ue3_dl_mcs': round(st.mean(m for m, _ in mac), 2) if mac else None, 'ue3_dl_bler': round(st.mean(x for _, x in mac), 3) if mac else None,
           'ue3_rtt_ms': {'mean': round(st.mean(rtt), 1), 'p95': rtt[int(0.95 * (len(rtt) - 1))], 'n': len(rtt)} if rtt else None,
           'sim_speed': round(st.mean(r['budget'] / 61200 for r in rows), 3) if rows else None,
           'trafficgen_memory_mib': [x and round(x / 2**20) for x in E.trafficgen_memory()], 'provenance': prov}
    json.dump(res, open(os.path.join(out, 'summary.json'), 'w'), indent=1)
    print(json.dumps({k_: v for k_, v in res.items() if k_ != 'provenance'}, indent=1))
    print('wrote', out)


if __name__ == '__main__':
    main()
