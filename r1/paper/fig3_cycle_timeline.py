"""Fig. 3: the typical Exp-03 cycle per arm over all 39 cycles ((a) No xApp, (b) Reactive, (c) xApp), aligned at ramp-up
start, 0-90 s. No single-cycle traces.
  top strip   reservation time after ramp-up start across the 39 cycles (exp03_cycles.csv reserve_s_after_ramp_start),
              horizontal box plots for Reactive and xApp (5-95 % whiskers, as Fig. 4)
  middle      gNB cell DL PRB: each cycle's 1 s series linearly interpolated onto a 0.5 s grid, then the median across
              cycles per bin for (a) and (c); IQR band for (a); 80 % line; vertical line at (a)'s median congestion
              onset (first time cell PRB >= 80 % for 3 consecutive 1 s rows within ramp-up..OFF start, per cycle)
  bottom      edge-UE (UE3) 10 Hz ping: per cycle the 95th percentile in each 1 s bin, then the median across cycles per
              arm with IQR band (log y)
Runs whose gNB log covers < 90 % of the run have no PRB data (exp03_analyze rule; Exp-03 b3) — only (a)/(c) PRB is
drawn, both complete.
usage: python3 r1/paper/fig3_cycle_timeline.py [batch_dir] -> r1/paper/cycle_timeline.pdf + cycle_timeline.json (medians)"""
import csv
import json
import os
import re
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'experiments'))
sys.path.insert(0, os.path.join(HERE, '..', 'collector'))
from exp03_analyze import congested, p95, parse_aura  # noqa: E402
from speed_rules import GNB_COVERAGE_MIN, gnb_log_coverage  # noqa: E402

BATCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'results', 'exp03', 'exp03-20261106b')
ARMS = [('a', 'No xApp'), ('b', 'Reactive'), ('c', 'xApp')]  # same as Fig. 4
COLORS = {'a': '#E69F00', 'b': '#56B4E9', 'c': '#009E73'}  # Okabe-Ito, same as Fig. 4
XMAX = 90
GRID = np.arange(0, XMAX + 0.01, 0.5)  # PRB bins (s)
SEC = np.arange(0, XMAX)  # RTT 1 s bins [k, k+1)

cyc_rows = list(csv.DictReader(open(os.path.join(BATCH, 'exp03_cycles.csv'))))
used = sorted({r['run_id'] for r in cyc_rows if r['arm'] in 'abc'})
prb, rtt, onset = {a: [] for a, _ in ARMS}, {a: [] for a, _ in ARMS}, {a: [] for a, _ in ARMS}
phases = []
for rid in used:
    d = os.path.join(BATCH, rid)
    m = json.load(open(os.path.join(d, 'run_meta.json')))
    arm, t0 = m['arm'], m['t0_wall_s']
    aura = parse_aura(open(os.path.join(d, 'raw', 'gnb_aura.log')).read())
    prb_ok = gnb_log_coverage([r['t'] for r in aura], t0, m['t_end_wall_s']) >= GNB_COVERAGE_MIN
    cell = [(r['t'], 100 * sum(x for _, x in r['ues'].values()) / r['budget']) for r in aura]
    cong = congested(cell)
    ping = [(float(a), float(b)) for a, b in
            re.findall(r'\[(\d+\.\d+)\].*?time=([\d.]+) ms', open(os.path.join(d, 'raw', 'ue3_ping.log')).read())]
    for c in m['schedule']['cycles']:
        ts = t0 + c['ramp_up'][0]
        phases.append([c[k][0] - c['ramp_up'][0] for k in ('hold', 'ramp_down', 'off')])
        if prb_ok:
            x = np.array([t - ts for t, _ in cell])
            y = np.array([v for _, v in cell])
            sel = (x >= -2) & (x <= XMAX + 2)
            prb[arm].append(np.interp(GRID, x[sel], y[sel]))
            onset[arm].append(next((t - ts for t, k in cong if k and ts <= t < t0 + c['off'][0]), None))
        pc = [(t - ts, v) for t, v in ping if 0 <= t - ts < XMAX]
        rtt[arm].append([p95([v for t, v in pc if k <= t < k + 1]) for k in SEC])

res = {a: [float(r['reserve_s_after_ramp_start']) for r in cyc_rows if r['arm'] == a and r['reserve_s_after_ramp_start']]
       for a in 'bc'}
q = lambda v: [round(float(x), 1) for x in np.percentile(v, [25, 50, 75])]
onset_a = [o for o in onset['a'] if o is not None]
ph = np.median(np.array(phases), axis=0)  # hold, down, OFF starts (identical schedule; OFF length varies ~1 s)
prb_q = {a: np.percentile(np.array(prb[a]), [25, 50, 75], axis=0) for a in 'ac'}
rtt_q = {}
for a, _ in ARMS:
    arr = np.array([[np.nan if v is None else v for v in cyc] for cyc in rtt[a]], dtype=float)
    rtt_q[a] = np.nanpercentile(arr, [25, 50, 75], axis=0)
win = lambda a, lo, hi: round(float(np.median(rtt_q[a][1][int(lo):int(hi)])), 1)  # median of the per-second medians

summary = {
    'batch': os.path.basename(os.path.normpath(BATCH)),
    'n_cycles': {a: len(rtt[a]) for a, _ in ARMS}, 'n_cycles_prb': {a: len(prb[a]) for a in 'ac'},
    'phase_starts_s': {'ramp_up': 0, 'hold': round(ph[0], 1), 'ramp_down': round(ph[1], 1), 'off': round(ph[2], 1)},
    'a_congestion_onset_s': {'q25_median_q75': q(onset_a), 'n_with_onset': len(onset_a), 'n': len(onset['a'])},
    'reserve_s_after_ramp_start': {a: {'q25_median_q75': q(res[a]), 'p5_p95': [round(float(x), 1) for x in np.percentile(res[a], [5, 95])],
                                       'n': len(res[a]), 'median_vs_a_onset_s': round(float(np.median(res[a])) - q(onset_a)[1], 1)}
                                   for a in 'bc'},
    'cell_prb_median_pct': {a: {'ramp_up_last_10s': round(float(np.median(prb_q[a][1][(GRID >= 34) & (GRID < 44)])), 1),
                                'hold': round(float(np.median(prb_q[a][1][(GRID >= ph[0]) & (GRID < ph[1])])), 1)}
                            for a in 'ac'},
    'ue3_rtt_p95_1s_median_ms': {a: {'ramp_up_0_20s': win(a, 0, 20), 'eval_22_54s': win(a, 22, 54),
                                     'peak': round(float(np.nanmax(rtt_q[a][1])), 1),
                                     'peak_at_s': float(SEC[int(np.nanargmax(rtt_q[a][1]))]),
                                     'off_65_90s': win(a, 65, 90)} for a, _ in ARMS},
    'method': {'prb': '1 s gNB rows interpolated to 0.5 s per cycle, median/IQR across cycles',
               'rtt': 'per cycle p95 per 1 s bin of the 10 Hz ping, median/IQR across cycles',
               'onset': 'first time cell PRB >= 80 % for 3 consecutive 1 s rows, ramp-up start..OFF start'},
}

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8, 'legend.fontsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif',
                     'font.sans-serif': ['Arial', 'Liberation Sans'], 'axes.linewidth': 0.6})
fig, (ax0, ax1, ax2) = plt.subplots(3, 1, figsize=(3.5, 3.7), sharex=True, gridspec_kw={'height_ratios': [0.38, 1, 1.1]})
bounds = [(0, ph[0], 'ramp-up', 'center'), (ph[0], ph[1], 'hold', 'center'), (ph[1], ph[2], 'down', 'left'),
          (ph[2], XMAX, 'OFF', 'center')]
for lo, hi, lab, ha in bounds:
    for ax in (ax0, ax1, ax2):
        if lo > 0:
            ax.axvline(lo, color='0.6', lw=0.4, zorder=0)
        if lab == 'OFF':
            ax.axvspan(lo, hi, color='0.5', alpha=0.06, lw=0, zorder=0)
    ax0.text(lo + 0.3 if ha == 'left' else (lo + hi) / 2, 1.04, lab, transform=ax0.get_xaxis_transform(), ha=ha,
             va='bottom', color='0.35')  # 'down' (5 s) is narrower than its label: left-aligned at its start

# top strip: reservation times
bp = ax0.boxplot([res['c'], res['b']], positions=[0, 1], vert=False, widths=0.55, whis=(5, 95), showfliers=False,
                 patch_artist=True, medianprops={'color': 'black', 'lw': 1.0}, whiskerprops={'lw': 0.7}, capprops={'lw': 0.7})
for patch, a in zip(bp['boxes'], 'cb'):
    patch.set(facecolor=COLORS[a], alpha=0.6, lw=0.7)
ax0.set_yticks([0, 1], ['xApp', 'Reactive'])
ax0.set_ylim(-0.6, 1.6)
ax0.tick_params(axis='y', length=0)
ax0.spines[['top', 'right', 'left']].set_visible(False)

# middle: cell PRB
lo25, med, hi75 = prb_q['a']
ax1.fill_between(GRID, lo25, hi75, color=COLORS['a'], alpha=0.25, lw=0)
for a, n in (('a', 'No xApp'), ('c', 'xApp')):
    ax1.plot(GRID, prb_q[a][1], color=COLORS[a], lw=1.2, label=n)
ax1.axhline(80, color='black', lw=0.6, ls=':')
ax1.text(XMAX - 1, 81, '80 %', ha='right', va='bottom')
for ax in (ax0, ax1, ax2):
    ax.axvline(q(onset_a)[1], color='black', lw=0.8, ls='--', zorder=4)
ax1.text(q(onset_a)[1] - 0.8, 4, 'onset', ha='right', va='bottom', rotation=90)
ax1.set_ylim(0, 105)
ax1.set_ylabel('Cell DL PRB (%)')
ax1.legend(loc='lower left', bbox_to_anchor=(0.38, 0.03), frameon=False, handlelength=1.6)

# bottom: UE3 RTT
for a, n in ARMS:
    lo25, med, hi75 = rtt_q[a]
    x = SEC + 0.5
    ax2.fill_between(x, lo25, hi75, color=COLORS[a], alpha=0.2, lw=0)
    ax2.plot(x, med, color=COLORS[a], lw=1.1, label=n)
ax2.set_yscale('log')
ax2.set_ylim(5, 1500)
ax2.set_yticks([10, 30, 100, 300, 1000], ['10', '30', '100', '300', '1000'])
ax2.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
ax2.set_ylabel('Edge-UE p95 RTT (ms)')
ax2.set_xlabel('Time from ramp-up start (s)')
ax2.set_xlim(0, XMAX)
ax2.legend(ncol=3, loc='upper left', frameon=False, handlelength=1.4, columnspacing=0.9, borderaxespad=0.1)
for ax in (ax1, ax2):
    ax.spines[['top', 'right']].set_visible(False)
fig.align_ylabels((ax0, ax1, ax2))
fig.tight_layout(pad=0.3, h_pad=0.35)
out = os.path.join(HERE, 'cycle_timeline.pdf')
fig.savefig(out)
json.dump(summary, open(os.path.join(HERE, 'cycle_timeline.json'), 'w'), indent=1)
print(out)
print(json.dumps(summary, indent=1))
