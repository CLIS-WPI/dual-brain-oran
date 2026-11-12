"""Fig. 4: Exp-03 arms (a) no xApp, (b) reactive (threshold), (c) xApp (ML), per cycle in the evaluation window.
Top: edge-UE (UE3) ping p95 RTT (log); bottom: center-UE combined goodput. Box + jittered points, 39 cycles per arm.
usage: python3 r1/paper/fig4_edge_ue_outcome.py [batch_dir]  ->  r1/paper/edge_ue_outcome.pdf (vector, 3.5 in wide)"""
import csv
import os
import sys

import matplotlib
import matplotlib.ticker

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
BATCH = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, '..', 'results', 'exp03', 'exp03-20261106b')
ARMS = [('a', 'No xApp'), ('b', 'Reactive'), ('c', 'xApp')]  # paper wording
COLORS = {'a': '#E69F00', 'b': '#56B4E9', 'c': '#009E73'}  # Okabe-Ito (colour-blind safe)

rows = list(csv.DictReader(open(os.path.join(BATCH, 'exp03_cycles.csv'))))
data = {a: {k: np.array([float(r[k]) for r in rows if r['arm'] == a])
            for k in ('eval_ue3_rtt_p95_ms', 'eval_center_goodput_mbps')} for a, _ in ARMS}
for a, _ in ARMS:
    assert len(data[a]['eval_ue3_rtt_p95_ms']) == 39, (a, len(data[a]['eval_ue3_rtt_p95_ms']))

plt.rcParams.update({'font.size': 8, 'axes.labelsize': 8, 'xtick.labelsize': 8, 'ytick.labelsize': 8,
                     'pdf.fonttype': 42, 'ps.fonttype': 42, 'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans'], 'axes.linewidth': 0.6})
fig, axes = plt.subplots(2, 1, figsize=(3.5, 3.3), sharex=True)
rng = np.random.default_rng(0)
for ax, key, ylab, log in ((axes[0], 'eval_ue3_rtt_p95_ms', 'Edge-UE p95 RTT (ms)', True),
                           (axes[1], 'eval_center_goodput_mbps', 'Center-UE goodput (Mbps)', False)):
    vals = [data[a][key] for a, _ in ARMS]
    ax.boxplot(vals, positions=range(3), widths=0.5, whis=(5, 95), showfliers=False, patch_artist=True,
               medianprops={'color': 'black', 'lw': 1.0},
               boxprops={'facecolor': 'none', 'lw': 0.7}, whiskerprops={'lw': 0.7}, capprops={'lw': 0.7})
    for i, (a, _) in enumerate(ARMS):
        ax.scatter(i + rng.uniform(-0.18, 0.18, len(data[a][key])), data[a][key], s=7, color=COLORS[a],
                   alpha=0.75, lw=0, zorder=3)
    ax.set_ylabel(ylab)
    if log:
        ax.set_yscale('log')
        ax.set_ylim(10, 1500)
        ax.set_yticks([10, 30, 100, 300, 1000], ['10', '30', '100', '300', '1000'])
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.grid(axis='y', lw=0.3, alpha=0.5, which='major')
    ax.spines[['top', 'right']].set_visible(False)
axes[1].set_xticks(range(3), [f'({a}) {n}' for a, n in ARMS])
fig.align_ylabels(axes)
fig.tight_layout(pad=0.3, h_pad=0.4)
out = os.path.join(HERE, 'edge_ue_outcome.pdf')
fig.savefig(out)
print(out, {a: (float(np.median(data[a]['eval_ue3_rtt_p95_ms'])), float(np.median(data[a]['eval_center_goodput_mbps'])))
            for a, _ in ARMS})
