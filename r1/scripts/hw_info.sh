#!/usr/bin/env bash
# Hardware record → r1/env/hardware.json (CPU model/cores, RAM, GPUs, driver, kernel). Run once per host.
set -euo pipefail
OUT="${1:-$(cd "$(dirname "$0")/.." && pwd)/env/hardware.json}"
python3 - "$OUT" <<'EOF'
import json, os, platform, subprocess, sys
def sh(*a):
    try:
        return subprocess.run(a, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ''
cpu = next((l.split(':', 1)[1].strip() for l in open('/proc/cpuinfo') if l.startswith('model name')), '')
mem_kb = next(int(l.split()[1]) for l in open('/proc/meminfo') if l.startswith('MemTotal'))
gpus = [dict(zip(('index', 'name', 'memory_total_mib', 'driver'), [x.strip() for x in l.split(',')]))
        for l in sh('nvidia-smi', '--query-gpu=index,name,memory.total,driver_version', '--format=csv,noheader,nounits').splitlines()]
rec = {'hostname': platform.node(), 'kernel': platform.release(), 'os': sh('lsb_release', '-ds') or platform.platform(),
       'cpu_model': cpu, 'cpu_logical_cores': os.cpu_count(), 'cpu_sockets': sh('sh', '-c', "lscpu | awk -F: '/Socket\\(s\\)/{print $2}'").strip(),
       'ram_gib': round(mem_kb / 2**20, 1), 'gpus': gpus, 'cuda_version': sh('sh', '-c', "nvidia-smi | grep -o 'CUDA Version: [0-9.]*'"),
       'docker': sh('docker', '--version'), 'kubernetes': sh('kubectl', 'version', '--client', '-o', 'json')[:400]}
json.dump(rec, open(sys.argv[1], 'w'), indent=2)
print('wrote', sys.argv[1], '|', cpu, '|', os.cpu_count(), 'cores |', rec['ram_gib'], 'GiB |', [g['name'] for g in gpus])
EOF
