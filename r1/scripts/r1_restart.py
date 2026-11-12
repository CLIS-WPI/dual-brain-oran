"""Full R1 RAN restart in the only working order: RIC → gNB → UE1 → UE2 → UE3 (each UE after the previous PDU).

usage: r1_restart.py [--keep-ric]
The gNB resolves the RIC's headless service at start; without a RIC pod it falls back to 127.0.0.1 and E2
never connects. --keep-ric leaves a running RIC in place (still restarts gNB + UEs). Needs /tmp/portal_token.
"""
import sys
import time

import r1env as E


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def main(keep_ric):
    st, _ = E.req('GET', f'/api/experiments/{E.EXP}')
    log('auth', st)
    assert st == 200, 'portal token missing or expired'
    imsis = [u['imsi'] for u in E.UES]
    log('ue stop', E.req('POST', '/api/ue/virtual/batch/stop', {'imsis': imsis, 'count': len(imsis)}, E.H, timeout=300)[0])
    log('ran stop', E.req('POST', '/api/ran/stop', {'mode': 'rfsim'}, E.H, timeout=300)[0])
    if not keep_ric:
        log('ric stop', E.req('POST', '/api/ric/flexric/stop', None, E.H, timeout=300)[0])
    E.k('wait', '--for=delete', 'pod', '-l',
        f'harmony.telecom/experiment={E.SLUG},app.kubernetes.io/component in (gnb-rfsim,virtual-ue)', '--timeout=180s')
    st, r = E.req('POST', '/api/ric/flexric/start', None, E.H, timeout=300)
    log('ric start', st)
    log(E.k('rollout', 'status', f'deploy/test-1-{E.SLUG}-ric', '--timeout=240s').strip())
    st, r = E.req('POST', '/api/ran/start', {'mode': 'rfsim', 'config_file': E.ENV['gnb_conf'], 'startup_cleanup': True},
                  E.H, timeout=300)
    log('ran start', st)
    log(E.k('rollout', 'status', f'deploy/test-1-{E.SLUG}-rfsim-gnb', '--timeout=240s').strip())
    ok = True
    for u in E.UES:
        ip = None
        for _ in range(20):
            st, r = E.req('POST', f"/api/ue/virtual/{u['imsi']}/start", {'interface': ''}, E.H, timeout=300)
            ip = (r or {}).get('ip') if isinstance(r, dict) else None
            if ip:
                break
            time.sleep(6)
        log(u['label'], 'PDU up' if ip else 'NO PDU')
        ok &= bool(ip)
    gnb = E.pod('gnb-rfsim')[0]
    e2 = [ln for ln in E.k('logs', gnb, '-c', 'gnb').splitlines() if 'nearRT-RIC IP Address' in ln or 'E2 SETUP RESPONSE' in ln]
    log('E2:', ' | '.join(x.strip() for x in e2[:2]) or 'no E2 lines')
    if not ok or not any('E2 SETUP RESPONSE' in x for x in e2) or any('127.0.0.1' in x for x in e2):
        sys.exit('restart incomplete: check PDUs / E2')


if __name__ == '__main__':
    main('--keep-ric' in sys.argv)
