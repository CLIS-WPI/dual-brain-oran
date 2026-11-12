"""Deploy an xApp configuration to the RIC byte for byte (Exp-04 registration, Exp-03 arms b/c/d).

Unlike services/dual_brain/ric.start (which builds config.json from a backend profile), this writes the given bytes
unchanged, verifies them on the RIC by sha256, writes the model files, starts the congestion xApp and waits until
KPM samples flow. Same retry policy as ric.start (stale RIC subscriptions). Returns (events offset, sha256 on RIC).
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import shlex

from services.dual_brain import ric


async def deploy(profile: dict, config_bytes: bytes, files: dict[str, bytes], attempts: int = 6) -> tuple[int, str]:
    wd = profile['ric']['workdir']
    lease = json.loads(config_bytes).get('lease_s')
    if not isinstance(lease, (int, float)) or not 10 <= lease <= 120:  # R1: never deploy without the lease rule
        raise RuntimeError(f'config.json lease_s {lease!r} missing or outside [10, 120] s')
    want = hashlib.sha256(config_bytes).hexdigest()
    await ric.stop()
    for name, data in files.items():
        await ric.write_file(f'{wd}/{name}', data)
    await ric.write_file(f'{wd}/config.json', config_bytes)
    got = (await ric._exec(f'sha256sum {shlex.quote(wd)}/config.json')).split()[0]
    if got != want:
        raise RuntimeError(f'config.json on the RIC differs from the rendered file ({got} != {want})')
    for name, data in files.items():  # model files must also arrive unchanged
        g = (await ric._exec(f'sha256sum {shlex.quote(wd)}/{shlex.quote(name)}')).split()[0]
        if g != hashlib.sha256(data).hexdigest():
            raise RuntimeError(f'{name} on the RIC differs from the local file')
    offset = int((await ric._exec(f'wc -c < {shlex.quote(wd)}/events.jsonl 2>/dev/null || echo 0')).strip() or 0)
    for attempt in range(1, attempts + 1):
        res = await ric.xapp_bridge.start_k8s_xapp(ric.XAPP, duration=-1)
        if not res.get('success'):
            raise RuntimeError(f"{res.get('message')}: {res.get('error') or ''}")
        if await ric._samples_flowing(profile, offset):
            return offset, got
        await ric.stop()
        if attempt < attempts:
            await asyncio.sleep(20)
    raise RuntimeError(f'{ric.XAPP} started but no KPM indications arrived ({attempts} tries)')
