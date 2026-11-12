"""Unit test of speed_rules (run: python3 -I r1/collector/test_speed_rules.py, or pytest)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from speed_rules import cycle_means, drift_check, start_gate_ok  # noqa: E402

CYCLE = 90.0  # s: 45 up, 10 hold, 5 down, 30 OFF (frozen profile)


def series(n_cycles, step_from=None, step_factor=1.0, base=1.45):
    """10 s speed means with the ~10 % within-cycle phase effect: ramp-up 1.0x, hold/down 1.11x, OFF 1.12x of base."""
    out = []
    for c in range(n_cycles):
        for k in range(9):  # nine 10 s windows per 90 s cycle
            t = c * CYCLE + k * 10
            phase = 1.0 if k < 5 else 1.11 if k < 6 else 1.12
            f = step_factor if step_from is not None and c >= step_from else 1.0
            out.append((t, base * phase * f))
    return out


def windows(n):
    return [(c * CYCLE, (c + 1) * CYCLE) for c in range(n)]


def test_start_gate():
    assert start_gate_ok(1.73, 1.70) and start_gate_ok(1.54, 1.70) is True  # 1.54 = -9.4 %
    assert not start_gate_ok(1.446, 1.70)  # exp04-dryrun4 start (-14.9 %): passed the old ±15 % gate
    assert not start_gate_ok(1.35, 1.70)  # the loaded calibration speed is NOT an idle speed (-21 %)
    assert not start_gate_ok(None, 1.70)
    assert start_gate_ok(1.886, 1.70) and start_gate_ok(2.5, 1.70)  # one-sided (2026-11-05): fast is not contention
    assert not start_gate_ok(1.529, 1.70)


def test_phase_effect_does_not_trip():
    s = series(13)
    within = [v for t, v in s if t < CYCLE]
    assert max(within) / min(within) - 1 > 0.10  # the phase effect alone exceeds the old ±10 % window rule
    assert drift_check(cycle_means(s, windows(13))) is None


def test_host_load_step_trips():
    s = series(13, step_from=5, step_factor=0.82)  # host load step from cycle 6: everything 18 % slower
    reason = drift_check(cycle_means(s, windows(13)))
    assert reason and 'cycles 6–7' in reason, reason


def test_single_bad_cycle_does_not_trip():
    m = [1.5, 1.5, 1.2, 1.5, 1.5]  # one transient cycle out of bounds
    assert drift_check(m) is None
    assert drift_check([1.5, 1.5, 1.2, 1.3]) is not None


def test_gnb_log_coverage():
    from speed_rules import GNB_COVERAGE_MIN, gnb_log_coverage
    full = [1000 + i for i in range(1168)]
    assert gnb_log_coverage(full, 1000, 2168) >= GNB_COVERAGE_MIN
    assert gnb_log_coverage(full[-41:], 1000, 2168) < GNB_COVERAGE_MIN  # Exp-03 b3: last 41 s only
    assert gnb_log_coverage(full + [999, 3000], 1000, 2168) <= 1.0  # rows outside the run are not counted
    assert gnb_log_coverage([], 1000, 1000) == 0.0


if __name__ == '__main__':
    for name, fn in list(globals().items()):
        if name.startswith('test_'):
            fn()
            print('ok', name)


def test_regime_check():
    from speed_rules import regime_check
    assert regime_check([1.6, 1.7], 1.635)['in_regime']
    assert not regime_check([1.22, 1.23], 1.635)['in_regime']
    assert not regime_check([None], 1.635)['in_regime']
