"""Sim-speed rules for R1 measurement runs (approved 2026-11-03). Pure functions; unit test: test_speed_rules.py.

RFsim is not paced to real time; its speed depends mostly on host load (measured 2026-11-03: an idle cell and a
loaded cell run at about the same speed, ~1.5-1.8x), and within one 90 s traffic cycle the phase moves the speed
by ~10 % (ramp-up slower than hold/OFF). So each rule compares like with like:

  start gate   idle speed (no traffic) vs the IDLE reference (r1/env/experiment.json idle_reference_sim_speed),
               within ±START_TOL (10 %). Otherwise wait and retry, without limit (every skipped start is logged).
  regime       run level: mean of the per-cycle loaded speeds within ±REGIME_TOL of the exp01 loaded mean
               (loaded_reference_sim_speed); outside → the run is marked off-regime (kept, excluded from the paper).
  drift guard  per-cycle mean speed (all phases of one traffic cycle) vs cycle 1's mean, within ±DRIFT_TOL;
               abort after DRIFT_STRIKES consecutive cycles out of bounds (host-load change, not the profile).
The 10 s series is recorded for the report but no longer used for the abort decision.
"""
from __future__ import annotations

START_TOL = 0.10  # 2026-11-04 (Milad): was 0.15; runs start only in the calibrated regime
REGIME_TOL = 0.10
DRIFT_TOL = 0.10
DRIFT_STRIKES = 2
# gNB ground-truth log coverage (2026-11-09, Milad, after Exp-03 b3: only the last 41 s of its gNB log survived, most
# likely container-log rotation): a run whose [AURA-PRB] 1 s rows cover < 90 % of its seconds has no valid PRB data.
GNB_COVERAGE_MIN = 0.90


def start_gate_ok(idle_speed: float | None, idle_reference: float, tol: float = START_TOL) -> bool:
    """Idle check, ONE-sided since 2026-11-05 17:15 EST (Milad): fail only below idle_reference·(1 − tol) = 1.53.
    Contention only slows the simulator; "too fast" came from our own NUMA pin (idle 1.85-1.89 > 1.87). The loaded
    probe and the run-level regime check stay two-sided."""
    return bool(idle_speed) and idle_speed / idle_reference - 1 >= -tol


def regime_check(means: list[float | None], loaded_reference: float, tol: float = REGIME_TOL) -> dict:
    """Run-level validity: the mean of the per-cycle loaded speeds must lie within ±tol of the exp01 loaded mean
    (env/experiment.json loaded_reference_sim_speed). Outside → 'off-regime': kept, excluded from paper numbers."""
    v = [m for m in means if m is not None]
    mean = sum(v) / len(v) if v else None
    ok = mean is not None and abs(mean / loaded_reference - 1) <= tol
    return {'mean_cycle_speed': mean and round(mean, 4), 'loaded_reference': loaded_reference, 'tol': tol,
            'deviation': mean and round(mean / loaded_reference - 1, 4), 'in_regime': ok}


def cycle_means(series: list[tuple[float, float]], cycles: list[tuple[float, float]]) -> list[float | None]:
    """Mean speed per cycle window [start, end) from a (t, speed) series (any resolution)."""
    out = []
    for a, b in cycles:
        v = [s for t, s in series if a <= t < b]
        out.append(sum(v) / len(v) if v else None)
    return out


def drift_check(means: list[float | None], tol: float = DRIFT_TOL, strikes: int = DRIFT_STRIKES) -> str | None:
    """Abort reason if `strikes` consecutive complete cycles deviate > tol from cycle 1's mean, else None."""
    if not means or means[0] is None:
        return None
    base, run = means[0], 0
    for i, m in enumerate(means[1:], 2):
        bad = m is not None and abs(m / base - 1) > tol
        run = run + 1 if bad else 0
        if run >= strikes:
            return (f'cycle mean sim/wall speed {m:.3f}x vs cycle 1 {base:.3f}x ({m / base - 1:+.0%}, limit '
                    f'±{tol:.0%}) for {strikes} consecutive cycles (cycles {i - strikes + 1}–{i}) — host load changed')
    return None


def gnb_log_coverage(times: list[float], t0: float, t_end: float) -> float:
    """Fraction of the run's seconds [t0, t_end] that have a gNB [AURA-PRB] 1 s row (capped at 1)."""
    if t_end <= t0:
        return 0.0
    return round(min(1.0, sum(1 for t in times if t0 <= t <= t_end) / (t_end - t0)), 4)
