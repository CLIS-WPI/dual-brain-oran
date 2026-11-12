"""Request-scoped experiment context for the per-experiment RAN.

Every Canvas experiment with ``ran_mode = 'isolated'`` gets its own gNB, UEs,
traffic generator and Near-RT RIC (an ``OaiExperiment`` installed by the Operator),
named ``<tenant>-<slug>-…`` and labelled ``harmony.telecom/experiment=<slug>``.
Legacy experiments keep the shared ``<tenant>-rfsim-*`` / ``<tenant>-ric`` stack.

Bridges never build RAN/RIC names themselves: they call the helpers below, which
read the experiment from a ContextVar set per request (``X-Harmony-Experiment``
header, see main.py) or explicitly via :func:`use_experiment`. No context = legacy.
See docs/specs/experiment-isolation.md.
"""
from __future__ import annotations

import hashlib
import re
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator

EXPERIMENT_LABEL = "harmony.telecom/experiment"
_SLUG_RE = re.compile(r"^[a-z][a-z0-9]{2,11}$")


@dataclass(frozen=True)
class ExperimentRef:
    experiment_id: str
    slug: str


_current: ContextVar[ExperimentRef | None] = ContextVar("harmony_experiment", default=None)


def slug_for(experiment_id: str) -> str:
    """Deterministic DNS-safe slug for an experiment id (matches the CRD pattern)."""
    digest = hashlib.sha1(str(experiment_id).encode()).hexdigest()
    return "x" + digest[:7]


def valid_slug(slug: str) -> bool:
    return bool(_SLUG_RE.match(slug or ""))


def current() -> ExperimentRef | None:
    return _current.get()


def current_slug() -> str:
    ref = _current.get()
    return ref.slug if ref else ""


def set_current(ref: ExperimentRef | None):
    """Set for the rest of the current task; returns a token for :func:`reset`."""
    return _current.set(ref)


def reset(token) -> None:
    _current.reset(token)


@contextmanager
def use_experiment(ref: ExperimentRef | None) -> Iterator[None]:
    token = _current.set(ref)
    try:
        yield
    finally:
        _current.reset(token)


# ── names ────────────────────────────────────────────────────────────────────

def rfsim_fullname(tenant: str, slug: str | None = None) -> str:
    """Matches the oai-rfsim chart's ``fullname`` helper."""
    s = current_slug() if slug is None else slug
    return f"{tenant}-{s}-rfsim" if s else f"{tenant}-rfsim"


def rfsim_gnb_name(tenant: str) -> str:
    return f"{rfsim_fullname(tenant)}-gnb"


def rfsim_trf_name(tenant: str) -> str:
    return f"{rfsim_fullname(tenant)}-trf"


def rfsim_ue_name(tenant: str, ue_name: str) -> str:
    return f"{rfsim_fullname(tenant)}-ue-{ue_name}"


def ric_name(tenant: str, slug: str | None = None) -> str:
    """Matches the oai-ric chart's ``fullname`` helper (Deployment + headless Service)."""
    s = current_slug() if slug is None else slug
    return f"{tenant}-{s}-ric" if s else f"{tenant}-ric"


# ── label selectors ──────────────────────────────────────────────────────────

def experiment_selector_clause(slug: str | None = None) -> str:
    """Selector term scoping pods to this experiment — or, for legacy, to *no*
    experiment (otherwise legacy lookups also match experiment pods)."""
    s = current_slug() if slug is None else slug
    return f"{EXPERIMENT_LABEL}={s}" if s else f"!{EXPERIMENT_LABEL}"


def selector(component: str, tenant: str, slug: str | None = None) -> str:
    return (
        f"app.kubernetes.io/component={component},harmony.telecom/tenant={tenant},"
        f"{experiment_selector_clause(slug)}"
    )
