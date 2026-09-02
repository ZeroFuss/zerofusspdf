"""Quality assurance: verification, render/structural diffing, and metrics.

Adversarial by design -- this package's job is to fail a run, not to bless it.
"""

from __future__ import annotations

from .metrics import MetricsDashboard, evaluate
from .verify import QAFinding, QAReport, verify_document

__all__ = ["QAFinding", "QAReport", "verify_document", "MetricsDashboard", "evaluate"]
