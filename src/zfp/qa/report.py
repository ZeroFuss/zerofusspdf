"""Plain-text and JSON rendering of QA reports and metric dashboards."""

from __future__ import annotations

import json
from typing import Any

from .metrics import MetricsDashboard
from .verify import QAReport


def render_qa_report(report: QAReport) -> str:
    lines = ["QA report: %s" % ("PASSED" if report.passed else "FAILED"), "=" * 40]
    for f in report.findings:
        loc = ""
        if f.page is not None:
            loc += " page=%d" % f.page
        if f.field is not None:
            loc += " field=%r" % f.field
        lines.append("[%s] %s: %s%s" % (f.severity.upper(), f.code, f.message, loc))
    if report.metrics:
        lines.append("-" * 40)
        for k, v in report.metrics.items():
            lines.append("%s: %s" % (k, v))
    return "\n".join(lines)


def render_dashboard(dash: MetricsDashboard) -> str:
    return dash.render_text()


def to_json(obj: Any) -> str:
    if hasattr(obj, "as_dict"):
        obj = obj.as_dict()
    return json.dumps(obj, indent=2, sort_keys=True, default=str)


__all__ = ["render_qa_report", "render_dashboard", "to_json"]
