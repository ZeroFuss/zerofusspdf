"""The quality dashboard: geometry, recall/precision, type F1, semantics, OCR, autofill.

Every function here is pure and deterministic. See ``docs/QA.md`` for the metric
definitions and the thresholds the integration gates enforce against the synthetic
corpus.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple


def match_fields(pred: Sequence[Any], truth: Sequence[Any],
                 iou_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """Greedy highest-IoU matching between predictions and ground truth, same page only."""
    candidates: List[Tuple[float, int, int]] = []
    for i, p in enumerate(pred):
        for j, t in enumerate(truth):
            if getattr(p, "page", 0) != getattr(t, "page", 0):
                continue
            p_rect = getattr(p, "rect", None)
            t_rect = getattr(t, "rect", None)
            if p_rect is None or t_rect is None:
                continue
            iou = p_rect.iou(t_rect)
            if iou >= iou_threshold:
                candidates.append((iou, i, j))
    candidates.sort(key=lambda c: (-c[0], c[1], c[2]))

    used_pred: set = set()
    used_truth: set = set()
    matches: List[Tuple[int, int, float]] = []
    for iou, i, j in candidates:
        if i in used_pred or j in used_truth:
            continue
        used_pred.add(i)
        used_truth.add(j)
        matches.append((i, j, iou))
    matches.sort(key=lambda m: (m[0], m[1]))
    return matches


def field_geometry_metrics(pred: Sequence[Any], truth: Sequence[Any],
                           iou_threshold: float = 0.5) -> Dict[str, Any]:
    matches = match_fields(pred, truth, iou_threshold)
    if not matches:
        return {"mean_iou": 0.0, "median_iou": 0.0, "mean_center_error": 0.0,
               "edge_mae": {"x0": 0.0, "y0": 0.0, "x1": 0.0, "y1": 0.0}, "n": 0}

    ious = [m[2] for m in matches]
    center_errors = []
    edge_errors = {"x0": [], "y0": [], "x1": [], "y1": []}
    for i, j, _iou in matches:
        p_rect = pred[i].rect
        t_rect = truth[j].rect
        pc, tc = p_rect.center, t_rect.center
        center_errors.append(pc.distance_to(tc))
        edge_errors["x0"].append(abs(p_rect.x0 - t_rect.x0))
        edge_errors["y0"].append(abs(p_rect.y0 - t_rect.y0))
        edge_errors["x1"].append(abs(p_rect.x1 - t_rect.x1))
        edge_errors["y1"].append(abs(p_rect.y1 - t_rect.y1))

    sorted_ious = sorted(ious)
    n = len(sorted_ious)
    median = sorted_ious[n // 2] if n % 2 else (sorted_ious[n // 2 - 1] + sorted_ious[n // 2]) / 2
    deciles = [sorted_ious[min(n - 1, int(n * d / 10))] for d in range(1, 10)]

    return {
        "mean_iou": sum(ious) / len(ious), "median_iou": median, "iou_deciles": deciles,
        "mean_center_error": sum(center_errors) / len(center_errors),
        "edge_mae": {k: (sum(v) / len(v) if v else 0.0) for k, v in edge_errors.items()},
        "n": len(matches),
    }


def recall_precision(pred: Sequence[Any], truth: Sequence[Any],
                     iou_threshold: float = 0.5) -> Dict[str, float]:
    matches = match_fields(pred, truth, iou_threshold)
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(truth) - tp
    recall = tp / len(truth) if truth else (1.0 if not pred else 0.0)
    precision = tp / len(pred) if pred else (1.0 if not truth else 0.0)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"recall": recall, "precision": precision, "f1": f1, "tp": tp, "fp": fp, "fn": fn}


def type_macro_f1(pred: Sequence[Any], truth: Sequence[Any],
                  iou_threshold: float = 0.5) -> Dict[str, Any]:
    matches = match_fields(pred, truth, iou_threshold)
    per_type: Dict[str, Dict[str, int]] = {}
    for i, j, _iou in matches:
        p_type = str(getattr(pred[i], "field_type", "unknown"))
        t_type = str(getattr(truth[j], "field_type", "unknown"))
        per_type.setdefault(t_type, {"tp": 0, "fp": 0, "fn": 0})
        per_type.setdefault(p_type, {"tp": 0, "fp": 0, "fn": 0})
        if p_type == t_type:
            per_type[t_type]["tp"] += 1
        else:
            per_type[t_type]["fn"] += 1
            per_type[p_type]["fp"] += 1

    f1s = []
    table = {}
    for t, counts in per_type.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * prec * rec / (prec + rec)) if (prec + rec) else 0.0
        table[t] = {"precision": prec, "recall": rec, "f1": f1, **counts}
        f1s.append(f1)
    macro = sum(f1s) / len(f1s) if f1s else 0.0
    return {"macro_f1": macro, "per_type": table}


def label_association_rate(pred: Sequence[Any], truth: Sequence[Any],
                           iou_threshold: float = 0.5) -> float:
    matches = match_fields(pred, truth, iou_threshold)
    if not matches:
        return 0.0
    correct = 0
    for i, j, _iou in matches:
        p_label = getattr(pred[i], "visible_label", None)
        t_label = getattr(truth[j], "visible_label", None) or getattr(truth[j], "label", None)
        if p_label is not None and t_label is not None and \
                p_label.strip().lower() == t_label.strip().lower():
            correct += 1
    return correct / len(matches)


def canonical_accuracy(pred: Sequence[Any], truth: Sequence[Any],
                       iou_threshold: float = 0.5) -> float:
    matches = match_fields(pred, truth, iou_threshold)
    if not matches:
        return 0.0
    correct = 0
    for i, j, _iou in matches:
        p_key = getattr(pred[i], "canonical_key", None)
        t_key = getattr(truth[j], "canonical_key", None)
        if p_key is not None and p_key == t_key:
            correct += 1
    return correct / len(matches)


def _levenshtein(a: Sequence[Any], b: Sequence[Any]) -> int:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        curr = [i] + [0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev = curr
    return prev[lb]


def ocr_cer_wer(hyp: str, ref: str) -> Dict[str, float]:
    char_dist = _levenshtein(hyp, ref)
    cer = char_dist / len(ref) if ref else (0.0 if not hyp else 1.0)
    hyp_words, ref_words = hyp.split(), ref.split()
    word_dist = _levenshtein(hyp_words, ref_words)
    wer = word_dist / len(ref_words) if ref_words else (0.0 if not hyp_words else 1.0)
    return {"cer": cer, "wer": wer, "char_distance": char_dist, "word_distance": word_dist}


def autofill_exact_match(filled: Sequence[Any], expected: Dict[str, str]) -> Dict[str, float]:
    if not expected:
        return {"exact_match_rate": 0.0, "n": 0}
    correct = 0
    n = 0
    for f in filled:
        key = getattr(f, "canonical_key", None) or getattr(f, "field_name", None)
        if key not in expected:
            continue
        n += 1
        value = getattr(f, "value", None)
        if value is not None and value == expected[key]:
            correct += 1
    return {"exact_match_rate": (correct / n) if n else 0.0, "n": n, "correct": correct}


def repeat_consistency(groups: Sequence[Sequence[Any]]) -> float:
    if not groups:
        return 1.0
    consistent = 0
    for group in groups:
        values = {getattr(m, "value", None) for m in group}
        values.discard(None)
        if len(values) <= 1:
            consistent += 1
    return consistent / len(groups)


@dataclass
class MetricsDashboard:
    geometry: Dict[str, Any] = field(default_factory=dict)
    recall_precision: Dict[str, float] = field(default_factory=dict)
    type_f1: Dict[str, Any] = field(default_factory=dict)
    label_association: float = 0.0
    canonical_accuracy: float = 0.0
    ocr: Dict[str, float] = field(default_factory=dict)
    autofill: Dict[str, float] = field(default_factory=dict)
    repeat_consistency: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        return {
            "geometry": self.geometry, "recall_precision": self.recall_precision,
            "type_f1": self.type_f1, "label_association": self.label_association,
            "canonical_accuracy": self.canonical_accuracy, "ocr": self.ocr,
            "autofill": self.autofill, "repeat_consistency": self.repeat_consistency,
        }

    def render_text(self) -> str:
        lines = ["ZFP quality dashboard", "=" * 40]
        rp = self.recall_precision
        lines.append("Field recall:          %.4f" % rp.get("recall", 0.0))
        lines.append("Field precision:       %.4f" % rp.get("precision", 0.0))
        lines.append("False positives (fp):  %d" % rp.get("fp", 0))
        lines.append("Field geometry IoU:    mean %.4f  median %.4f" % (
            self.geometry.get("mean_iou", 0.0), self.geometry.get("median_iou", 0.0)))
        lines.append("Field geometry center error: %.4f" % self.geometry.get("mean_center_error", 0.0))
        lines.append("Field type macro-F1:   %.4f" % self.type_f1.get("macro_f1", 0.0))
        lines.append("Label association:     %.4f" % self.label_association)
        lines.append("Canonical semantics:   %.4f" % self.canonical_accuracy)
        lines.append("OCR CER/WER:           %.4f / %.4f" % (
            self.ocr.get("cer", 0.0), self.ocr.get("wer", 0.0)))
        lines.append("Auto-fill exact match: %.4f" % self.autofill.get("exact_match_rate", 0.0))
        lines.append("Repeat consistency:    %.4f" % self.repeat_consistency)
        return "\n".join(lines)

    def compare(self, other: "MetricsDashboard") -> str:
        lines = ["Metric deltas (this - other)", "=" * 40]
        a, b = self.as_dict(), other.as_dict()
        for key in ("label_association", "canonical_accuracy", "repeat_consistency"):
            lines.append("%-24s %+.4f" % (key, a[key] - b[key]))
        for key in ("recall", "precision", "f1"):
            lines.append("%-24s %+.4f" % (key, a["recall_precision"].get(key, 0.0) -
                                          b["recall_precision"].get(key, 0.0)))
        lines.append("%-24s %+.4f" % ("mean_iou", a["geometry"].get("mean_iou", 0.0) -
                                      b["geometry"].get("mean_iou", 0.0)))
        return "\n".join(lines)


def evaluate(pred_candidates: Sequence[Any], truth_fields: Sequence[Any], *,
            filled: Optional[Sequence[Any]] = None,
            expected: Optional[Dict[str, str]] = None,
            ocr_hyp: str = "", ocr_ref: str = "",
            repeat_groups: Optional[Sequence[Sequence[Any]]] = None,
            iou_threshold: float = 0.5) -> MetricsDashboard:
    dash = MetricsDashboard()
    dash.geometry = field_geometry_metrics(pred_candidates, truth_fields, iou_threshold)
    dash.recall_precision = recall_precision(pred_candidates, truth_fields, iou_threshold)
    dash.type_f1 = type_macro_f1(pred_candidates, truth_fields, iou_threshold)
    dash.label_association = label_association_rate(pred_candidates, truth_fields, iou_threshold)
    dash.canonical_accuracy = canonical_accuracy(pred_candidates, truth_fields, iou_threshold)
    dash.ocr = ocr_cer_wer(ocr_hyp, ocr_ref) if (ocr_hyp or ocr_ref) else {}
    dash.autofill = autofill_exact_match(filled or [], expected or {})
    dash.repeat_consistency = repeat_consistency(repeat_groups or [])
    return dash


__all__ = [
    "match_fields", "field_geometry_metrics", "recall_precision", "type_macro_f1",
    "label_association_rate", "canonical_accuracy", "ocr_cer_wer", "autofill_exact_match",
    "repeat_consistency", "MetricsDashboard", "evaluate",
]
