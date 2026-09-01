"""Unit tests for :mod:`zfp.core.config`, :mod:`zfp.core.logging` and
:mod:`zfp.core.optional`."""

from __future__ import annotations

import dataclasses
import io
import json
import logging as std_logging
import os
import tempfile
import unittest

from zfp.core import logging as zlog
from zfp.core import optional
from zfp.core.config import (
    SCORING_WEIGHT_NAMES,
    AutofillConfig,
    CouncilConfig,
    DetectionConfig,
    OcrConfig,
    OrchestratorConfig,
    PrivacyConfig,
    ScoringWeights,
    ZfpConfig,
)
from zfp.core.errors import UnsupportedFeatureError, ValidationError


class SectionDefaultsTest(unittest.TestCase):
    def test_detection_defaults(self) -> None:
        d = DetectionConfig()
        self.assertEqual(d.min_line_length_pt, 24.0)
        self.assertEqual(d.max_line_thickness_pt, 3.0)
        self.assertEqual(d.line_merge_tolerance_pt, 1.5)
        self.assertEqual(d.field_height_pt, 12.0)
        self.assertEqual(d.underline_gap_pt, 2.0)
        self.assertEqual(d.label_max_distance_pt, 120.0)
        self.assertEqual(d.checkbox_min_pt, 5.0)
        self.assertEqual(d.checkbox_max_pt, 22.0)
        self.assertEqual(d.checkbox_aspect_tolerance, 0.35)
        self.assertEqual(d.blank_min_width_pt, 40.0)
        self.assertEqual(d.blank_min_height_pt, 9.0)
        self.assertEqual(d.comb_cell_tolerance_pt, 2.0)
        self.assertEqual(d.min_candidate_confidence, 0.35)
        self.assertEqual(d.dedup_iou_threshold, 0.55)

    def test_ocr_defaults_and_isolation(self) -> None:
        a, b = OcrConfig(), OcrConfig()
        self.assertTrue(a.enabled)
        self.assertEqual(a.dpi, 300)
        self.assertEqual(a.engines, ["tesseract", "paddle"])
        self.assertEqual(a.languages, ["eng"])
        self.assertEqual(a.min_word_confidence, 0.55)
        self.assertEqual(a.escalate_below, 0.70)
        a.engines.append("kraken")
        a.languages.append("deu")
        self.assertEqual(b.engines, ["tesseract", "paddle"])
        self.assertEqual(b.languages, ["eng"])

    def test_privacy_defaults(self) -> None:
        a, b = PrivacyConfig(), PrivacyConfig()
        self.assertFalse(a.allow_external_inference)
        self.assertTrue(a.require_zero_data_retention)
        self.assertEqual(a.provider_allowlist, [])
        self.assertEqual(a.max_context_chars, 2000)
        self.assertTrue(a.redact_values_in_prompts)
        a.provider_allowlist.append("openrouter")
        self.assertEqual(b.provider_allowlist, [])

    def test_council_defaults(self) -> None:
        a, b = CouncilConfig(), CouncilConfig()
        self.assertTrue(a.enabled)
        self.assertEqual(a.quorum, 3)
        self.assertEqual(a.agreement_threshold, 0.66)
        self.assertEqual(a.escalate_below_confidence, 0.80)
        self.assertEqual(a.max_rounds, 2)
        self.assertEqual(a.providers, ["rules", "heuristic", "ontology"])
        a.providers.append("openrouter")
        self.assertEqual(b.providers, ["rules", "heuristic", "ontology"])

    def test_autofill_and_orchestrator_defaults(self) -> None:
        a = AutofillConfig()
        self.assertEqual(a.mode, "conservative")
        self.assertEqual(a.min_fill_confidence, 0.90)
        self.assertEqual(a.min_completion_confidence, 0.55)
        self.assertTrue(a.propagate_repeats)
        self.assertTrue(a.require_validation)
        o = OrchestratorConfig()
        self.assertEqual(o.max_workers, 8)
        self.assertEqual(o.page_shard_size, 4)
        self.assertEqual(o.stage_timeout_s, 300.0)
        self.assertFalse(o.fail_fast)
        self.assertTrue(o.deterministic)


class ScoringWeightsTest(unittest.TestCase):
    def test_default_weights_match_the_spec_formula(self) -> None:
        w = ScoringWeights()
        self.assertEqual(w.geometric_evidence, 0.30)
        self.assertEqual(w.blank_region_evidence, 0.20)
        self.assertEqual(w.nearby_label_evidence, 0.15)
        self.assertEqual(w.layout_consistency, 0.10)
        self.assertEqual(w.repeated_pattern_evidence, 0.10)
        self.assertEqual(w.semantic_type_confidence, 0.10)
        self.assertEqual(w.model_consensus, 0.05)
        self.assertAlmostEqual(w.total(), 1.0)

    def test_weight_names_match_dataclass_fields(self) -> None:
        self.assertEqual(SCORING_WEIGHT_NAMES, tuple(f.name for f in dataclasses.fields(ScoringWeights)))

    def test_normalized_sums_to_one(self) -> None:
        w = ScoringWeights(1, 2, 3, 4, 5, 6, 7)
        n = w.normalized()
        self.assertAlmostEqual(n.total(), 1.0)
        self.assertAlmostEqual(n.geometric_evidence, 1.0 / 28.0)
        self.assertAlmostEqual(n.model_consensus, 7.0 / 28.0)

    def test_normalized_preserves_ratios(self) -> None:
        w = ScoringWeights(0.6, 0.4, 0, 0, 0, 0, 0)
        n = w.normalized()
        self.assertAlmostEqual(n.geometric_evidence, 0.6)
        self.assertAlmostEqual(n.blank_region_evidence, 0.4)

    def test_normalized_of_default_is_stable(self) -> None:
        w = ScoringWeights()
        self.assertEqual(w.normalized(), w.normalized().normalized())

    def test_all_zero_weights_become_uniform(self) -> None:
        n = ScoringWeights(0, 0, 0, 0, 0, 0, 0).normalized()
        self.assertAlmostEqual(n.total(), 1.0)
        for name in SCORING_WEIGHT_NAMES:
            self.assertAlmostEqual(getattr(n, name), 1.0 / 7.0)

    def test_negative_weights_are_clamped(self) -> None:
        n = ScoringWeights(-1, 1, 0, 0, 0, 0, 0).normalized()
        self.assertEqual(n.geometric_evidence, 0.0)
        self.assertAlmostEqual(n.blank_region_evidence, 1.0)

    def test_score_is_the_weighted_sum(self) -> None:
        w = ScoringWeights()
        self.assertAlmostEqual(w.score({name: 1.0 for name in SCORING_WEIGHT_NAMES}), 1.0)
        self.assertAlmostEqual(w.score({}), 0.0)
        self.assertAlmostEqual(w.score({"geometric_evidence": 1.0}), 0.30)
        self.assertAlmostEqual(
            w.score({"geometric_evidence": 0.5, "nearby_label_evidence": 1.0, "unknown": 9.0}),
            0.15 + 0.15,
        )

    def test_score_uses_normalized_weights(self) -> None:
        w = ScoringWeights(3, 2, 1.5, 1, 1, 1, 0.5)  # 10x the defaults
        self.assertAlmostEqual(w.score({name: 1.0 for name in SCORING_WEIGHT_NAMES}), 1.0)
        self.assertAlmostEqual(w.score({"geometric_evidence": 1.0}), 0.30)

    def test_score_tolerates_bad_values(self) -> None:
        w = ScoringWeights()
        self.assertAlmostEqual(w.score({"geometric_evidence": None}), 0.0)
        self.assertAlmostEqual(w.score({"geometric_evidence": "nope"}), 0.0)


class ZfpConfigTest(unittest.TestCase):
    def test_default_is_constructible_without_arguments(self) -> None:
        self.assertEqual(ZfpConfig.default(), ZfpConfig())
        self.assertEqual(ZfpConfig.default().seed, 0)

    def test_sections_are_independent_between_instances(self) -> None:
        a, b = ZfpConfig.default(), ZfpConfig.default()
        a.ocr.engines.append("kraken")
        a.detection.field_height_pt = 99.0
        self.assertEqual(b.ocr.engines, ["tesseract", "paddle"])
        self.assertEqual(b.detection.field_height_pt, 12.0)

    def test_to_dict_is_json_serializable(self) -> None:
        d = ZfpConfig.default().to_dict()
        self.assertEqual(
            sorted(d),
            [
                "autofill",
                "council",
                "detection",
                "ocr",
                "orchestrator",
                "privacy",
                "scoring",
                "seed",
            ],
        )
        json.dumps(d)  # must not raise
        self.assertEqual(d["ocr"]["languages"], ["eng"])

    def test_round_trip(self) -> None:
        cfg = ZfpConfig.default()
        cfg.seed = 42
        cfg.detection.dedup_iou_threshold = 0.7
        cfg.privacy.provider_allowlist = ["local"]
        self.assertEqual(ZfpConfig.from_dict(cfg.to_dict()), cfg)

    def test_from_dict_accepts_partial_input(self) -> None:
        cfg = ZfpConfig.from_dict({"detection": {"field_height_pt": 20.0}})
        self.assertEqual(cfg.detection.field_height_pt, 20.0)
        self.assertEqual(cfg.detection.underline_gap_pt, 2.0)
        self.assertEqual(cfg.ocr, OcrConfig())
        self.assertEqual(cfg.seed, 0)

    def test_from_dict_ignores_unknown_keys(self) -> None:
        cfg = ZfpConfig.from_dict({"nonsense": 1, "detection": {"nonsense": 2}})
        self.assertEqual(cfg, ZfpConfig.default())

    def test_from_dict_rejects_non_mapping(self) -> None:
        with self.assertRaises(ValidationError):
            ZfpConfig.from_dict([1, 2, 3])  # type: ignore[arg-type]
        with self.assertRaises(ValidationError):
            ZfpConfig.from_dict({"detection": [1, 2]})
        with self.assertRaises(ValidationError):
            ZfpConfig.from_dict({"seed": "abc"})

    def test_from_dict_of_none_is_default(self) -> None:
        self.assertEqual(ZfpConfig.from_dict(None), ZfpConfig.default())  # type: ignore[arg-type]

    def test_from_file(self) -> None:
        cfg = ZfpConfig.default()
        cfg.seed = 11
        cfg.council.quorum = 5
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "zfp.json")
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(cfg.to_dict(), handle)
            self.assertEqual(ZfpConfig.from_file(path), cfg)

    def test_from_file_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            missing = os.path.join(tmp, "nope.json")
            with self.assertRaises(ValidationError):
                ZfpConfig.from_file(missing)
            bad = os.path.join(tmp, "bad.json")
            with open(bad, "w", encoding="utf-8") as handle:
                handle.write("{not json")
            with self.assertRaises(ValidationError):
                ZfpConfig.from_file(bad)

    def test_as_dict_alias(self) -> None:
        cfg = ZfpConfig.default()
        self.assertEqual(cfg.as_dict(), cfg.to_dict())


class LoggingTest(unittest.TestCase):
    def tearDown(self) -> None:
        zlog.reset()

    def test_get_logger_lives_under_the_zfp_root(self) -> None:
        self.assertEqual(zlog.get_logger("zfp.core.geometry").name, "zfp.core.geometry")
        self.assertEqual(zlog.get_logger("core.geometry").name, "zfp.core.geometry")
        self.assertEqual(zlog.get_logger().name, "zfp")
        self.assertEqual(zlog.get_logger("zfp").name, "zfp")

    def test_root_has_a_null_handler_by_default(self) -> None:
        root = zlog.get_logger()
        self.assertTrue(any(isinstance(h, std_logging.NullHandler) for h in root.handlers))

    def test_configure_is_idempotent(self) -> None:
        buf = io.StringIO()
        zlog.configure(level="INFO", stream=buf)
        zlog.configure(level="INFO", stream=buf)
        zlog.get_logger("zfp.core.test").info("once")
        self.assertEqual(buf.getvalue().count("once"), 1)

    def test_json_formatter_emits_context(self) -> None:
        buf = io.StringIO()
        zlog.configure(level="DEBUG", json=True, stream=buf)
        log = zlog.get_logger("zfp.core.test")
        with zlog.LogContext(document_id="doc-1", page=4, agent="OcrAgent"):
            log.info("recognized %d words", 12)
        record = json.loads(buf.getvalue().strip())
        self.assertEqual(record["document_id"], "doc-1")
        self.assertEqual(record["page"], 4)
        self.assertEqual(record["agent"], "OcrAgent")
        self.assertEqual(record["message"], "recognized 12 words")
        self.assertEqual(record["level"], "INFO")
        self.assertEqual(record["logger"], "zfp.core.test")

    def test_context_is_restored_on_exit(self) -> None:
        buf = io.StringIO()
        zlog.configure(level="DEBUG", json=True, stream=buf)
        log = zlog.get_logger("zfp.core.test")
        with zlog.LogContext(document_id="doc-1"):
            pass
        log.info("after")
        record = json.loads(buf.getvalue().strip())
        self.assertNotIn("document_id", record)
        self.assertEqual(zlog.current_context(), {})

    def test_context_is_restored_after_an_exception(self) -> None:
        try:
            with zlog.LogContext(document_id="doc-1"):
                raise RuntimeError("boom")
        except RuntimeError:
            pass
        self.assertEqual(zlog.current_context(), {})

    def test_contexts_nest_and_inner_wins(self) -> None:
        with zlog.LogContext(document_id="doc-1", page=1):
            with zlog.LogContext(page=2, agent="Inner"):
                ctx = zlog.current_context()
                self.assertEqual(ctx["document_id"], "doc-1")
                self.assertEqual(ctx["page"], 2)
                self.assertEqual(ctx["agent"], "Inner")
            self.assertEqual(zlog.current_context()["page"], 1)

    def test_plain_formatter_writes_text(self) -> None:
        buf = io.StringIO()
        zlog.configure(level="WARNING", stream=buf)
        log = zlog.get_logger("zfp.core.test")
        log.debug("hidden")
        log.warning("shown")
        out = buf.getvalue()
        self.assertIn("shown", out)
        self.assertNotIn("hidden", out)


class OptionalImportTest(unittest.TestCase):
    def test_present_module(self) -> None:
        mod = optional.optional_import("json")
        self.assertTrue(mod)
        self.assertIsNotNone(mod.module)
        self.assertIsNone(mod.error)
        self.assertEqual(mod.require("anything").__name__, "json")

    def test_attribute_import(self) -> None:
        mod = optional.optional_import("json", attr="dumps")
        self.assertTrue(mod)
        self.assertTrue(callable(mod.module))

    def test_missing_attribute_is_not_fatal(self) -> None:
        mod = optional.optional_import("json", attr="definitely_not_here")
        self.assertFalse(mod)
        self.assertIn("AttributeError", mod.error or "")

    def test_missing_module_never_raises(self) -> None:
        mod = optional.optional_import("zfp_definitely_missing_module")
        self.assertFalse(mod)
        self.assertIsNone(mod.module)
        self.assertIn("ModuleNotFoundError", mod.error or "")
        self.assertIsNone(mod.version)

    def test_results_are_cached(self) -> None:
        first = optional.optional_import("zfp_definitely_missing_module")
        second = optional.optional_import("zfp_definitely_missing_module")
        self.assertIs(first, second)
        self.assertIsNot(first, optional.optional_import("json"))

    def test_require_names_the_pip_extra(self) -> None:
        mod = optional.optional_import("numpy")
        if mod:
            self.skipTest("numpy is installed in this environment")
        with self.assertRaises(UnsupportedFeatureError) as ctx:
            mod.require("raster shape detection")
        message = str(ctx.exception)
        self.assertIn("raster shape detection", message)
        self.assertIn("numpy", message)
        self.assertIn("zerofusspdf[vision]", message)

    def test_extras_cover_every_reported_module(self) -> None:
        for name in optional.OPTIONAL_MODULES:
            self.assertIn(name, optional.EXTRA_FOR_MODULE, name)

    def test_have(self) -> None:
        self.assertTrue(optional.have("json"))
        self.assertFalse(optional.have("zfp_definitely_missing_module"))

    def test_capability_report_shape(self) -> None:
        report = optional.capability_report()
        for name in ("pikepdf", "pypdf", "fitz", "pypdfium2", "numpy", "cv2", "PIL",
                     "pytesseract", "paddleocr", "fastapi"):
            self.assertIn(name, report, name)
            entry = report[name]
            self.assertEqual(entry["kind"], "module")
            self.assertIsInstance(entry["available"], bool)
            self.assertIn("version", entry)
            self.assertIn("extra", entry)
        for name in ("tesseract", "pdftoppm", "qpdf"):
            self.assertIn(name, report, name)
            self.assertEqual(report[name]["kind"], "binary")
            self.assertIsInstance(report[name]["available"], bool)
            self.assertIn("path", report[name])

    def test_capability_report_is_json_serializable(self) -> None:
        json.dumps(optional.capability_report())


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
