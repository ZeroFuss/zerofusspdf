"""Unit tests for :mod:`zfp.semantics.graph`.

Every fixture is hand-built geometry: a label beside a blank, two blanks on one row, a
section covering half a page, a radio group.  The tests assert the exact relations that
come back, their direction, their weight, and that the build stays near-linear -- a page
with fifteen hundred spans must not quietly become quadratic.
"""

from __future__ import annotations

import time
import unittest
from typing import List, Optional

from zfp.core.config import DetectionConfig, ZfpConfig
from zfp.core.geometry import PageGeometry, Rect
from zfp.core.types import FieldCandidate, FieldType, TextSpan
from zfp.semantics.graph import (
    DEFAULT_LINE_HEIGHT,
    GRID_CELL_PT,
    NODE_FIELD,
    NODE_LABEL,
    NODE_SECTION,
    Edge,
    Node,
    RelationKind,
    SpatialGraph,
    build_graph,
    decay_weight,
    median_line_height,
    spatial_relation,
)
from zfp.semantics.sections import Section

PAGE = PageGeometry(0, Rect(0, 0, 612, 792), Rect(0, 0, 612, 792))
MAX_D = ZfpConfig.default().detection.label_max_distance_pt


def span(text: str, x0: float, y0: float, x1: float, y1: float, page: int = 0) -> TextSpan:
    """A native text span with a plausible baseline."""
    return TextSpan(text, Rect(x0, y0, x1, y1), page, "Helvetica", y1 - y0, baseline=y0)


def field(
    ident: str,
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    page: int = 0,
    kind: FieldType = FieldType.TEXT,
    group: Optional[str] = None,
) -> FieldCandidate:
    """A field candidate at an exact rectangle."""
    return FieldCandidate(ident, page, Rect(x0, y0, x1, y1), kind, group_id=group)


def kinds_between(graph: SpatialGraph, src: str, dst: str) -> List[str]:
    """Every relation stored from ``src`` to ``dst``, sorted."""
    return sorted(e.kind.value for e in graph.neighbors(src) if e.dst == dst)


class NodeTest(unittest.TestCase):
    """Nodes exist for every input, with the right kind and text."""

    def test_one_node_per_input(self) -> None:
        spans = [span("Name:", 50, 700, 90, 712), span("City:", 50, 680, 85, 692)]
        fields = [field("f1", 100, 699, 300, 713)]
        sections = [Section("Applicant", Rect(40, 600, 560, 740), 1, 0)]
        graph = build_graph(spans, fields, sections, PAGE)

        self.assertEqual(len(graph), 4)
        self.assertEqual(len(graph.nodes_of(NODE_LABEL)), 2)
        self.assertEqual(len(graph.nodes_of(NODE_FIELD)), 1)
        self.assertEqual(len(graph.nodes_of(NODE_SECTION)), 1)
        self.assertEqual(graph.nodes_of(NODE_FIELD)[0].id, "f1")
        self.assertEqual(graph.nodes_of(NODE_SECTION)[0].text, "Applicant")

    def test_blank_spans_are_ignored(self) -> None:
        graph = build_graph([span("   ", 50, 700, 90, 712)], [])
        self.assertEqual(len(graph), 0)

    def test_payload_round_trip(self) -> None:
        text = span("Name:", 50, 700, 90, 712)
        graph = build_graph([text], [])
        node = graph.nodes_of(NODE_LABEL)[0]
        self.assertIs(graph.payload(node.id), text)
        self.assertIsNone(graph.payload("missing"))

    def test_duplicate_spans_get_distinct_ids(self) -> None:
        twin = span("Date", 50, 700, 90, 712)
        graph = build_graph([twin, span("Date", 50, 700, 90, 712)], [])
        self.assertEqual(len(graph), 2)

    def test_candidate_without_id_still_gets_a_node(self) -> None:
        graph = build_graph([], [FieldCandidate("", 0, Rect(10, 10, 40, 20))])
        self.assertEqual(len(graph.nodes_of(NODE_FIELD)), 1)


class AdjacencyTest(unittest.TestCase):
    """Left/right and above/below, with their inverses."""

    def test_label_left_of_field(self) -> None:
        label = span("First Name:", 50, 700, 110, 712)
        blank = field("f1", 115, 699, 300, 713)
        graph = build_graph([label], [blank])
        node = graph.nodes_of(NODE_LABEL)[0]

        self.assertEqual(kinds_between(graph, node.id, "f1"), ["left_of", "precedes"])
        self.assertEqual(kinds_between(graph, "f1", node.id), ["right_of"])
        incoming = graph.incoming("f1", RelationKind.LEFT_OF)
        self.assertEqual([e.src for e in incoming], [node.id])

    def test_row_needs_forty_percent_vertical_overlap(self) -> None:
        # The label overlaps the blank by 3 pt of its own 12 pt height: 25%, not enough.
        label = span("Name:", 50, 709, 90, 721)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([label], [blank])
        self.assertEqual(graph.incoming("f1", RelationKind.LEFT_OF), [])

    def test_row_accepts_sufficient_overlap(self) -> None:
        label = span("Name:", 50, 703, 90, 715)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([label], [blank])
        self.assertEqual(len(graph.incoming("f1", RelationKind.LEFT_OF)), 1)

    def test_beyond_label_max_distance_no_edge(self) -> None:
        label = span("Name:", 10, 700, 50, 712)
        blank = field("f1", 50 + MAX_D + 5, 700, 400, 712)
        graph = build_graph([label], [blank])
        self.assertEqual(graph.incoming("f1", RelationKind.LEFT_OF), [])

    def test_above_and_below(self) -> None:
        header = span("Street Address", 100, 716, 200, 728)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([header], [blank])
        node = graph.nodes_of(NODE_LABEL)[0]
        self.assertIn("above", kinds_between(graph, node.id, "f1"))
        self.assertIn("below", kinds_between(graph, "f1", node.id))

    def test_above_needs_horizontal_overlap(self) -> None:
        header = span("Street Address", 400, 716, 500, 728)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([header], [blank])
        self.assertEqual(graph.incoming("f1", RelationKind.ABOVE), [])

    def test_above_limited_to_two_and_a_half_lines(self) -> None:
        # Body line height is 12 pt, so the reach is 30 pt.
        near = span("Near", 100, 736, 200, 748)
        far = span("Far", 100, 748, 200, 760)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([near, far], [blank])
        sources = {graph.node(e.src).text for e in graph.incoming("f1", RelationKind.ABOVE)}
        self.assertEqual(sources, {"Near"})

    def test_overlapping_rects_get_no_adjacency(self) -> None:
        inside = span("MM/DD/YYYY", 120, 702, 180, 710)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([inside], [blank])
        self.assertEqual(graph.incoming("f1", RelationKind.LEFT_OF), [])
        self.assertEqual(graph.incoming("f1", RelationKind.ABOVE), [])

    def test_pages_never_mix(self) -> None:
        label = span("Name:", 50, 700, 90, 712, page=0)
        blank = field("f1", 100, 700, 300, 712, page=1)
        graph = build_graph([label], [blank])
        self.assertEqual(graph.edge_count, 0)


class WeightTest(unittest.TestCase):
    """Weights are the documented distance decay."""

    def test_decay_formula(self) -> None:
        self.assertAlmostEqual(decay_weight(0.0, MAX_D), 1.0)
        self.assertAlmostEqual(decay_weight(MAX_D, MAX_D), 0.5)
        self.assertAlmostEqual(decay_weight(3 * MAX_D, MAX_D), 0.25)
        self.assertAlmostEqual(decay_weight(-5.0, MAX_D), 1.0)
        self.assertAlmostEqual(decay_weight(10.0, 0.0), 1.0)

    def test_edge_weight_matches_gap(self) -> None:
        label = span("Name:", 50, 700, 90, 712)
        blank = field("f1", 90 + MAX_D / 2.0, 700, 400, 712)
        graph = build_graph([label], [blank])
        edge = graph.incoming("f1", RelationKind.LEFT_OF)[0]
        self.assertAlmostEqual(edge.weight, 1.0 / 1.5, places=6)

    def test_nearer_label_outranks_further_one(self) -> None:
        near = span("Near:", 60, 700, 95, 712)
        far = span("Far:", 10, 700, 45, 712)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([near, far], [blank])
        found = graph.incoming("f1", RelationKind.LEFT_OF)
        self.assertEqual([graph.node(e.src).text for e in found], ["Near:", "Far:"])


class StructureTest(unittest.TestCase):
    """Rows, columns, containment, groups and reading order."""

    def test_same_row_between_fields(self) -> None:
        a = field("a", 100, 700, 200, 712)
        b = field("b", 300, 700, 400, 712)
        graph = build_graph([], [a, b])
        self.assertIn("same_row", kinds_between(graph, "a", "b"))
        self.assertIn("same_row", kinds_between(graph, "b", "a"))

    def test_same_column_between_rows(self) -> None:
        a = field("a", 100, 700, 200, 712)
        b = field("b", 100, 660, 200, 672)
        graph = build_graph([], [a, b])
        self.assertIn("same_column", kinds_between(graph, "a", "b"))

    def test_same_row_is_not_same_column(self) -> None:
        a = field("a", 100, 700, 200, 712)
        b = field("b", 100.5, 700, 200, 712)
        graph = build_graph([], [a, b])
        self.assertNotIn("same_column", kinds_between(graph, "a", "b"))

    def test_section_contains_nodes(self) -> None:
        section = Section("Applicant", Rect(40, 600, 560, 740), 1, 0)
        label = span("Name:", 50, 700, 90, 712)
        inside = field("f1", 100, 700, 300, 712)
        outside = field("f2", 100, 500, 300, 512)
        graph = build_graph([label], [inside, outside], [section], PAGE)
        section_node = graph.nodes_of(NODE_SECTION)[0]
        contained = {e.dst for e in graph.neighbors(section_node.id, RelationKind.CONTAINS)}
        self.assertIn("f1", contained)
        self.assertNotIn("f2", contained)

    def test_member_of_within_a_group(self) -> None:
        a = field("a", 100, 700, 110, 710, kind=FieldType.RADIO, group="g")
        b = field("b", 160, 700, 170, 710, kind=FieldType.RADIO, group="g")
        c = field("c", 220, 700, 230, 710, kind=FieldType.RADIO, group="other")
        graph = build_graph([], [a, b, c])
        self.assertIn("member_of", kinds_between(graph, "a", "b"))
        self.assertIn("member_of", kinds_between(graph, "b", "a"))
        self.assertNotIn("member_of", kinds_between(graph, "a", "c"))

    def test_precedes_follows_reading_order(self) -> None:
        first = span("One", 50, 700, 80, 712)
        second = span("Two", 100, 700, 130, 712)
        third = span("Three", 50, 680, 90, 692)
        graph = build_graph([third, second, first], [])
        chain = graph.edges_of(RelationKind.PRECEDES)
        pairs = [(graph.node(e.src).text, graph.node(e.dst).text) for e in chain]
        self.assertEqual(sorted(pairs), [("One", "Two"), ("Two", "Three")])


class AccessorTest(unittest.TestCase):
    """The read side is sorted, total and deterministic."""

    def test_neighbors_sorted_by_weight_then_destination(self) -> None:
        graph = SpatialGraph()
        for name in ("a", "b", "c", "d"):
            graph.add_node(Node(name, NODE_LABEL, Rect(0, 0, 1, 1), 0, name))
        graph.add_edge(Edge("a", "d", RelationKind.LEFT_OF, 0.5))
        graph.add_edge(Edge("a", "b", RelationKind.LEFT_OF, 0.5))
        graph.add_edge(Edge("a", "c", RelationKind.LEFT_OF, 0.9))
        self.assertEqual([e.dst for e in graph.neighbors("a")], ["c", "b", "d"])

    def test_add_edge_is_idempotent_and_rejects_self_loops(self) -> None:
        graph = SpatialGraph()
        graph.add_node(Node("a", NODE_LABEL, Rect(0, 0, 1, 1), 0))
        graph.add_node(Node("b", NODE_LABEL, Rect(0, 0, 1, 1), 0))
        self.assertTrue(graph.add_edge(Edge("a", "b", RelationKind.LEFT_OF, 1.0)))
        self.assertFalse(graph.add_edge(Edge("a", "b", RelationKind.LEFT_OF, 0.2)))
        self.assertFalse(graph.add_edge(Edge("a", "a", RelationKind.LEFT_OF, 1.0)))
        self.assertEqual(graph.edge_count, 1)

    def test_nearest_returns_the_strongest_of_the_right_kind(self) -> None:
        near = span("Near:", 60, 700, 95, 712)
        far = span("Far:", 10, 700, 45, 712)
        blank = field("f1", 100, 700, 300, 712)
        graph = build_graph([near, far], [blank])
        found = graph.nearest("f1", NODE_LABEL, RelationKind.LEFT_OF)
        self.assertIsNotNone(found)
        self.assertEqual(found[0].text, "Near:")
        self.assertGreater(found[1], 0.9)
        self.assertIsNone(graph.nearest("f1", NODE_FIELD, RelationKind.LEFT_OF))
        self.assertIsNone(graph.nearest("nope", NODE_LABEL, RelationKind.LEFT_OF))

    def test_to_dict_shape(self) -> None:
        graph = build_graph([span("Name:", 50, 700, 90, 712)], [field("f1", 100, 700, 300, 712)])
        payload = graph.to_dict()
        self.assertEqual({"nodes", "edges"}, set(payload))
        self.assertEqual(len(payload["nodes"]), 2)
        first = payload["nodes"][0]
        self.assertEqual(set(first), {"id", "kind", "rect", "page", "text"})
        self.assertEqual(len(first["rect"]), 4)
        edge = payload["edges"][0]
        self.assertEqual(set(edge), {"src", "dst", "kind", "weight"})

    def test_config_forms_are_interchangeable(self) -> None:
        spans = [span("Name:", 50, 700, 90, 712)]
        fields = [field("f1", 100, 700, 300, 712)]
        default = build_graph(spans, fields).to_dict()
        for config in (ZfpConfig.default(), DetectionConfig(), None, object()):
            self.assertEqual(build_graph(spans, fields, config=config).to_dict(), default)

    def test_tight_config_drops_the_edge(self) -> None:
        spans = [span("Name:", 50, 700, 90, 712)]
        fields = [field("f1", 200, 700, 300, 712)]
        graph = build_graph(spans, fields, config=DetectionConfig(label_max_distance_pt=4.0))
        self.assertEqual(graph.incoming("f1", RelationKind.LEFT_OF), [])


class HelperTest(unittest.TestCase):
    """The primitives the rest of the semantic layer reuses."""

    def test_spatial_relation_directions(self) -> None:
        left = Rect(0, 0, 10, 10)
        right = Rect(20, 0, 30, 10)
        self.assertEqual(spatial_relation(left, right, 120.0, 30.0)[0], RelationKind.LEFT_OF)
        self.assertEqual(spatial_relation(right, left, 120.0, 30.0)[0], RelationKind.RIGHT_OF)
        self.assertAlmostEqual(spatial_relation(left, right, 120.0, 30.0)[1], 10.0)

        top = Rect(0, 20, 10, 30)
        self.assertEqual(spatial_relation(top, left, 120.0, 30.0)[0], RelationKind.ABOVE)
        self.assertEqual(spatial_relation(left, top, 120.0, 30.0)[0], RelationKind.BELOW)
        self.assertIsNone(spatial_relation(left, Rect(400, 400, 410, 410), 120.0, 30.0))

    def test_degenerate_rect_still_relates(self) -> None:
        rule = Rect(20, 5, 200, 5)  # a zero-height underline
        label = Rect(0, 0, 10, 10)
        self.assertEqual(spatial_relation(label, rule, 120.0, 30.0)[0], RelationKind.LEFT_OF)

    def test_median_line_height(self) -> None:
        self.assertEqual(median_line_height([]), DEFAULT_LINE_HEIGHT)
        self.assertAlmostEqual(
            median_line_height([span("a", 0, 0, 10, 10), span("b", 0, 0, 10, 30)]), 20.0
        )


class ScaleTest(unittest.TestCase):
    """The build must stay near-linear and free of ordering artefacts."""

    @staticmethod
    def _dense_page(rows: int = 60, columns: int = 25) -> List[TextSpan]:
        """A dense, realistic page: ``rows * columns`` short words."""
        out: List[TextSpan] = []
        for row in range(rows):
            y = 760.0 - row * 12.5
            for column in range(columns):
                x = 20.0 + column * 23.0
                out.append(span("w%d_%d" % (row, column), x, y, x + 18.0, y + 9.0))
        return out

    def test_fifteen_hundred_spans_stay_fast(self) -> None:
        spans = self._dense_page()
        self.assertEqual(len(spans), 1500)
        started = time.monotonic()
        graph = build_graph(spans, [])
        elapsed = time.monotonic() - started
        self.assertEqual(len(graph), 1500)
        self.assertLess(elapsed, 2.0, "graph build took %.2fs" % elapsed)
        # Bounded degree: never the ~2.2M edges a quadratic build would produce.
        self.assertLess(graph.edge_count, 40 * len(graph))

    def test_result_does_not_depend_on_input_order(self) -> None:
        spans = self._dense_page(rows=6, columns=6)
        fields = [field("f%d" % i, 30.0 + i * 60.0, 600, 80.0 + i * 60.0, 612) for i in range(4)]
        forward = build_graph(spans, fields, [], PAGE).to_dict()
        backward = build_graph(list(reversed(spans)), list(reversed(fields)), [], PAGE).to_dict()
        self.assertEqual(forward, backward)

    def test_grid_pitch_is_sane(self) -> None:
        self.assertGreater(GRID_CELL_PT, 1.0)
        self.assertLess(GRID_CELL_PT, MAX_D)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
