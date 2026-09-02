"""The spatial graph: geometry expressed as relations instead of coordinates.

Detection hands the semantic layer three piles of rectangles -- text spans, field
candidates and section headings -- and nothing that says which belongs to which.  This
module turns that into a graph a reader would recognise::

    label   --left-of--->   field
    label   --above----->   field
    section --contains-->   label
    field   --same-row-->   peer field
    radio   --member-of->   radio

Everything downstream (the linker, the type inferencer, the normalizer's sibling pass)
asks the graph rather than re-deriving geometry, so every stage agrees about what is
next to what.

**Edge direction.**  ``Edge(src, dst, kind)`` reads *"src is <kind> dst"*: a label left
of a field is ``Edge(label, field, LEFT_OF)``.  Every spatial relation is stored with its
inverse as well (``LEFT_OF``/``RIGHT_OF``, ``ABOVE``/``BELOW``), and the symmetric ones
(``SAME_ROW``, ``SAME_COLUMN``, ``MEMBER_OF``) are stored in both directions, so
:meth:`SpatialGraph.incoming` and :meth:`SpatialGraph.neighbors` always agree.  To ask
"what is left of this field?" use ``graph.incoming(field_id, RelationKind.LEFT_OF)``.

**Cost.**  The edge build is bucketed through a fixed-pitch grid and capped at
:data:`MAX_EDGES_PER_RELATION` edges per node and relation, so a page with two thousand
spans stays near-linear instead of quadratic.

All rectangles are PDF user space: y-up, origin at the page origin, points.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from ..core.config import DetectionConfig, ZfpConfig
from ..core.geometry import EPS, PageGeometry, Rect
from ..core.ids import candidate_id, stable_id
from ..core.logging import get_logger
from ..core.types import FieldCandidate, TextSpan

__all__ = [
    "RelationKind",
    "Node",
    "Edge",
    "SpatialGraph",
    "build_graph",
    "spatial_relation",
    "decay_weight",
    "median_line_height",
    "NODE_LABEL",
    "NODE_FIELD",
    "NODE_SECTION",
    "VERTICAL_OVERLAP_RATIO",
    "HORIZONTAL_OVERLAP_RATIO",
    "VERTICAL_GAP_LINES",
    "MAX_EDGES_PER_RELATION",
    "GRID_CELL_PT",
    "DEFAULT_LINE_HEIGHT",
]

LOG = get_logger(__name__)

#: Node kind for a text span.
NODE_LABEL = "label"
#: Node kind for a field candidate.
NODE_FIELD = "field"
#: Node kind for a section heading.
NODE_SECTION = "section"

#: Two rects are on the same row when they share this fraction of the shorter height.
VERTICAL_OVERLAP_RATIO = 0.40
#: Two rects are stacked when they share this fraction of the narrower width.
HORIZONTAL_OVERLAP_RATIO = 0.30
#: ``ABOVE``/``BELOW`` reach this many median line heights.
VERTICAL_GAP_LINES = 2.5
#: Per node and relation, only the strongest this-many edges are kept.
MAX_EDGES_PER_RELATION = 12
#: Side of one spatial-index bucket, in points.
GRID_CELL_PT = 24.0
#: Line height assumed when the input carries no usable text metrics.
DEFAULT_LINE_HEIGHT = 12.0
#: Rows/columns cluster within this fraction of a line height.
ROW_TOLERANCE_RATIO = 0.55
#: Fields share a column when their left edges agree to this many points.
COLUMN_TOLERANCE_PT = 8.0
#: Clusters larger than this are chained instead of fully connected.
MAX_CLIQUE = 12


class RelationKind(str, Enum):
    """How two nodes stand to one another."""

    LEFT_OF = "left_of"
    RIGHT_OF = "right_of"
    ABOVE = "above"
    BELOW = "below"
    SAME_ROW = "same_row"
    SAME_COLUMN = "same_column"
    CONTAINS = "contains"
    MEMBER_OF = "member_of"
    PRECEDES = "precedes"


#: The inverse of every relation, used to store both directions of one observation.
INVERSE: Dict[RelationKind, RelationKind] = {
    RelationKind.LEFT_OF: RelationKind.RIGHT_OF,
    RelationKind.RIGHT_OF: RelationKind.LEFT_OF,
    RelationKind.ABOVE: RelationKind.BELOW,
    RelationKind.BELOW: RelationKind.ABOVE,
    RelationKind.SAME_ROW: RelationKind.SAME_ROW,
    RelationKind.SAME_COLUMN: RelationKind.SAME_COLUMN,
    RelationKind.MEMBER_OF: RelationKind.MEMBER_OF,
}


@dataclass
class Node:
    """One thing on the page: a text span, a field candidate or a section heading."""

    id: str
    kind: str
    rect: Rect
    page: int
    text: str = ""

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "id": self.id,
            "kind": self.kind,
            "rect": self.rect.as_list(),
            "page": self.page,
            "text": self.text,
        }


@dataclass
class Edge:
    """A directed relation, read as ``"src is <kind> dst"``."""

    src: str
    dst: str
    kind: RelationKind
    weight: float = 1.0

    def as_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary."""
        return {
            "src": self.src,
            "dst": self.dst,
            "kind": self.kind.value,
            "weight": self.weight,
        }


class SpatialGraph:
    """Nodes plus directed, weighted relations, with deterministic iteration order.

    Every accessor returns a freshly sorted list, so two runs over the same page emit
    byte-identical output regardless of dictionary insertion order.
    """

    def __init__(self) -> None:
        self._nodes: Dict[str, Node] = {}
        self._payloads: Dict[str, Any] = {}
        self._out: Dict[str, List[Edge]] = {}
        self._in: Dict[str, List[Edge]] = {}
        self._edges: List[Edge] = []
        self._seen: Set[Tuple[str, str, str]] = set()

    # ------------------------------------------------------------------ building
    def add_node(self, node: Node, payload: Any = None) -> Node:
        """Insert ``node`` (last write wins) and remember an optional payload object.

        The payload is the original :class:`~zfp.core.types.TextSpan`,
        :class:`~zfp.core.types.FieldCandidate` or ``Section`` the node stands for; it
        lets later stages recover the full object from a node id without re-indexing.
        """
        self._nodes[node.id] = node
        if payload is not None:
            self._payloads[node.id] = payload
        self._out.setdefault(node.id, [])
        self._in.setdefault(node.id, [])
        return node

    def add_edge(self, edge: Edge) -> bool:
        """Insert ``edge``; return ``False`` when an identical edge already exists."""
        key = (edge.src, edge.dst, edge.kind.value)
        if edge.src == edge.dst or key in self._seen:
            return False
        self._seen.add(key)
        self._edges.append(edge)
        self._out.setdefault(edge.src, []).append(edge)
        self._in.setdefault(edge.dst, []).append(edge)
        return True

    def relate(self, src: str, dst: str, kind: RelationKind, weight: float) -> None:
        """Add ``src -kind-> dst`` together with its inverse edge, when one exists."""
        self.add_edge(Edge(src, dst, kind, weight))
        inverse = INVERSE.get(kind)
        if inverse is not None:
            self.add_edge(Edge(dst, src, inverse, weight))

    # ------------------------------------------------------------------- reading
    def node(self, node_id: str) -> Optional[Node]:
        """Return the node with this id, or ``None``."""
        return self._nodes.get(node_id)

    def payload(self, node_id: str) -> Any:
        """Return the object a node stands for, or ``None`` when none was recorded."""
        return self._payloads.get(node_id)

    def has_node(self, node_id: str) -> bool:
        """True when ``node_id`` is present."""
        return node_id in self._nodes

    def nodes(self) -> List[Node]:
        """Every node, ordered page by page, top to bottom, left to right."""
        return sorted(self._nodes.values(), key=_node_sort_key)

    def nodes_of(self, kind: str) -> List[Node]:
        """Every node of ``kind`` (``"label"``, ``"field"``, ``"section"``)."""
        return [n for n in self.nodes() if n.kind == kind]

    def edges(self) -> List[Edge]:
        """Every edge, strongest first."""
        return sorted(self._edges, key=_edge_sort_key)

    def edges_of(self, kind: RelationKind) -> List[Edge]:
        """Every edge of one relation, strongest first."""
        return [e for e in self.edges() if e.kind == kind]

    def neighbors(self, node_id: str, kind: Optional[RelationKind] = None) -> List[Edge]:
        """Outgoing edges of ``node_id`` -- what this node *is* to other nodes.

        Sorted by ``(-weight, dst)``, so the closest relation comes first and ties break
        on the destination id.
        """
        found = self._out.get(node_id, ())
        picked = [e for e in found if kind is None or e.kind == kind]
        return sorted(picked, key=lambda e: (-e.weight, e.dst, e.kind.value))

    def incoming(self, node_id: str, kind: Optional[RelationKind] = None) -> List[Edge]:
        """Incoming edges of ``node_id`` -- what other nodes are *to* this node.

        ``incoming(field_id, RelationKind.LEFT_OF)`` is the set of nodes lying to the
        left of that field, closest first.
        """
        found = self._in.get(node_id, ())
        picked = [e for e in found if kind is None or e.kind == kind]
        return sorted(picked, key=lambda e: (-e.weight, e.src, e.kind.value))

    def nearest(
        self, node_id: str, kind: str, relation: RelationKind
    ) -> Optional[Tuple[Node, float]]:
        """Return the strongest node of node-kind ``kind`` standing in ``relation`` to
        ``node_id``, with the edge weight.

        ``nearest(field, "label", RelationKind.LEFT_OF)`` answers "which label is
        immediately to the left of this field?".  Returns ``None`` when nothing matches.
        """
        best: Optional[Tuple[Node, float]] = None
        for edge in self.incoming(node_id, relation):
            node = self._nodes.get(edge.src)
            if node is None or node.kind != kind:
                continue
            if best is None or edge.weight > best[1]:
                best = (node, edge.weight)
        return best

    # -------------------------------------------------------------------- export
    def to_dict(self) -> Dict[str, Any]:
        """Return a JSON-ready dictionary of the whole graph."""
        return {
            "nodes": [n.as_dict() for n in self.nodes()],
            "edges": [e.as_dict() for e in self.edges()],
        }

    @property
    def edge_count(self) -> int:
        """Number of stored edges."""
        return len(self._edges)

    def __len__(self) -> int:
        """Number of nodes."""
        return len(self._nodes)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return "SpatialGraph(nodes=%d, edges=%d)" % (len(self._nodes), len(self._edges))


# ---------------------------------------------------------------------- helpers
def _node_sort_key(node: Node) -> Tuple[int, float, float, str]:
    return (node.page, -node.rect.y1, node.rect.x0, node.id)


def _edge_sort_key(edge: Edge) -> Tuple[float, str, str, str]:
    return (-edge.weight, edge.src, edge.dst, edge.kind.value)


def detection_config(config: Any) -> DetectionConfig:
    """Accept a :class:`ZfpConfig`, a :class:`DetectionConfig` or ``None``."""
    if isinstance(config, DetectionConfig):
        return config
    if isinstance(config, ZfpConfig):
        return config.detection
    detection = getattr(config, "detection", None)
    if isinstance(detection, DetectionConfig):
        return detection
    return ZfpConfig.default().detection


def decay_weight(gap: float, scale: float) -> float:
    """Return ``1 / (1 + gap / scale)`` clamped into ``(0, 1]``.

    A zero gap scores ``1.0``; a gap of one full ``scale`` scores ``0.5``.  ``scale`` is
    normally ``DetectionConfig.label_max_distance_pt``.
    """
    if scale <= EPS:
        return 1.0
    value = 1.0 / (1.0 + max(0.0, float(gap)) / float(scale))
    return min(1.0, max(1e-6, value))


def median_line_height(spans: Sequence[TextSpan]) -> float:
    """Median text line height of ``spans``, falling back to ``DEFAULT_LINE_HEIGHT``."""
    heights: List[float] = []
    for span in spans:
        if span is None:
            continue
        height = span.rect.height
        if height <= EPS:
            height = span.font_size
        if height > EPS:
            heights.append(height)
    if not heights:
        return DEFAULT_LINE_HEIGHT
    heights.sort()
    middle = len(heights) // 2
    if len(heights) % 2:
        return heights[middle]
    return 0.5 * (heights[middle - 1] + heights[middle])


def _row_overlap_ok(a: Rect, b: Rect) -> bool:
    """True when two rects share enough vertical extent to count as one row."""
    shorter = min(a.height, b.height)
    if shorter <= EPS:
        return a.y0 <= b.y1 + EPS and b.y0 <= a.y1 + EPS
    return a.vertical_overlap(b) >= VERTICAL_OVERLAP_RATIO * shorter


def _column_overlap_ok(a: Rect, b: Rect) -> bool:
    """True when two rects share enough horizontal extent to count as stacked."""
    narrower = min(a.width, b.width)
    if narrower <= EPS:
        return a.x0 <= b.x1 + EPS and b.x0 <= a.x1 + EPS
    return a.horizontal_overlap(b) >= HORIZONTAL_OVERLAP_RATIO * narrower


def spatial_relation(
    a: Rect, b: Rect, max_gap: float, vertical_gap: float
) -> Optional[Tuple[RelationKind, float]]:
    """Classify how ``a`` stands to ``b``, or ``None`` when they are unrelated.

    Horizontal adjacency is tried first: the two must overlap vertically by at least
    :data:`VERTICAL_OVERLAP_RATIO` of the shorter height and be separated in x by less
    than ``max_gap``.  Vertical adjacency needs :data:`HORIZONTAL_OVERLAP_RATIO` of the
    narrower width and a y separation below ``vertical_gap``.  Overlapping rectangles
    (one drawn on top of the other) get no relation.

    Returns:
        ``(relation, gap)`` where ``relation`` is one of ``LEFT_OF``, ``RIGHT_OF``,
        ``ABOVE``, ``BELOW`` and ``gap`` is the separation in points.
    """
    if _row_overlap_ok(a, b):
        right_gap = b.x0 - a.x1
        if -EPS <= right_gap < max_gap:
            return (RelationKind.LEFT_OF, max(0.0, right_gap))
        left_gap = a.x0 - b.x1
        if -EPS <= left_gap < max_gap:
            return (RelationKind.RIGHT_OF, max(0.0, left_gap))
    if _column_overlap_ok(a, b):
        above_gap = a.y0 - b.y1
        if -EPS <= above_gap < vertical_gap:
            return (RelationKind.ABOVE, max(0.0, above_gap))
        below_gap = b.y0 - a.y1
        if -EPS <= below_gap < vertical_gap:
            return (RelationKind.BELOW, max(0.0, below_gap))
    return None


class _Grid:
    """A fixed-pitch bucket index over rectangles, for near-linear neighbour queries."""

    def __init__(self, cell: float = GRID_CELL_PT) -> None:
        self.cell = max(1.0, float(cell))
        self._buckets: Dict[Tuple[int, int], List[int]] = {}

    def _cells(self, rect: Rect) -> Iterable[Tuple[int, int]]:
        cell = self.cell
        x0 = int(rect.x0 // cell)
        x1 = int(rect.x1 // cell)
        y0 = int(rect.y0 // cell)
        y1 = int(rect.y1 // cell)
        # A pathological rect (a page-wide rule) must not explode the index.
        if (x1 - x0) > 512:
            x1 = x0 + 512
        if (y1 - y0) > 512:
            y1 = y0 + 512
        for cx in range(x0, x1 + 1):
            for cy in range(y0, y1 + 1):
                yield (cx, cy)

    def insert(self, index: int, rect: Rect) -> None:
        """Record ``index`` under every cell ``rect`` touches."""
        for key in self._cells(rect):
            self._buckets.setdefault(key, []).append(index)

    def query(self, rect: Rect) -> List[int]:
        """Return every index whose rect may touch ``rect``, ascending and deduplicated."""
        found: Set[int] = set()
        for key in self._cells(rect):
            bucket = self._buckets.get(key)
            if bucket:
                found.update(bucket)
        return sorted(found)


def _span_node_id(span: TextSpan, taken: Set[str]) -> str:
    """A deterministic, collision-free id for a span node."""
    base = stable_id(int(span.page), span.rect.rounded(3), span.text, prefix="sp")
    if base not in taken:
        return base
    bump = 1
    while True:
        alt = stable_id(int(span.page), span.rect.rounded(3), span.text, bump, prefix="sp")
        if alt not in taken:
            return alt
        bump += 1


def _section_node_id(section: Any, taken: Set[str]) -> str:
    """A deterministic, collision-free id for a section node."""
    page = int(getattr(section, "page", 0) or 0)
    rect = getattr(section, "rect", None) or Rect(0.0, 0.0, 0.0, 0.0)
    title = str(getattr(section, "title", "") or "")
    base = stable_id(page, rect.rounded(3), title, prefix="sec")
    if base not in taken:
        return base
    bump = 1
    while True:
        alt = stable_id(page, rect.rounded(3), title, bump, prefix="sec")
        if alt not in taken:
            return alt
        bump += 1


def _candidate_node_id(candidate: FieldCandidate, taken: Set[str]) -> str:
    """The candidate's own id when it has one, else a derived stable id."""
    node_id = candidate.id or candidate_id(
        candidate.page, candidate.rect, candidate.field_type.value
    )
    if node_id not in taken:
        return node_id
    bump = 1
    while "%s#%d" % (node_id, bump) in taken:
        bump += 1
    return "%s#%d" % (node_id, bump)


def _keep_best(
    graph: SpatialGraph, found: Dict[Tuple[str, RelationKind], List[Tuple[float, str]]]
) -> None:
    """Emit at most :data:`MAX_EDGES_PER_RELATION` edges per (node, relation)."""
    for (src, kind), entries in sorted(found.items(), key=lambda kv: (kv[0][0], kv[0][1].value)):
        entries.sort(key=lambda item: (-item[0], item[1]))
        for weight, dst in entries[:MAX_EDGES_PER_RELATION]:
            graph.relate(src, dst, kind, weight)


def _cluster(values: Sequence[Tuple[float, int]], tolerance: float) -> List[List[int]]:
    """Group ``(coordinate, index)`` pairs into runs no wider than ``tolerance``."""
    if not values:
        return []
    ordered = sorted(values)
    groups: List[List[int]] = [[ordered[0][1]]]
    reference = ordered[0][0]
    for value, index in ordered[1:]:
        if abs(value - reference) <= tolerance:
            groups[-1].append(index)
            continue
        groups.append([index])
        reference = value
    return groups


def _pairs(members: Sequence[int]) -> List[Tuple[int, int]]:
    """All pairs of a small cluster; consecutive pairs only for a large one."""
    count = len(members)
    if count < 2:
        return []
    if count <= MAX_CLIQUE:
        return [(members[i], members[j]) for i in range(count) for j in range(i + 1, count)]
    return [(members[i], members[i + 1]) for i in range(count - 1)]


def _reading_chain(nodes: Sequence[Node], line_h: float) -> List[Tuple[Node, Node]]:
    """Consecutive node pairs in reading order: lines top to bottom, left to right."""
    if len(nodes) < 2:
        return []
    tolerance = max(EPS, ROW_TOLERANCE_RATIO * line_h)
    ordered = sorted(nodes, key=lambda n: (-n.rect.center.y, n.rect.x0, n.id))
    lines: List[List[Node]] = []
    reference = 0.0
    for node in ordered:
        centre = node.rect.center.y
        if lines and abs(centre - reference) <= tolerance:
            lines[-1].append(node)
            continue
        lines.append([node])
        reference = centre
    flat: List[Node] = []
    for line in lines:
        line.sort(key=lambda n: (n.rect.x0, n.rect.y1, n.id))
        flat.extend(line)
    return [(flat[i], flat[i + 1]) for i in range(len(flat) - 1)]


# ------------------------------------------------------------------------ build
def build_graph(
    spans: Sequence[TextSpan],
    candidates: Sequence[FieldCandidate],
    sections: Sequence[Any] = (),
    geometry: Optional[PageGeometry] = None,
    *,
    config: Any = None,
) -> SpatialGraph:
    """Build the spatial graph of one page (or of several, keyed by ``span.page``).

    Args:
        spans: Text spans; blank ones are ignored.  Each becomes a ``"label"`` node.
        candidates: Field candidates.  Each becomes a ``"field"`` node keyed by its own
            :attr:`~zfp.core.types.FieldCandidate.id`.
        sections: Objects exposing ``title``/``rect``/``level``/``page`` -- normally
            :class:`zfp.semantics.sections.Section`.  Each becomes a ``"section"`` node.
        geometry: The page geometry, used only to bound the grid; optional.
        config: A :class:`~zfp.core.config.ZfpConfig` or
            :class:`~zfp.core.config.DetectionConfig`; the default config is used when
            omitted.  Only ``label_max_distance_pt`` is read.

    Returns:
        A :class:`SpatialGraph`.  Nodes and edges never cross a page boundary.

    Examples:
        >>> from zfp.core.geometry import Rect
        >>> from zfp.core.types import FieldCandidate, TextSpan
        >>> span = TextSpan("Name:", Rect(50, 700, 90, 712), 0, font_size=10)
        >>> field = FieldCandidate("f1", 0, Rect(100, 700, 260, 712))
        >>> g = build_graph([span], [field])
        >>> [e.kind.value for e in g.incoming("f1", RelationKind.LEFT_OF)]
        ['left_of']
    """
    det = detection_config(config)
    max_gap = float(det.label_max_distance_pt)
    graph = SpatialGraph()

    live_spans = [s for s in spans or () if s is not None and not s.is_blank()]
    live_candidates = [c for c in candidates or () if c is not None]
    live_sections = [s for s in sections or () if s is not None]

    taken: Set[str] = set()
    nodes: List[Node] = []
    span_nodes: List[Node] = []
    field_nodes: List[Node] = []
    field_of_node: Dict[str, FieldCandidate] = {}

    for span in live_spans:
        node_id = _span_node_id(span, taken)
        taken.add(node_id)
        node = Node(node_id, NODE_LABEL, span.rect.normalized(), int(span.page), span.text)
        graph.add_node(node, span)
        nodes.append(node)
        span_nodes.append(node)

    for candidate in live_candidates:
        node_id = _candidate_node_id(candidate, taken)
        taken.add(node_id)
        node = Node(
            node_id,
            NODE_FIELD,
            candidate.rect.normalized(),
            int(candidate.page),
            candidate.visible_label or "",
        )
        graph.add_node(node, candidate)
        nodes.append(node)
        field_nodes.append(node)
        field_of_node[node_id] = candidate

    section_nodes: List[Node] = []
    for section in live_sections:
        node_id = _section_node_id(section, taken)
        taken.add(node_id)
        rect = getattr(section, "rect", None) or Rect(0.0, 0.0, 0.0, 0.0)
        node = Node(
            node_id,
            NODE_SECTION,
            rect.normalized(),
            int(getattr(section, "page", 0) or 0),
            str(getattr(section, "title", "") or ""),
        )
        graph.add_node(node, section)
        section_nodes.append(node)

    line_h = median_line_height(live_spans)
    vertical_gap = max(VERTICAL_GAP_LINES * line_h, 1.0)

    _build_adjacency(graph, nodes, max_gap, vertical_gap)
    _build_rows_and_columns(graph, field_nodes, max_gap, line_h)
    _build_containment(graph, section_nodes, nodes, max_gap)
    _build_groups(graph, field_nodes, field_of_node, max_gap)
    _build_reading_order(graph, nodes, line_h, max_gap)

    LOG.debug(
        "spatial graph: %d nodes, %d edges (%d spans, %d fields, %d sections)",
        len(graph),
        graph.edge_count,
        len(span_nodes),
        len(field_nodes),
        len(section_nodes),
    )
    return graph


def _build_adjacency(
    graph: SpatialGraph, nodes: Sequence[Node], max_gap: float, vertical_gap: float
) -> None:
    """LEFT_OF / RIGHT_OF / ABOVE / BELOW between labels and fields, grid-bucketed.

    Two narrow window queries instead of one square one: the row pass only reaches
    sideways and the column pass only reaches up and down, which is what keeps a
    two-thousand-span page linear.  The inner test works on raw floats -- building a
    ``Rect`` per comparison is what a quadratic-looking profile is usually made of.
    """
    by_page: Dict[int, List[Node]] = {}
    for node in nodes:
        by_page.setdefault(node.page, []).append(node)

    for page in sorted(by_page):
        page_nodes = sorted(by_page[page], key=_node_sort_key)
        boxes = [
            (n.rect.x0, n.rect.y0, n.rect.x1, n.rect.y1, n.rect.height, n.rect.width)
            for n in page_nodes
        ]
        grid = _Grid()
        for index, node in enumerate(page_nodes):
            grid.insert(index, node.rect)
        found: Dict[Tuple[str, RelationKind], List[Tuple[float, str]]] = {}

        for index, node in enumerate(page_nodes):
            ax0, ay0, ax1, ay1, ah, aw = boxes[index]
            src = node.id

            for other in grid.query(node.rect.inflated(max_gap, 0.0)):
                if other == index:
                    continue
                bx0, by0, bx1, by1, bh, bw = boxes[other]
                overlap = (ay1 if ay1 < by1 else by1) - (ay0 if ay0 > by0 else by0)
                shorter = ah if ah < bh else bh
                if shorter > EPS:
                    if overlap < VERTICAL_OVERLAP_RATIO * shorter:
                        continue
                elif overlap < -EPS:
                    continue
                gap = bx0 - ax1
                if -EPS <= gap < max_gap:
                    kind = RelationKind.LEFT_OF
                else:
                    gap = ax0 - bx1
                    if not (-EPS <= gap < max_gap):
                        continue
                    kind = RelationKind.RIGHT_OF
                if gap < 0.0:
                    gap = 0.0
                found.setdefault((src, kind), []).append(
                    (decay_weight(gap, max_gap), page_nodes[other].id)
                )

            for other in grid.query(node.rect.inflated(0.0, vertical_gap)):
                if other == index:
                    continue
                bx0, by0, bx1, by1, bh, bw = boxes[other]
                overlap = (ax1 if ax1 < bx1 else bx1) - (ax0 if ax0 > bx0 else bx0)
                narrower = aw if aw < bw else bw
                if narrower > EPS:
                    if overlap < HORIZONTAL_OVERLAP_RATIO * narrower:
                        continue
                elif overlap < -EPS:
                    continue
                gap = ay0 - by1
                if -EPS <= gap < vertical_gap:
                    kind = RelationKind.ABOVE
                else:
                    gap = by0 - ay1
                    if not (-EPS <= gap < vertical_gap):
                        continue
                    kind = RelationKind.BELOW
                if gap < 0.0:
                    gap = 0.0
                found.setdefault((src, kind), []).append(
                    (decay_weight(gap, max_gap), page_nodes[other].id)
                )

        _keep_best(graph, found)


def _build_rows_and_columns(
    graph: SpatialGraph, field_nodes: Sequence[Node], max_gap: float, line_h: float
) -> None:
    """SAME_ROW for fields on one baseline band, SAME_COLUMN for one x band."""
    by_page: Dict[int, List[Node]] = {}
    for node in field_nodes:
        by_page.setdefault(node.page, []).append(node)

    for page in sorted(by_page):
        page_nodes = sorted(by_page[page], key=_node_sort_key)
        row_tolerance = max(ROW_TOLERANCE_RATIO * line_h, 1.0)
        rows = _cluster(
            [(-node.rect.center.y, index) for index, node in enumerate(page_nodes)],
            row_tolerance,
        )
        row_of: Dict[int, int] = {}
        for row_index, members in enumerate(rows):
            for member in members:
                row_of[member] = row_index
            for left, right in _pairs(sorted(members, key=lambda i: page_nodes[i].rect.x0)):
                a, b = page_nodes[left], page_nodes[right]
                gap = max(0.0, b.rect.x0 - a.rect.x1, a.rect.x0 - b.rect.x1)
                graph.relate(a.id, b.id, RelationKind.SAME_ROW, decay_weight(gap, max_gap))

        columns = _cluster(
            [(node.rect.x0, index) for index, node in enumerate(page_nodes)],
            COLUMN_TOLERANCE_PT,
        )
        for members in columns:
            ordered = sorted(members, key=lambda i: -page_nodes[i].rect.y1)
            for top, bottom in _pairs(ordered):
                a, b = page_nodes[top], page_nodes[bottom]
                if row_of.get(top) == row_of.get(bottom):
                    continue  # a column needs distinct rows
                gap = max(0.0, a.rect.y0 - b.rect.y1, b.rect.y0 - a.rect.y1)
                graph.relate(a.id, b.id, RelationKind.SAME_COLUMN, decay_weight(gap, max_gap))


def _build_containment(
    graph: SpatialGraph, section_nodes: Sequence[Node], nodes: Sequence[Node], max_gap: float
) -> None:
    """CONTAINS from a section rect to every node it encloses."""
    if not section_nodes:
        return
    by_page: Dict[int, List[Node]] = {}
    for node in nodes:
        by_page.setdefault(node.page, []).append(node)

    for section in sorted(section_nodes, key=_node_sort_key):
        page_nodes = by_page.get(section.page, ())
        for node in page_nodes:
            if node.id == section.id:
                continue
            if not _encloses(section.rect, node.rect):
                continue
            distance = max(0.0, section.rect.y1 - node.rect.y1)
            graph.add_edge(
                Edge(section.id, node.id, RelationKind.CONTAINS, decay_weight(distance, max_gap))
            )


def _encloses(outer: Rect, inner: Rect) -> bool:
    """True when ``outer`` holds the bulk of ``inner``."""
    if inner.area <= EPS:
        return outer.inflated(EPS, EPS).contains_point(inner.center)
    overlap = outer.intersection(inner)
    if overlap is None:
        return False
    return overlap.area >= 0.6 * inner.area


def _build_groups(
    graph: SpatialGraph,
    field_nodes: Sequence[Node],
    field_of_node: Dict[str, FieldCandidate],
    max_gap: float,
) -> None:
    """MEMBER_OF between candidates sharing a ``group_id``."""
    groups: Dict[str, List[Node]] = {}
    for node in field_nodes:
        candidate = field_of_node.get(node.id)
        if candidate is None or not candidate.group_id:
            continue
        groups.setdefault(str(candidate.group_id), []).append(node)

    for group_id in sorted(groups):
        members = sorted(groups[group_id], key=_node_sort_key)
        indexes = list(range(len(members)))
        for left, right in _pairs(indexes):
            a, b = members[left], members[right]
            gap = a.rect.center.distance_to(b.rect.center)
            graph.relate(a.id, b.id, RelationKind.MEMBER_OF, decay_weight(gap, max_gap))


def _build_reading_order(
    graph: SpatialGraph, nodes: Sequence[Node], line_h: float, max_gap: float
) -> None:
    """PRECEDES along the order a person reads the page."""
    by_page: Dict[int, List[Node]] = {}
    for node in nodes:
        by_page.setdefault(node.page, []).append(node)
    for page in sorted(by_page):
        for first, second in _reading_chain(by_page[page], line_h):
            gap = first.rect.center.distance_to(second.rect.center)
            graph.add_edge(
                Edge(first.id, second.id, RelationKind.PRECEDES, decay_weight(gap, max_gap))
            )
