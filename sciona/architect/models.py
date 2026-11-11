"""Core Pydantic models for the Conceptual Dependency Graph (CDG)."""

from __future__ import annotations

from enum import Enum
import re
from collections.abc import Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class ConceptType(str, Enum):
    """Algorithmic paradigm categories."""

    SORTING = "sorting"
    SEARCHING = "searching"
    DIVIDE_AND_CONQUER = "divide_and_conquer"
    GREEDY = "greedy"
    DYNAMIC_PROGRAMMING = "dynamic_programming"
    GRAPH_TRAVERSAL = "graph_traversal"
    GRAPH_OPTIMIZATION = "graph_optimization"
    STRING_MATCHING = "string_matching"
    GEOMETRY = "geometry"
    ARITHMETIC = "arithmetic"
    NUMBER_THEORY = "number_theory"
    COMBINATORICS = "combinatorics"
    ALGEBRA = "algebra"
    OPTIMIZATION = "optimization"
    ANALYSIS = "analysis"
    SET_THEORY = "set_theory"
    SIGNAL_TRANSFORM = "signal_transform"
    SIGNAL_FILTER = "signal_filter"
    GRAPH_SIGNAL_PROCESSING = "graph_signal_processing"
    NEURAL_NETWORK = "neural_network"
    CLUSTERING = "clustering"
    DIMENSIONALITY_REDUCTION = "dimensionality_reduction"
    ODE_SOLVER = "ode_solver"
    QUADRATURE = "quadrature"
    RANDOMIZED = "randomized"
    INFORMATION_THEORY = "information_theory"
    COMPRESSION = "compression"
    # Bayesian / probabilistic inference
    SAMPLER = "sampler"
    LOG_PROB = "log_prob"
    POSTERIOR_UPDATE = "posterior_update"
    VARIATIONAL_INFERENCE = "variational_inference"
    PRIOR_INIT = "prior_init"
    PRIOR_DISTRIBUTION = "prior_distribution"
    LIKELIHOOD_EVALUATION = "likelihood_evaluation"
    PROBABILISTIC_ORACLE = "probabilistic_oracle"
    ORACLE_GRADIENT = "oracle_gradient"
    MCMC_KERNEL = "mcmc_kernel"
    MCMC_PROPOSAL = "mcmc_proposal"
    VI_ELBO = "vi_elbo"
    SEQUENTIAL_FILTER = "sequential_filter"
    SMC_REWEIGHT = "smc_reweight"
    MESSAGE_PASSING = "message_passing"
    CONJUGATE_UPDATE = "conjugate_update"
    FIXED_POINT = "fixed_point"
    MAP_OVER = "map_over"
    BASELINE_ANALYSIS = "baseline_analysis"
    # ML model selection
    ML_MODEL_SELECTION = "ml_model_selection"
    # Data flow / orchestration
    STATE_INIT = "state_init"
    DATA_ASSEMBLY = "data_assembly"
    CONDITIONAL_ROUTING = "conditional_routing"
    DATA_EXTRACTION = "data_extraction"
    # Presentation / observability
    VISUALIZATION = "visualization"
    OBSERVABILITY = "observability"
    # Loss / objective functions (passed as callables to optimizers/trainers)
    LOSS_FUNCTION = "loss_function"
    # Domain knowledge not derivable from code (e.g., physics-specific feature sets)
    EXTERNAL_KNOWLEDGE = "external_knowledge"
    CUSTOM = "custom"
    EXTERNAL_TOOL = "external_tool"


def split_top_level_type_args(text: str, sep: str = ",") -> list[str]:
    """Split *text* on *sep* only at bracket depth 0."""
    parts: list[str] = []
    current: list[str] = []
    depth = 0
    quote_char: str | None = None
    for ch in text:
        if quote_char is not None:
            current.append(ch)
            if ch == quote_char:
                quote_char = None
            continue
        if ch in {"'", '"'}:
            quote_char = ch
            current.append(ch)
            continue
        if ch in "([{":
            depth += 1
            current.append(ch)
        elif ch in ")]}":
            depth = max(0, depth - 1)
            current.append(ch)
        elif ch == sep and depth == 0:
            token = "".join(current).strip()
            if token:
                parts.append(token)
            current = []
        else:
            current.append(ch)
    tail = "".join(current).strip()
    if tail:
        parts.append(tail)
    return parts


_ABSTRACT_WITNESS_TAG_MAP: dict[str, tuple[str, int | None]] = {
    "abstractsignal": ("signal", 1),
    "abstractpeaks": ("peaks", 1),
    "abstractarray": ("array", None),
    "abstractmatrix": ("matrix", 2),
    "abstractvector": ("vector", 1),
    "abstractscalar": ("scalar", 0),
    "abstractdistribution": ("distribution", None),
    "abstractgraph": ("graph", None),
    "abstractstate": ("state", None),
    "abstractimage": ("image", 2),
    "abstractspectrum": ("spectrum", 1),
}

_NUMERIC_DTYPES: frozenset[str] = frozenset(
    {
        "int",
        "int8",
        "int16",
        "int32",
        "int64",
        "uint8",
        "uint16",
        "uint32",
        "uint64",
        "float",
        "float16",
        "float32",
        "float64",
        "complex",
        "complex64",
        "complex128",
        "numeric",
        "number",
    }
)

_INT_DTYPES: frozenset[str] = frozenset(
    {"int", "int8", "int16", "int32", "int64", "uint8", "uint16", "uint32", "uint64"}
)
_FLOAT_DTYPES: frozenset[str] = frozenset(
    {"float", "float16", "float32", "float64", "numeric", "number"}
)


def _extract_dtype_token(text: str) -> str:
    lowered = text.lower()
    for token in (
        "float64",
        "float32",
        "float16",
        "int64",
        "int32",
        "int16",
        "int8",
        "uint64",
        "uint32",
        "uint16",
        "uint8",
        "complex128",
        "complex64",
        "bool_",
        "bool",
        "float",
        "int",
        "str",
    ):
        if re.search(rf"(?<![a-z0-9_]){re.escape(token)}(?![a-z0-9_])", lowered):
            return "bool" if token == "bool_" else token
    return ""


def _extract_ndim_and_shapes(constraints: str, witness_type_desc: str = "") -> tuple[int | None, list[str], list[str]]:
    tags: list[str] = []
    ndim: int | None = None
    shapes: list[str] = []

    w_norm = re.sub(r"[^a-z0-9]", "", witness_type_desc.lower())
    for key, (tag, w_ndim) in _ABSTRACT_WITNESS_TAG_MAP.items():
        if key in w_norm:
            if tag not in tags:
                tags.append(tag)
            if ndim is None and w_ndim is not None:
                ndim = w_ndim

    c_text = str(constraints or "")
    if c_text:
        m_dim = re.search(r"\b([1234])d\b", c_text.lower())
        if m_dim:
            ndim = int(m_dim.group(1))
        m_ndim = re.search(r"ndim\s*==\s*(\d+)", c_text.lower())
        if m_ndim:
            ndim = int(m_ndim.group(1))
        m_shape = re.search(r"\(([A-Za-z0-9_+\-*/\s,]+)\)", c_text)
        if m_shape and "," in m_shape.group(1):
            dims = [d.strip() for d in m_shape.group(1).split(",") if d.strip()]
            if dims:
                shapes = dims
                if ndim is None:
                    ndim = len(dims)
    return ndim, shapes, tags


class PortTypeSpec(BaseModel):
    """Structured, normalized port type specification derived from Python annotations."""

    raw: str = "Any"
    kind: str = "any"
    dtype: str = ""
    ndim: int | None = None
    shape_symbols: list[str] = Field(default_factory=list)
    domain_tags: list[str] = Field(default_factory=list)
    optional: bool = False
    default_json: str = ""
    tuple_elements: list[PortTypeSpec] = Field(default_factory=list)

    @classmethod
    def from_type_desc(
        cls,
        type_desc: str,
        *,
        constraints: str = "",
        data_kind: str = "",
        required: bool = True,
        default_value_repr: str = "",
        witness_type_desc: str = "",
    ) -> "PortTypeSpec":
        raw = str(type_desc or "Any").strip().strip("'\"") or "Any"
        optional = not required
        cleaned = raw

        # Unwrap Optional[T] or T | None / Union[T, None]
        if cleaned.startswith("Optional[") and cleaned.endswith("]"):
            optional = True
            cleaned = cleaned[len("Optional[") : -1].strip()
        elif cleaned.startswith("Union[") and cleaned.endswith("]"):
            union_args = split_top_level_type_args(cleaned[len("Union[") : -1], ",")
            non_none = [a for a in union_args if a not in {"None", "NoneType"}]
            if len(non_none) < len(union_args):
                optional = True
            if len(non_none) == 1:
                cleaned = non_none[0]
        elif "|" in cleaned:
            union_args = split_top_level_type_args(cleaned, "|")
            non_none = [a for a in union_args if a not in {"None", "NoneType"}]
            if len(non_none) < len(union_args):
                optional = True
            if len(non_none) == 1:
                cleaned = non_none[0]

        lower = cleaned.lower()
        ndim, shape_symbols, domain_tags = _extract_ndim_and_shapes(
            constraints, f"{raw} {witness_type_desc}".strip()
        )
        if data_kind and data_kind not in domain_tags:
            domain_tags.append(data_kind)

        # Tuple[A, B, ...]
        if (lower.startswith("tuple[") or lower.startswith("typing.tuple[")) and cleaned.endswith("]"):
            inner = cleaned[cleaned.index("[") + 1 : -1].strip()
            elts = split_top_level_type_args(inner, ",")
            if len(elts) == 2 and elts[1] == "...":
                return cls(
                    raw=raw,
                    kind="sequence",
                    dtype=_extract_dtype_token(elts[0]),
                    ndim=ndim,
                    shape_symbols=shape_symbols,
                    domain_tags=domain_tags,
                    optional=optional,
                    default_json=default_value_repr,
                )
            return cls(
                raw=raw,
                kind="tuple",
                optional=optional,
                default_json=default_value_repr,
                domain_tags=domain_tags,
                tuple_elements=[cls.from_type_desc(e) for e in elts],
            )

        if lower in {"any", "object", "unknown", ""}:
            # Witness can still supply semantic tags without replacing concrete python type
            if witness_type_desc:
                w_lower = witness_type_desc.lower()
                if any(tok in w_lower for tok in ("signal", "peaks", "array", "matrix", "vector", "spectrum", "image")):
                    return cls(
                        raw=raw,
                        kind="array",
                        ndim=ndim,
                        shape_symbols=shape_symbols,
                        domain_tags=domain_tags,
                        optional=optional,
                        default_json=default_value_repr,
                    )
            return cls(
                raw=raw,
                kind="any",
                ndim=ndim,
                shape_symbols=shape_symbols,
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        if lower in {"none", "nonetype", "void"}:
            return cls(raw=raw, kind="none", optional=True, default_json=default_value_repr)

        # DataFrame / Series
        if "dataframe" in lower:
            return cls(
                raw=raw,
                kind="dataframe",
                ndim=2 if ndim is None else ndim,
                shape_symbols=shape_symbols,
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )
        if re.search(r"(?<![a-z0-9_])series(?![a-z0-9_])", lower):
            return cls(
                raw=raw,
                kind="series",
                dtype=_extract_dtype_token(cleaned),
                ndim=1 if ndim is None else ndim,
                shape_symbols=shape_symbols,
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        # Arrays / Tensors / Sparse matrices
        if any(
            tok in lower
            for tok in (
                "ndarray",
                "arraylike",
                "array_like",
                "np.array",
                "numpy.array",
                "tensor",
                "csr_matrix",
                "csc_matrix",
                "coo_matrix",
                "csr_array",
                "csc_array",
                "spmatrix",
                "sparse",
                "abstractsignal",
                "abstractarray",
                "abstractmatrix",
                "abstractvector",
                "abstractpeaks",
                "abstractimage",
                "abstractspectrum",
            )
        ):
            if "sparse" in lower or any(s in lower for s in ("csr_", "csc_", "coo_", "spmatrix")):
                if "sparse" not in domain_tags:
                    domain_tags.append("sparse")
                if ndim is None:
                    ndim = 2
            return cls(
                raw=raw,
                kind="array",
                dtype=_extract_dtype_token(cleaned),
                ndim=ndim,
                shape_symbols=shape_symbols,
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        # Scalars
        if lower in {"float", "int", "bool", "str", "bytes", "complex", "number"} or lower.startswith("np.float") or lower.startswith("np.int") or lower.startswith("Literal[".lower()):
            dtype = _extract_dtype_token(cleaned) or (lower if lower in {"bytes", "complex", "number"} else "literal")
            return cls(
                raw=raw,
                kind="scalar",
                dtype=dtype,
                ndim=0,
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        # Callable
        if lower.startswith("callable") or "->" in cleaned:
            return cls(
                raw=raw,
                kind="callable",
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        # Mapping / dict
        if lower == "dict" or lower.startswith("dict[") or lower.startswith("mapping[") or lower == "mapping":
            return cls(
                raw=raw,
                kind="mapping",
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        # Sequence / list / set / iterable
        if any(
            lower == p or lower.startswith(p + "[")
            for p in ("list", "sequence", "iterable", "set", "frozenset", "deque")
        ):
            inner_dtype = ""
            if "[" in cleaned and cleaned.endswith("]"):
                inner = cleaned[cleaned.index("[") + 1 : -1].strip()
                inner_dtype = _extract_dtype_token(inner)
            return cls(
                raw=raw,
                kind="sequence",
                dtype=inner_dtype,
                ndim=1 if ndim is None else ndim,
                shape_symbols=shape_symbols,
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        # Explicit state / model / graph objects
        if lower.endswith("state") or "statemodel" in lower or "model" in lower:
            return cls(
                raw=raw,
                kind="state",
                dtype=cleaned,
                domain_tags=domain_tags,
                optional=optional,
                default_json=default_value_repr,
            )

        return cls(
            raw=raw,
            kind="object",
            dtype=cleaned.lower(),
            ndim=ndim,
            shape_symbols=shape_symbols,
            domain_tags=domain_tags,
            optional=optional,
            default_json=default_value_repr,
        )

    def is_compatible_with(self, target: "PortTypeSpec") -> bool:
        """Return whether *self* can safely satisfy *target* without type breakage."""
        if self.kind == "any" or target.kind == "any":
            return True
        if self.kind == "tuple" or target.kind == "tuple":
            if self.kind != target.kind or len(self.tuple_elements) != len(target.tuple_elements):
                return False
            return all(
                a.is_compatible_with(b)
                for a, b in zip(self.tuple_elements, target.tuple_elements)
            )
        if self.kind != target.kind:
            return False

        # Check dimensionality when both ports constrain ndim
        if self.ndim is not None and target.ndim is not None and self.ndim != target.ndim:
            return False

        # Check dtype compatibility
        if not self.dtype or not target.dtype or self.dtype == target.dtype:
            return True
        if self.kind == "scalar":
            if self.dtype in _INT_DTYPES and target.dtype in (_INT_DTYPES | _FLOAT_DTYPES):
                return True
            if self.dtype in _FLOAT_DTYPES and target.dtype in _FLOAT_DTYPES:
                return True
            return False
        if self.kind in {"array", "series", "sequence"}:
            if self.dtype in _NUMERIC_DTYPES and target.dtype in _NUMERIC_DTYPES:
                if target.dtype in _INT_DTYPES and self.dtype in _FLOAT_DTYPES:
                    return False
                return True
            return False
        return self.dtype == target.dtype


class StatePortSpec(BaseModel):
    """State or artifact port contract on stateful / pretrained CDG nodes."""

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    port_name: str = ""
    direction: str = "consume"  # consume | produce | update
    type_desc: str = "Any"
    accepted_formats: list[str] = Field(default_factory=list)
    required: bool = True
    path_semantics: str = ""
    multi_file: bool = False
    files: list[dict[str, Any]] = Field(default_factory=list)

    @model_validator(mode="before")
    @classmethod
    def _normalize_port_name(cls, value: Any) -> Any:
        if isinstance(value, dict):
            data = dict(value)
            if not data.get("port_name") and data.get("name"):
                data["port_name"] = str(data["name"])
            return data
        return value


class IOSpec(BaseModel):
    """Type specification for a node's inputs/outputs."""

    name: str
    type_desc: str = "Any"  # e.g., "list[int]", "Graph", "nat -> nat -> Prop"
    constraints: str = ""  # e.g., "sorted", "non-empty", "n > 0"
    data_kind: str = ""
    time_basis: str = ""
    provenance: str = ""
    required: bool = True
    default_value_repr: str = ""
    dim_signature: str = ""  # compact dimensional signature, e.g. "M1L2T-3" for Power
    port_type: PortTypeSpec | None = None

    def resolved_port_type(self, *, witness_type_desc: str = "") -> PortTypeSpec:
        """Return explicit ``port_type`` if present, else parse from ``type_desc``."""
        if self.port_type is not None:
            return self.port_type
        return PortTypeSpec.from_type_desc(
            self.type_desc,
            constraints=self.constraints,
            data_kind=self.data_kind,
            required=self.required,
            default_value_repr=self.default_value_repr,
            witness_type_desc=witness_type_desc,
        )


def format_canonical_port(port: IOSpec) -> str:
    """Format a single IOSpec in canonical signature notation."""
    type_text = str(port.type_desc or "Any").strip() or "Any"
    rendered = f"{port.name}: {type_text}"
    if not port.required:
        if port.default_value_repr:
            rendered += f" = {port.default_value_repr}"
        else:
            rendered += " = ..."
    return rendered


def format_canonical_type_signature(
    inputs: Sequence[IOSpec],
    outputs: Sequence[IOSpec],
) -> str:
    """Format canonical ``(x: T1, y: T2 = d) -> R`` signature across all stages."""
    in_sig = ", ".join(format_canonical_port(port) for port in inputs)
    if not outputs:
        out_sig = "None"
    elif len(outputs) == 1:
        out_sig = str(outputs[0].type_desc or "Any").strip() or "Any"
    else:
        out_sig = "tuple[" + ", ".join(
            str(port.type_desc or "Any").strip() or "Any" for port in outputs
        ) + "]"
    return f"({in_sig}) -> {out_sig}"


class NodeStatus(str, Enum):
    """Status of a CDG node through the decomposition process."""

    PENDING = "pending"  # Not yet decomposed
    DECOMPOSED = "decomposed"  # Has children
    ATOMIC = "atomic"  # Leaf — maps to a known primitive
    REJECTED = "rejected"  # Critic rejected this decomposition
    HIGH_RISK = "high_risk"  # Requires novel proof, flagged by Critic
    BLOCKED = "blocked"  # Decomposition terminated without a valid handoff


class BoundaryKind(str, Enum):
    """Whether a node represents a semantic graph boundary."""

    NONE = "none"
    ROOT_INPUT = "root_input"
    ROOT_OUTPUT = "root_output"


class CommonPattern(BaseModel):
    """A composition pattern declaring companion atoms for multi-step stages."""

    pattern_id: str
    description: str = ""
    atoms: list[str]
    ordering: str = "sequential"  # sequential, parallel, alternatives
    when: str = ""


class AlgorithmicNode(BaseModel):
    """A node in the Conceptual Dependency Graph."""

    node_id: str
    parent_id: str | None = None
    name: str  # e.g., "Sort the List"
    description: str  # Natural language spec
    concept_type: ConceptType
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    status: NodeStatus = NodeStatus.PENDING
    children: list[str] = Field(default_factory=list)  # child node_ids
    depth: int = 0
    type_signature: str = ""  # Formal type sig for Round 2 handoff
    matched_primitive: str | None = None  # e.g., "Nat.add_comm" or "heapsort"
    primitive_binding_confidence: float = 0.0
    primitive_binding_source: str = ""
    action_class: str = "replace_stage"
    resolution_reason: str = ""
    resolved_by: str = ""
    is_optional: bool = False  # Config-gated branches
    is_opaque: bool = False  # DL boundary: skip internal decomposition
    is_external: bool = False  # External tool call
    parallelizable: bool = False  # Supports parallel execution (e.g., particle swarms)
    conceptual_summary: str = ""
    critic_notes: str = ""
    decomposition_rationale: str = ""
    fixed_point_max_iterations: int = 0  # 0 means not a fixed-point node
    fixed_point_convergence_field: str = ""  # name of output field signaling convergence
    map_window_size: int = 0  # 0 means not a MAP node; >0 = window length
    map_hop_size: int = 0  # 0 means not a MAP node; >0 = hop between windows
    boundary_kind: BoundaryKind = BoundaryKind.NONE
    boundary_port_name: str = ""
    phase: str = "stateless"  # stateless | fit | transform | predict | score
    state_ports: list[StatePortSpec] = Field(default_factory=list)
    semantic_role: str = ""
    time_complexity: str = ""
    space_complexity: str = ""
    aliases: list[str] = Field(default_factory=list)
    common_patterns: list[CommonPattern] = Field(default_factory=list)


class EdgeLossClass(str, Enum):
    """Semantic information-loss classification for a data-flow edge."""

    PRESERVING = "preserving"
    LOSSY_ALLOWED = "lossy_allowed"
    IRREVERSIBLE = "irreversible"


class EdgeKind(str, Enum):
    """How the source node's output is consumed by the target node.

    DATA_FLOW (default): the source produces data that the target consumes
        as input.  Standard pipeline edge.
    CALLABLE_INJECTION: the source node is a pure function that the target
        node accepts as a callable parameter (e.g., a loss function passed
        to an optimizer, a kernel function passed to a GP, a custom
        objective passed to XGBoost).  The source node is not *called
        before* the target — it is *passed to* the target for the target
        to call repeatedly during its own execution.

    The synthesizer uses this to decide whether to wire data or pass a
    function reference.  Both kinds participate in topological ordering
    (the callable must be defined before the caller can reference it).
    """

    DATA_FLOW = "data_flow"
    CALLABLE_INJECTION = "callable_injection"


class DependencyEdge(BaseModel):
    """A data-flow or callable-injection edge between CDG nodes."""

    source_id: str
    target_id: str
    output_name: str  # which output of source
    input_name: str  # which input of target
    source_type: str  # type of the data flowing
    target_type: str  # expected type at target
    edge_kind: EdgeKind = EdgeKind.DATA_FLOW
    requires_glue: bool = False  # True if types don't match
    data_kind: str = ""
    provenance: str = ""
    time_basis: str = ""
    loss_class: EdgeLossClass = EdgeLossClass.PRESERVING
    alignment_expectation: str = ""


class SkeletonGraph(BaseModel):
    """A pre-fabricated graph template for an algorithmic paradigm."""

    paradigm: ConceptType
    name: str  # e.g., "Divide and Conquer"
    description: str
    template_nodes: list[AlgorithmicNode] = Field(default_factory=list)
    template_edges: list[DependencyEdge] = Field(default_factory=list)
    variants: list[str] = Field(
        default_factory=list
    )  # e.g., ["merge_sort", "quicksort"]
    metadata: dict[str, Any] = Field(default_factory=dict)


class ParamStatus(str, Enum):
    """Audit status of a primitive's tunable parameters."""

    APPROVED = "approved"
    FIXED = "fixed"
    BLOCKED = "blocked"
    DEPRECATED = "deprecated"


class PrimitiveParamSpec(BaseModel):
    """Schema for a single tunable parameter on a primitive."""

    name: str
    kind: Literal["int", "float", "categorical", "bool"]
    default: float | int | str | bool
    min_value: float | int | None = None
    max_value: float | int | None = None
    step: float | int | None = None
    log_scale: bool = False
    choices: list[str | int | float] | None = None
    constraints: str = ""
    semantic_role: str = ""
    safe_to_optimize: bool = True

    # Provenance
    range_source: str = ""
    source_reference: str = ""
    source_confidence: str = ""

    @model_validator(mode="after")
    def _validate_range(self) -> "PrimitiveParamSpec":
        if (
            self.min_value is not None
            and self.max_value is not None
            and self.min_value > self.max_value
        ):
            raise ValueError(
                f"min_value ({self.min_value}) must be <= max_value ({self.max_value})"
            )
        return self


class AlgorithmicPrimitive(BaseModel):
    """A known atomic operation from CLRS or a library."""

    name: str
    source: str  # "clrs-30", "coq-100-theorems", "mathlib"
    category: ConceptType
    description: str
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    type_signature: str = ""  # Formal type for Round 2
    clrs_spec: dict = Field(default_factory=dict)  # Raw CLRS spec if from CLRS-30
    uncertainty_factor: float | None = None
    uncertainty_confidence: float = 0.0
    uncertainty_mode: str = ""
    tunable_params: list[PrimitiveParamSpec] = Field(default_factory=list)
    param_status: ParamStatus = ParamStatus.FIXED
    aliases: list[str] = Field(default_factory=list)
    common_patterns: list[CommonPattern] = Field(default_factory=list)
    phase: str = "stateless"
    state_ports: list[StatePortSpec] = Field(default_factory=list)
    semantic_role: str = ""
    time_complexity: str = ""
    space_complexity: str = ""


class BaselineComponentShape(str, Enum):
    """Supported analyzer-level baseline component topologies."""

    WINDOWED = "windowed"
    COMBINER = "combiner"


class BaselineStageSpec(BaseModel):
    """Declarative node spec for a baseline analyzer stage."""

    key: str
    name: str
    template_name: str | None = None
    description: str = ""
    concept_type: ConceptType = ConceptType.BASELINE_ANALYSIS
    input_name: str = "signal"
    input_type: str = "np.ndarray"
    output_name: str = "signal"
    output_type: str = "np.ndarray"
    matched_primitive: str | None = None
    status: NodeStatus = NodeStatus.ATOMIC
    is_opaque: bool = False
    is_optional: bool = False


class BaselineWindowSpec(BaseModel):
    """Sliding-window envelope for a windowed baseline component."""

    size: int = Field(gt=0)
    hop: int = Field(gt=0)
    name: str = "Windowed Analysis"
    description: str = "Sliding window iteration over component input."
    input_name: str = "signal"
    input_type: str = "np.ndarray"
    output_name: str = "accumulated"
    output_type: str = "list[np.ndarray]"


class BaselineComponentOutputRef(BaseModel):
    """Reference to a named stage output produced by a component."""

    component: str
    stage_key: str


class BaselinePredictorAliasSpec(BaseModel):
    """Analyzer-level predictor alias that exposes a component output."""

    alias: str
    source: BaselineComponentOutputRef
    name: str | None = None
    description: str = ""


class BaselineAnalyzerComponentSpec(BaseModel):
    """Component declaration within a heterogeneous baseline analyzer."""

    name: str
    shape: BaselineComponentShape
    source_key: str | None = None
    window: BaselineWindowSpec | None = None
    window_stages: list[BaselineStageSpec] = Field(default_factory=list)
    post_stages: list[BaselineStageSpec] = Field(default_factory=list)
    combine_stage: BaselineStageSpec | None = None
    combine_inputs: list[BaselineComponentOutputRef] = Field(default_factory=list)
    default_output_stage: str

    @model_validator(mode="after")
    def _validate_shape(self) -> "BaselineAnalyzerComponentSpec":
        if self.shape == BaselineComponentShape.WINDOWED:
            if self.window is None:
                raise ValueError("windowed components require a window spec")
            if not self.window_stages:
                raise ValueError("windowed components require at least one window stage")
            if self.combine_stage is not None or self.combine_inputs:
                raise ValueError("windowed components cannot declare combiner inputs")
        else:
            if self.window is not None or self.window_stages:
                raise ValueError("combiner components cannot declare window stages")
            if self.source_key is not None:
                raise ValueError("combiner components cannot declare a source_key")
            if self.combine_stage is None:
                raise ValueError("combiner components require a combine_stage")
            if not self.combine_inputs:
                raise ValueError("combiner components require at least one combine input")

        stage_keys = [stage.key for stage in self.window_stages]
        stage_keys.extend(stage.key for stage in self.post_stages)
        if self.combine_stage is not None:
            stage_keys.append(self.combine_stage.key)
        if len(stage_keys) != len(set(stage_keys)):
            raise ValueError(f"component '{self.name}' has duplicate stage keys")

        available_outputs = set(stage_keys)
        available_outputs.add("windowed")
        if self.default_output_stage not in available_outputs:
            raise ValueError(
                f"component '{self.name}' default_output_stage "
                f"'{self.default_output_stage}' does not match any declared stage"
            )
        return self

    @property
    def available_stage_keys(self) -> set[str]:
        """Return the stage keys that may be referenced externally."""
        stage_keys = {stage.key for stage in self.window_stages}
        stage_keys.update(stage.key for stage in self.post_stages)
        if self.combine_stage is not None:
            stage_keys.add(self.combine_stage.key)
        stage_keys.add("windowed")
        return stage_keys


class BaselineAnalyzerSpec(BaseModel):
    """Analyzer-level baseline assembly spec with heterogeneous components."""

    preprocessors: list[BaselineStageSpec] = Field(default_factory=list)
    components: list[BaselineAnalyzerComponentSpec] = Field(default_factory=list)
    predictor_aliases: list[BaselinePredictorAliasSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_references(self) -> "BaselineAnalyzerSpec":
        preprocessor_keys = [stage.key for stage in self.preprocessors]
        if len(preprocessor_keys) != len(set(preprocessor_keys)):
            raise ValueError("baseline analyzer preprocessors must have unique keys")

        component_names = [component.name for component in self.components]
        if len(component_names) != len(set(component_names)):
            raise ValueError("baseline analyzer components must have unique names")

        alias_names = [alias.alias for alias in self.predictor_aliases]
        if len(alias_names) != len(set(alias_names)):
            raise ValueError("baseline analyzer predictor aliases must be unique")

        known_preprocessors = set(preprocessor_keys)
        known_components = {component.name: component for component in self.components}

        for component in self.components:
            if component.source_key is not None and component.source_key not in known_preprocessors:
                raise ValueError(
                    f"component '{component.name}' references unknown preprocessor "
                    f"'{component.source_key}'"
                )
            for ref in component.combine_inputs:
                target = known_components.get(ref.component)
                if target is None:
                    raise ValueError(
                        f"component '{component.name}' references unknown component "
                        f"'{ref.component}'"
                    )
                if ref.stage_key not in target.available_stage_keys:
                    raise ValueError(
                        f"component '{component.name}' references unknown stage "
                        f"'{ref.stage_key}' on component '{ref.component}'"
                    )

        for alias in self.predictor_aliases:
            target = known_components.get(alias.source.component)
            if target is None:
                raise ValueError(
                    f"predictor alias '{alias.alias}' references unknown component "
                    f"'{alias.source.component}'"
                )
            if alias.source.stage_key not in target.available_stage_keys:
                raise ValueError(
                    f"predictor alias '{alias.alias}' references unknown stage "
                    f"'{alias.source.stage_key}' on component '{alias.source.component}'"
                )

        return self
