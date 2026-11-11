"""Data models for the synthesizer pipeline (Round 3)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from sciona.architect.models import IOSpec
from sciona.judge.models import CompilerFeedback


class AssemblyUnit(BaseModel):
    """A single verified leaf fused from CDG node + MatchResult."""

    node_id: str
    name: str
    declaration_name: str
    type_signature: str
    matched_primitive: str = ""
    raw_code: str = ""
    inputs: list[IOSpec] = Field(default_factory=list)
    outputs: list[IOSpec] = Field(default_factory=list)
    requires_glue: bool = False
    tunable_param_names: list[str] = Field(default_factory=list)


class GlueEdge(BaseModel):
    """A data-flow edge that may need a type cast."""

    source_id: str
    target_id: str
    output_name: str
    input_name: str
    source_type: str
    target_type: str
    cast_expr: str = ""
    source_dim: str = ""  # compact dimensional signature of source output
    target_dim: str = ""  # compact dimensional signature of target input


class SkeletonFile(BaseModel):
    """A generated Lean 4 / Coq source file with sorry placeholders."""

    prover: str
    source_code: str
    units: list[AssemblyUnit] = Field(default_factory=list)
    glue_edges: list[GlueEdge] = Field(default_factory=list)
    sorry_count: int = 0
    metadata: dict[str, Any] = Field(default_factory=dict)


class AssemblyResult(BaseModel):
    """Result of assembling and optionally compiling a skeleton."""

    skeleton: SkeletonFile
    feedback: CompilerFeedback | None = None
    compiled_ok: bool = False

    model_config = {"arbitrary_types_allowed": True}


class SynthesisResult(BaseModel):
    """Result of the repair agent's synthesis loop."""

    skeleton: SkeletonFile
    compiled_ok: bool = False
    sorry_remaining: int = 0
    patches_applied: int = 0
    iterations_used: int = 0
    error_history: list[tuple[int, str, str]] = Field(default_factory=list)


class VerificationCertificate(BaseModel):
    """Cryptographic certificate tying a binary artifact back to its proof."""

    source_hash: str  # SHA-256 of the verified source file
    artifact_hash: str = ""  # SHA-256 of the compiled artifact
    prover: str  # "lean4" or "coq"
    prover_version: str
    goal: str = ""
    node_count: int = 0
    sorry_count: int = 0
    timestamp: str = ""
    certificate_version: str = "1.0"


class ExportBundle(BaseModel):
    """Result of the extractor's export operation."""

    target: str
    output_dir: Path
    source_path: Path
    compiled_artifact: Path | None = None
    executable_artifact: Path | None = None
    ffi_files: list[Path] = Field(default_factory=list)
    certificate: VerificationCertificate | None = None
    errors: list[str] = Field(default_factory=list)
    parameter_assignments: dict[str, dict[str, Any]] = Field(default_factory=dict)
