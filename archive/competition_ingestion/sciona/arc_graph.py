"""Full adaptive ARC execution graph, draft pending Tier 3 publication review."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus
from sciona.arc_runtime import IDENTITY_SHA256


def build_arc_graph():
    nodes = [AlgorithmicNode(node_id=name, name=name, description="ARC " + name,
                             concept_type="custom", status=NodeStatus.ATOMIC,
                             matched_primitive="sciona.atoms.ml.arc_execution.arc_" + name,
                             inputs=[IOSpec(name=inp, type_desc=intype)],
                             outputs=[IOSpec(name=out, type_desc=outtype)])
             for name, inp, out, intype, outtype in [
                 ("prepare", "payload", "prepared", "dict", "object"),
                 ("execute", "prepared", "result", "object", "dict")]]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(source_id="prepare", target_id="execute",
                      output_name="prepared", input_name="prepared", source_type="object", target_type="object")],
                     metadata={"artifact_source": "competition_execution_reconstruction", "publication_status": "draft",
                               "source_version_id": "4a5eaed3-fbc0-564f-b12b-120a76304e05",
                               "source_commits": ["9407072659de1270358c2ba34c527785214dd68b"],
                               "scope": "Full C++ color normalization, output-size deduction, cost-limited DAG expansion, piece composition, candidate scoring and reconstruction; adaptive 3/23/33/4 phases and top-three merging.",
                               "runtime_contract": "Version 1 strict JSON task and explicit time/RSS budget within source defaults; locally provisioned reviewed macOS arm64 build. No test labels or payload file paths.",
                               "build_identity_sha256": IDENTITY_SHA256,
                               "source_corrections": "Actual C++ piece solver, not Python A*. Candidate scoring permits partial training agreement. Exports predictions rather than saved programs.",
                               "output_contract": "Per-input ranked grids and scores, attempt statuses, explicit missing predictions and full-run completion flag. Sampled RSS is not an instantaneous OS memory ceiling.",
                               "exclusions": ["Historical competition accuracy", "Historical GCC binary equivalence",
                                              "Unreviewed binaries or platforms", "Tier 1 human certification", "Tier 2 usage qualification"],
                               "num_nodes": 2, "num_edges": 1})
