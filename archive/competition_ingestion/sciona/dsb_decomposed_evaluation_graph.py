"""Draft graph exposing spatial recipes in training, validation and inference."""
from sciona.dsb_decomposed_preparation_graph import build_dsb_preparation_graph


def build_dsb_evaluation_graph():
    graph = build_dsb_preparation_graph()
    bindings = {'validation_batches':'validation_batches', 'classifier_crops':'prepare_classifier_crops'}
    for node in graph.nodes:
        if node.node_id in bindings:
            node.matched_primitive = 'sciona.atoms.dl.detection.dsb_evaluation_sampling.' + bindings[node.node_id]
    return graph
