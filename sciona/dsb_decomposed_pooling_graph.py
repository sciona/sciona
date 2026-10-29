"""Draft runtime graph with reusable noisy-OR in training and inference."""
from sciona.dsb_decomposed_runtime import build_dsb_runtime_graph


def build_dsb_pooling_graph():
    graph=build_dsb_runtime_graph()
    replacements={
        'training_runtime':'restore_shared_training_runtime',
        'models':'restore_inference_models',
    }
    for node in graph.nodes:
        if node.node_id in replacements:
            node.matched_primitive='sciona.atoms.dl.detection.dsb_pooling_runtime.'+replacements[node.node_id]
    graph.metadata.update(scope='Explicit runtime/inference graph with reusable noisy-OR pooling, sampling and tiling.',publication_status='draft')
    return graph
