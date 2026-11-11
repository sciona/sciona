"""Draft DSB graph with reusable resampling in all preparation populations."""
from sciona.dsb_decomposed_loss_graph import build_dsb_loss_graph


def build_dsb_preparation_graph():
    graph = build_dsb_loss_graph()
    bindings = {'raw_preparation': 'prepare_raw_collections', 'preprocess': 'preprocess_volume'}
    for node in graph.nodes:
        if node.node_id in bindings:
            node.matched_primitive = 'sciona.atoms.dl.detection.dsb_spatial_preparation.' + bindings[node.node_id]
    graph.metadata.update(scope='Reusable resampling, sampling, tiling, pooling and losses with explicit shared runtime.', publication_status='draft')
    return graph
