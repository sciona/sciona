"""Draft graph using reusable probability pooling and log penalty in training."""
from sciona.dsb_decomposed_pooling_graph import build_dsb_pooling_graph


def build_dsb_loss_graph():
    graph=build_dsb_pooling_graph()
    node=next(n for n in graph.nodes if n.node_id=='training')
    node.matched_primitive='sciona.atoms.dl.detection.dsb_loss_training.train_shared_runtime'
    graph.metadata.update(scope='Explicit runtime and inference with reusable sampling, tiling, pooling and classifier penalty.',publication_status='draft')
    return graph
