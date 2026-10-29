"""Incremental DSB reconstruction; additional operation decomposition is pending."""
from sciona.dsb_execution import build_dsb_raw_training_graph


def build_dsb_sampling_graph():
    graph = build_dsb_raw_training_graph()
    node = next(n for n in graph.nodes if n.node_id == 'training_batches')
    node.matched_primitive = 'sciona.atoms.dl.detection.dsb_decomposed_training.dsb_decomposed_training_batches'
    node.description = 'Construct stateful batches using reusable threshold repetition and sequential score sampling.'
    graph.metadata.update(
        publication_status='draft',
        scope='Raw-volume workflow with reusable sampling within the stateful training factory.',
        remaining_training_obligations=[
            'Separate remaining preparation, inference, loss and state-management operations.',
            'Qualify complete stored execution and exact provider interfaces/dependencies before promotion.',
        ],
    )
    return graph
