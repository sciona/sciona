"""Complete corrected Cornell execution candidate."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode, DependencyEdge, IOSpec, NodeStatus


def build_cornell_graph():
    nodes = [AlgorithmicNode(
        node_id=name, name=name, description='Cornell ' + name,
        concept_type='custom', status=NodeStatus.ATOMIC,
        matched_primitive='sciona.atoms.ml.cornell_execution.cornell_' + name,
        inputs=[IOSpec(name=inp, type_desc=intype)],
        outputs=[IOSpec(name=out, type_desc=outtype)])
        for name, inp, out, intype, outtype in [
            ('prepare', 'payload', 'prepared', 'dict', 'object'),
            ('execute', 'prepared', 'result', 'object', 'dict')]]
    return CDGExport(nodes=nodes, edges=[DependencyEdge(
        source_id='prepare', target_id='execute', output_name='prepared',
        input_name='prepared', source_type='object', target_type='object')], metadata={
        'artifact_source': 'competition_execution_reconstruction',
        'publication_status': 'draft',
        'source_version_id': '5326bafb-d26f-5202-85c9-6f903df0fc77',
        'source_commits': ['4ad1aa4ed99bc097289c7593c55bc09234e0fc59'],
        'scope': 'Thirteen full DenseNet121 SED outputs, seventeen training phases with four model-only continuations, source augmentation, sampling, losses, scheduling, validation, checkpoint reload and ten-copy inference with fixed voting.',
        'runtime_contract': 'Version1 private32kHz mono audio and explicit four/fivefold populations, noise banks, epochs, batch size, seed and initialization. Provisioned source/dependencies; optional complete backbone state.',
        'source_corrections': 'Independent array voting supports singleton/empty events and half-open chunk/padding boundaries; modern librosa keyword; source component semantics retained. Explicit best/latest-tie checkpoint choice replaces hand-selected historical checkpoints.',
        'exclusions': ['Historical competition accuracy','Original pretrained prediction parity','Tier1 human certification','Tier2 usage qualification'],
        'num_nodes':2,'num_edges':1})
