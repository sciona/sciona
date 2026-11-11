"""Source-faithful HuBMAP lifecycle execution graph, pending publication review."""
from sciona.architect.handoff import CDGExport
from sciona.architect.models import AlgorithmicNode,DependencyEdge,IOSpec,NodeStatus


def build_hubmap_lifecycle_graph():
    prefix='sciona.atoms.dl.segmentation.hubmap_execution.'
    def node(identity,primitive,inputs,outputs,optional=None):
        optional=optional or {}
        return AlgorithmicNode(node_id=identity,name=identity.replace('_',' '),description=primitive,
            concept_type='custom',status=NodeStatus.ATOMIC,matched_primitive=prefix+primitive,
            inputs=[IOSpec(name=name,type_desc='int' if name in {'tile_size','input_resolution'} else 'object',
                           required=name not in optional,default_value_repr=optional.get(name,'')) for name in inputs],
            outputs=[IOSpec(name=name,type_desc='object') for name in outputs])
    nodes=[
        node('initial_training','hubmap_initial_training',['slides','groups','initial_stages','initial_selectors','tile_size'],
             ['initial_checkpoints','initial_metrics'],{'tile_size':'1024'}),
        node('initial_models','hubmap_restore_models',['checkpoints','input_resolution'],['models'],{'input_resolution':'320'}),
        node('pseudo_labels','hubmap_pseudo_labels',['models','pseudo_sources','pseudo_rng','inference_geometry'],
             ['pseudo_labeled_sources'],{'inference_geometry':'None'}),
        node('retraining','hubmap_retraining',['slides','groups','retraining_stages','final_selectors','pseudo_labeled_sources','tile_size'],
             ['final_checkpoints','retraining_metrics'],{'tile_size':'1024'}),
        node('final_models','hubmap_restore_models',['checkpoints','input_resolution'],['models'],{'input_resolution':'320'}),
        node('final_prediction','hubmap_final_prediction',['models','final_images','final_rng','inference_geometry'],
             ['predictions'],{'inference_geometry':'None'}),
    ]
    links=[('initial_training','initial_models','initial_checkpoints','checkpoints'),
           ('initial_models','pseudo_labels','models','models'),
           ('pseudo_labels','retraining','pseudo_labeled_sources','pseudo_labeled_sources'),
           ('retraining','final_models','final_checkpoints','checkpoints'),
           ('final_models','final_prediction','models','models')]
    edges=[DependencyEdge(source_id=a,target_id=b,output_name=c,input_name=d,source_type='object',target_type='object') for a,b,c,d in links]
    return CDGExport(nodes=nodes,edges=edges,metadata=dict(
        artifact_source='competition_execution_reconstruction',publication_status='draft',
        source_version_id='325c30ff-ea2a-515c-8d40-110a5fd6991d',
        source_commit='615444e86f4fb5ab916c0b66d311a307f4e4dd27',final_notebook_version=59475269,
        scope='Decoded RGB/binary runtime arrays and explicit encoder states, folds and RNGs through initial training, pseudo labels, fresh retraining and final inference.',
        pseudo_tta=4,final_tta=3,retraining_initialization='fresh encoder-state model and optimizer',
        exclusions=['TIFF/subdataset codecs','historical GPU/worker scheduling','learned predictive accuracy',
                    'unsupported stain-normalization intake claim'],num_nodes=len(nodes),num_edges=len(edges)))
