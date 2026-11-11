"""Complete metadata on new DSB drafts without changing approval or history."""
import argparse
import hashlib
import json
from pathlib import Path
from uuid import UUID
from psycopg.types.json import Jsonb
from scripts.import_dsb_provider_drafts import reviewed_plan, connect, ROOT
from scripts.plan_dsb_provider_interfaces import plan as interfaces
from scripts.inventory_dsb_decomposed_providers import sha
from scripts.import_residual_execution_drafts import comparable, ensure_row
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score

RUNNER='dsb-draft-metadata.v1'
PLAN=ROOT/'docs/reviews/dsb_draft_metadata_plan.json'
OPERATIONS={
 'dsb_decomposed_training_batches':'Build repeatable training batches while keeping the caller’s random streams continuous across passes.',
 'prepare_classifier_crops':'Select high-scoring proposals and form padded image crops and coordinate grids for the source classifier.',
 'validation_batches':'Build the source recipe’s three evaluation groups using caller-supplied arrays and random streams.',
 'decode_detector_proposals':'Convert the source detector’s output grid into scored cubic proposals using its fixed anchor sizes and spacing.',
 'filter_classifier_proposals':'Keep proposals whose source detector score is strictly greater than minus one.',
 'normalize_and_align_tiles':'Check that image and coordinate tiles align, then apply the source image intensity scaling.',
 'predict_classifier_crops':'Evaluate caller-supplied classifier crops and return both the combined score and each proposal’s score.',
 'predict_detector_tiles':'Evaluate ordered image tiles with a caller-supplied detector already in evaluation mode.',
 'prepare_inference_geometry':'Pad an image to the detector spacing and build coordinates over the entire padded image before tiling.',
 'suppress_classifier_proposals':'Remove overlapping cubic proposals using the source recipe’s fixed overlap threshold.',
 'export_model_checkpoint':'Copy model weights to CPU memory and return serialized weights and an epoch number.',
 'validate_classifier':'Evaluate the source classifier and combine loss and accuracy using the number of cases in each batch.',
 'validate_detector':'Evaluate the source detector, average batch losses equally, and add classification counts.',
 'train_shared_runtime':'Run consecutive source training epochs while keeping shared weights, optimizers, and random streams alive.',
 'restore_inference_models':'Restore separate detector and classifier weights and set both models to evaluation mode.',
 'restore_shared_training_runtime':'Restore shared detector and classifier weights and create fresh optimizers for a model warm restart.',
 'prepare_raw_collections':'Prepare the source recipe’s five input groups using explicit coordinate conventions.',
 'preprocess_volume':'Apply the source image segmentation, cleanup, resizing, and cropping recipe.',
 'assign_anchor_labels':'Assign source detector targets to anchor boxes, using a separate caller-owned random stream.',
 'augment_classifier':'Apply the source classifier’s configured random flips, rotation, and axis swaps to a crop.',
 'augment_detection':'Apply the source detector’s configured random transforms to a crop, target, boxes, and coordinates.',
 'crop_classifier_proposal':'Crop around a proposal using the source classifier’s padding, scale, and jitter rules.',
 'crop_detection':'Crop an image and transform its detector targets using the source training rules.',
 'highest_scoring_indices':'Return up to the requested number of candidate indices in descending score order.',
 'thresholded_log_penalty':'Compute a weighted logarithmic penalty for probabilities below a chosen threshold.',
 'noisy_or_with_baseline':'Combine each row of probabilities with a caller-supplied baseline score using the complement of their product.',
 'centered_window':'Return a centered view of a tensor along selected axes without copying its values.',
 'scale_coordinate_rows':'Multiply, divide, then shift coordinate rows using explicit values for each axis.',
 'resample_regular_volume':'Resize a three-dimensional array to a requested spacing and return the spacing actually achieved.',
 'combine_volume_tiles':'Remove tile margins and reconstruct the padded volume in the required tile order.',
 'split_volume_tiles':'Split a volume into overlapping cubic tiles, extending edge values where padding is needed.',
 'repeat_rows_by_threshold':'Repeat each row by adding the configured counts for thresholds its selected value strictly exceeds.',
 'sequential_softmax_indices':'Choose indices without replacement using scores, recomputing selection probabilities after every draw.',
}
RECIPE_LIMIT='Source-policy adapter: model, coordinate, threshold, and sampling conventions remain specific to this recipe. Cross-domain use requires checking those assumptions.'
GENERIC_LIMIT='Reusable numerical operation: the caller supplies axis meanings, units, selection policy, and interpretation. No application domain is inferred.'


def protected(db,ids):
    result={}
    for table,key in [('atoms','atom_id'),('artifacts','artifact_id'),('atom_versions','atom_id'),
                      ('artifact_versions','artifact_id'),('atom_io_specs','atom_id'),('artifact_io_specs','artifact_id'),
                      ('artifact_audit_evidence','artifact_id'),('catalog_atoms_served','atom_id'),
                      ('catalog_artifacts_served','artifact_id')]:
        value="to_jsonb(t)-'updated_at'" if table in ('atoms','artifacts') else 'to_jsonb(t)'
        rows=db.execute(f'SELECT {value} AS row FROM {table} t WHERE {key}=ANY(%s) ORDER BY {value}::text',(ids,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(comparable(rows),sort_keys=True).encode()).hexdigest()
    return result


def expected():
    intake=reviewed_plan(); contract=interfaces()
    contracts={r['runtime_fqdn']:r for r in contract['provider_contracts']}
    rows=[]
    for record in intake['new_providers']:
        runtime=record['runtime_fqdn']; item=contracts[runtime]
        operation=OPERATIONS[runtime.rsplit('.',1)[1]]
        limits=GENERIC_LIMIT if record['classification']=='generic_operation' else RECIPE_LIMIT
        technical=item['operation_contract']
        if runtime.endswith('.augment_classifier'):
            technical+=' Source rotation changes the image but leaves its coordinate grid unchanged; rotation is disabled by default.'
        if runtime.endswith('.augment_detection'):
            technical+=' Source flips transform positions as size-coordinate; this convention is preserved without correction.'
        for prefix,key in [('atom','atom_id'),('artifact','artifact_id')]:
            for port in item['ports']:
                if port['direction']!='input':continue
                row={key:record['atom_id'],'version_id':record['version_id'],'name':port['name'],
                     'position':port['ordinal'],'kind':port['parameter_kind'],'type_desc':port['type_desc'],
                     'required':port['required'],'default_value_repr':port['default_value_repr'],
                     'technical_description':'Input '+port['name']+' for: '+operation,
                     'dejargonized_description':'Caller supplies '+port['name'].replace('_',' ')+'.',
                     'constraints_json':dict(port_contract=port,operation_contract=technical,reuse_limits=limits,
                         provider_sha256=record['provider_sha256'],review_status='draft_pending_semantic_approval')}
                rows.append(dict(table=prefix+'_parameters',key={k:row[k] for k in (key,'version_id','name')},row=row))
            for kind,content in [('technical',technical+' '+limits),('dejargonized',operation+' '+limits)]:
                key_fields={key:record['atom_id'],'kind':kind,'language':'en'}
                score=round(estimate_jargon_score(content),6)
                if kind=='dejargonized' and score>=.4:raise ValueError('Description needs plainer language')
                rows.append(dict(table=prefix+'_descriptions',key=key_fields,
                    row=dict(key_fields,content=content,generated_by=RUNNER,reviewed=False,jargon_score=score)))
    return intake,rows


def plan():
    intake,rows=expected()
    ids=[UUID(r['atom_id']) for r in intake['new_providers']]
    ids += [UUID(r['atom_latest'][0]['identity']) for r in intake['existing_providers']]
    with connect() as db:
        db.execute('SET TRANSACTION READ ONLY')
        for item in rows:
            if db.execute('SELECT 1 FROM '+item['table']+' WHERE '+' AND '.join(k+'=%s' for k in item['key']),tuple(item['key'].values())).fetchone():
                raise ValueError('Metadata already exists; preserve and review it')
        snapshot=protected(db,ids)
    return dict(approved=False,catalog_mutations=0,providers=33,rows=rows,identities=list(map(str,ids)),
        preserved=snapshot,intake_sha256=sha(ROOT/'docs/reviews/dsb_provider_intake_plan.json'),
        interface_sha256=sha(ROOT/'docs/reviews/dsb_provider_interface_plan.json'),script_sha256=sha(Path(__file__)))


class InjectedFailure(RuntimeError):pass


def stage(apply=False,fail_after=None):
    frozen=json.loads(PLAN.read_text());_,rows=expected()
    if rows!=frozen['rows'] or sha(Path(__file__))!=frozen['script_sha256']:
        raise ValueError('Metadata implementation drift')
    for field,path in [('intake_sha256','dsb_provider_intake_plan.json'),('interface_sha256','dsb_provider_interface_plan.json')]:
        if frozen[field]!=sha(ROOT/'docs/reviews'/path):raise ValueError('Evidence drift')
    ids=list(map(UUID,frozen['identities']));writes=0
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        if protected(db,ids)!=frozen['preserved']:raise ValueError('Protected state drift')
        for item in rows:
            row=dict(item['row'])
            if 'constraints_json' in row:row['constraints_json']=Jsonb(row['constraints_json'])
            writes+=int(ensure_row(db,item['table'],item['key'],row))
            if writes==fail_after:raise InjectedFailure('Injected metadata failure')
        if protected(db,ids)!=frozen['preserved']:raise ValueError('Publication or history changed')
        if fail_after is not None:raise ValueError('Failure point not reached')
        if not apply:db.rollback()
    return dict(applied=apply,approved=False,providers=33,rows_created=writes,
                publication_and_history_preserved=True,plan_sha256=sha(PLAN))


def validate():
    frozen=json.loads(PLAN.read_text())
    def unchanged():
        with connect() as db:
            for item in frozen['rows']:
                if db.execute('SELECT 1 FROM '+item['table']+' WHERE '+' AND '.join(k+'=%s' for k in item['key']),tuple(item['key'].values())).fetchone():
                    raise ValueError('Rollback left metadata')
            if protected(db,list(map(UUID,frozen['identities'])))!=frozen['preserved']:
                raise ValueError('Rollback changed protected state')
    unchanged();dry=stage();unchanged()
    points=[1,dry['rows_created']//2,dry['rows_created']]
    for point in points:
        try:stage(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected failure')
        unchanged()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,rollback_points=points,
                plan_sha256=sha(PLAN),script_sha256=sha(Path(__file__)))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    action=parser.parse_args().action
    if action=='plan':report=plan();path=PLAN
    elif action=='validate':report=validate();path=ROOT/'docs/reviews/dsb_draft_metadata_transactions.json'
    else:
        gate=json.loads((ROOT/'docs/reviews/dsb_draft_metadata_transactions.json').read_text())
        if not gate['passed'] or gate['plan_sha256']!=sha(PLAN) or gate['script_sha256']!=sha(Path(__file__)):
            raise ValueError('Matching rollback validation required')
        report=stage(apply=True)
        if action=='repeat' and report['rows_created']!=0:raise ValueError('Repeat changed metadata')
        path=ROOT/f'docs/reviews/dsb_draft_metadata_{action}.json'
    path.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ('rows','identities','preserved')}))
