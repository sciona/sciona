"""Synthetic identifier/vocabulary checks for pinned entity feature extraction."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import re
import numpy as np
import pandas as pd
from sciona.unique_entity_features import unique_entity_features
from sciona.structured_identity_features import structured_identity_features

ROOT=Path(__file__).resolve().parents[1]
SOURCE='1st Place/Phase 1/submission/utilities.py'
PIN='8b9776ae0e7fdbb48e2bee64ee442ca1d2d4023309f953092dba108f93e6896c'


def validate(source):
    data=(source/SOURCE).read_bytes()
    if hashlib.sha256(data).hexdigest()!=PIN:raise ValueError('Pinned source differs')
    function=next(n for n in ast.parse(data).body if isinstance(n,ast.FunctionDef) and n.name=='extract_mfs_features')
    identity,time=ast.literal_eval(function.body[0].value.func.value.slice)
    source_columns=sorted({n.slice.value for n in ast.walk(function) if isinstance(n,ast.Subscript) and isinstance(n.value,ast.Name) and n.value.id=='mfs' and isinstance(n.slice,ast.Constant)})
    vocabularies={n.args[0].id for n in ast.walk(function) if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=='isin' and isinstance(n.args[0],ast.Name)}
    namespace={'pd':pd,'np':np,'re':re,**{v:['synthetic_A','synthetic_B'] for v in vocabularies}}
    namespace.update(FLIGHT_IDS=['SYN123'],FLIGHT_NUMBERS=['123'],AIRPORTS=['DST'])
    exec(compile(ast.Module(body=[function],type_ignores=[]),'pinned_entity_features','exec'),namespace)
    known='SYN123.SRC.DST.000101.1200.tail'
    unknown='UNK999.SRC.BAD.000101.1200.tail'
    master=pd.DataFrame({c:([known,unknown] if c==identity else ['synthetic_A']*2) for c in source_columns})
    queries=pd.DataFrame({identity:[known,known,unknown],time:pd.to_datetime(['2000-01-01 11:00','2000-01-01 13:00','2000-01-01 12:00'])})
    run=namespace[function.name]
    result=run(queries.copy(),master.copy())
    duration=next(c for c in result if c.endswith('_time_diff'))
    number=next(c for c in result if c.endswith('_cat_flightno'))
    last=next(c for c in result if c.endswith('_cat_no_last'))
    np.testing.assert_array_equal(result[duration],[-60.,60.,0.])
    if result[number].tolist()!=['123','123','OTHER'] or result[last].tolist()!=['3','3','R']:
        raise ValueError('Known/unknown identifier derivation differs')
    duplicate=master.iloc[[0]].copy()
    for c in source_columns:
        if c!=identity:duplicate[c]='synthetic_B'
    forward=run(queries.copy(),pd.concat([master,duplicate],ignore_index=True))
    reverse=run(queries.copy(),pd.concat([duplicate,master],ignore_index=True))
    differing=[c for c in forward if not forward[c].equals(reverse[c])]
    if not differing:raise ValueError('Duplicate lookup order counterexample disappeared')
    # Derive public software field mappings rather than depending on data files.
    attribute_map={}
    for node in function.body:
        if not isinstance(node,ast.Assign) or not isinstance(node.targets[0],ast.Subscript):continue
        for call in ast.walk(node.value):
            if (isinstance(call,ast.Call) and isinstance(call.func,ast.Name) and call.func.id=='zip'
                    and len(call.args)==2 and isinstance(call.args[1],ast.Subscript)
                    and isinstance(call.args[1].value,ast.Name) and call.args[1].value.id=='mfs'):
                attribute_map[node.targets[0].slice.value]=call.args[1].slice.value
    if len(attribute_map)!=5:raise ValueError('Expected complete attribute projection differs')
    controls={}
    for call in ast.walk(function):
        if (isinstance(call,ast.Call) and isinstance(call.func,ast.Attribute) and call.func.attr=='isin'
                and isinstance(call.func.value,ast.Subscript) and isinstance(call.args[0],ast.Name)):
            output=call.func.value.slice.value
            if output in attribute_map:controls[attribute_map[output]]=namespace[call.args[0].id]
    corrected=unique_entity_features(queries[identity].to_numpy(),master[identity].to_numpy(),
        master[list(attribute_map.values())],controls,'OTHER')
    for output,column in attribute_map.items():
        np.testing.assert_array_equal(corrected[column].to_numpy(),result[output].to_numpy())
    lexical=structured_identity_features(queries[identity].to_numpy(),pd.DatetimeIndex(queries[time]),
        master[identity].to_numpy(),delimiter='.',minimum_parts=6,primary_part=0,category_part=2,
        reference_parts=[3,4],reference_format='%y%m%d%H%M',
        vocabularies={'primary':namespace['FLIGHT_IDS'],'numeric':namespace['FLIGHT_NUMBERS'],'category':namespace['AIRPORTS']},
        fallback='OTHER',seconds_per_unit=60)
    suffixes={'primary':'_cat_flightid','prefix':'_cat_airline','numeric':'_cat_flightno',
        'numeric_length':'_flightno_length','numeric_last':'_cat_no_last','category':'_cat_arrival',
        'last':'_cat_last','penultimate':'_cat_sectolast','reference_hour':'_cat_hourdep','elapsed':'_time_diff'}
    covered=set(attribute_map)
    for generic,suffix in suffixes.items():
        matches=[column for column in result if column.endswith(suffix)]
        if len(matches)!=1:raise ValueError('Expected lexical feature mapping differs')
        np.testing.assert_array_equal(lexical[generic].to_numpy(),result[matches[0]].to_numpy())
        covered.add(matches[0])
    if covered!=set(result)-{identity,time}:raise ValueError('Entity feature coverage incomplete')
    try:
        duplicated=pd.concat([master,duplicate],ignore_index=True)
        unique_entity_features(queries[identity].to_numpy(),duplicated[identity].to_numpy(),
            duplicated[list(attribute_map.values())],controls,'OTHER')
    except ValueError:pass
    else:raise ValueError('Corrected lookup accepted ambiguous entity records')
    malformed=queries.iloc[[0]].copy();malformed[identity]='synthetic_invalid'
    try:run(malformed,master.copy())
    except IndexError:pass
    else:raise ValueError('Malformed identifier failure not reproduced')
    return dict(passed=True,approved=False,catalog_mutations=0,synthetic_only=True,source_sha256=PIN,
        synthetic_queries=3,signed_time_checks=3,vocabulary_controls=len(vocabularies),duplicate_order_dependent_columns=len(differing),
        corrected_attribute_values_checked=len(corrected)*len(attribute_map),corrected_duplicate_rejection=True,
        corrected_total_feature_values_checked=len(queries)*len(covered),complete_synthetic_feature_column_coverage=True,
        findings=['Derived time differences preserve sign before and after the encoded reference time.',
            'Unknown identifiers are replaced before numeric-suffix extraction; the fallback label consequently yields its final character as another category.',
            'Dictionary construction chooses the last input record for duplicate identities without an observation-time contract.',
            'Malformed structured identifiers raise an index error; a corrected adapter requires explicit format validation.'],
        requirements=['Caller supplies explicit reviewed vocabularies and well-formed identifier semantics; none may be inferred from inference rows.',
            'Require unique authoritative entity records or an explicit available-at lookup policy for repeated records.',
            'Preserve source fallback-derived categories or record and requalify a correction consistently in training and inference.'],
        limitations=['Injected synthetic vocabularies; original configuration and real entity records are not inspected.',
            'Structured identifier parsing and unique-snapshot attribute projection are checked under explicit synthetic policies; snapshot availability remains a caller obligation.',
            'No complete entity adapter, full raw-feature workflow or publication approval applied.'],
        implementation_sha256={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in ['sciona/unique_entity_features.py','tests/test_unique_entity_features.py','sciona/structured_identity_features.py','tests/test_structured_identity_features.py']},
        validator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    report=validate(parser.parse_args().source_directory)
    (ROOT/'docs/reviews/competition_nasa_first_entity_features.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:report[k] for k in ['passed','signed_time_checks','duplicate_order_dependent_columns','approved']}))
