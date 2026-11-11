"""Execute the version-selected original-intake draft through the production converter."""
import argparse
import json
from pathlib import Path
from unittest.mock import patch

import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row

from scripts.plan_nasa_first_corrected_intake import ROOT,plan,sha,require
from scripts.verify_nasa_first_corrected_draft import check_candidate
import scripts.validate_nasa_first_complete_graph as complete


def validate(source,checkpoint):
    proposed=plan()
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        graph=check_candidate(db,proposed)
    # Only choose the retrieved graph. The executor, all non-fit operations and
    # the checked checkpoint-replay qualification are unchanged.
    with patch.object(complete,'build_nasa_first_complete_graph',side_effect=lambda:graph.model_copy(deep=True)):
        execution=complete.validate(source,checkpoint)
    require(execution['passed'] and execution['graph_sha256']==proposed['graph_sha256'],'Catalog execution graph differs')
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        require(check_candidate(db,proposed)==graph,'Stored graph changed during execution')
    return dict(passed=True,approved=False,catalog_mutations=0,version_id=proposed['version_id'],
        graph_sha256=proposed['graph_sha256'],version_selected_production_converter=True,
        execution=execution,original_history_preserved=True,
        implementation_sha256={name:sha(ROOT/name) for name in ['scripts/validate_nasa_first_corrected_catalog.py',
            'scripts/verify_nasa_first_corrected_draft.py','sciona/services/catalog_artifact_retrieval.py']},
        plan_sha256=sha(ROOT/'docs/reviews/competition_nasa_first_corrected_intake_plan.json'),
        limitations=['Draft version-selected execution, not served-view or HTTP transport verification.',
                     'The 21 native fits retain their prior qualification; this full traversal checks every operand before checkpoint replay.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--source-directory',type=Path,required=True)
    parser.add_argument('--checkpoint-directory',type=Path,required=True);args=parser.parse_args()
    report=validate(args.source_directory,args.checkpoint_directory)
    (ROOT/'docs/reviews/competition_nasa_first_corrected_catalog_execution.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,executed_nodes=report['execution']['executed_outer_nodes'],
        exact_predictions=report['execution']['exact_native_prediction_comparisons'],approved=False)))
