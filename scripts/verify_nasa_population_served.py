"""Fresh served-catalog verification for the approved population graph family."""
import argparse
import json
from pathlib import Path

from scripts.validate_nasa_population_catalog import validate,ROOT,sha


def verify(source):
    report=validate(source,approved=True)
    report['execution']['approved']=True
    report['execution']['limitations']=[item.replace('catalog promotion and original-intake completion remain pending',
        'original-intake completion remains pending') for item in report['execution']['limitations']]
    report['catalog_validator_sha256']=report.pop('validator_sha256')
    report.update(validator_sha256=sha(Path(__file__)),trust_tier=3,source_intake_remains_draft=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--source-directory',type=Path,required=True)
    args=parser.parse_args()
    report=verify(args.source_directory)
    (ROOT/'docs/reviews/nasa_population_served_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({key:value for key,value in report.items() if key not in ('execution','version_ids','graph_sha256')}))
