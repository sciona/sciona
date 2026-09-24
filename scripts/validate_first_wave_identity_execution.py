"""Execute the stored original-identity revision against analytic references."""
import copy
import json
import math
from unittest.mock import patch
import psycopg
from dotenv import dotenv_values
from psycopg.rows import dict_row
from scripts.check_first_wave_identity_revision import check
from scripts.stage_first_wave_identity_revisions import ROOT,sha,require
from scripts import validate_first_wave_dynamics_graph as oracle


def selected(approved=False):
    with psycopg.connect(dotenv_values(ROOT/'.env')['SCIONA_DATA_CATALOG_DATABASE_URL'],row_factory=dict_row,
                        options='-c default_transaction_read_only=on -c statement_timeout=30000') as db:
        return check(db,'first_wave_dynamics',approved)


def validate(approved=False):
    graph,record=selected(approved)
    oracle.runner._ensure_atoms_imported()
    with patch.object(oracle,'build_first_wave_dynamics_graph',return_value=graph):
        inherited=oracle.validate(ROOT)
    require(inherited['serialized_graph_sha256']==record['graph_sha256'],'Stored graph digest differs')
    # These closed-form references use ordinary arithmetic/math, independently
    # of the provider's symbolic parser and differentiation implementation.
    cases=[('constant',0,lambda t:(0.,0.)),('linear','t',lambda t:(t,0.)),
        ('quadratic',{'op':'add','args':[{'op':'mul','args':[3,{'op':'pow','args':['t',2]}]},{'op':'mul','args':[-2,'t']},4]},lambda t:(3*t*t-2*t+4,6.)),
        ('quartic',{'op':'add','args':[{'op':'pow','args':['t',4]},{'op':'mul','args':[2,'t']}]},lambda t:(t**4+2*t,12*t*t)),
        ('sine',{'op':'sin','args':['t']},lambda t:(math.sin(t),-math.sin(t))),
        ('exponential',{'op':'exp','args':['t']},lambda t:(math.exp(t),math.exp(t)))]
    times=[-2.,-.5,0.,.5,2.];count=0;components=0;maximum=0.
    for name,tree,reference in cases:
        for mass in [.25,2.,7.]:
            payload=dict(version=1,mass=mass,position=tree,times=times);before=copy.deepcopy(payload)
            outcome,outputs=oracle.run_graph(graph,payload)
            require(outcome['status']=='completed' and payload==before,'Runner failed or mutated input')
            result=outputs[('dynamics','result')]
            require(result['times']==times and result['sample_columns']==['position','acceleration','force'] and len(result['samples'])==len(times),'Sample contract changed')
            for t,actual in zip(times,result['samples']):
                x,a=reference(t);expected=[x,a,mass*a]
                require(len(actual)==3,'Sample shape changed')
                for value,wanted in zip(actual,expected):
                    require(math.isfinite(value),'Nonfinite result')
                    error=abs(value-wanted)/math.ulp(wanted)
                    require(error<=4,'Independent analytic mismatch: '+name)
                    maximum=max(maximum,error);components+=1
            count+=1
    invalid=[('zero_mass',dict(version=1,mass=0,position='t',times=[0])),
        ('negative_mass',dict(version=1,mass=-1,position='t',times=[0])),
        ('singular_position',dict(version=1,mass=1,position={'op':'log','args':['t']},times=[0])),
        ('unevaluable_position',dict(version=1,mass=1,position={'op':'position','args':[]},times=[0]))]
    rejected=[]
    for name,payload in invalid:
        try:oracle.run_graph(graph,payload)
        except RuntimeError as error:
            require('first_wave_dynamics' in str(error),'Unexpected runner error');rejected.append(name)
        else:raise ValueError('Invalid case accepted: '+name)
    require(selected(approved)==(graph,record),'Stored state changed during execution')
    return dict(passed=True,approved=approved,catalog_mutations=0,version_id=record['version_id'],graph_sha256=record['graph_sha256'],
        inherited_execution=inherited,independent_runner_cases=count,independent_numeric_components=components,maximum_ulp_error=maximum,
        invalid_cases_rejected=rejected,native_threads=1,validator_sha256=sha(__file__),
        checker_sha256=sha(ROOT/'scripts/check_first_wave_identity_revision.py'),
        limitations=['Synthetic analytic cases validate sampled dynamics; historical malformed proof remains unapproved.',
            'Caller must supply a twice-differentiable position in one inertial Cartesian component with consistent SI coefficients.'])


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--approved',action='store_true');args=p.parse_args()
    report=validate(args.approved)
    name='first_wave_identity_served_execution.json' if args.approved else 'first_wave_identity_catalog_execution.json'
    (ROOT/'docs/reviews'/name).write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({k:v for k,v in report.items() if k not in ['inherited_execution','limitations']}))
