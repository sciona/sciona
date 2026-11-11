"""Add reviewed plain descriptions to already-approved physics graphs only."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from scripts.import_flavours_provider_drafts import ROOT,connect
from scripts.audit_remaining_cdg_readiness import inspect
from scripts.import_residual_execution_drafts import ensure_row
from scripts.stage_remaining_legacy_physics_revisions import history
from scripts.backfill_dejargonized_descriptions import estimate_jargon_score
from scripts.promote_arc_execution import sha

BASELINE='docs/reviews/remaining_cdg_readiness_post_reference_repair.json'
PLAN=ROOT/'docs/reviews/physics_plain_description_plan.json'
GATES=ROOT/'docs/reviews/physics_plain_description_gates.json'
RUNNER='physics-plain-description-repair.v1'
PLAIN={
'first_wave':'Find motion rate, net force and acceleration for a positive constant mass under the stated motion assumptions.',
'series':'Add the resistance of two ideal resistors connected in series. Resistances must be nonnegative and the common current must be nonzero. Return ohms for a single input or compatible arrays.',
'langmuir':'Find the fraction of occupied surface sites, occupied and empty site densities, and the shared equilibrium event rate from the supplied adsorption and release rates under the stated surface model.',
'work_energy':'Find starting and ending motion energy and the signed net work between them for a constant mass moving along one axis.',
'projectile_range':'Find flight time, horizontal distance and the greatest possible distance at a fixed launch speed for ideal motion landing at its launch height.',
'hermitian':'Find the normalized average value of a finite Hermitian operator in a nonzero pure state, using exact arithmetic for the converted inputs.',
'vacuum_wave':'Build the two sides of the electric-field wave equation in empty space and a separate check of its divergence condition. Accept sufficiently smooth Cartesian fields without assuming they solve the equations.',
'isobaric_energy':'Find how internal energy changes with temperature at fixed pressure from heat capacity, internal pressure, volume and thermal expansion.',
'variance':'Find the mean and population variance of a finite probability distribution after normalization, using exact arithmetic for the converted inputs.',
'brewster_angle':'Find the incidence angle for the stated Brewster condition and its complementary refraction angle using the corrected ratio of incident and transmitted refractive indices.',
'lorentz_boost':'Convert an event time and position between inertial frames moving relative to each other along the x axis, using the positive Lorentz factor.',
'sine_double_angle':'Find the sine of twice a real angle in radians using the double-angle product identity.',
'infall_speed':'Find inward speed, signed inward velocity, gravitational work and potential for ideal radial fall from rest at infinity toward a fixed central source.',
'euler_formula':'Build the relation between a complex exponential and sine and cosine for a real angle, together with a checkable differential-equation proof.',
'mechanical_energy':'Find position, velocity, motion energy, potential energy, total energy and signed work for motion under a constant force.',
'spring_mass':'Find natural frequency, displacement, velocity and acceleration for the stated ideal spring and mass released from rest.',
'frequency':'Convert a positive finite cycle duration in seconds to cycles per second. Preserve input shape; this does not estimate a cycle duration or return angular frequency.',
'newton_force':'Find force magnitude, each acceleration magnitude and interaction potential energy under the assumed Newtonian gravity model. The retained source inference limits still apply.',
'escape_speed':'Find ideal escape speed and the launch energy needed in the stated gravity model. Also return an explicitly labeled inward-fall counterpart.',
'sine_squared':'Build equivalent sine-squared and half-angle expressions with a checkable symbolic certificate.',
'gravity_mass':'Infer the mass of a spherical gravity source from gravitational acceleration, radius and the supplied gravitational constant.',
'quadratic':'Find both ordered roots of a real quadratic using a corrected derivation, exact classification of the converted discriminant and stable high-precision evaluation.',
'constant_acceleration':'Find signed final velocity, displacement and average velocity for constant acceleration, including zero elapsed time and direction reversal.',
'two_body':'Find the period of a circular Newtonian two-body orbit from separation, both masses and an explicit gravitational constant in SI units.',
'wave_interference':'Find the combined intensity of two waves at fixed relative phase, their incoherent mean, maximum and minimum intensity, cross term and ratio under the stated normalization.',
'projectile':'Find elapsed time and height at a horizontal position for the stated ideal projectile model.',
'trig_exponential':'Build sine and cosine from complex exponentials with a checked Euler premise and a checkable linear-equation certificate.',
'curl_curl':'Build the curl of the curl, the gradient of divergence and the component Laplacian for a sufficiently smooth Cartesian vector field, with the corrected symbolic identity.',
'euler_interpreted':'Evaluate the exact Euler identity at pi and return its symbolic proof under the reviewed interpretation of the constants.',
'momentum':'Find the recoil momentum vector and its squared length from two real three-dimensional momentum vectors.',
'projectile_motion':'Find signed position and velocity at an elapsed time for ideal projectile motion with any launch direction and constant downward gravity.',
'orbit_radius':'Find the radius measured from the center and circular speed from a central mass, period and gravitational constant, treating the orbiting mass as negligible.',
'free_schrodinger':'Build a three-dimensional free-particle wave, its energy and spatial and time derivatives, with a checkable certificate for the governing equation.',
'circular_orbit_speed':'Find the circumference and average speed of circular motion from its radius and period, with explicit unit conversion.',
'hyperbolic_identities':'Build six hyperbolic and imaginary-angle function forms and eighteen identities, with a certificate based on exponential definitions.',
'parallel_resistance':'Find equivalent resistance, each branch current and total current for two ideal parallel resistors at a shared signed voltage, including zero voltage.',
'ideal_gas_expansion':'Find the fractional volume change per temperature change for an ideal gas held at fixed pressure, using absolute temperature.',
'eigenstate_orthogonality':'Check exact eigenstates of a Hermitian operator and find their complex overlap, matrix element and eigenvalue-gap product. Equal eigenvalues can retain nonzero overlap.',
'sound_scale':'Find the conditional sound-speed scale, its unit-prefactor value and the stated prefactor and shear diagnostics. This does not establish a universal speed bound for materials.',
'ideal_gas_compressibility':'Find how an ideal gas volume responds fractionally to pressure changes at fixed temperature, using absolute pressure.',
'wave_relations':'Find the four periodic-wave parameters from cycles per second and phase-speed magnitude using explicit angular conventions.',
'dirichlet_mode':'Build normalized interval modes that vanish at both ends, including both overall signs, their wavenumber, spatial derivatives and a boundary and normalization certificate.',
'helmholtz':'Build a vector field with harmonic time dependence, its spatial Laplacian and its Helmholtz equation residual. Retain fields that do not solve the equation and allow zero frequency.',
'schwarzschild':'Find the Schwarzschild length scale from a positive mass and explicit constants in SI units. Its physical interpretation is limited to the stated regime.',
'integration_parts':'Rewrite an indefinite integral using integration by parts, keeping the remaining integral and an independent integration constant. Check the result by differentiation within the stated smooth domain.',
'residual':'Check the difference between the two sides of the selected source equation for supplied values with the stated dimensions. This evaluates a residual; it does not solve for an unknown.',
}


def targets():
    raw=subprocess.check_output(['git','show','007443a0:'+BASELINE],cwd=ROOT)
    if raw!=(ROOT/BASELINE).read_bytes():raise ValueError('Baseline inventory changed')
    rows=[r for r in json.loads(raw)['physics'] if not r['native_eligible']]
    if len(rows)!=55 or any(r['missing_native_requirements']!=['plain_description'] for r in rows):
        raise ValueError('Exact description-only backlog required')
    return rows


def frozen(db,r):
    identity=r['artifact_id']
    result={'version_history':history(db,r['version_id'])}
    for table in ('artifacts','artifact_descriptions','artifact_parameters','artifact_references','artifact_audit_rollups'):
        condition=" AND NOT (kind='dejargonized' AND language='en')" if table=='artifact_descriptions' else ''
        rows=db.execute(f"SELECT to_jsonb(t)-'updated_at' AS row FROM {table} t WHERE artifact_id=%s{condition} ORDER BY (to_jsonb(t)-'updated_at')::text",(identity,)).fetchall()
        result[table]=hashlib.sha256(json.dumps(rows,sort_keys=True,default=str).encode()).hexdigest()
    return result


def make_plan():
    records=[]
    with connect() as db:
        db.execute('SET TRANSACTION ISOLATION LEVEL REPEATABLE READ READ ONLY')
        for r in targets():
            if inspect(db,r['artifact_id'])!=r:raise ValueError('Baseline state changed')
            if r['status']!='approved' or not r['served'] or r['trust_tier']!=3 or r['failed_selected_audits']:
                raise ValueError('Existing clean Tier 3 approval required')
            if not db.execute("SELECT 1 FROM artifact_audit_evidence WHERE version_id=%s AND audit_type='semantic_audit' AND passed AND status='completed'",(r['version_id'],)).fetchone():
                raise ValueError('Existing passing semantic evidence required')
            key=r['fqdn'].rsplit('.',1)[1].removesuffix('_corrected_execution').removesuffix('_interpreted_execution').removesuffix('_execution')
            if key=='execution':key='first_wave' if '.first_wave.' in r['fqdn'] else 'series'
            if key=='euler':key='euler_interpreted'
            technical=db.execute('SELECT description FROM artifacts WHERE artifact_id=%s',(r['artifact_id'],)).fetchone()['description']
            if key=='series' and not technical.startswith('Compute equivalent resistance of two nonnegative ohmic resistors in series'):
                raise ValueError('Ambiguous execution family')
            records.append(dict(identity=r,technical_description=technical,plain_description=PLAIN[key],preserved=frozen(db,r)))
    return dict(format='physics-plain-description-plan.v1',records=records,script_sha256=sha(Path(__file__)),
        scope='Plain wording of existing approved technical descriptions only. No numerical, semantic, tier or serving change.',
        runtime_rerun=False)


class InjectedFailure(RuntimeError):pass


def repair(apply=False,fail_after=None):
    plan=json.loads(PLAN.read_text());writes=0
    if plan['script_sha256']!=sha(Path(__file__)):raise ValueError('Reviewed plan changed')
    if apply:
        gates=json.loads(GATES.read_text())
        if gates.get('passed') is not True or gates['plan_sha256']!=sha(PLAN):raise ValueError('Current rollback gates required')
    with connect() as db:
        db.execute('SELECT pg_advisory_xact_lock(hashtext(%s))',(RUNNER,))
        for item in plan['records']:
            r=item['identity'];identity=r['artifact_id']
            db.execute('SELECT artifact_id FROM artifacts WHERE artifact_id=%s FOR UPDATE',(identity,)).fetchone()
            current=inspect(db,identity)
            expected=dict(r,native_eligible=True,missing_native_requirements=[])
            if current not in (r,expected) or frozen(db,r)!=item['preserved']:raise ValueError('Frozen approved state changed')
            keys=dict(artifact_id=identity,kind='dejargonized',language='en')
            writes+=int(ensure_row(db,'artifact_descriptions',keys,dict(keys,content=item['plain_description'],
                generated_by=RUNNER,reviewed=False,jargon_score=round(estimate_jargon_score(item['plain_description']),6))))
            if fail_after is not None and writes>=fail_after:raise InjectedFailure('Injected description repair failure')
            if inspect(db,identity)!=expected or frozen(db,r)!=item['preserved']:raise ValueError('Unrelated approved state changed')
        if fail_after is not None:raise ValueError('Fault point not reached')
        if not apply:db.rollback()
    return dict(applied=apply,writes=writes,native_eligible_repaired=55,existing_approvals_preserved=True,runtime_rerun=False)


def validate():
    plan=json.loads(PLAN.read_text())
    def unchanged():
        with connect() as db:
            for item in plan['records']:
                r=item['identity']
                if inspect(db,r['artifact_id'])!=r or frozen(db,r)!=item['preserved']:raise ValueError('Repair survived rollback')
    unchanged();dry=repair();unchanged()
    for point in (1,27,55):
        try:repair(fail_after=point)
        except InjectedFailure:pass
        else:raise ValueError('Expected rollback')
        unchanged()
    return dict(passed=True,catalog_mutations=0,dry_run=dry,rollback_points=[1,27,55],plan_sha256=sha(PLAN))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('action',choices=['plan','validate','apply','repeat'])
    action=parser.parse_args().action
    result=make_plan() if action=='plan' else validate() if action=='validate' else repair(apply=True)
    if action=='repeat' and result['writes']:raise ValueError('Repeat changed descriptions')
    path=PLAN if action=='plan' else GATES if action=='validate' else ROOT/('docs/reviews/physics_plain_description_'+action+'.json')
    path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(planned=True,descriptions=len(result['records'])) if action=='plan' else result))
