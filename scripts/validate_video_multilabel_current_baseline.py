"""Probe existing served qualification without rewriting historical evidence."""
import contextlib
import io
import json
from scripts.audit_decomposed_video_multilabel_readiness import ROOT, audit, sha
from scripts.verify_video_multilabel_publication import verify


def validate():
    before=audit()
    qualification_passed=False
    result=None
    failure=None
    with contextlib.redirect_stdout(io.StringIO()),contextlib.redirect_stderr(io.StringIO()):
        try:
            result=verify(ROOT)
            qualification_passed=True
        except (ValueError,AssertionError) as error:
            # Record qualification failure, never reinterpret a failed verifier as a pass.
            failure=dict(type=type(error).__name__,message=str(error),
                cause_type=type(error.__cause__).__name__ if error.__cause__ else None)
    if audit()!=before:
        raise ValueError('Source or catalog audit changed during probe')
    return dict(format='video-multilabel-current-baseline.v1',read_only=True,catalog_mutations=0,
        qualification_passed=qualification_passed,approved=False,synthetic_only=True,
        existing_served_verification=result,failure=failure,runtime_drift=before['runtime_drift'],
        implementation_drift=before['implementation_drift'],validator_sha256=sha(__file__),
        retained_verifier_sha256=sha(ROOT/'scripts/verify_video_multilabel_publication.py'),
        limits=['A failed verifier does not requalify the existing derivative or the original. Historical reports remain unchanged.',
                'Current runtime qualification and original sharded-access/reuse gaps require separate work.'])


if __name__=='__main__':
    result=validate()
    (ROOT/'docs/reviews/video_multilabel_current_baseline.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))
