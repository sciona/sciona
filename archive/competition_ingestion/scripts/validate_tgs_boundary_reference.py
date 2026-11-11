"""Compare independent boundary mathematics with base R using synthetic arrays."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from sciona.tgs_boundaries import boundary_features


def main():
    reports = []
    with tempfile.TemporaryDirectory(prefix='tgs-scale-') as temp:
        work = Path(temp)
        # Independent base-R evaluation of the documented two-pass scaling.
        program = work / 'reference.R'
        program.write_text("""
args <- commandArgs(trailingOnly=TRUE)
a <- as.matrix(read.csv(args[1], header=FALSE))
z <- array(a, dim=c(4,5,6))
e <- list(u=2*z[,1,]-z[,2,], d=2*z[,5,]-z[,4,],
          l=2*z[,,1]-z[,,2], r=2*z[,,6]-z[,,5])
for (name in names(e)) {
  b <- scale(e[[name]])
  b <- t(apply(b, 1, scale))
  b[is.na(b)] <- 0
  write.table(b, file=paste0(args[2], '/', name, '.csv'),
              row.names=FALSE, col.names=FALSE, sep=',')
}
m <- matrix(NA, nrow=1, ncol=1)
m[0,0] <- 7
stopifnot(is.na(m[1,1]))
cat(R.version.string)
""")
        values = np.arange(120.).reshape(4,5,6)
        ordinary = np.sin(values / 7) + np.cos(values / 13)
        partly_constant = ordinary.copy()
        partly_constant[:,:,0] = 1
        partly_constant[:,:,1] = 1
        for name, population in [('ordinary', ordinary), ('constant', np.ones_like(ordinary)),
                                 ('partly_constant', partly_constant)]:
            np.savetxt(work / 'input.csv', population.transpose(0,2,1).reshape(4,-1), delimiter=',')
            process = subprocess.run(['Rscript', str(program), str(work/'input.csv'), str(work)],
                                     text=True, capture_output=True, check=True, timeout=30)
            actual = boundary_features(population)
            errors = {}
            for direction in actual:
                expected = np.loadtxt(work / (direction+'.csv'), delimiter=',')
                np.testing.assert_allclose(actual[direction], expected, atol=1e-12, rtol=1e-12)
                errors[direction] = float(np.max(np.abs(actual[direction]-expected)))
            reports.append(dict(case=name, maximum_absolute_errors=errors))
    root = Path(__file__).resolve().parents[1]
    report = dict(passed=True, synthetic_only=True, catalog_mutations=0,
                  r_version=process.stdout.strip(), cases=reports,
                  zero_based_matrix_assignment_drops_origin=True,
                  scope='Boundary features only; full mosaic construction not yet implemented.',
                  sha256={name:hashlib.sha256((root/name).read_bytes()).hexdigest() for name in
                          ['sciona/tgs_boundaries.py','scripts/validate_tgs_boundary_reference.py']})
    (root/'docs/reviews/competition_tgs_boundary_reference.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
