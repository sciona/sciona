# Agent Instructions for sciona-matcher

## Python Environment

Use the virtual environment at `/Users/conrad/personal/sciona-matcher/.venv` for all Python operations (tests, installs, imports).

```bash
/Users/conrad/personal/sciona-matcher/.venv/bin/python -m pytest ...
```

All sibling provider repos (`sciona-atoms`, `sciona-atoms-signal`, `sciona-atoms-ml`, etc.) are installed into this venv. Do not use the system Python or conda environments.

## Shared CPU budget

The user requires at most **four workers/CPU cores total across all agent-run
scripts**, including concurrent training jobs, test processes, data-loader
workers and native numerical thread pools. This is a shared budget, not a
per-script allowance.

- Check active jobs before launching compute. Reserve their allocations first.
- Run foreground compute serially and explicitly bound native thread pools and
  subprocess/data-loader worker counts within the remaining allocation.
- Do not resume an older job whose configured worker count exceeds this budget.
  Suspend it while arranging a qualified continuation under the current limit.
- The Salt snapshot pilot has completed its three scheduled phases and its
  process has exited; it no longer reserves two cores. The Wheat full-fit
  process group remains suspended because it uses 16 workers; do not resume
  that configuration. Reinspect process state before relying on these
  observations.

## Templated Dataset Confidentiality

Treat templated datasets and their metadata as non-public. Never commit dataset
contents, names, recording or subject identifiers, source paths, filenames,
directory layouts, schemas, channel inventories, excerpts, timestamps,
checksums, URLs, or generated adapters derived from a real template.

Use synthetic committed fixtures. Real-data evaluations must receive their
inputs through local runtime configuration excluded from Git, and committed
benchmark evidence may contain only opaque aliases and aggregate metrics.
Inspect staged evaluation and generated files for identifying metadata before
every commit. If publication appears necessary, stop and request an explicitly
approved public representation.
