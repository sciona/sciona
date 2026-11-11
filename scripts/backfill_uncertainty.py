#!/usr/bin/env python3
"""Compatibility wrapper for the provider-owned uncertainty backfill."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from sciona.atoms.supabase_backfill import (
    backfill_uncertainty,
    build_uncertainty_rows,
    build_uncertainty_rows_from_payload,
    create_supabase_client_from_env,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command-line flags."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Build rows and log stats without inserting")
    parser.add_argument(
        "--atoms-root",
        type=Path,
        default=None,
        help="Optional single artifact root. When omitted, all configured provider roots are searched.",
    )
    return parser.parse_args()


def main() -> None:
    """CLI entrypoint."""
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    result = backfill_uncertainty(
        create_supabase_client_from_env(),
        atoms_root=args.atoms_root,
        dry_run=args.dry_run,
    )
    logger.info("Uncertainty backfill complete: %s", result)


if __name__ == "__main__":
    main()
