-- Phase 2: Upgrade atom_uncertainty_estimates to store v2 Mixture-of-Regimes,
-- SufficientStatProjector specs, ConformalEnvelopes, and LowRankPlusTailOperator
-- spectral channels alongside backward-compatible v1 scalar columns.

ALTER TABLE public.atom_uncertainty_estimates
    ADD COLUMN IF NOT EXISTS schema_version TEXT NOT NULL DEFAULT '1.0',
    ADD COLUMN IF NOT EXISTS input_regimes JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS output_projectors JSONB NOT NULL DEFAULT '{}'::jsonb,
    ADD COLUMN IF NOT EXISTS components JSONB NOT NULL DEFAULT '[]'::jsonb;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'public.atom_uncertainty_estimates'::regclass
          AND conname = 'atom_uncertainty_estimates_schema_version_check'
    ) THEN
        ALTER TABLE public.atom_uncertainty_estimates
            ADD CONSTRAINT atom_uncertainty_estimates_schema_version_check
            CHECK (schema_version IN ('1.0', '2.0'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = 'public.atom_uncertainty_estimates'::regclass
          AND conname = 'atom_uncertainty_estimates_jsonb_shape_check'
    ) THEN
        ALTER TABLE public.atom_uncertainty_estimates
            ADD CONSTRAINT atom_uncertainty_estimates_jsonb_shape_check
            CHECK (
                jsonb_typeof(input_regimes) = 'object'
                AND jsonb_typeof(output_projectors) = 'object'
                AND jsonb_typeof(components) = 'array'
            );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS idx_uncertainty_atom_schema_version
    ON public.atom_uncertainty_estimates (atom_id, schema_version);

COMMENT ON COLUMN public.atom_uncertainty_estimates.schema_version IS
    'Uncertainty schema version: 1.0 legacy scalar estimate; 2.0 mixture-of-regimes + spectral channel profile.';
COMMENT ON COLUMN public.atom_uncertainty_estimates.input_regimes IS
    'Per-input-port PortRegimeSpec map containing SufficientStatProjector, ExponentialFamilyMixture, and ConformalEnvelope.';
COMMENT ON COLUMN public.atom_uncertainty_estimates.output_projectors IS
    'Per-output-port ProjectorSpec map defining output sufficient-statistic coordinates.';
COMMENT ON COLUMN public.atom_uncertainty_estimates.components IS
    'Array of ComponentChannel specifications storing per-regime LowRankPlusTailOperator Jacobians and intrinsic noise covariances.';
