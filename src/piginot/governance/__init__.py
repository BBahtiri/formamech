"""Population-centric PLM, EU AI Act Annex IV & Veteran-Engineer HITL Governance (V5-4)."""

from piginot.governance.hitl_signoff import (
    HITL_SIGNOFF_MACHINE,
    HITL_SIGNOFF_SCHEMA,
    HandCalculationSanityCheck,
    HitlSignoffRecord,
    HitlSignoffState,
    ReviewerRole,
    create_hitl_signoff,
    require_valid_hitl_signoff,
)
from piginot.governance.plm_dossier import (
    EU_AI_ACT_ANNEX_IV_VERSION,
    PLM_POPULATION_PACKAGE_SCHEMA,
    DesignPopulationPackage,
    DesignVariantSummary,
    PlmExportResult,
    export_plm_compliance_package,
    verify_plm_compliance_package,
)

__all__ = (
    "EU_AI_ACT_ANNEX_IV_VERSION",
    "HITL_SIGNOFF_MACHINE",
    "HITL_SIGNOFF_SCHEMA",
    "PLM_POPULATION_PACKAGE_SCHEMA",
    "DesignPopulationPackage",
    "DesignVariantSummary",
    "HandCalculationSanityCheck",
    "HitlSignoffRecord",
    "HitlSignoffState",
    "PlmExportResult",
    "ReviewerRole",
    "create_hitl_signoff",
    "export_plm_compliance_package",
    "require_valid_hitl_signoff",
    "verify_plm_compliance_package",
)
