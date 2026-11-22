"""Interactive 2D Engineering Workbench backend (Feature 4 / M2)."""

from piginot.ui.workbench.cad_viewer import (
    ROLE_COLORS,
    build_cad_boundary_figure,
    extract_selected_region,
    render_workbench_cad_png,
)
from piginot.ui.workbench.job_runner import (
    JOB_STATUS_SCHEMA,
    JobKind,
    JobRecord,
    JobState,
    WorkbenchJobRunner,
)
from piginot.ui.workbench.physics_forms import (
    MATERIAL_PRESETS,
    BoundaryAssignment,
    BoundaryKind,
    MaterialPreset,
    ParameterSpecDraft,
    ParamRole,
    WorkbenchPhysicsDraft,
    build_settings_document,
    default_boundaries_from_preview,
    default_draft_from_preview,
    prepare_and_plan_workspace,
    validate_boundaries_against_preview,
)
from piginot.ui.workbench.state import (
    SUPPORTED_CAD_EXTENSIONS,
    WORKBENCH_STATE_SCHEMA,
    StageProgress,
    WorkbenchWorkspace,
    canonical_settings_bytes,
    generate_sample_bracket_step,
)

__all__ = (
    "JOB_STATUS_SCHEMA",
    "MATERIAL_PRESETS",
    "ROLE_COLORS",
    "SUPPORTED_CAD_EXTENSIONS",
    "WORKBENCH_STATE_SCHEMA",
    "BoundaryAssignment",
    "BoundaryKind",
    "JobKind",
    "JobRecord",
    "JobState",
    "MaterialPreset",
    "ParamRole",
    "ParameterSpecDraft",
    "StageProgress",
    "WorkbenchJobRunner",
    "WorkbenchPhysicsDraft",
    "WorkbenchWorkspace",
    "build_cad_boundary_figure",
    "build_settings_document",
    "canonical_settings_bytes",
    "default_boundaries_from_preview",
    "default_draft_from_preview",
    "extract_selected_region",
    "generate_sample_bracket_step",
    "prepare_and_plan_workspace",
    "render_workbench_cad_png",
    "validate_boundaries_against_preview",
)
