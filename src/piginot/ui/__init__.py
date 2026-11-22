"""Interactive Engineering Workbench & Verification Inspector (`formamech[ui]`)."""

from __future__ import annotations

from piginot.ui.inspector import (
    INSPECTOR_SCHEMA,
    InspectorSnapshot,
    build_inspector_snapshot,
    build_plotly_figures,
    render_trust_dossier_markdown,
)
from piginot.ui.workbench import (
    MATERIAL_PRESETS,
    BoundaryAssignment,
    JobRecord,
    ParameterSpecDraft,
    StageProgress,
    WorkbenchJobRunner,
    WorkbenchPhysicsDraft,
    WorkbenchWorkspace,
    build_cad_boundary_figure,
    build_settings_document,
    default_draft_from_preview,
    extract_selected_region,
    generate_sample_bracket_step,
    prepare_and_plan_workspace,
    render_workbench_cad_png,
    validate_boundaries_against_preview,
)

__all__ = (
    "INSPECTOR_SCHEMA",
    "MATERIAL_PRESETS",
    "BoundaryAssignment",
    "InspectorSnapshot",
    "JobRecord",
    "ParameterSpecDraft",
    "StageProgress",
    "WorkbenchJobRunner",
    "WorkbenchPhysicsDraft",
    "WorkbenchWorkspace",
    "build_cad_boundary_figure",
    "build_inspector_snapshot",
    "build_plotly_figures",
    "build_settings_document",
    "default_draft_from_preview",
    "extract_selected_region",
    "generate_sample_bracket_step",
    "prepare_and_plan_workspace",
    "render_trust_dossier_markdown",
    "render_workbench_cad_png",
    "validate_boundaries_against_preview",
)
