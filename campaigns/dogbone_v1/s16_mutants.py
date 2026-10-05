"""The twenty S16 mutants (ADR 0022 §1.13) as one-line source edits.

Each mutant edits the production function that holds the defect; nothing is
simulated beside it. The primary gate is the one ADR 0022 names.
"""

from __future__ import annotations

from piginot.mutation import Mutant, SourceEdit


def _edit(target: str, old: str, new: str) -> SourceEdit:
    return SourceEdit(target, old, new)


MUTANTS: tuple[Mutant, ...] = (
    Mutant(
        "BODY_FORCE_SIGN",
        "G8.seeded-defects",
        "the registered body force enters the weak form with the wrong sign",
        (
            _edit(
                "piginot.fem.binding:_registered_body_force",
                "return dot(data.body_force, test)",
                "return -dot(data.body_force, test)",
            ),
        ),
    ),
    Mutant(
        "BOUNDARY_NORMAL",
        "G8.seeded-defects",
        "traction-field providers receive the inward normal",
        (
            _edit(
                "piginot.fem.binding:_traction_field",
                "provider.evaluate(points, normals)",
                "provider.evaluate(points, -normals)",
            ),
        ),
    ),
    Mutant(
        "TAG_SWAP",
        "G8.seeded-defects",
        "the left and right boundary tags resolve to each other's facets",
        (
            _edit(
                "piginot.fem.binding:facets_for_tag",
                "tagged_facets = mesh.semantic_tags[tag]",
                "tagged_facets = mesh.semantic_tags["
                '{"left": "right", "right": "left"}.get(tag, tag)]',
            ),
        ),
    ),
    Mutant(
        "CLOSURE_WRONG_ROOT",
        "G8.seeded-defects",
        "the plane-stress closure returns the negative root",
        (
            _edit(
                "piginot.physics.plane_stress:_nearest_representable_root",
                "return torch.gather(candidates, 0, best).squeeze(0)",
                "return -torch.gather(candidates, 0, best).squeeze(0)",
            ),
        ),
    ),
    Mutant(
        "INCONSISTENT_TANGENT",
        "G8.seeded-defects",
        "the consistent tangent is scaled by 0.9",
        (
            _edit(
                "piginot.physics.neo_hookean:consistent_tangent",
                "return material_term + volumetric_term - geometric_term",
                "return 0.9 * (material_term + volumetric_term - geometric_term)",
            ),
        ),
    ),
    Mutant(
        "ANISOTROPIC_SCALING",
        "G8.seeded-defects",
        "reported solve coordinates are stretched by 1.2 along x",
        (
            _edit(
                "piginot.fem.backend:_dof_coordinates",
                "(float(basis.doflocs[0, dof]), float(basis.doflocs[1, dof]))",
                "(1.2 * float(basis.doflocs[0, dof]), float(basis.doflocs[1, dof]))",
            ),
        ),
    ),
    Mutant(
        "NODE_SPLIT",
        "G9.no-node-split",
        "nodes of one case may carry different evidence roles",
        (
            _edit(
                "piginot.doe.roles:CaseRoleRegistry.validate_node_observations",
                "if len(observed_roles) != 1:",
                "if not observed_roles:",
            ),
        ),
    ),
    Mutant(
        "DROP_PROJECTION_ERROR",
        "G10.projection-error",
        "projected fields carry a zero transfer-error envelope",
        (
            _edit(
                "piginot.datasets.projection:_sample_envelope",
                "max(abs(sample[component] - value[component]) for sample in samples)",
                "0.0 * max(abs(sample[component] - value[component]) "
                "for sample in samples)",
            ),
        ),
    ),
    Mutant(
        "STRESS_NODE_PAIRING",
        "G10.cross-mesh",
        "cross-mesh stress integrals sum samples without their weights",
        (
            _edit(
                "piginot.datasets.projection:_physical_integral",
                "result[component] += weight * value",
                "result[component] += value",
            ),
        ),
    ),
    Mutant(
        "NORMALIZE_ON_ALL_SPLITS",
        "G10.normalization",
        "the S10 close-out fits normalisation on every role",
        (
            _edit(
                "campaigns.dogbone_v1.s10_closeout:_normalization_evidence",
                "training = tuple(record for record in records "
                "if record.role is EvidenceRole.TRAIN)",
                "training = tuple(records)",
            ),
        ),
    ),
    Mutant(
        "IGNORE_DIRICHLET_LIFTING",
        "G11.lifting-mutant",
        "the tension lifting drops the prescribed end displacement",
        (
            _edit(
                "piginot.operators.energy_operator:AxisAlignedTensionLifting.__call__",
                "offset = torch.stack((d * xi, torch.zeros_like(eta)), dim=-1)",
                "offset = torch.stack((0.0 * d * xi, torch.zeros_like(eta)), dim=-1)",
            ),
        ),
    ),
    Mutant(
        "COORDINATE_ANISOTROPY",
        "G12.anisotropy",
        "the strong residual converts every derivative with the x scale",
        (
            _edit(
                "piginot.objectives.equilibrium:strong_equilibrium_residual",
                "gradient[..., column] / conversion.to_mm[column]",
                "gradient[..., column] / conversion.to_mm[0]",
            ),
        ),
    ),
    Mutant(
        "HOLDOUT_ACCESS",
        "G12.role-isolation",
        "an objective accepts HOLDOUT evidence",
        (
            _edit(
                "piginot.objectives.registry:evaluate_objective",
                "if role not in _OBJECTIVE_ROLES:",
                "if role not in _OBJECTIVE_ROLES | {EvidenceRole.HOLDOUT}:",
            ),
        ),
    ),
    Mutant(
        "SELECT_ON_HOLDOUT",
        "G13.holdout",
        "checkpoint selection accepts non-TRAIN results",
        (
            _edit(
                "piginot.training.selection:_validate_result_matrix",
                "if result.role is not EvidenceRole.TRAIN:",
                "if result.role is EvidenceRole.LEGACY:",
            ),
        ),
    ),
    Mutant(
        "CHOOSE_NONRANKED_CHECKPOINT",
        "G13.selection",
        "the decision serves the last-ranked eligible checkpoint",
        (
            _edit(
                "piginot.training.selection:SelectionDecision.selected_result",
                "selected_id = self.ranked_eligible_trial_ids[0]",
                "selected_id = self.ranked_eligible_trial_ids[-1]",
            ),
        ),
    ),
    Mutant(
        "ZERO_MISSING_BUDGET_COMPONENT",
        "G14.budget",
        "a missing budget component counts as zero",
        (
            _edit(
                "piginot.verification.budget:CaseBudgetEntry.total",
                "            if value is None:\n                return None\n",
                "            value = 0.0 if value is None else value\n",
            ),
        ),
    ),
    Mutant(
        "PASS_ON_NOT_RUN",
        "G14.reducer",
        "the reducer counts NOT_RUN checks as PASS",
        (
            _edit(
                "piginot.verification.reducer:reduce_families",
                "statuses.get(check, CheckStatus.NOT_RUN) for check in family.check_ids",
                "(CheckStatus.PASS if statuses.get(check, CheckStatus.NOT_RUN) "
                "is CheckStatus.NOT_RUN else statuses[check]) "
                "for check in family.check_ids",
            ),
        ),
    ),
    Mutant(
        "REVEAL_BEFORE_CONSUME",
        "G14.holdout-order",
        "the holdout is read before its consumption record is checked",
        (
            _edit(
                "piginot.verification.holdout:reveal_after_consumption",
                "    if not record_path.is_file():",
                "    revealed = reader()\n    if not record_path.is_file():",
            ),
            _edit(
                "piginot.verification.holdout:reveal_after_consumption",
                "    return reader()",
                "    return revealed",
            ),
        ),
    ),
    Mutant(
        "SERVE_P_MATERIAL",
        "G15.substitution",
        "predict serves the material law in place of the declared stress",
        (
            _edit(
                "piginot.bundle.inference:_fields",
                "P_served=served.stress,",
                "P_served=material_stress,",
            ),
        ),
    ),
    Mutant(
        "UNDETECTED_BUNDLE_TAMPER",
        "G15.manifest",
        "bundle verification compares hashes only when sizes differ",
        (
            _edit(
                "piginot.bundle.manifest:verify_bundle",
                "if sha256_bytes(payload) != entry.sha256:",
                "if len(payload) != entry.bytes "
                "and sha256_bytes(payload) != entry.sha256:",
            ),
        ),
    ),
)

__all__ = ("MUTANTS",)
