"""S13 for the notched campaign: protocol, geometry-only data and the trial.

The S13c recipe of the DogBone campaign, unchanged in architecture, objective,
schedule and seeds (ADR 0025 decision 5), with the four inputs of this family:
``width``, ``notch_radius``, ``notch_depth`` and ``nominal_axial_strain``.
Training reads the morphed meshes of the TRAIN-fit requests only, so it binds
to the S10 request plan rather than to FEM results; the z-score of the inputs
is fitted on the TRAIN-fit coordinates. Selection admits the single trial with
preregistered absolute guardrails (decision 6).

Usage (clean committed revision)::

    python -m campaigns.notched_v1.s13 protocol          # write S13-PROTOCOL.json
    python -m campaigns.notched_v1.s13 train --run-root <abs path outside repo>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import platform
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path
from types import MappingProxyType
from typing import ClassVar, Final

import torch

from campaigns.dogbone_v1.s10_production import default_s10_execution_profile
from campaigns.dogbone_v1.s13_pilot import _absolute_directory, _git_provenance
from campaigns.dogbone_v1.s13c_trial import (
    _log_writer,
    latest_checkpoint,
    segment_targets,
)
from campaigns.notched_v1.family import (
    HALF_LENGTH_MM,
    INPUTS,
    LEFT_TAG,
    LOADED_END_TAG,
    LOWER_TAG,
    prescribed_displacement_mm,
)
from campaigns.notched_v1.pilot import ARTIFACTS, write_exact
from campaigns.notched_v1.s10 import load_public_sources, public_requests, request_plan
from piginot.contracts import (
    ExecutionProfile,
    MaterialSpec,
    Unit,
    canonical_json_bytes,
    hash_spec,
)
from piginot.doe import EvidenceRole, PublishedCase, PublishedCaseStatus
from piginot.fem import FemRequest
from piginot.objectives import (
    PotentialEnergyConfig,
    PotentialEnergyTerms,
    TriangleDomain,
    plane_stress_potential_energy,
    stratified_triangle_quadrature,
)
from piginot.operators import (
    ENERGY_ARCHITECTURE_ID,
    AxisAlignedTensionLifting,
    EnergyFormOperator,
    EnergyOperatorConfig,
    QueryFrame,
    energy_module_digest,
)
from piginot.training import (
    AbsoluteSelectionGuardrails,
    CosineTailSchedule,
    DeterministicSamplerPlan,
    EnergyTrainerConfig,
    EnergyTrainingResult,
    GridSearchSpace,
    SelectionPolicy,
    TrainingPopulation,
    TrainingRunBinding,
    model_state_hash,
    train_energy_to_checkpoint,
)

PROTOCOL_ID: Final = "notched-s13-energy-form-operator-v1"
SPLIT_POLICY_ID: Final = "notched-s13-derived-strata-quarter-screen-v1"
SAMPLER_POLICY_ID: Final = "notched-s13-blake2b-epoch-permutation-v1"
LIFTING_POLICY_ID: Final = "notched-quarter-axis-aligned-tension-lifting-v1"
CONDITIONING_POLICY_ID: Final = "notched-train-fit-zscore-geometry-and-strain-v1"
QUADRATURE_POLICY_ID: Final = "s10-morph-mesh-triangles-torch-cpu-stratified-v1"
SCREEN_METRIC_POLICY_ID: Final = (
    "notched-s13-screen-32-constitutive-vertex-stress-train-orientation-v1"
)
GEOMETRY_SNAPSHOT_SCHEMA: Final = "notched-s13-train-fit-geometry-v1"
CONDITIONING: Final = INPUTS
PROTOCOL_PATH = ARTIFACTS / "S13-PROTOCOL.json"
STRESS_REFERENCE_MPA: Final = 760.0
# Pinned from this machine's initialisation; see the DogBone S13c note on
# CPU-dependent normal() kernels.
INITIAL_WEIGHTS_HASH: Final = (
    "5ddc674536efcf02f3807f71601990509cf45f5c238caf76913981a7318bc913"
)
EXECUTION_DTYPE: Final = "float32"
EXECUTION_BLAS_THREADS: Final = 1
FIT_COUNT, SCREEN_COUNT = 96, 32
TRAINING_SEED, SAMPLER_SEED, INITIALIZATION_SEED = 13_101, 13_102, 13_103
ENERGY_EVALUATION_SEED = 13_104
GUARDRAILS = AbsoluteSelectionGuardrails(
    maximum_median_absolute_peak_error=0.05,
    maximum_p_relative_l2=0.05,
    maximum_u_relative_l2=0.02,
)
RUN_BINDING_NAME = "run-binding.json"
TRAINING_LOG_NAME = "training-log.jsonl"
_SAMPLER_PERSON = b"piginot-s13c-v1"
_SPLIT_PERSON = b"piginot-ntv1-sp"
_DERIVED = ("net_section_factor", "notch_sharpness")
_FEM_MATERIAL_MODEL = "finite_strain_plane_stress_neo_hookean"
_KERNEL_MATERIAL_MODEL = "compressible_neo_hookean"
_GEOMETRY_TOLERANCE_MM = 1.0e-9
Progress = Callable[[str], None]


def operator_config() -> EnergyOperatorConfig:
    return EnergyOperatorConfig(parameter_names=CONDITIONING)


def energy_config() -> PotentialEnergyConfig:
    return PotentialEnergyConfig(
        points_per_cell=3,
        barrier_weight=1.0e3,
        barrier_threshold=0.05,
        admissibility_floor=1.0e-6,
    )


def search_space() -> GridSearchSpace:
    return GridSearchSpace(
        learning_rates=(3.0e-4,),
        physics_weight_scales=(1.0,),
        optimizer_steps=2_400,
        case_batch_size=4,
        maximum_trials=1,
    )


def schedule() -> CosineTailSchedule:
    return CosineTailSchedule(
        peak_learning_rate=3.0e-4,
        final_learning_rate=3.0e-6,
        constant_steps=1_600,
        total_steps=2_400,
    )


def build_initial_operator(seed: int = INITIALIZATION_SEED) -> EnergyFormOperator:
    state = torch.random.get_rng_state()
    try:
        torch.manual_seed(seed)
        model = EnergyFormOperator(operator_config()).to(torch.float32)
    finally:
        torch.random.set_rng_state(state)
    return model


@dataclass(frozen=True, slots=True)
class ConditioningNormalization:
    """Z-score of the conditioning inputs, fitted on the TRAIN-fit cases."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    parameter_names: tuple[str, ...]
    mean: tuple[float, ...]
    standard_deviation: tuple[float, ...]
    fit_case_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        width = len(self.parameter_names)
        if len(self.mean) != width or len(self.standard_deviation) != width:
            raise ValueError("normalization statistics must match the names")
        if any(not math.isfinite(v) for v in (*self.mean, *self.standard_deviation)):
            raise ValueError("normalization statistics must be finite")
        if any(value <= 0.0 for value in self.standard_deviation):
            raise ValueError("standard deviations must be positive")

    @property
    def content_hash(self) -> str:
        return hash_spec(self, "content")

    def vector(self, coordinates: Mapping[str, float]) -> torch.Tensor:
        return torch.tensor(
            [
                (float(coordinates[name]) - mean) / deviation
                for name, mean, deviation in zip(
                    self.parameter_names,
                    self.mean,
                    self.standard_deviation,
                    strict=True,
                )
            ],
            dtype=torch.float64,
        )


def fit_normalization(cases: Sequence[PublishedCase]) -> ConditioningNormalization:
    ordered = sorted(cases, key=lambda case: case.case_id)
    if any(
        case.role is not EvidenceRole.TRAIN
        or case.status is not PublishedCaseStatus.ACTIVE
        for case in ordered
    ):
        raise PermissionError("normalization is fitted on ACTIVE TRAIN cases only")
    means, deviations = [], []
    for name in CONDITIONING:
        values = [float(case.primitive_coordinates[name]) for case in ordered]
        mean = math.fsum(values) / len(values)
        variance = math.fsum((value - mean) ** 2 for value in values) / len(values)
        means.append(mean)
        deviations.append(math.sqrt(variance) if variance > 0.0 else 1.0)
    return ConditioningNormalization(
        parameter_names=CONDITIONING,
        mean=tuple(means),
        standard_deviation=tuple(deviations),
        fit_case_ids=tuple(case.case_id for case in ordered),
    )


def _rank(case_id: str) -> bytes:
    return hashlib.blake2b(
        f"{SPLIT_POLICY_ID}\0{case_id}".encode(), digest_size=32, person=_SPLIT_PERSON
    ).digest()


def split_train_cases(
    cases: Sequence[PublishedCase],
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """A quarter of each derived-strata cell to TRAIN-screen, by keyed rank."""

    groups: dict[tuple[int, ...], list[PublishedCase]] = {}
    for case in cases:
        groups.setdefault(tuple(case.strata[name] for name in _DERIVED), []).append(
            case
        )
    quotas = {key: len(group) // 4 for key, group in groups.items()}
    remaining = SCREEN_COUNT - sum(quotas.values())
    for key in sorted(groups, key=lambda key: (-(len(groups[key]) % 4), key))[
        :remaining
    ]:
        quotas[key] += 1
    screen: set[str] = set()
    for key, group in groups.items():
        ranked = sorted(group, key=lambda case: (_rank(case.case_id), case.case_id))
        screen.update(case.case_id for case in ranked[: quotas[key]])
    all_ids = {case.case_id for case in cases}
    if len(screen) != SCREEN_COUNT or len(all_ids) != FIT_COUNT + SCREEN_COUNT:
        raise ValueError("TRAIN split did not produce the fixed 96/32 population")
    return tuple(sorted(all_ids - screen)), tuple(sorted(screen))


@dataclass(frozen=True, slots=True)
class NotchedObjectivePolicy:
    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    energy: PotentialEnergyConfig
    lifting_policy_id: str
    quadrature_policy_id: str
    conditioning_policy_id: str
    conditioning_parameter_names: tuple[str, ...]
    normalization_hash: str

    @property
    def policy_hash(self) -> str:
        return hash_spec(self, "recipe")


@dataclass(frozen=True, slots=True)
class NotchedS13Protocol:
    """Complete preregistration surface before any notched training."""

    __canonical_namespace__: ClassVar[str] = "piginot.campaigns"

    protocol_id: str
    campaign_hash: str
    public_manifest_hash: str
    request_plan_hash: str
    dataset_snapshot_hash: str
    normalization: ConditioningNormalization
    normalization_hash: str
    architecture_id: str
    architecture_config: EnergyOperatorConfig
    architecture_module_digest: str
    initial_weights_hash: str
    objective: NotchedObjectivePolicy
    split_policy_id: str
    fit_case_ids: tuple[str, ...]
    screen_case_ids: tuple[str, ...]
    search_space: GridSearchSpace
    schedule: CosineTailSchedule
    gradient_clip_norm: float
    training_seed: int
    sampler_seed: int
    initialization_seed: int
    sampler_policy_id: str
    execution_dtype: str
    execution_blas_threads: int
    screen_metric_policy_id: str
    selection_policy: SelectionPolicy

    def __post_init__(self) -> None:
        if self.protocol_id != PROTOCOL_ID:
            raise ValueError("unexpected notched S13 protocol_id")
        if self.architecture_id != ENERGY_ARCHITECTURE_ID:
            raise ValueError("architecture_id must be the energy-form operator")
        if self.architecture_config != operator_config():
            raise ValueError("architecture_config differs from the registered operator")
        if self.normalization_hash != self.normalization.content_hash:
            raise ValueError("normalization_hash does not match the statistics")
        if self.normalization.fit_case_ids != self.fit_case_ids:
            raise ValueError("normalization must be fitted on the TRAIN-fit cases")
        if self.objective != objective_policy(self.normalization_hash):
            raise ValueError("objective policy differs from the preregistration")
        if (
            len(self.fit_case_ids) != FIT_COUNT
            or len(self.screen_case_ids) != SCREEN_COUNT
        ):
            raise ValueError("TRAIN split must be 96 fit and 32 screen cases")
        if set(self.fit_case_ids) & set(self.screen_case_ids):
            raise ValueError("TRAIN-fit and TRAIN-screen must be disjoint")
        if self.search_space != search_space() or self.schedule != schedule():
            raise ValueError("search space or schedule differs from the S13c recipe")
        if self.selection_policy != SelectionPolicy(self.search_space, GUARDRAILS):
            raise ValueError("selection policy differs from the absolute guardrails")
        if (
            self.training_seed,
            self.sampler_seed,
            self.initialization_seed,
        ) != (TRAINING_SEED, SAMPLER_SEED, INITIALIZATION_SEED):
            raise ValueError("seeds differ from the S13c recipe")
        if self.gradient_clip_norm != 1.0:
            raise ValueError("gradient_clip_norm must remain 1.0")

    @property
    def protocol_hash(self) -> str:
        return hash_spec(self, "recipe")

    @property
    def objective_hash(self) -> str:
        return self.objective.policy_hash

    def to_bytes(self) -> bytes:
        return canonical_json_bytes(self)


def objective_policy(normalization_hash: str) -> NotchedObjectivePolicy:
    return NotchedObjectivePolicy(
        energy=energy_config(),
        lifting_policy_id=LIFTING_POLICY_ID,
        quadrature_policy_id=QUADRATURE_POLICY_ID,
        conditioning_policy_id=CONDITIONING_POLICY_ID,
        conditioning_parameter_names=CONDITIONING,
        normalization_hash=normalization_hash,
    )


def train_cases() -> tuple[PublishedCase, ...]:
    _, manifest = load_public_sources()
    cases = tuple(
        sorted(
            (
                case
                for case in manifest.cases
                if case.role is EvidenceRole.TRAIN
                and case.status is PublishedCaseStatus.ACTIVE
            ),
            key=lambda case: case.case_id,
        )
    )
    if len(cases) != FIT_COUNT + SCREEN_COUNT:
        raise ValueError("the notched manifest must expose 128 active TRAIN cases")
    return cases


def geometry_snapshot_hash(
    fit_case_ids: Sequence[str], profile: ExecutionProfile
) -> str:
    """Identity of the TRAIN-fit geometry: the request recipes in the S10 plan."""

    plan = request_plan(profile)
    fit = set(fit_case_ids)
    entries = [
        {
            "case_id": entry.case_id,
            "leakage_group_id": entry.leakage_group_id,
            "request_recipe_hash": entry.request_recipe_hash,
            "mesh_content_hash": entry.mesh_content_hash,
        }
        for entry in plan.entries
        if entry.case_id in fit
    ]
    if len(entries) != len(fit) or any(
        plan.entry_by_case_id(case_id).role is not EvidenceRole.TRAIN for case_id in fit
    ):
        raise ValueError("TRAIN-fit cases are missing from the request plan")
    return hash_spec(
        {
            "schema": GEOMETRY_SNAPSHOT_SCHEMA,
            "request_plan_hash": plan.content_hash,
            "entries": entries,
        },
        "content",
    )


def build_protocol(*, initial_weights_hash: str | None = None) -> NotchedS13Protocol:
    """Build the protocol from the committed public S9 artifacts only."""

    checkpoint, manifest = load_public_sources()
    cases = train_cases()
    fit_ids, screen_ids = split_train_cases(cases)
    fit_cases = tuple(case for case in cases if case.case_id in set(fit_ids))
    normalization = fit_normalization(fit_cases)
    profile = default_s10_execution_profile()
    model = build_initial_operator()
    observed = model_state_hash(model)
    pinned = (
        INITIAL_WEIGHTS_HASH if initial_weights_hash is None else initial_weights_hash
    )
    if pinned != observed:
        raise ValueError(f"initial weights {observed} differ from the pinned {pinned}")
    space = search_space()
    return NotchedS13Protocol(
        protocol_id=PROTOCOL_ID,
        campaign_hash=checkpoint.campaign_hash,
        public_manifest_hash=manifest.content_hash,
        request_plan_hash=request_plan(profile).content_hash,
        dataset_snapshot_hash=geometry_snapshot_hash(fit_ids, profile),
        normalization=normalization,
        normalization_hash=normalization.content_hash,
        architecture_id=model.architecture_id,
        architecture_config=model.config,
        architecture_module_digest=energy_module_digest(model),
        initial_weights_hash=observed,
        objective=objective_policy(normalization.content_hash),
        split_policy_id=SPLIT_POLICY_ID,
        fit_case_ids=fit_ids,
        screen_case_ids=screen_ids,
        search_space=space,
        schedule=schedule(),
        gradient_clip_norm=1.0,
        training_seed=TRAINING_SEED,
        sampler_seed=SAMPLER_SEED,
        initialization_seed=INITIALIZATION_SEED,
        sampler_policy_id=SAMPLER_POLICY_ID,
        execution_dtype=EXECUTION_DTYPE,
        execution_blas_threads=EXECUTION_BLAS_THREADS,
        screen_metric_policy_id=SCREEN_METRIC_POLICY_ID,
        selection_policy=SelectionPolicy(space, GUARDRAILS),
    )


def load_committed_protocol(root: Path | None = None) -> NotchedS13Protocol:
    base = Path(__file__).resolve().parents[2] if root is None else root
    protocol = build_protocol()
    if (base / PROTOCOL_PATH).read_bytes() != protocol.to_bytes():
        raise ValueError("S13 protocol differs from the committed preregistration")
    return protocol


def _epoch_order(population: int, seed: int, epoch: int) -> tuple[int, ...]:
    prefix = seed.to_bytes(8, "big") + epoch.to_bytes(8, "big")
    return tuple(
        sorted(
            range(population),
            key=lambda index: (
                hashlib.blake2b(
                    prefix + index.to_bytes(8, "big"),
                    digest_size=32,
                    person=_SAMPLER_PERSON,
                ).digest(),
                index,
            ),
        )
    )


def sampler_plan(protocol: NotchedS13Protocol) -> DeterministicSamplerPlan:
    population = len(protocol.fit_case_ids)
    batch = protocol.search_space.case_batch_size
    if population % batch:
        raise ValueError("TRAIN-fit population must divide exactly into batches")
    batches: list[tuple[int, ...]] = []
    epoch = 0
    while len(batches) < protocol.search_space.optimizer_steps:
        order = _epoch_order(population, protocol.sampler_seed, epoch)
        batches.extend(
            order[start : start + batch] for start in range(0, population, batch)
        )
        epoch += 1
    return DeterministicSamplerPlan(
        sampler_id=f"{protocol.sampler_policy_id}:{protocol.sampler_seed}",
        population_size=population,
        batches=tuple(batches[: protocol.search_space.optimizer_steps]),
    )


def trainer_config(protocol: NotchedS13Protocol) -> EnergyTrainerConfig:
    (trial,) = protocol.search_space.trials
    return EnergyTrainerConfig(
        search_space=protocol.search_space,
        trial=trial,
        schedule=protocol.schedule,
        gradient_clip_norm=protocol.gradient_clip_norm,
    )


def execution_profile(protocol: NotchedS13Protocol) -> ExecutionProfile:
    return ExecutionProfile(
        python_version=platform.python_version(),
        library_versions={
            "torch": torch.__version__,
            "safetensors": version("safetensors"),
            "rfc8785": version("rfc8785"),
        },
        blas_threads=protocol.execution_blas_threads,
        device="cpu",
        dtype=protocol.execution_dtype,
        rng_algorithm="torch-cpu-mt19937",
        seed=protocol.training_seed,
        container_digest=None,
    )


def run_binding(
    protocol: NotchedS13Protocol, *, implementation_revision: str
) -> TrainingRunBinding:
    if model_state_hash(build_initial_operator()) != protocol.initial_weights_hash:
        raise ValueError("initial operator weights differ from the protocol")
    return TrainingRunBinding(
        dataset_snapshot_hash=protocol.dataset_snapshot_hash,
        public_manifest_hash=protocol.public_manifest_hash,
        normalization_hash=protocol.normalization_hash,
        architecture_id=protocol.architecture_id,
        architecture_module_digest=protocol.architecture_module_digest,
        objective_registry_hash=protocol.objective_hash,
        trainer_config_hash=trainer_config(protocol).config_hash,
        sampler_plan_hash=sampler_plan(protocol).plan_hash,
        initial_weights_hash=protocol.initial_weights_hash,
        implementation_revision=implementation_revision,
        seed=protocol.training_seed,
        execution_profile=execution_profile(protocol),
    )


def physics_material(material: MaterialSpec) -> MaterialSpec:
    if material.model != _FEM_MATERIAL_MODEL:
        raise ValueError(f"material model must be {_FEM_MATERIAL_MODEL!r}")
    modulus = material.properties["elastic_modulus"]
    if modulus.unit is not Unit.MPA or modulus.value != STRESS_REFERENCE_MPA:
        raise ValueError("the stress scale must match the 760 MPa material")
    return replace(material, model=_KERNEL_MATERIAL_MODEL)


@dataclass(frozen=True, slots=True)
class NotchedGeometryCase:
    """One case reduced to what the energy objective and serving need."""

    case_id: str
    leakage_group_id: str
    request: FemRequest
    material: MaterialSpec
    domain: TriangleDomain
    conditioning: torch.Tensor
    half_length_mm: float
    half_width_mm: float
    prescribed_displacement_mm: float

    @property
    def dtype(self) -> torch.dtype:
        dtype: torch.dtype = self.domain.vertices.dtype
        return dtype

    def frame(self) -> QueryFrame:
        return QueryFrame(
            torch.zeros((1, 2), dtype=self.dtype),
            torch.tensor([[self.half_length_mm, self.half_width_mm]], dtype=self.dtype),
        )

    def lifting(self) -> AxisAlignedTensionLifting:
        return AxisAlignedTensionLifting(
            torch.tensor([[self.half_length_mm, self.half_width_mm]], dtype=self.dtype),
            torch.tensor([self.prescribed_displacement_mm], dtype=self.dtype),
        )

    def parameters(self) -> torch.Tensor:
        return self.conditioning.to(self.dtype).unsqueeze(0)


def loaded_end_displacement(request: FemRequest) -> float:
    kinds = {
        condition.boundary_tag: (condition.kind, condition.components)
        for condition in request.boundary_conditions
    }
    if kinds != {
        LEFT_TAG: ("symmetry", ("x",)),
        LOWER_TAG: ("symmetry", ("y",)),
        LOADED_END_TAG: ("dirichlet", ("x",)),
    }:
        raise ValueError("boundary conditions differ from the quarter-model lifting")
    (loaded,) = (
        c for c in request.boundary_conditions if c.boundary_tag == LOADED_END_TAG
    )
    (value,) = loaded.values
    if value.unit is not Unit.MM:
        raise ValueError("loaded-end displacement must be in mm")
    return float(value.value)


def geometry_case(
    case_id: str,
    leakage_group_id: str,
    coordinates: Mapping[str, float],
    request: FemRequest,
    normalization: ConditioningNormalization,
    *,
    dtype: torch.dtype,
) -> NotchedGeometryCase:
    if request.request_id != case_id:
        raise ValueError("request does not belong to the case")
    nodes = torch.tensor(request.mesh.nodes, dtype=torch.float64)
    cells = torch.tensor(request.mesh.cells, dtype=torch.int64)
    half_width = 0.5 * float(coordinates["width"])
    error = max(
        abs(float(nodes[:, 0].min())),
        abs(float(nodes[:, 1].min())),
        abs(float(nodes[:, 0].max()) - HALF_LENGTH_MM),
        abs(float(nodes[:, 1].max()) - half_width),
    )
    if error > _GEOMETRY_TOLERANCE_MM:
        raise ValueError(f"{case_id} mesh extent differs from the quarter model")
    displacement = loaded_end_displacement(request)
    expected = prescribed_displacement_mm(float(coordinates["nominal_axial_strain"]))
    if not math.isclose(displacement, expected, rel_tol=1.0e-12, abs_tol=0.0):
        raise ValueError(f"{case_id} loaded-end displacement differs from strain")
    return NotchedGeometryCase(
        case_id=case_id,
        leakage_group_id=leakage_group_id,
        request=request,
        material=physics_material(request.material),
        domain=TriangleDomain(nodes.to(dtype), cells),
        conditioning=normalization.vector(coordinates),
        half_length_mm=HALF_LENGTH_MM,
        half_width_mm=half_width,
        prescribed_displacement_mm=displacement,
    )


def case_energy(
    model: EnergyFormOperator,
    case: NotchedGeometryCase,
    config: PotentialEnergyConfig,
    *,
    generator: torch.Generator | None = None,
    create_graph: bool = True,
) -> PotentialEnergyTerms:
    points, weights = stratified_triangle_quadrature(
        case.domain, config.points_per_cell, generator=generator
    )
    _, gradient = model.displacement_and_gradient(
        query_xy=points.unsqueeze(0),
        parameters=case.parameters(),
        frame=case.frame(),
        lifting=case.lifting(),
        create_graph=create_graph,
    )
    return plane_stress_potential_energy(
        gradient[0],
        weights,
        case.material,
        config=config,
        reference_area=case.domain.area,
    )


class EnergyObjective:
    """Potential energy of one registered TRAIN-fit case at fresh points."""

    __slots__ = ("_cases", "_config")

    def __init__(
        self, cases: tuple[NotchedGeometryCase, ...], config: PotentialEnergyConfig
    ) -> None:
        self._cases: Mapping[str, NotchedGeometryCase] = MappingProxyType(
            {case.case_id: case for case in cases}
        )
        self._config = config

    def __call__(
        self, model: torch.nn.Module, sample_id: str, step: int
    ) -> PotentialEnergyTerms:
        if type(model) is not EnergyFormOperator:
            raise TypeError("model must be an exact EnergyFormOperator")
        case = self._cases.get(sample_id)
        if case is None:
            raise PermissionError(f"{sample_id} is not a registered TRAIN-fit case")
        return case_energy(model, case, self._config)


def published_cases(case_ids: Sequence[str]) -> tuple[PublishedCase, ...]:
    _, manifest = load_public_sources()
    by_id = {case.case_id: case for case in manifest.cases}
    return tuple(by_id[case_id] for case_id in case_ids)


def requests_for(
    case_ids: Sequence[str], profile: ExecutionProfile
) -> Mapping[str, FemRequest]:
    """S10 requests of the named cases, checked against the request plan."""

    plan = request_plan(profile)
    wanted = set(case_ids)
    requests = {
        r.request_id: r for r in public_requests(profile) if r.request_id in wanted
    }
    for case_id, request in requests.items():
        if plan.entry_by_case_id(case_id).request_recipe_hash != request.recipe_hash:
            raise ValueError(f"{case_id} request differs from the request plan")
    if set(requests) != wanted:
        raise ValueError("requested cases are missing from the request plan")
    return requests


def train_geometry(
    case_ids: Sequence[str],
    normalization: ConditioningNormalization,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[NotchedGeometryCase, ...]:
    """Geometry of ACTIVE TRAIN cases only; any other role is refused."""

    cases = published_cases(case_ids)
    for case in cases:
        if case.role is not EvidenceRole.TRAIN or case.status is not (
            PublishedCaseStatus.ACTIVE
        ):
            raise PermissionError(f"{case.role.value} evidence cannot enter S13")
    requests = requests_for(case_ids, default_s10_execution_profile())
    return tuple(
        geometry_case(
            case.case_id,
            case.leakage_group_id,
            case.primitive_coordinates,
            requests[case.case_id],
            normalization,
            dtype=dtype,
        )
        for case in cases
    )


def training_cases(
    protocol: NotchedS13Protocol, *, dtype: torch.dtype = torch.float32
) -> tuple[NotchedGeometryCase, ...]:
    """Geometry of the TRAIN-fit cases; no FEM result is read."""

    profile = default_s10_execution_profile()
    if (
        geometry_snapshot_hash(protocol.fit_case_ids, profile)
        != protocol.dataset_snapshot_hash
    ):
        raise ValueError("TRAIN-fit geometry differs from the protocol")
    return train_geometry(protocol.fit_case_ids, protocol.normalization, dtype=dtype)


def _prepare_run_root(run_root: Path) -> Path:
    if not run_root.is_absolute():
        raise ValueError("run_root must be absolute")
    parent = _absolute_directory(run_root.parent, "run_root parent")
    output = parent / run_root.name
    if output.is_relative_to(Path(__file__).resolve().parents[2]):
        raise ValueError("run_root must remain outside the repository")
    output.mkdir(exist_ok=True)
    return _absolute_directory(output, "run_root")


def run_trial(
    *,
    run_root: Path,
    segment_steps: int = 400,
    stop_after: int | None = None,
    progress: Progress | None = None,
) -> EnergyTrainingResult | None:
    """Train (or resume) the single registered trial on TRAIN-fit geometry."""

    protocol = load_committed_protocol()
    provenance = _git_provenance()
    if provenance.dirty_worktree:
        raise RuntimeError("registered training requires a clean revision")
    binding = run_binding(protocol, implementation_revision=provenance.head)
    config = trainer_config(protocol)
    plan = sampler_plan(protocol)
    output = _prepare_run_root(run_root)
    payload = canonical_json_bytes(binding)
    path = output / RUN_BINDING_NAME
    if path.exists() and path.read_bytes() != payload:
        raise ValueError("run root belongs to a different run binding")
    path.write_bytes(payload)
    resume_from, last_step = latest_checkpoint(
        output, expected_run_hash=binding.run_hash
    )
    targets = segment_targets(
        last_step, config.trial.optimizer_steps, segment_steps, stop_after=stop_after
    )
    say = progress or (lambda message: None)
    say(
        f"phase=PREFLIGHT role=TRAIN trial={config.trial.trial_id} "
        f"revision={provenance.head} protocol={protocol.protocol_hash} "
        f"run={binding.run_hash} resume_step={last_step} targets={targets}"
    )
    if not targets:
        return None
    cases = training_cases(protocol)
    objective = EnergyObjective(cases, protocol.objective.energy)
    population = TrainingPopulation(
        dataset_snapshot_hash=protocol.dataset_snapshot_hash,
        public_manifest_hash=protocol.public_manifest_hash,
        role=EvidenceRole.TRAIN,
        sample_ids=protocol.fit_case_ids,
    )
    say(f"phase=TRAIN_FIT role=TRAIN cases={len(cases)}")
    result: EnergyTrainingResult | None = None
    with (output / TRAINING_LOG_NAME).open("a", encoding="utf-8") as log:
        writer = _log_writer(log)
        for target in targets:
            checkpoint = output / f"checkpoint-step-{target:04d}"
            result = train_energy_to_checkpoint(
                checkpoint,
                model=build_initial_operator(),
                population=population,
                plan=plan,
                binding=binding,
                config=config,
                objective=objective,
                objective_hash=protocol.objective_hash,
                target_step=target,
                resume_from=resume_from,
                progress=writer,
            )
            last = result.history[-1]
            say(
                f"phase=CHECKPOINT role=TRAIN step={result.step} "
                f"model={result.model_state_hash} "
                f"manifest={result.checkpoint.manifest.manifest_hash} "
                f"energy={last.energy:.9g} barrier={last.barrier:.3g} "
                f"min_det={last.minimum_determinant:.4f} lr={last.learning_rate:.3g}"
            )
            resume_from = checkpoint
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m campaigns.notched_v1.s13")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("protocol")
    sub.add_parser("initial-weights")
    train = sub.add_parser("train")
    train.add_argument("--run-root", type=Path, required=True)
    train.add_argument("--segment-steps", type=int, default=400)
    train.add_argument("--stop-after", type=int)
    arguments = parser.parse_args(argv)
    if arguments.command == "initial-weights":
        print(model_state_hash(build_initial_operator()))
        return 0
    if arguments.command == "protocol":
        protocol = build_protocol()
        root = Path(__file__).resolve().parents[2]
        write_exact(root / PROTOCOL_PATH, protocol.to_bytes())
        print(json.dumps({"protocol_hash": protocol.protocol_hash}))
        return 0
    run_trial(
        run_root=arguments.run_root,
        segment_steps=arguments.segment_steps,
        stop_after=arguments.stop_after,
        progress=lambda message: print(message, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))


__all__ = (
    "CONDITIONING",
    "GUARDRAILS",
    "PROTOCOL_PATH",
    "ConditioningNormalization",
    "NotchedGeometryCase",
    "NotchedS13Protocol",
    "build_initial_operator",
    "build_protocol",
    "case_energy",
    "geometry_case",
    "load_committed_protocol",
    "run_binding",
    "run_trial",
    "sampler_plan",
    "split_train_cases",
    "trainer_config",
)
