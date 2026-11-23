import type { Data, Layout } from "plotly.js";

export const COMPONENTS = ["x", "y"] as const;
export type Component = (typeof COMPONENTS)[number];
export type ConditionKind = "dirichlet" | "symmetry" | "traction_free";

export const KIND_LABELS: Record<ConditionKind, string> = {
  dirichlet: "Prescribed displacement",
  symmetry: "Symmetry",
  traction_free: "Traction-free",
};

export type Assignment = {
  region: string;
  name: string;
  kind: ConditionKind;
  components?: Component[];
  values?: { expression: string; unit: string }[];
};

export type Mesh = {
  nodes_mm: [number, number][];
  triangles: [number, number, number][];
};

export type Figure = { data: Data[]; layout: Partial<Layout> };
export type Field = "name" | "kind" | "components" | Component | "region" | "form";
export type FieldErrors = Partial<Record<Field, string>>;

export type ApplyResult = {
  event_id: string;
  region: string;
  status: "accepted" | "invalid" | "stale";
  message?: string;
  errors?: FieldErrors;
  draft_revision?: string;
  reload?: boolean;
};

export type PickerData = {
  workspace_id: string;
  geometry_id: string;
  draft_revision: string;
  preview: { mesh: Mesh; boundaries: { name: string; kind: string }[] };
  figure: Figure;
  assignments: Assignment[];
  suggested_regions: string[];
  readiness_errors: string[];
  last_apply: ApplyResult | null;
};

export type Revision = Pick<PickerData, "geometry_id" | "draft_revision">;
export type ApplyEvent = Revision & {
  workspace_id: string;
  event_id: string;
  region: string;
  assignment: Assignment;
};
