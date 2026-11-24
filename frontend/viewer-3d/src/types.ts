export type FaceBcKind =
  | "fixed"
  | "dirichlet"
  | "traction"
  | "contact"
  | "traction_free";

export interface FaceAssignment3D {
  face_tag: string;
  name: string;
  kind: FaceBcKind;
  components: ("x" | "y" | "z")[];
  values_mm: number[];
  traction_mpa: [number, number, number];
}

export interface SurfaceFaceGroup3D {
  face_tag: string;
  triangle_indices: number[];
  area_mm2: number;
  centroid_mm: [number, number, number];
}

export interface ScalarField3D {
  name: string;
  label: string;
  unit: string;
  values: number[];
}

export interface ApplyFaceEvent3D {
  workspace_id: string;
  geometry_id: string;
  draft_revision: string;
  event_id: string;
  face_tag: string;
  assignment: FaceAssignment3D;
}

export interface ApplyFaceResult3D {
  event_id: string;
  face_tag: string;
  status: "accepted" | "invalid" | "stale";
  draft_revision?: string;
  errors?: Record<string, string>;
}

export interface Viewer3DData {
  workspace_id: string;
  geometry_id: string;
  draft_revision: string;
  nodes_mm: [number, number, number][];
  triangles: [number, number, number][];
  face_groups: SurfaceFaceGroup3D[];
  assignments: FaceAssignment3D[];
  displacements_mm?: [number, number, number][];
  fields?: Record<string, ScalarField3D>;
  readiness_errors?: string[];
  last_apply?: ApplyFaceResult3D | null;
}

export interface ProbeReading3D {
  face_tag: string;
  point_mm: [number, number, number];
  field_label: string;
  field_value: number | null;
  field_unit: string;
}
