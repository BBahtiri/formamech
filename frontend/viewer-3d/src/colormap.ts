import type { FaceAssignment3D, FaceBcKind } from "./types";

export const ROLE_HEX_COLORS: Record<FaceBcKind | "selected", string> = {
  fixed: "#1d4ed8",
  dirichlet: "#0f766e",
  traction: "#dc2626",
  contact: "#d97706",
  traction_free: "#64748b",
  selected: "#9333ea",
};

const VIRIDIS_STOPS: readonly [number, number, number][] = [
  [0.267, 0.005, 0.329],
  [0.23, 0.322, 0.546],
  [0.128, 0.567, 0.551],
  [0.369, 0.789, 0.383],
  [0.993, 0.906, 0.144],
];

export function sampleViridisRgb(tRaw: number): [number, number, number] {
  const t = Number.isFinite(tRaw) ? Math.min(1, Math.max(0, tRaw)) : 0;
  const scaled = t * (VIRIDIS_STOPS.length - 1);
  const idx = Math.min(VIRIDIS_STOPS.length - 2, Math.floor(scaled));
  const frac = scaled - idx;
  const a = VIRIDIS_STOPS[idx];
  const b = VIRIDIS_STOPS[idx + 1];
  return [
    a[0] + (b[0] - a[0]) * frac,
    a[1] + (b[1] - a[1]) * frac,
    a[2] + (b[2] - a[2]) * frac,
  ];
}

export function hexToRgbTuple(hex: string): [number, number, number] {
  const clean = hex.replace("#", "");
  const num = Number.parseInt(clean, 16);
  if (!Number.isFinite(num)) return [0.39, 0.45, 0.55];
  return [((num >> 16) & 255) / 255, ((num >> 8) & 255) / 255, (num & 255) / 255];
}

export function roleColorHex(
  assignment: FaceAssignment3D | undefined,
  selected: boolean,
): string {
  if (selected) return ROLE_HEX_COLORS.selected;
  if (!assignment) return ROLE_HEX_COLORS.traction_free;
  return ROLE_HEX_COLORS[assignment.kind] ?? ROLE_HEX_COLORS.traction_free;
}

export function fieldRange(values: readonly number[]): [number, number] {
  let lo = Number.POSITIVE_INFINITY;
  let hi = Number.NEGATIVE_INFINITY;
  for (const v of values) {
    if (Number.isFinite(v)) {
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    }
  }
  if (!Number.isFinite(lo) || !Number.isFinite(hi)) return [0, 1];
  if (hi <= lo) return [lo, lo + 1e-12];
  return [lo, hi];
}
