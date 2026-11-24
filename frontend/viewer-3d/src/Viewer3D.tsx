import { useEffect, useMemo, useRef, useState } from "react";
import { fieldRange } from "./colormap";
import { FacePickerCanvas, type ClipState } from "./FacePickerCanvas";
import type {
  ApplyFaceEvent3D,
  FaceAssignment3D,
  FaceBcKind,
  ProbeReading3D,
  Viewer3DData,
} from "./types";

export interface Viewer3DProps {
  data: Viewer3DData;
  onApply: (event: ApplyFaceEvent3D) => void;
}

const BC_KINDS: { value: FaceBcKind; label: string }[] = [
  { value: "fixed", label: "Fixed (u = 0)" },
  { value: "dirichlet", label: "Prescribed Displacement (mm)" },
  { value: "traction", label: "Surface Traction (MPa)" },
  { value: "contact", label: "Contact Surface" },
  { value: "traction_free", label: "Traction Free" },
];

function dirichletAxisValue(
  assign: FaceAssignment3D | undefined,
  axis: "x" | "y" | "z",
): number {
  if (!assign) return 0;
  const idx = assign.components.indexOf(axis);
  if (idx < 0 || idx >= assign.values_mm.length) return 0;
  const val = assign.values_mm[idx];
  return Number.isFinite(val) ? val : 0;
}

export function Viewer3D({ data, onApply }: Viewer3DProps) {
  const assignmentsByTag = useMemo(() => {
    const map: Record<string, FaceAssignment3D> = {};
    for (const a of data.assignments) {
      map[a.face_tag] = a;
    }
    return map;
  }, [data.assignments]);

  const initialTag = data.face_groups[0]?.face_tag ?? null;
  const [selectedFaceTag, setSelectedFaceTag] = useState<string | null>(initialTag);
  const [fieldKey, setFieldKey] = useState<string>("__roles__");
  const [deformScale, setDeformScale] = useState<number>(1.0);
  const [showWireframeGhost, setShowWireframeGhost] = useState<boolean>(true);
  const [clip, setClip] = useState<ClipState>({ xFrac: 1.0, yFrac: 1.0, zFrac: 1.0 });
  const [probe, setProbe] = useState<ProbeReading3D | null>(null);
  const [eventSeq, setEventSeq] = useState<number>(1);
  const instanceNonceRef = useRef<string>(Math.random().toString(36).slice(2, 8));

  const selectedAssign = selectedFaceTag ? assignmentsByTag[selectedFaceTag] : undefined;
  const [draftKind, setDraftKind] = useState<FaceBcKind>(selectedAssign?.kind ?? "traction_free");
  const [draftName, setDraftName] = useState<string>(
    selectedAssign?.name ?? (selectedFaceTag ?? "face_bc"),
  );
  const [draftUx, setDraftUx] = useState<string>(
    String(dirichletAxisValue(selectedAssign, "x")),
  );
  const [draftUy, setDraftUy] = useState<string>(
    String(dirichletAxisValue(selectedAssign, "y")),
  );
  const [draftUz, setDraftUz] = useState<string>(
    String(dirichletAxisValue(selectedAssign, "z")),
  );
  const [draftTx, setDraftTx] = useState<string>(String(selectedAssign?.traction_mpa[0] ?? 0));
  const [draftTy, setDraftTy] = useState<string>(String(selectedAssign?.traction_mpa[1] ?? 0));
  const [draftTz, setDraftTz] = useState<string>(String(selectedAssign?.traction_mpa[2] ?? 0));

  const syncDraftFromTag = (tag: string | null) => {
    const cur = tag ? assignmentsByTag[tag] : undefined;
    setDraftKind(cur?.kind ?? "traction_free");
    setDraftName(cur?.name ?? (tag ?? "face_bc"));
    setDraftUx(String(dirichletAxisValue(cur, "x")));
    setDraftUy(String(dirichletAxisValue(cur, "y")));
    setDraftUz(String(dirichletAxisValue(cur, "z")));
    setDraftTx(String(cur?.traction_mpa[0] ?? 0));
    setDraftTy(String(cur?.traction_mpa[1] ?? 0));
    setDraftTz(String(cur?.traction_mpa[2] ?? 0));
  };

  // Sync face selection, draft inputs, and active overlay when geometry or fields change
  useEffect(() => {
    const hasCurrent =
      selectedFaceTag !== null &&
      data.face_groups.some((fg) => fg.face_tag === selectedFaceTag);
    const nextTag = hasCurrent ? selectedFaceTag : (data.face_groups[0]?.face_tag ?? null);
    setSelectedFaceTag(nextTag);
    syncDraftFromTag(nextTag);
    setProbe(null);
  }, [data.geometry_id, data.draft_revision]);

  useEffect(() => {
    if (fieldKey !== "__roles__" && (!data.fields || !data.fields[fieldKey])) {
      setFieldKey("__roles__");
    }
  }, [data.fields, fieldKey]);

  const selectFace = (tag: string) => {
    setSelectedFaceTag(tag);
    syncDraftFromTag(tag);
  };

  const activeField =
    fieldKey !== "__roles__" && data.fields ? (data.fields[fieldKey] ?? null) : null;
  const activeRange = activeField ? fieldRange(activeField.values) : null;

  const handleApply = () => {
    if (!selectedFaceTag) return;
    const isFixed = draftKind === "fixed";
    const isDirichlet = draftKind === "dirichlet";
    const components: ("x" | "y" | "z")[] = isFixed || isDirichlet ? ["x", "y", "z"] : [];
    const valuesMm = isFixed
      ? [0, 0, 0]
      : isDirichlet
        ? [Number(draftUx) || 0, Number(draftUy) || 0, Number(draftUz) || 0]
        : [];
    const tractionMpa: [number, number, number] =
      draftKind === "traction"
        ? [Number(draftTx) || 0, Number(draftTy) || 0, Number(draftTz) || 0]
        : [0, 0, 0];

    const event: ApplyFaceEvent3D = {
      workspace_id: data.workspace_id,
      geometry_id: data.geometry_id,
      draft_revision: data.draft_revision,
      event_id: `apply-3d-${data.draft_revision}-${instanceNonceRef.current}-${eventSeq}`,
      face_tag: selectedFaceTag,
      assignment: {
        face_tag: selectedFaceTag,
        name: draftName.trim() || selectedFaceTag,
        kind: draftKind,
        components,
        values_mm: valuesMm,
        traction_mpa: tractionMpa,
      },
    };
    setEventSeq((n) => n + 1);
    onApply(event);
  };

  return (
    <div className="fm3d-root">
      <div className="fm3d-viewport">
        <div className="fm3d-toolbar">
          <label>
            Overlay
            <select
              aria-label="Field Overlay"
              value={fieldKey}
              onChange={(e) => setFieldKey(e.target.value)}
            >
              <option value="__roles__">Boundary Roles</option>
              {data.fields &&
                Object.values(data.fields).map((f) => (
                  <option key={f.name} value={f.name}>
                    {f.label} ({f.unit})
                  </option>
                ))}
            </select>
          </label>
          <label>
            Deform ×{deformScale.toFixed(1)}
            <input
              aria-label="Deformation Scale"
              type="range"
              min={0}
              max={10}
              step={0.5}
              value={deformScale}
              onChange={(e) => setDeformScale(Number(e.target.value))}
            />
          </label>
          <label className="fm3d-checkbox">
            <input
              type="checkbox"
              checked={showWireframeGhost}
              onChange={(e) => setShowWireframeGhost(e.target.checked)}
            />
            Ghost Wireframe
          </label>
        </div>

        <FacePickerCanvas
          nodesMm={data.nodes_mm}
          triangles={data.triangles}
          faceGroups={data.face_groups}
          assignmentsByTag={assignmentsByTag}
          selectedFaceTag={selectedFaceTag}
          activeField={activeField}
          displacementsMm={data.displacements_mm}
          deformScale={deformScale}
          showWireframeGhost={showWireframeGhost}
          clip={clip}
          onSelectFace={selectFace}
          onProbe={setProbe}
        />

        <div className="fm3d-clip-bar">
          <label>
            Clip X ({Math.round(clip.xFrac * 100)}%)
            <input
              aria-label="Clip X"
              type="range"
              min={0.05}
              max={1.0}
              step={0.05}
              value={clip.xFrac}
              onChange={(e) => setClip((c) => ({ ...c, xFrac: Number(e.target.value) }))}
            />
          </label>
          <label>
            Clip Y ({Math.round(clip.yFrac * 100)}%)
            <input
              aria-label="Clip Y"
              type="range"
              min={0.05}
              max={1.0}
              step={0.05}
              value={clip.yFrac}
              onChange={(e) => setClip((c) => ({ ...c, yFrac: Number(e.target.value) }))}
            />
          </label>
          <label>
            Clip Z ({Math.round(clip.zFrac * 100)}%)
            <input
              aria-label="Clip Z"
              type="range"
              min={0.05}
              max={1.0}
              step={0.05}
              value={clip.zFrac}
              onChange={(e) => setClip((c) => ({ ...c, zFrac: Number(e.target.value) }))}
            />
          </label>
        </div>

        {activeField && activeRange && (
          <div className="fm3d-colorbar" data-testid="viewer-3d-colorbar">
            <span>{activeField.label}:</span>
            <span>
              {activeRange[0].toFixed(3)} → {activeRange[1].toFixed(3)} {activeField.unit}
            </span>
          </div>
        )}

        {probe && (
          <div className="fm3d-probe" data-testid="viewer-3d-probe">
            <strong>3D Probe</strong> · Face <code>{probe.face_tag}</code> · (
            {probe.point_mm.map((v) => v.toFixed(2)).join(", ")}) mm
            {probe.field_value !== null && (
              <>
                {" "}
                · {probe.field_label}: <strong>{probe.field_value.toFixed(4)}</strong>{" "}
                {probe.field_unit}
              </>
            )}
          </div>
        )}
      </div>

      <aside className="fm3d-panel">
        <h3>3D B-Rep Face Boundary</h3>
        <label>
          Selected Face
          <select
            aria-label="Selected Face"
            value={selectedFaceTag ?? ""}
            onChange={(e) => selectFace(e.target.value)}
          >
            {data.face_groups.map((fg) => (
              <option key={fg.face_tag} value={fg.face_tag}>
                {fg.face_tag} ({fg.area_mm2.toFixed(1)} mm²)
              </option>
            ))}
          </select>
        </label>

        <label>
          BC Name
          <input
            aria-label="BC Name"
            type="text"
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
          />
        </label>

        <label>
          Condition Type
          <select
            aria-label="Condition Type"
            value={draftKind}
            onChange={(e) => setDraftKind(e.target.value as FaceBcKind)}
          >
            {BC_KINDS.map((k) => (
              <option key={k.value} value={k.value}>
                {k.label}
              </option>
            ))}
          </select>
        </label>

        {draftKind === "dirichlet" && (
          <div className="fm3d-vec-grid">
            <label>
              ux (mm)
              <input aria-label="ux" type="number" step="0.05" value={draftUx} onChange={(e) => setDraftUx(e.target.value)} />
            </label>
            <label>
              uy (mm)
              <input aria-label="uy" type="number" step="0.05" value={draftUy} onChange={(e) => setDraftUy(e.target.value)} />
            </label>
            <label>
              uz (mm)
              <input aria-label="uz" type="number" step="0.05" value={draftUz} onChange={(e) => setDraftUz(e.target.value)} />
            </label>
          </div>
        )}

        {draftKind === "traction" && (
          <div className="fm3d-vec-grid">
            <label>
              tx (MPa)
              <input aria-label="tx" type="number" step="0.5" value={draftTx} onChange={(e) => setDraftTx(e.target.value)} />
            </label>
            <label>
              ty (MPa)
              <input aria-label="ty" type="number" step="0.5" value={draftTy} onChange={(e) => setDraftTy(e.target.value)} />
            </label>
            <label>
              tz (MPa)
              <input aria-label="tz" type="number" step="0.5" value={draftTz} onChange={(e) => setDraftTz(e.target.value)} />
            </label>
          </div>
        )}

        <button
          type="button"
          className="fm3d-apply-btn"
          disabled={!selectedFaceTag}
          onClick={handleApply}
        >
          Apply 3D Face BC
        </button>

        {data.last_apply && (
          <div
            className={`fm3d-status fm3d-status-${data.last_apply.status}`}
            data-testid="viewer-3d-apply-status"
          >
            Last apply ({data.last_apply.face_tag}): {data.last_apply.status}
          </div>
        )}

        {data.readiness_errors && data.readiness_errors.length > 0 && (
          <ul className="fm3d-errors" data-testid="viewer-3d-readiness-errors">
            {data.readiness_errors.map((err) => (
              <li key={err}>{err}</li>
            ))}
          </ul>
        )}
      </aside>
    </div>
  );
}
