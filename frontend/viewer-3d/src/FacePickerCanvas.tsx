import { useEffect, useMemo, useRef, useState } from "react";
import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { fieldRange, hexToRgbTuple, roleColorHex, sampleViridisRgb } from "./colormap";
import type { FaceAssignment3D, ProbeReading3D, ScalarField3D, SurfaceFaceGroup3D } from "./types";

export interface ClipState {
  xFrac: number;
  yFrac: number;
  zFrac: number;
}

export interface RendererLike {
  domElement: HTMLCanvasElement;
  localClippingEnabled: boolean;
  setSize: (w: number, h: number) => void;
  render: (scene: THREE.Scene, camera: THREE.Camera) => void;
  dispose: () => void;
}

export interface FacePickerCanvasProps {
  nodesMm: [number, number, number][];
  triangles: [number, number, number][];
  faceGroups: SurfaceFaceGroup3D[];
  assignmentsByTag: Record<string, FaceAssignment3D>;
  selectedFaceTag: string | null;
  activeField: ScalarField3D | null;
  displacementsMm?: [number, number, number][];
  deformScale: number;
  showWireframeGhost: boolean;
  clip: ClipState;
  onSelectFace: (faceTag: string) => void;
  onProbe: (reading: ProbeReading3D) => void;
  rendererFactory?: () => RendererLike;
}

export function computeBoundingBox(nodes: readonly [number, number, number][]): {
  min: [number, number, number];
  max: [number, number, number];
} {
  if (nodes.length === 0) {
    return { min: [0, 0, 0], max: [1, 1, 1] };
  }
  let minX = nodes[0][0], minY = nodes[0][1], minZ = nodes[0][2];
  let maxX = minX, maxY = minY, maxZ = minZ;
  for (const [x, y, z] of nodes) {
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }
  return { min: [minX, minY, minZ], max: [maxX, maxY, maxZ] };
}

export function buildDeformedNodes(
  nodes: readonly [number, number, number][],
  displacements: readonly [number, number, number][] | undefined,
  scale: number,
): [number, number, number][] {
  if (!displacements || displacements.length !== nodes.length || scale === 0) {
    return nodes.map(([x, y, z]) => [x, y, z]);
  }
  return nodes.map(([x, y, z], i) => {
    const [ux, uy, uz] = displacements[i];
    return [x + scale * ux, y + scale * uy, z + scale * uz];
  });
}

export function canCreateWebGLContext(): boolean {
  try {
    const canvas = document.createElement("canvas");
    return Boolean(canvas.getContext("webgl2") || canvas.getContext("webgl"));
  } catch {
    return false;
  }
}

export function FacePickerCanvas(props: FacePickerCanvasProps) {
  const {
    nodesMm,
    triangles,
    faceGroups,
    assignmentsByTag,
    selectedFaceTag,
    activeField,
    displacementsMm,
    deformScale,
    showWireframeGhost,
    clip,
    onSelectFace,
    onProbe,
    rendererFactory,
  } = props;

  const containerRef = useRef<HTMLDivElement | null>(null);
  const [webglReady, setWebglReady] = useState<boolean>(
    () => Boolean(rendererFactory) || canCreateWebGLContext(),
  );
  const bbox = useMemo(() => computeBoundingBox(nodesMm), [nodesMm]);
  const deformed = useMemo(
    () => buildDeformedNodes(nodesMm, displacementsMm, deformScale),
    [nodesMm, displacementsMm, deformScale],
  );

  useEffect(() => {
    const host = containerRef.current;
    if (!host || !webglReady || nodesMm.length === 0 || triangles.length === 0) {
      return;
    }

    let renderer: RendererLike;
    try {
      renderer = rendererFactory
        ? rendererFactory()
        : new THREE.WebGLRenderer({ antialias: true, alpha: true });
    } catch {
      setWebglReady(false);
      return;
    }

    const width = Math.max(host.clientWidth || 560, 320);
    const height = Math.max(host.clientHeight || 420, 320);
    renderer.setSize(width, height);
    renderer.localClippingEnabled = true;
    host.innerHTML = "";
    host.appendChild(renderer.domElement);

    const scene = new THREE.Scene();
    scene.background = new THREE.Color("#f8fafc");
    const camera = new THREE.PerspectiveCamera(42, width / height, 0.1, 1e5);

    const cx = 0.5 * (bbox.min[0] + bbox.max[0]);
    const cy = 0.5 * (bbox.min[1] + bbox.max[1]);
    const cz = 0.5 * (bbox.min[2] + bbox.max[2]);
    const diag = Math.max(
      1,
      Math.hypot(bbox.max[0] - bbox.min[0], bbox.max[1] - bbox.min[1], bbox.max[2] - bbox.min[2]),
    );
    camera.position.set(cx + diag * 1.4, cy + diag * 1.1, cz + diag * 1.4);
    camera.lookAt(cx, cy, cz);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.target.set(cx, cy, cz);
    controls.enableDamping = true;

    scene.add(new THREE.AmbientLight(0xffffff, 0.65));
    const dirLight = new THREE.DirectionalLight(0xffffff, 0.85);
    dirLight.position.set(cx + diag * 2, cy + diag * 2, cz + diag * 2);
    scene.add(dirLight);

    const clipPlanes = [
      new THREE.Plane(
        new THREE.Vector3(-1, 0, 0),
        bbox.min[0] + clip.xFrac * (bbox.max[0] - bbox.min[0]) + 1e-4,
      ),
      new THREE.Plane(
        new THREE.Vector3(0, -1, 0),
        bbox.min[1] + clip.yFrac * (bbox.max[1] - bbox.min[1]) + 1e-4,
      ),
      new THREE.Plane(
        new THREE.Vector3(0, 0, -1),
        bbox.min[2] + clip.zFrac * (bbox.max[2] - bbox.min[2]) + 1e-4,
      ),
    ];

    const [fLo, fHi] = activeField ? fieldRange(activeField.values) : [0, 1];
    const pickableMeshes: THREE.Mesh[] = [];

    for (const fg of faceGroups) {
      const positions: number[] = [];
      const colors: number[] = [];
      const [rr, rg, rb] = hexToRgbTuple(
        roleColorHex(assignmentsByTag[fg.face_tag], fg.face_tag === selectedFaceTag),
      );

      for (const triIdx of fg.triangle_indices) {
        const tri = triangles[triIdx];
        if (!tri) continue;
        for (const nid of tri) {
          const p = deformed[nid] ?? [0, 0, 0];
          positions.push(p[0], p[1], p[2]);
          if (activeField && nid < activeField.values.length) {
            const norm = (activeField.values[nid] - fLo) / (fHi - fLo);
            const [vr, vg, vb] = sampleViridisRgb(norm);
            colors.push(vr, vg, vb);
          } else {
            colors.push(rr, rg, rb);
          }
        }
      }

      if (positions.length === 0) continue;
      const geom = new THREE.BufferGeometry();
      geom.setAttribute("position", new THREE.Float32BufferAttribute(positions, 3));
      geom.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      geom.computeVertexNormals();

      const mat = new THREE.MeshStandardMaterial({
        vertexColors: true,
        side: THREE.DoubleSide,
        roughness: 0.45,
        metalness: 0.08,
        clippingPlanes: clipPlanes,
      });
      const mesh = new THREE.Mesh(geom, mat);
      mesh.userData = { faceTag: fg.face_tag };
      scene.add(mesh);
      pickableMeshes.push(mesh);
    }

    if (showWireframeGhost) {
      const ghostPos: number[] = [];
      for (const [i0, i1, i2] of triangles) {
        const p0 = nodesMm[i0], p1 = nodesMm[i1], p2 = nodesMm[i2];
        if (!p0 || !p1 || !p2) continue;
        ghostPos.push(p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], p2[0], p2[1], p2[2]);
      }
      const gGeom = new THREE.BufferGeometry();
      gGeom.setAttribute("position", new THREE.Float32BufferAttribute(ghostPos, 3));
      const gMat = new THREE.MeshBasicMaterial({
        color: 0x334155,
        wireframe: true,
        transparent: true,
        opacity: 0.25,
        clippingPlanes: clipPlanes,
      });
      scene.add(new THREE.Mesh(gGeom, gMat));
    }

    const raycaster = new THREE.Raycaster();
    const pointer = new THREE.Vector2();
    const handlePointerDown = (ev: MouseEvent) => {
      const rect = renderer.domElement.getBoundingClientRect();
      pointer.x = ((ev.clientX - rect.left) / rect.width) * 2 - 1;
      pointer.y = -((ev.clientY - rect.top) / rect.height) * 2 + 1;
      raycaster.setFromCamera(pointer, camera);
      const hits = raycaster.intersectObjects(pickableMeshes, false);
      if (hits.length > 0) {
        const hit = hits[0];
        const tag = String(hit.object.userData.faceTag ?? "");
        if (tag) {
          onSelectFace(tag);
          let val: number | null = null;
          if (activeField && activeField.values.length > 0) {
            let bestIdx = 0;
            let bestD2 = Number.POSITIVE_INFINITY;
            for (let i = 0; i < deformed.length; i += 1) {
              const dx = deformed[i][0] - hit.point.x;
              const dy = deformed[i][1] - hit.point.y;
              const dz = deformed[i][2] - hit.point.z;
              const d2 = dx * dx + dy * dy + dz * dz;
              if (d2 < bestD2) {
                bestD2 = d2;
                bestIdx = i;
              }
            }
            val = activeField.values[bestIdx] ?? null;
          }
          onProbe({
            face_tag: tag,
            point_mm: [hit.point.x, hit.point.y, hit.point.z],
            field_label: activeField?.label ?? "Boundary Role",
            field_value: val,
            field_unit: activeField?.unit ?? "",
          });
        }
      }
    };
    renderer.domElement.addEventListener("pointerdown", handlePointerDown);

    let rafId = 0;
    const renderLoop = () => {
      controls.update();
      renderer.render(scene, camera);
      rafId = window.requestAnimationFrame(renderLoop);
    };
    renderLoop();

    return () => {
      window.cancelAnimationFrame(rafId);
      renderer.domElement.removeEventListener("pointerdown", handlePointerDown);
      controls.dispose();
      renderer.dispose();
    };
  }, [
    webglReady,
    nodesMm,
    triangles,
    faceGroups,
    assignmentsByTag,
    selectedFaceTag,
    activeField,
    deformed,
    showWireframeGhost,
    clip,
    bbox,
    onSelectFace,
    onProbe,
    rendererFactory,
  ]);

  if (!webglReady) {
    return (
      <div className="fm3d-fallback" data-testid="viewer-3d-fallback">
        <div className="fm3d-fallback-header">
          Interactive 3D Face Map ({nodesMm.length} nodes, {triangles.length} surface triangles)
        </div>
        <div className="fm3d-face-grid">
          {faceGroups.map((fg) => {
            const assign = assignmentsByTag[fg.face_tag];
            const isSel = fg.face_tag === selectedFaceTag;
            return (
              <button
                key={fg.face_tag}
                type="button"
                className={`fm3d-face-chip${isSel ? " selected" : ""}`}
                style={{ borderColor: roleColorHex(assign, isSel) }}
                onClick={() => {
                  onSelectFace(fg.face_tag);
                  onProbe({
                    face_tag: fg.face_tag,
                    point_mm: fg.centroid_mm,
                    field_label: activeField?.label ?? "Boundary Role",
                    field_value:
                      activeField && activeField.values.length > 0 ? activeField.values[0] : null,
                    field_unit: activeField?.unit ?? "",
                  });
                }}
              >
                <strong>{fg.face_tag}</strong>
                <span>{assign?.kind ?? "traction_free"}</span>
                <span>{fg.area_mm2.toFixed(1)} mm²</span>
              </button>
            );
          })}
        </div>
      </div>
    );
  }

  return <div ref={containerRef} className="fm3d-canvas-host" data-testid="viewer-3d-webgl" />;
}
