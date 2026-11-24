import "@testing-library/jest-dom/vitest";
import { cleanup, fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import * as THREE from "three";
import { fieldRange, hexToRgbTuple, roleColorHex, sampleViridisRgb } from "./colormap";
import {
  buildDeformedNodes,
  computeBoundingBox,
  FacePickerCanvas,
  type RendererLike,
} from "./FacePickerCanvas";
import type { ApplyFaceEvent3D, Viewer3DData } from "./types";
import { Viewer3D } from "./Viewer3D";

afterEach(cleanup);

function sampleData(): Viewer3DData {
  return {
    workspace_id: "ws-3d-001",
    geometry_id: "geom-box-001",
    draft_revision: "rev-1",
    nodes_mm: [
      [0, 0, 0],
      [10, 0, 0],
      [10, 5, 0],
      [0, 5, 0],
    ],
    triangles: [
      [0, 1, 2],
      [0, 2, 3],
    ],
    face_groups: [
      { face_tag: "face_x_min", triangle_indices: [0], area_mm2: 25.0, centroid_mm: [0, 2.5, 2.5] },
      { face_tag: "face_x_max", triangle_indices: [1], area_mm2: 25.0, centroid_mm: [10, 2.5, 2.5] },
    ],
    assignments: [
      {
        face_tag: "face_x_min",
        name: "left_clamp",
        kind: "fixed",
        components: ["x", "y", "z"],
        values_mm: [0, 0, 0],
        traction_mpa: [0, 0, 0],
      },
      {
        face_tag: "face_x_max",
        name: "right_free",
        kind: "traction_free",
        components: [],
        values_mm: [],
        traction_mpa: [0, 0, 0],
      },
    ],
    displacements_mm: [
      [0, 0, 0],
      [0.2, 0, 0],
      [0.2, -0.05, 0],
      [0, -0.05, 0],
    ],
    fields: {
      von_mises_mpa: {
        name: "von_mises_mpa",
        label: "von Mises Stress",
        unit: "MPa",
        values: [10.0, 25.0, 40.0, 15.0],
      },
    },
    readiness_errors: ["Assign at least one load."],
    last_apply: { event_id: "init", face_tag: "face_x_min", status: "accepted" },
  };
}

describe("colormap and geometry helpers", () => {
  it("samples viridis, converts hex colors, and computes ranges/bounds", () => {
    expect(sampleViridisRgb(0)[0]).toBeCloseTo(0.267, 2);
    expect(sampleViridisRgb(1)[0]).toBeCloseTo(0.993, 2);
    expect(sampleViridisRgb(Number.NaN)[0]).toBeCloseTo(0.267, 2);
    expect(hexToRgbTuple("#ffffff")).toEqual([1, 1, 1]);
    expect(hexToRgbTuple("nothex")[0]).toBeCloseTo(0.39, 2);
    expect(roleColorHex(undefined, false)).toBe("#64748b");
    expect(roleColorHex(undefined, true)).toBe("#9333ea");
    expect(fieldRange([5, 5])).toEqual([5, 5 + 1e-12]);
    expect(fieldRange([])).toEqual([0, 1]);
    expect(computeBoundingBox([])).toEqual({ min: [0, 0, 0], max: [1, 1, 1] });
    const def = buildDeformedNodes(
      [[1, 2, 3]],
      [[0.1, 0.2, 0.3]],
      2.0,
    );
    expect(def[0][0]).toBeCloseTo(1.2, 6);
    expect(buildDeformedNodes([[1, 2, 3]], undefined, 2.0)).toEqual([[1, 2, 3]]);
  });
});

describe("Viewer3D interactive workbench component", () => {
  it("selects faces, probes field values, edits dirichlet/traction BCs, and emits ApplyFaceEvent3D", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn<(ev: ApplyFaceEvent3D) => void>();
    render(<Viewer3D data={sampleData()} onApply={onApply} />);

    // Fallback face map renders cleanly in jsdom without WebGL
    expect(screen.getByTestId("viewer-3d-fallback")).toBeInTheDocument();
    expect(screen.getByTestId("viewer-3d-apply-status")).toHaveTextContent("accepted");
    expect(screen.getByTestId("viewer-3d-readiness-errors")).toHaveTextContent("Assign at least one load.");

    // Switch field overlay -> colorbar appears
    await user.selectOptions(screen.getByLabelText("Field Overlay"), "von_mises_mpa");
    expect(screen.getByTestId("viewer-3d-colorbar")).toHaveTextContent("10.000 → 40.000 MPa");

    // Click second face chip -> updates selection and 3D probe readout
    await user.click(screen.getByRole("button", { name: /face_x_max/i }));
    expect(screen.getByTestId("viewer-3d-probe")).toHaveTextContent("face_x_max");

    // Configure Dirichlet displacement and apply
    await user.selectOptions(screen.getByLabelText("Condition Type"), "dirichlet");
    fireEvent.change(screen.getByLabelText("ux"), { target: { value: "0.75" } });
    fireEvent.change(screen.getByLabelText("uy"), { target: { value: "0.0" } });
    fireEvent.change(screen.getByLabelText("uz"), { target: { value: "-0.1" } });
    await user.click(screen.getByRole("button", { name: "Apply 3D Face BC" }));

    expect(onApply).toHaveBeenCalledTimes(1);
    const ev1 = onApply.mock.calls[0][0];
    expect(ev1.face_tag).toBe("face_x_max");
    expect(ev1.assignment.kind).toBe("dirichlet");
    expect(ev1.assignment.values_mm).toEqual([0.75, 0, -0.1]);

    // Switch to traction BC and apply
    await user.selectOptions(screen.getByLabelText("Condition Type"), "traction");
    fireEvent.change(screen.getByLabelText("tx"), { target: { value: "12.5" } });
    fireEvent.change(screen.getByLabelText("ty"), { target: { value: "0" } });
    fireEvent.change(screen.getByLabelText("tz"), { target: { value: "-2.0" } });
    await user.click(screen.getByRole("button", { name: "Apply 3D Face BC" }));
    expect(onApply).toHaveBeenCalledTimes(2);
    expect(onApply.mock.calls[1][0].assignment.traction_mpa).toEqual([12.5, 0, -2]);

    // Adjust deformation scale, wireframe ghost, and X/Y/Z clip sliders
    fireEvent.change(screen.getByLabelText("Deformation Scale"), { target: { value: "2.5" } });
    await user.click(screen.getByRole("checkbox"));
    fireEvent.change(screen.getByLabelText("Clip X"), { target: { value: "0.5" } });
    fireEvent.change(screen.getByLabelText("Clip Y"), { target: { value: "0.6" } });
    fireEvent.change(screen.getByLabelText("Clip Z"), { target: { value: "0.7" } });
    expect(screen.getByLabelText("Clip X")).toHaveValue("0.5");

    // Change face via dropdown and BC name input, apply fixed BC
    await user.selectOptions(screen.getByLabelText("Selected Face"), "face_x_min");
    fireEvent.change(screen.getByLabelText("BC Name"), { target: { value: "clamped_end" } });
    await user.selectOptions(screen.getByLabelText("Condition Type"), "fixed");
    await user.click(screen.getByRole("button", { name: "Apply 3D Face BC" }));
    expect(onApply).toHaveBeenCalledTimes(3);
    expect(onApply.mock.calls[2][0].assignment.name).toBe("clamped_end");
    expect(onApply.mock.calls[2][0].assignment.values_mm).toEqual([0, 0, 0]);
  });

  it("builds Three.js meshes, clip planes, ghost wireframe, and handles raycast face pick", () => {
    const data = sampleData();
    const onSelectFace = vi.fn();
    const onProbe = vi.fn();
    const canvas = document.createElement("canvas");
    canvas.setPointerCapture = vi.fn();
    canvas.releasePointerCapture = vi.fn();
    canvas.getBoundingClientRect = () =>
      ({ left: 0, top: 0, width: 400, height: 300, right: 400, bottom: 300, x: 0, y: 0, toJSON: () => ({}) }) as DOMRect;
    let renderedScene: THREE.Scene | null = null;
    const fakeRenderer: RendererLike = {
      domElement: canvas,
      localClippingEnabled: false,
      setSize: vi.fn(),
      render: (scene) => {
        renderedScene = scene;
      },
      dispose: vi.fn(),
    };
    const intersectSpy = vi
      .spyOn(THREE.Raycaster.prototype, "intersectObjects")
      .mockImplementation((objs) => [
        {
          distance: 1,
          point: new THREE.Vector3(10, 2.5, 0),
          object: objs[0],
        } as THREE.Intersection,
      ]);

    const assignmentsByTag = Object.fromEntries(data.assignments.map((a) => [a.face_tag, a]));
    const { unmount } = render(
      <FacePickerCanvas
        nodesMm={data.nodes_mm}
        triangles={data.triangles}
        faceGroups={data.face_groups}
        assignmentsByTag={assignmentsByTag}
        selectedFaceTag="face_x_min"
        activeField={data.fields?.von_mises_mpa ?? null}
        displacementsMm={data.displacements_mm}
        deformScale={1.0}
        showWireframeGhost={true}
        clip={{ xFrac: 0.8, yFrac: 0.9, zFrac: 1.0 }}
        onSelectFace={onSelectFace}
        onProbe={onProbe}
        rendererFactory={() => fakeRenderer}
      />,
    );

    expect(screen.getByTestId("viewer-3d-webgl")).toBeInTheDocument();
    expect(renderedScene).not.toBeNull();
    expect(fakeRenderer.localClippingEnabled).toBe(true);
    fireEvent.pointerDown(canvas, { clientX: 200, clientY: 150 });
    expect(onSelectFace).toHaveBeenCalledWith("face_x_min");
    expect(onProbe).toHaveBeenCalledTimes(1);
    expect(onProbe.mock.calls[0][0].field_value).toBeCloseTo(40.0, 5);
    unmount();
    intersectSpy.mockRestore();

    // Role-colored WebGL pass (activeField=null, ghost=false) + pointerDown without hit
    const emptyHitSpy = vi
      .spyOn(THREE.Raycaster.prototype, "intersectObjects")
      .mockReturnValue([]);
    const { unmount: unmountRoles } = render(
      <FacePickerCanvas
        nodesMm={data.nodes_mm}
        triangles={data.triangles}
        faceGroups={data.face_groups}
        assignmentsByTag={assignmentsByTag}
        selectedFaceTag={null}
        activeField={null}
        deformScale={0}
        showWireframeGhost={false}
        clip={{ xFrac: 1, yFrac: 1, zFrac: 1 }}
        onSelectFace={onSelectFace}
        onProbe={onProbe}
        rendererFactory={() => fakeRenderer}
      />,
    );
    fireEvent.pointerDown(canvas, { clientX: 50, clientY: 50 });
    unmountRoles();
    emptyHitSpy.mockRestore();

    // Renderer factory failure gracefully falls back to interactive face map
    render(
      <FacePickerCanvas
        nodesMm={data.nodes_mm}
        triangles={data.triangles}
        faceGroups={data.face_groups}
        assignmentsByTag={assignmentsByTag}
        selectedFaceTag="face_x_min"
        activeField={null}
        deformScale={0}
        showWireframeGhost={false}
        clip={{ xFrac: 1, yFrac: 1, zFrac: 1 }}
        onSelectFace={onSelectFace}
        onProbe={onProbe}
        rendererFactory={() => {
          throw new Error("no webgl");
        }}
      />,
    );
    expect(screen.getByTestId("viewer-3d-fallback")).toBeInTheDocument();

    // Empty geometry renders cleanly with Apply disabled
    render(
      <Viewer3D
        data={{
          workspace_id: "empty",
          geometry_id: "g0",
          draft_revision: "r0",
          nodes_mm: [],
          triangles: [],
          face_groups: [],
          assignments: [],
        }}
        onApply={vi.fn()}
      />,
    );
    expect(screen.getByRole("button", { name: "Apply 3D Face BC" })).toBeDisabled();
  });
});
