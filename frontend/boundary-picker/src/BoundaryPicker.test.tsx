import "@testing-library/jest-dom/vitest";
import { act, cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { BoundaryPicker } from "./BoundaryPicker";
import mount from "./index";
import type { PickerData } from "./types";

const plotly = vi.hoisted(() => ({
  react: vi.fn(),
  restyle: vi.fn().mockResolvedValue(undefined),
  purge: vi.fn(),
  click: undefined as ((event: { points: { customdata: unknown }[] }) => void) | undefined,
  Plots: { resize: vi.fn().mockResolvedValue(undefined) },
}));

vi.mock("plotly.js-gl2d-dist-min", () => ({ default: plotly }));

const E1 = "edge-000001";
const E2 = "edge-000002";
const left = { region: E1, name: "left_free", kind: "traction_free" as const };
const right = {
  region: E2, name: "right_load", kind: "dirichlet" as const,
  components: ["x" as const], values: [{ expression: "ux", unit: "mm" }],
};
const rightSide = { type: "scatter" as const, mode: "lines" as const, x: [1, 1], y: [0, 1], customdata: [E2, E2] };
const data: PickerData = {
  workspace_id: "workspace",
  geometry_id: "geometry",
  draft_revision: "revision",
  preview: {
    mesh: { nodes_mm: [[0, 0], [1, 0], [1, 1]], triangles: [[0, 1, 2]] },
    boundaries: [{ name: E1, kind: "line" }, { name: E2, kind: "line" }],
  },
  figure: {
    data: [
      { ...rightSide, x: [0, 0], customdata: [E1, E1] },
      rightSide,
      { type: "scatter", mode: "text+markers", x: [0, 1], y: [0.5, 0.5], customdata: [E1, E2] },
    ],
    layout: { xaxis: { title: { text: "x (mm)" } }, yaxis: { title: { text: "y (mm)" } } },
  },
  assignments: [left, right],
  suggested_regions: [E1, E2],
  readiness_errors: [],
  last_apply: null,
};

const selector = () => screen.getByLabelText("Selected CAD Boundary Group");
const name = () => screen.getByLabelText("Condition Name");
const displacement = () => screen.getByLabelText("u_x expression (mm)");
const apply = () => screen.getByRole("button", { name: "Apply boundary condition" });
const activeRow = () => screen.getByRole("row", { current: true });

beforeEach(() => {
  class TestResizeObserver {
    observe = vi.fn();
    disconnect = vi.fn();
  }
  vi.stubGlobal("ResizeObserver", TestResizeObserver);
  plotly.click = undefined;
  plotly.react.mockImplementation(async (element: HTMLElement) => Object.assign(element, {
    on: (_name: string, callback: typeof plotly.click) => { plotly.click = callback; },
    removeAllListeners: () => { plotly.click = undefined; },
  }));
});

afterEach(() => {
  cleanup();
  vi.unstubAllGlobals();
  vi.clearAllMocks();
});

describe("Boundary Picker", () => {
  it("keeps the selector, drawing, editor and accepted table on the same edge", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    render(<BoundaryPicker data={data} onApply={onApply} />);
    const plot = screen.getByTestId("boundary-plot");
    expect(activeRow()).toHaveTextContent(E1);
    expect(activeRow()).toHaveTextContent("Suggested from geometry");
    await user.selectOptions(selector(), E2);
    expect(name()).toHaveValue("right_load");
    expect(activeRow()).toHaveTextContent(E2);
    await waitFor(() => expect(plot).toHaveAttribute("data-highlighted-region", E2));
    await act(async () => plotly.click?.({ points: [{ customdata: E1 }] }));
    expect(selector()).toHaveValue(E1);
    expect(name()).toHaveValue("left_free");
    expect(activeRow()).toHaveTextContent(E1);
    await waitFor(() => expect(plot).toHaveAttribute("data-highlighted-region", E1));
    expect(onApply).not.toHaveBeenCalled();
  });

  it("keeps per-edge draft edits and focus across navigation and fresh host props", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    const view = render(<BoundaryPicker data={data} onApply={onApply} />);
    await user.selectOptions(selector(), E2);
    await user.clear(displacement());
    await user.type(displacement(), "2 * ux");
    await user.selectOptions(selector(), E1);
    await user.clear(name());
    await user.type(name(), "another_name");
    await user.selectOptions(selector(), E2);
    expect(displacement()).toHaveValue("2 * ux");
    expect(activeRow()).toHaveTextContent("u_x = ux mm");
    await user.click(displacement());
    view.rerender(<BoundaryPicker data={structuredClone(data)} onApply={onApply} />);
    expect(displacement()).toHaveFocus();
    expect(displacement()).toHaveValue("2 * ux");
    expect(onApply).not.toHaveBeenCalled();
    await user.click(apply());
    expect(onApply).toHaveBeenCalledOnce();
    const event = onApply.mock.lastCall![0];
    expect(event).toEqual({
      event_id: expect.any(String), workspace_id: "workspace", geometry_id: "geometry",
      draft_revision: "revision", region: E2,
      assignment: { ...right, values: [{ expression: "2 * ux", unit: "mm" }] },
    });
    view.rerender(<BoundaryPicker data={{
      ...data, assignments: [left, event.assignment], suggested_regions: [E1], draft_revision: "updated",
      last_apply: { event_id: event.event_id, region: E2, status: "accepted", draft_revision: "updated", message: "Applied boundary condition to edge-000002." },
    }} onApply={onApply} />);
    expect(screen.getByRole("status")).toHaveTextContent("Applied boundary condition");
    expect(activeRow()).toHaveTextContent("u_x = 2 * ux mm");
    await user.selectOptions(selector(), E1);
    expect(name()).toHaveValue("another_name");
    await user.click(apply());
    expect(onApply.mock.lastCall![0]).toMatchObject({ draft_revision: "updated", region: E1 });
  });

  it("links an invalid field to the diagnostic and preserves the proposed value", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    const view = render(<BoundaryPicker data={data} onApply={onApply} />);
    await user.selectOptions(selector(), E2);
    await user.clear(displacement());
    await user.type(displacement(), "bad expression");
    await user.click(apply());
    view.rerender(<BoundaryPicker data={{
      ...data, readiness_errors: ["Missing another region"],
      last_apply: { event_id: onApply.mock.lastCall![0].event_id, region: E2, status: "invalid", errors: { x: "Unknown symbol" } },
    }} onApply={onApply} />);
    expect(displacement()).toHaveValue("bad expression");
    expect(displacement()).toHaveAttribute("aria-invalid", "true");
    expect(displacement()).toHaveAccessibleDescription("Unknown symbol");
    expect(activeRow()).toHaveTextContent("u_x = ux mm");
    expect(screen.getByText("Missing another region")).toBeVisible();
  });

  it("retains a stale proposal until the engineer reloads the current accepted revision", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    const view = render(<BoundaryPicker data={data} onApply={onApply} />);
    await user.clear(name());
    await user.type(name(), "my_proposal");
    const latest = { ...data, assignments: [{ ...left, name: "remote_edit" }, right], draft_revision: "remote", geometry_id: "fresh-geometry" };
    view.rerender(<BoundaryPicker data={latest} onApply={onApply} />);
    await user.click(apply());
    expect(onApply.mock.lastCall![0]).toMatchObject({ geometry_id: "geometry", draft_revision: "revision" });
    const eventId = onApply.mock.lastCall![0].event_id;
    view.rerender(<BoundaryPicker data={{
      ...latest, last_apply: { event_id: eventId, region: E1, status: "stale", reload: true, message: "Conditions changed" },
    }} onApply={onApply} />);
    expect(name()).toHaveValue("my_proposal");
    expect(activeRow()).toHaveTextContent("remote_edit");
    expect(apply()).toBeDisabled();
    await user.click(screen.getByRole("button", { name: "Reload accepted conditions" }));
    expect(name()).toHaveValue("my_proposal");
    await user.click(apply());
    expect(onApply.mock.lastCall![0]).toMatchObject({ geometry_id: "fresh-geometry", draft_revision: "remote" });
  });

  it("submits symmetry components and can toggle mesh visibility", async () => {
    const user = userEvent.setup();
    const onApply = vi.fn();
    render(<BoundaryPicker data={data} onApply={onApply} />);
    await user.selectOptions(screen.getByLabelText("Boundary Condition Type"), "symmetry");
    await user.click(screen.getByLabelText("Constrain y"));
    expect(screen.queryByLabelText("u_x expression (mm)")).toBeNull();
    await user.click(screen.getByLabelText("Constrain x"));
    await user.click(screen.getByLabelText("Constrain x"));
    await user.click(screen.getByLabelText("Show mesh wireframe"));
    expect(plotly.restyle).toHaveBeenCalledWith(expect.any(HTMLElement), { visible: false }, [0]);
    await user.click(apply());
    expect(onApply.mock.lastCall![0].assignment).toEqual({ ...left, kind: "symmetry", components: ["y"] });
  });

  it("retains keyboard access when plotting fails", async () => {
    plotly.react.mockRejectedValueOnce(new Error("Plot unavailable"));
    const user = userEvent.setup();
    render(<BoundaryPicker data={data} onApply={vi.fn()} />);
    expect(await screen.findByRole("alert")).toHaveTextContent("The drawing could not be displayed");
    await user.selectOptions(selector(), E2);
    expect(activeRow()).toHaveTextContent(E2);
  });

  it("uses a WebGL mesh for large previews and ignores click events without an edge", async () => {
    const mesh = { ...data.preview.mesh, triangles: Array.from({ length: 1001 }, () => [0, 1, 2] as [number, number, number]) };
    render(<BoundaryPicker data={{ ...data, preview: { ...data.preview, mesh } }} onApply={vi.fn()} />);
    await waitFor(() => expect(screen.getByTestId("boundary-plot")).toHaveAttribute("data-highlighted-region", E1));
    expect(plotly.react.mock.lastCall![1][0].type).toBe("scattergl");
    await act(async () => {
      plotly.click?.({ points: [] });
      plotly.click?.({ points: [{ customdata: null }] });
    });
    expect(selector()).toHaveValue(E1);
  });

  it("handles an empty boundary roster and reports a missing mount element", () => {
    render(<BoundaryPicker data={{ ...data, assignments: [] }} onApply={vi.fn()} />);
    expect(screen.getByText("No boundary groups are available.")).toBeVisible();
    expect(() => mount({ name: "picker", key: "none", data, parentElement: document.createElement("div"),
      setTriggerValue: vi.fn(), setStateValue: vi.fn() })).toThrow("mount element is missing");
  });

  it("uses a stable React root across Streamlit renders and cleans up on unmount", async () => {
    const container = document.createElement("div");
    container.innerHTML = '<div class="react-root"></div>';
    document.body.append(container);
    const setTriggerValue = vi.fn();
    const args: Parameters<typeof mount>[0] = {
      name: "picker", data, parentElement: container, setTriggerValue, setStateValue: vi.fn(), key: "component",
    };
    let unmount: ReturnType<typeof mount>;
    await act(async () => { unmount = mount(args); });
    fireEvent.change(name(), { target: { value: "new_name" } });
    await act(async () => { mount({ ...args, data: structuredClone(data) }); });
    expect(name()).toHaveValue("new_name");
    fireEvent.click(within(container).getByRole("button", { name: "Apply boundary condition" }));
    expect(setTriggerValue).toHaveBeenCalledWith("apply", expect.objectContaining({ region: E1 }));
    await act(async () => { unmount?.(); });
    expect(container).not.toHaveTextContent("new_name");
    container.remove();
  });
});
