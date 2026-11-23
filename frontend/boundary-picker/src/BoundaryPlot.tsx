import { useEffect, useMemo, useRef, useState } from "react";
import type { Data, PlotlyHTMLElement, PlotMouseEvent } from "plotly.js";
import Plotly from "plotly.js-gl2d-dist-min";
import type { Figure, Mesh, Revision } from "./types";

function wireframe(mesh: Mesh): Data {
  const useWebGL = mesh.triangles.length > 1000;
  const size = mesh.triangles.length * 5;
  const x = useWebGL ? new Float64Array(size) : new Array<number | null>(size);
  const y = useWebGL ? new Float64Array(size) : new Array<number | null>(size);
  const gap = useWebGL ? NaN : null;
  for (const [index, [a, b, c]] of mesh.triangles.entries()) {
    const offset = index * 5;
    const first = mesh.nodes_mm[a];
    const second = mesh.nodes_mm[b];
    const third = mesh.nodes_mm[c];
    x[offset] = x[offset + 3] = first[0];
    y[offset] = y[offset + 3] = first[1];
    x[offset + 1] = second[0];
    y[offset + 1] = second[1];
    x[offset + 2] = third[0];
    y[offset + 2] = third[1];
    x[offset + 4] = y[offset + 4] = gap;
  }
  return {
    type: useWebGL ? "scattergl" : "scatter",
    mode: "lines", x, y, hoverinfo: "skip", showlegend: false,
    line: { color: "rgba(120, 130, 145, 0.42)", width: 0.8 },
  };
}

type Props = Revision & {
  figure: Figure;
  mesh: Mesh;
  selected: string;
  onSelect: (region: string) => void;
};

export function BoundaryPlot({ geometry_id, draft_revision, figure, mesh, selected, onSelect }: Props) {
  const element = useRef<HTMLDivElement>(null);
  const draw = useRef<Promise<PlotlyHTMLElement> | null>(null);
  const [showMesh, setShowMesh] = useState(true);
  const [plotError, setPlotError] = useState(false);

  // The host can deliver a fresh object on every rerun. These identities change only
  // when the corresponding mesh or accepted boundary traces change.
  const meshTrace = useMemo(() => wireframe(mesh), [geometry_id]);
  const boundaryTraces = useMemo(() => figure.data, [geometry_id, draft_revision]);
  const plottingLayout = useMemo(() => figure.layout, [geometry_id]);

  useEffect(() => {
    const node = element.current!;
    const style = getComputedStyle(node);
    const color = style.color;
    const background = style.backgroundColor;
    let active = true;
    let graph: PlotlyHTMLElement | undefined;
    const select = (event: PlotMouseEvent) => {
      const region = event.points[0]?.customdata;
      if (typeof region === "string") onSelect(region);
    };
    const resize = () => {
      if (node.offsetWidth && graph) void Promise.resolve(Plotly.Plots.resize(graph)).catch(() => {});
    };
    const observer = new ResizeObserver(resize);
    const outline: Data = {
      type: "scatter", mode: "lines", x: [], y: [], hoverinfo: "skip",
      line: { color, width: 8 }, showlegend: false,
    };
    const plot = Plotly.react(node, [meshTrace, outline, ...boundaryTraces], {
      ...plottingLayout,
      uirevision: geometry_id,
      height: undefined,
      title: undefined,
      autosize: true,
      clickmode: "event",
      showlegend: false,
      xaxis: { ...plottingLayout.xaxis, autorange: true },
      yaxis: { ...plottingLayout.yaxis, autorange: true },
      font: { family: style.fontFamily, color },
      paper_bgcolor: background,
      plot_bgcolor: background,
      margin: { l: 46, r: 15, t: 28, b: 47 },
    }, { displaylogo: false, responsive: false, modeBarButtonsToRemove: ["toImage", "select2d", "lasso2d", "sendChartToCloud"] });
    draw.current = plot;
    void plot.then((mounted) => {
      if (!active) return;
      graph = mounted;
      graph.on("plotly_click", select);
      observer.observe(node);
      resize();
    }).catch(() => { if (active) setPlotError(true); });
    return () => {
      active = false;
      graph?.removeAllListeners("plotly_click");
      observer.disconnect();
    };
  }, [geometry_id, meshTrace, boundaryTraces, plottingLayout, onSelect]);

  useEffect(() => {
    let active = true;
    const outline = boundaryTraces.find((trace) => (
      "mode" in trace && trace.mode === "lines" && Array.isArray(trace.customdata)
      && trace.customdata[0] === selected
    ));
    void draw.current?.then(async (graph) => {
      if (!active || !outline || !("x" in outline) || !("y" in outline)) return;
      await Plotly.restyle(graph, { x: [outline.x], y: [outline.y] }, [1]);
      if (active) graph.dataset.highlightedRegion = selected;
    }).catch(() => { if (active) setPlotError(true); });
    return () => { active = false; };
  }, [geometry_id, boundaryTraces, selected]);

  useEffect(() => {
    let active = true;
    void draw.current?.then((graph) => {
      if (active) return Plotly.restyle(graph, { visible: showMesh }, [0]);
    }).catch(() => { if (active) setPlotError(true); });
    return () => { active = false; };
  }, [geometry_id, boundaryTraces, showMesh]);

  useEffect(() => {
    const node = element.current!;
    return () => Plotly.purge(node);
  }, []);

  return (
    <figure className="boundary-figure">
      <label className="boundary-mesh-toggle">
        <input type="checkbox" checked={showMesh} onChange={(event) => setShowMesh(event.target.checked)} />
        Show mesh wireframe
      </label>
      <div ref={element} className="boundary-plot" data-testid="boundary-plot" />
      <figcaption>Selected on drawing <strong>{selected}</strong></figcaption>
      {plotError && <p role="alert">The drawing could not be displayed. Use the boundary selector to continue.</p>}
    </figure>
  );
}
