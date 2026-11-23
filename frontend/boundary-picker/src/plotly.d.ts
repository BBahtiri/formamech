declare module "plotly.js-gl2d-dist-min" {
  const plotly: Omit<typeof import("plotly.js"), "restyle"> & {
    restyle(
      root: import("plotly.js").Root,
      update: Record<string, unknown>,
      traces: number[],
    ): Promise<import("plotly.js").PlotlyHTMLElement>;
  };
  export default plotly;
}
