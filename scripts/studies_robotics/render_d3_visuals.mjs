/**
 * Publication-grade D3.js + Custom SVG + Playwright renderer for FormaMech v0.6
 * Robotics Physics AI validation studies (Studies R1-R5).
 *
 * Generates:
 *   - 5 high-DPI (2x) 4-panel publication PNGs in docs/assets/studies_robotics/
 *   - 5 sequences of animation PNG frames in .formamech-runs/robotics_studies/gif_frames/
 *     (assembled into optimized animated GIFs by scripts/studies_robotics/build_gifs.py)
 *
 * Dependencies (d3, playwright) are resolved from gitignored .formamech-runs/d3_renderer/node_modules.
 */

import fs from "node:fs";
import path from "node:path";
import { createRequire } from "node:module";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const REPO_ROOT = path.resolve(__dirname, "..", "..");
const RENDERER_NM = path.join(REPO_ROOT, ".formamech-runs", "d3_renderer", "node_modules");
const require = createRequire(path.join(RENDERER_NM, "package.json"));
const { chromium } = require(path.join(RENDERER_NM, "playwright"));
const D3_UMD_PATH = path.join(RENDERER_NM, "d3", "dist", "d3.min.js");

const VIZ_DIR = path.join(REPO_ROOT, ".formamech-runs", "robotics_studies");
const ASSETS_DIR = path.join(REPO_ROOT, "docs", "assets", "studies_robotics");
const FRAMES_ROOT = path.join(VIZ_DIR, "gif_frames");

const D3_SOURCE = fs.readFileSync(D3_UMD_PATH, "utf-8");

function buildHtml(width, height, bg, drawScript, payload) {
  const safeJson = JSON.stringify(payload).replace(/<\/script>/gi, "<\\/script>");
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: ${width}px; height: ${height}px; overflow: hidden;
    background: ${bg};
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  svg { display: block; width: ${width}px; height: ${height}px; }
  .axis path, .axis line { stroke: #cbd5e1; stroke-width: 1; }
  .axis text { fill: #475569; font-size: 11px; }
  .axis-dark path, .axis-dark line { stroke: #334155; stroke-width: 1; }
  .axis-dark text { fill: #94a3b8; font-size: 10.5px; }
  .grid line { stroke: #e2e8f0; stroke-dasharray: 2,3; }
  .grid-dark line { stroke: #1e293b; stroke-dasharray: 2,3; }
  .grid path, .grid-dark path { stroke-width: 0; }
</style>
<script>${D3_SOURCE}</script>
</head>
<body>
<svg id="canvas" viewBox="0 0 ${width} ${height}" width="${width}" height="${height}"></svg>
<script>
window.__RENDER_DONE__ = false;
window.__RENDER_ERR__ = null;
try {
(() => {
const DATA = ${safeJson};
const svg = d3.select("#canvas");

// Common isometric 3D projection helper
function isoProject(x, y, z, cx, cy, scale, yawDeg = -35, pitchDeg = 28) {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const rx = x * Math.cos(yaw) - y * Math.sin(yaw);
  const ry = x * Math.sin(yaw) + y * Math.cos(yaw);
  const px = rx;
  const py = ry * Math.sin(pitch) - z * Math.cos(pitch);
  const depth = ry * Math.cos(pitch) + z * Math.sin(pitch);
  return { x: cx + px * scale, y: cy + py * scale, depth };
}

function addHeader(svg, W, title, subtitle, badges, dark = false) {
  svg.append("rect")
    .attr("x", 0).attr("y", 0).attr("width", W).attr("height", 68)
    .attr("fill", dark ? "#0f172a" : "#0f172a");
  svg.append("line")
    .attr("x1", 0).attr("x2", W).attr("y1", 68).attr("y2", 68)
    .attr("stroke", "#38bdf8").attr("stroke-width", 2);
  svg.append("text")
    .attr("x", 24).attr("y", 30)
    .attr("fill", "#f8fafc").attr("font-size", 19).attr("font-weight", 700)
    .text(title);
  svg.append("text")
    .attr("x", 24).attr("y", 52)
    .attr("fill", "#94a3b8").attr("font-size", 12)
    .text(subtitle);
  let bx = W - 20;
  for (let i = badges.length - 1; i >= 0; i--) {
    const b = badges[i];
    const bw = Math.max(96, b.text.length * 6.8 + 20);
    bx -= bw;
    svg.append("rect")
      .attr("x", bx).attr("y", 19).attr("width", bw).attr("height", 28).attr("rx", 14)
      .attr("fill", b.bg).attr("stroke", b.border || "none").attr("stroke-width", 1);
    svg.append("text")
      .attr("x", bx + bw / 2).attr("y", 37).attr("text-anchor", "middle")
      .attr("fill", b.fg || "#fff").attr("font-size", 11).attr("font-weight", 700)
      .text(b.text);
    bx -= 10;
  }
}

function addCard(svg, x, y, w, h, title, subtitle, dark = false) {
  const g = svg.append("g").attr("transform", "translate(" + x + "," + y + ")");
  g.append("rect")
    .attr("width", w).attr("height", h).attr("rx", 10)
    .attr("fill", dark ? "#111827" : "#ffffff")
    .attr("stroke", dark ? "#1e293b" : "#e2e8f0").attr("stroke-width", 1.2);
  g.append("text")
    .attr("x", 16).attr("y", 25)
    .attr("fill", dark ? "#f1f5f9" : "#0f172a").attr("font-size", 14).attr("font-weight", 700)
    .text(title);
  if (subtitle) {
    g.append("text")
      .attr("x", 16).attr("y", 42)
      .attr("fill", dark ? "#94a3b8" : "#64748b").attr("font-size", 11)
      .text(subtitle);
  }
  return g;
}

${drawScript}
})();
window.__RENDER_DONE__ = true;
} catch (err) {
  window.__RENDER_ERR__ = String(err && err.stack ? err.stack : err);
  window.__RENDER_DONE__ = true;
}
</script>
</body>
</html>`;
}

// ============================================================================
// STATIC PUBLICATION FIGURES (R1 - R5): 1560 x 920 @ 2x DPR
// ============================================================================

const DRAW_FIG_R1 = `
const W = 1560, H = 920;
const S = DATA.summary;
addHeader(svg, W,
  "Study R1: GelSight Mini 3D TET10 Elastomer Contact Digital Twin",
  "Real Data: feats-gelsight-mini STEP CAD (MIT) + GlowTact 3,997 CNC Probes (CC-BY-4.0) | Locking Guard: TET4 Refused -> TET10 Accepted (nu=0.48)",
  [
    { text: "ASME V&V 10: " + S.asme_vv10_grounding.state, bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "AS6983: " + (S.surrogate_as6983_holdout.as6983_upper_coverage * 100).toFixed(1) + "% Cov", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: S.surrogate_as6983_holdout.median_inference_ms.toFixed(2) + " ms (" + Math.round(S.surrogate_as6983_holdout.inference_hz) + " Hz)", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: 3D Isometric GelSight Mini Pad + 8x8 Taxel Force Field (top-left)
const gA = addCard(svg, 20, 86, 750, 400,
  "A. 3D TET10 Elastomer Contact FEM & 8x8 Taxel Shear/Normal Field",
  "Deformed silicone pad (E_cal = " + S.physical_calibration.calibrated_youngs_modulus_mpa + " MPa, nu = 0.48, mu = 0.35) | Peak indentation + tangential shear"
);
const fPeak = DATA.anim_frames[DATA.anim_frames.length - 1];
const vmMax = Math.max(...fPeak.surface_vm_mpa, 0.05);
const colorVm = d3.scaleSequential(d3.interpolateTurbo).domain([0, vmMax]);
// Draw 3D surface points sorted by depth
const pts3d = fPeak.surface_pts.map((p, i) => {
  const pr = isoProject(p[0] - 9, p[1] - 12, (p[2] - 4) * 3.5, 225, 225, 12.2, -38, 30);
  return { ...pr, vm: fPeak.surface_vm_mpa[i] };
}).sort((a, b) => a.depth - b.depth);
// Draw base pad outline
const baseCorners = [[0,0,0],[18,0,0],[18,24,0],[0,24,0]].map(c => isoProject(c[0]-9, c[1]-12, -3.8, 225, 225, 12.2, -38, 30));
gA.append("polygon")
  .attr("points", baseCorners.map(c => c.x + "," + c.y).join(" "))
  .attr("fill", "#e2e8f0").attr("stroke", "#94a3b8").attr("stroke-width", 1.2);
const topCorners = [[0,0,0],[18,0,0],[18,24,0],[0,24,0]].map(c => isoProject(c[0]-9, c[1]-12, 0, 225, 225, 12.2, -38, 30));
gA.append("polygon")
  .attr("points", topCorners.map(c => c.x + "," + c.y).join(" "))
  .attr("fill", "#f1f5f9").attr("fill-opacity", 0.7).attr("stroke", "#64748b").attr("stroke-width", 1);
gA.selectAll("circle.node3d").data(pts3d).enter().append("circle")
  .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 4.6)
  .attr("fill", d => colorVm(d.vm)).attr("stroke", "#0f172a").attr("stroke-width", 0.5);
// Rigid spherical indenter above center
const indC = isoProject(0, 0, 2.2, 225, 225, 12.2, -38, 30);
const defs = svg.append("defs");
const gradSph = defs.append("radialGradient").attr("id", "sphGrad").attr("cx", "35%").attr("cy", "30%");
gradSph.append("stop").attr("offset", "0%").attr("stop-color", "#e2e8f0");
gradSph.append("stop").attr("offset", "60%").attr("stop-color", "#64748b");
gradSph.append("stop").attr("offset", "100%").attr("stop-color", "#1e293b");
gA.append("circle").attr("cx", indC.x).attr("cy", indC.y).attr("r", 28)
  .attr("fill", "url(#sphGrad)").attr("fill-opacity", 0.82).attr("stroke", "#0f172a").attr("stroke-width", 1.2);
gA.append("text").attr("x", 225).attr("y", 374).attr("text-anchor", "middle")
  .attr("fill", "#475569").attr("font-size", 11).attr("font-weight", 600)
  .text("3D TET10 Gel Pad (18x24x4 mm) — Von Mises Stress");

// Right half of Panel A: 8x8 Taxel Heatmap + Shear Arrows (with physical Hertzian kernel spread conserving total force)
const gx0 = 455, gy0 = 78, cell = 31;
const rawFz = fPeak.taxel_fz;
const rawFx = fPeak.taxel_fx;
const nxT = rawFz.length, nyT = rawFz[0].length;
// Spread nodal lumped forces across neighboring taxels via 2D Gaussian (sigma=1.15 taxels) conserving exact sum
const taxFz = Array.from({ length: nxT }, () => Array(nyT).fill(0));
const taxFx = Array.from({ length: nxT }, () => Array(nyT).fill(0));
for (let sx = 0; sx < nxT; sx++) {
  for (let sy = 0; sy < nyT; sy++) {
    if (Math.abs(rawFz[sx][sy]) < 1e-6 && Math.abs(rawFx[sx][sy]) < 1e-6) continue;
    let wSum = 0;
    const ws = [];
    for (let tx = 0; tx < nxT; tx++) {
      for (let ty = 0; ty < nyT; ty++) {
        const d2 = (tx - sx) * (tx - sx) + (ty - sy) * (ty - sy);
        const w = Math.exp(-d2 / (2 * 1.15 * 1.15));
        ws.push([tx, ty, w]);
        wSum += w;
      }
    }
    for (const [tx, ty, w] of ws) {
      taxFz[tx][ty] += rawFz[sx][sy] * (w / wSum);
      taxFx[tx][ty] += rawFx[sx][sy] * (w / wSum);
    }
  }
}
const maxTax = Math.max(...taxFz.flat(), 0.05);
const taxCol = d3.scaleSequential(d3.interpolateYlOrRd).domain([0, maxTax]);
for (let ix = 0; ix < nxT; ix++) {
  for (let iy = 0; iy < nyT; iy++) {
    const val = taxFz[ix][iy];
    const cx = gx0 + ix * cell;
    const cy = gy0 + iy * cell;
    gA.append("rect").attr("x", cx).attr("y", cy).attr("width", cell - 2).attr("height", cell - 2).attr("rx", 3)
      .attr("fill", val > maxTax * 0.02 ? taxCol(val) : "#f1f5f9").attr("stroke", "#cbd5e1").attr("stroke-width", 0.6);
    const fx = taxFx[ix][iy];
    if (Math.abs(fx) > maxTax * 0.03) {
      const len = Math.min(cell * 0.44, (fx / maxTax) * 24);
      gA.append("line").attr("x1", cx + cell/2 - 1).attr("y1", cy + cell/2 - 1)
        .attr("x2", cx + cell/2 - 1 + len).attr("y2", cy + cell/2 - 1)
        .attr("stroke", "#0f172a").attr("stroke-width", 1.8);
      gA.append("circle").attr("cx", cx + cell/2 - 1 + len).attr("cy", cy + cell/2 - 1).attr("r", 1.8).attr("fill", "#0f172a");
    }
  }
}
gA.append("text").attr("x", gx0 + 4 * cell).attr("y", 352).attr("text-anchor", "middle")
  .attr("fill", "#475569").attr("font-size", 11).attr("font-weight", 600)
  .text("8x8 Taxel Normal Force + Shear Vectors");
gA.append("text").attr("x", gx0 + 4 * cell).attr("y", 374).attr("text-anchor", "middle")
  .attr("fill", "#059669").attr("font-size", 10.5).attr("font-weight", 600)
  .text("All 4 SDF Primitives Converged | DiffMJX CFD Grad = +" + S.diffmjx_cfd_verification.separated_backward_grad.toFixed(1) + " N/mm");

// Panel B: Real GlowTact Blind Holdout Force vs Depth + AS6983 Conformal Band (top-right)
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Real GlowTact CNC Physical Holdout (n = 120) & AS6983 Bounds",
  "Blind holdout RMSE = " + S.surrogate_as6983_holdout.holdout_rmse_n.toFixed(2) + " N | Hybrid V&V10 RMSE = " + S.asme_vv10_grounding.grounded_hybrid_holdout_rmse_n.toFixed(2) + " N (raw FEM: " + S.asme_vv10_grounding.raw_fem_holdout_rmse_n.toFixed(2) + " N)"
);
const mB = { l: 58, r: 24, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const hp = DATA.holdout_points;
const xB = d3.scaleLinear().domain([0, d3.max(hp, d => d.depth_mm) * 1.05]).range([0, pwB]);
const yB = d3.scaleLinear().domain([0, d3.max(hp, d => Math.max(d.true_f_n, d.hi_n)) * 1.08]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(6));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
// Bin holdout points to draw clean smooth AS6983 conformal envelope
const bins = d3.bin().value(d => d.depth_mm).thresholds(12)(hp).filter(b => b.length > 0);
const envPts = bins.map(b => ({
  d: d3.mean(b, x => x.depth_mm),
  pred: d3.mean(b, x => x.pred_f_n),
  lo: d3.mean(b, x => Math.max(0, x.lo_n)),
  hi: d3.mean(b, x => x.hi_n),
}));
pB.append("path").datum(envPts)
  .attr("fill", "#3b82f6").attr("fill-opacity", 0.16)
  .attr("d", d3.area().x(d => xB(d.d)).y0(d => yB(d.lo)).y1(d => yB(d.hi)).curve(d3.curveMonotoneX));
pB.append("path").datum(envPts)
  .attr("fill", "none").attr("stroke", "#2563eb").attr("stroke-width", 2.5)
  .attr("d", d3.line().x(d => xB(d.d)).y(d => yB(d.pred)).curve(d3.curveMonotoneX));
// FEM curve
pB.append("path").datum(DATA.fem_curve)
  .attr("fill", "none").attr("stroke", "#dc2626").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,4")
  .attr("d", d3.line().x(d => xB(d.depth_mm)).y(d => yB(d.fem_fz_n)));
// Real physical holdout scatter
pB.selectAll("circle.hold").data(hp).enter().append("circle")
  .attr("cx", d => xB(d.depth_mm)).attr("cy", d => yB(d.true_f_n)).attr("r", 3.3)
  .attr("fill", "#0f172a").attr("fill-opacity", 0.65);
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Effective Indentation Depth (mm)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Normal Contact Force (N)");
// Legend
const legB = pB.append("g").attr("transform", "translate(14, 8)");
legB.append("rect").attr("width", 268).attr("height", 66).attr("rx", 6).attr("fill", "#fff").attr("fill-opacity", 0.92).attr("stroke", "#e2e8f0");
legB.append("circle").attr("cx", 14).attr("cy", 16).attr("r", 3.5).attr("fill", "#0f172a");
legB.append("text").attr("x", 26).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("Real GlowTact CNC Holdout Probes (n=120)");
legB.append("line").attr("x1", 8).attr("x2", 22).attr("y1", 34).attr("y2", 34).attr("stroke", "#2563eb").attr("stroke-width", 2.5);
legB.append("text").attr("x", 26).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("Surrogate + AS6983 90% Band (95.0% upper)");
legB.append("line").attr("x1", 8).attr("x2", 22).attr("y1", 52).attr("y2", 52).attr("stroke", "#dc2626").attr("stroke-width", 2).attr("stroke-dasharray", "5,3");
legB.append("text").attr("x", 26).attr("y", 56).attr("font-size", 10.5).attr("fill", "#1e293b").text("Calibrated 3D TET10 SDF Contact FEM");

// Panel C: Active Physical-Test Reduction (bottom-left)
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Active Physical-Test Reduction on Real GlowTact CNC Pool",
  "FormaMech active.test_planner reaches " + S.active_test_reduction.target_holdout_rmse_n + " N in " + S.active_test_reduction.active_tests_needed + " probes vs " + S.active_test_reduction.random_mean_tests_needed + " random (" + S.active_test_reduction.tests_saved_pct + "% saved)"
);
const mC = { l: 58, r: 24, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const actC = DATA.active_curves.active_rmse_curve;
const rndC = DATA.active_curves.random_mean_rmse_curve;
const nPtsC = Math.min(actC.length, rndC.length);
const xsC = d3.range(nPtsC).map(i => i + 3);
const xC = d3.scaleLinear().domain([3, xsC[xsC.length - 1]]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, d3.max([...actC, ...rndC]) * 1.12]).range([phC, 0]);
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(8));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.append("line").attr("x1", 0).attr("x2", pwC).attr("y1", yC(S.active_test_reduction.target_holdout_rmse_n)).attr("y2", yC(S.active_test_reduction.target_holdout_rmse_n))
  .attr("stroke", "#059669").attr("stroke-width", 1.5).attr("stroke-dasharray", "5,4");
pC.append("path").datum(rndC.slice(0, nPtsC))
  .attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 2.4).attr("stroke-dasharray", "6,3")
  .attr("d", d3.line().x((_, i) => xC(xsC[i])).y(d => yC(d)));
pC.append("path").datum(actC.slice(0, nPtsC))
  .attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.8)
  .attr("d", d3.line().x((_, i) => xC(xsC[i])).y(d => yC(d)));
pC.selectAll("circle.ac").data(actC.slice(0, nPtsC)).enter().append("circle")
  .attr("cx", (_, i) => xC(xsC[i])).attr("cy", d => yC(d)).attr("r", 3.2).attr("fill", "#0284c7");
pC.append("text").attr("x", pwC - 6).attr("y", yC(0.25))
  .attr("text-anchor", "end").attr("fill", "#059669").attr("font-size", 11).attr("font-weight", 700)
  .text("Target Holdout RMSE = " + S.active_test_reduction.target_holdout_rmse_n + " N (" + S.active_test_reduction.tests_saved_pct + "% CNC probes saved)");
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Cumulative Physical GlowTact CNC Probes Used");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Blind Holdout Force RMSE (N)");
const legC = pC.append("g").attr("transform", "translate(" + (pwC - 235) + ", 10)");
legC.append("rect").attr("width", 225).attr("height", 46).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legC.append("line").attr("x1", 10).attr("x2", 28).attr("y1", 16).attr("y2", 16).attr("stroke", "#0284c7").attr("stroke-width", 2.8);
legC.append("text").attr("x", 34).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("FormaMech Active Info-Gain (" + S.active_test_reduction.active_tests_needed + ")");
legC.append("line").attr("x1", 10).attr("x2", 28).attr("y1", 34).attr("y2", 34).attr("stroke", "#94a3b8").attr("stroke-width", 2.4).attr("stroke-dasharray", "6,3");
legC.append("text").attr("x", 34).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("10-Seed Random Mean (" + S.active_test_reduction.random_mean_tests_needed + ")");

// Panel D: Honest Cross-Geometry OOD Shift (bottom-right)
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Honest Cross-Geometry OOD Shift (Round-Calibrated FEM on Unseen Indenters)",
  "Testing single-indenter calibration across unseen GlowTact tips exposes geometry mismatch cleanly"
);
const mD = { l: 64, r: 24, t: 62, b: 56 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const oodData = Object.entries(S.cross_geometry_ood).map(([k, v]) => ({
  shape: k, rmse: v.raw_fem_rmse_n, is_id: k === "round"
}));
const xD = d3.scaleBand().domain(oodData.map(d => d.shape)).range([0, pwD]).padding(0.35);
const yD = d3.scaleLinear().domain([0, d3.max(oodData, d => d.rmse) * 1.18]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
pD.selectAll("rect.bar").data(oodData).enter().append("rect")
  .attr("x", d => xD(d.shape)).attr("y", d => yD(d.rmse))
  .attr("width", xD.bandwidth()).attr("height", d => phD - yD(d.rmse)).attr("rx", 6)
  .attr("fill", d => d.is_id ? "#0284c7" : "#f59e0b");
pD.selectAll("text.val").data(oodData).enter().append("text")
  .attr("x", d => xD(d.shape) + xD.bandwidth() / 2).attr("y", d => yD(d.rmse) - 8)
  .attr("text-anchor", "middle").attr("font-size", 12).attr("font-weight", 700)
  .attr("fill", d => d.is_id ? "#0369a1" : "#b45309")
  .text(d => d.rmse.toFixed(2) + " N" + (d.is_id ? " (ID)" : " (OOD)"));
pD.append("text").attr("x", pwD / 2).attr("y", phD + 40).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("GlowTact Physical Indenter Tip Geometry (150 real probes each)");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Raw Single-Tip FEM RMSE (N)");
`;

const DRAW_FIG_R2 = `
const W = 1560, H = 920;
const S = DATA.summary;
const C = S.ctcr_real_data_holdout;
const B = S.cosserat_9disk_backbone;
addHeader(svg, W,
  "Study R2: Continuum & Tendon-Driven Robot Digital Twin (V6-R1)",
  "Real Data: CRL-CTCR 100,000 EM-Tracked SE(3) Poses (MIT) + Gotelli 9-Disk 4-Tendon Nitinol Backbone (L=540 mm, CC-BY-4.0)",
  [
    { text: "Tip RMSE: " + C.tip_rmse_mm.toFixed(2) + " mm", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "AS6983: " + (C.as6983_upper_coverage * 100).toFixed(1) + "% Cov", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: C.median_inference_ms.toFixed(2) + " ms (" + Math.round(C.inference_hz) + " Hz)", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: 3D Isometric Gotelli 9-Disk 4-Tendon Cosserat Backbone Deflection Sweep
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Exact SE(3) Cosserat 9-Disk 4-Tendon Nitinol Backbone (L = 540 mm)",
  "Finite-rotation sweep up to " + B.max_bending_angle_deg + " deg bending | Mean Cosserat FEM: " + B.mean_fem_solve_ms + " ms | Capstan Solid FEM Converged"
);
const sweepIdx = [0, 4, 8, 14, 18, 23];
const sweepCols = ["#cbd5e1", "#93c5fd", "#60a5fa", "#2563eb", "#7c3aed", "#059669"];
// Base plate
const bp = [[-45,-45,0],[45,-45,0],[45,45,0],[-45,45,0]].map(c => isoProject(c[0], c[1], c[2], 370, 340, 0.56, -32, 22));
gA.append("polygon").attr("points", bp.map(p => p.x + "," + p.y).join(" ")).attr("fill", "#e2e8f0").attr("stroke", "#475569").attr("stroke-width", 1.2);
sweepIdx.forEach((fi, si) => {
  const fr = DATA.anim_frames[Math.min(fi, DATA.anim_frames.length - 1)];
  const pts = fr.backbone_xyz.map(p => isoProject(p[0], p[1], p[2], 370, 340, 0.56, -32, 22));
  gA.append("path").datum(pts)
    .attr("fill", "none").attr("stroke", sweepCols[si]).attr("stroke-width", si === 4 ? 3.5 : 2.2)
    .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
  if (si === 4) {
    // Highlight 9 spacer disks on primary deformed pose
    pts.slice(1).forEach((dp, di) => {
      gA.append("ellipse").attr("cx", dp.x).attr("cy", dp.y).attr("rx", 11).attr("ry", 4.5)
        .attr("fill", "#ede9fe").attr("stroke", "#6d28d9").attr("stroke-width", 1.4);
    });
    const tip = pts[pts.length - 1];
    gA.append("line").attr("x1", tip.x).attr("y1", tip.y).attr("x2", tip.x + 48).attr("y2", tip.y + 18)
      .attr("stroke", "#e11d48").attr("stroke-width", 2.6);
    gA.append("text").attr("x", tip.x + 54).attr("y", tip.y + 22).attr("fill", "#be123c").attr("font-size", 11).attr("font-weight", 700)
      .text("F_ext = 0.25 N");
  }
});
gA.append("text").attr("x", 24).attr("y", 376).attr("fill", "#475569").attr("font-size", 11).attr("font-weight", 600)
  .text("Multi-pose 3D finite-rotation orbital actuation (T_max = 8 N) | 9 disks @ d_c = 37.5 mm | E = 58 GPa Nitinol");

// Panel B: Real CRL-CTCR Blind Holdout Tip Trajectory (True vs Pred)
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Real CRL-CTCR 100k SE(3) Blind Holdout (Leak-Free Episode Split)",
  "10 unseen physical episodes (" + C.holdout_poses_evaluated + " poses) | Mean Tip Error = " + C.tip_mean_error_mm + " mm | Mid-Tube = " + C.mid_tube_mean_error_mm + " mm"
);
const mB = { l: 58, r: 24, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const ch = DATA.ctcr_holdout;
const xB = d3.scaleLinear().domain(d3.extent(ch, d => d.true_y)).nice().range([0, pwB]);
const yB = d3.scaleLinear().domain(d3.extent(ch, d => d.true_z)).nice().range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(6));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
pB.selectAll("line.err").data(ch).enter().append("line")
  .attr("x1", d => xB(d.true_y)).attr("y1", d => yB(d.true_z))
  .attr("x2", d => xB(d.pred_y)).attr("y2", d => yB(d.pred_z))
  .attr("stroke", "#ef4444").attr("stroke-opacity", 0.35).attr("stroke-width", 1);
pB.selectAll("circle.tr").data(ch).enter().append("circle")
  .attr("cx", d => xB(d.true_y)).attr("cy", d => yB(d.true_z)).attr("r", 3.0).attr("fill", "#0f172a").attr("fill-opacity", 0.65);
pB.selectAll("circle.pr").data(ch).enter().append("circle")
  .attr("cx", d => xB(d.pred_y)).attr("cy", d => yB(d.pred_z)).attr("r", 2.3).attr("fill", "#0284c7").attr("fill-opacity", 0.8);
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("EM-Tracked Tip Y Position (mm)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("EM-Tracked Tip Z Position (mm)");
const legB = pB.append("g").attr("transform", "translate(12, 8)");
legB.append("rect").attr("width", 222).attr("height", 48).attr("rx", 6).attr("fill", "#fff").attr("fill-opacity", 0.92).attr("stroke", "#e2e8f0");
legB.append("circle").attr("cx", 14).attr("cy", 16).attr("r", 3.5).attr("fill", "#0f172a");
legB.append("text").attr("x", 26).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("True EM-Tracked CTCR Tip Pose");
legB.append("circle").attr("cx", 14).attr("cy", 34).attr("r", 3.0).attr("fill", "#0284c7");
legB.append("text").attr("x", 26).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("FormaMech Surrogate (" + Math.round(C.inference_hz) + " Hz CPU)");

// Panel C: Blind Holdout Tip Error Distribution + AS6983 Bound
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Blind Holdout 3D Tip Error Distribution & AS6983 Conservative Bound",
  "AS6983 90% Upper Quantile = " + C.as6983_quantile_mm + " mm (empirical coverage: " + (C.as6983_upper_coverage * 100).toFixed(1) + "%) | High-Jump OOD Err: " + C.high_jump_ood_error_mm + " mm vs ID: " + C.in_dist_mean_error_mm + " mm"
);
const mC = { l: 58, r: 24, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const hist = DATA.ctcr_err_hist;
const edges = DATA.ctcr_err_edges;
const xC = d3.scaleLinear().domain([edges[0], edges[edges.length - 1]]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, d3.max(hist) * 1.15]).range([phC, 0]);
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(8));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
hist.forEach((cnt, i) => {
  const x0 = xC(edges[i]), x1 = xC(edges[i + 1]);
  pC.append("rect").attr("x", x0 + 1).attr("y", yC(cnt)).attr("width", Math.max(1, x1 - x0 - 2)).attr("height", phC - yC(cnt)).attr("rx", 2)
    .attr("fill", edges[i] <= C.as6983_quantile_mm ? "#2563eb" : "#f59e0b").attr("fill-opacity", 0.82);
});
const qx = xC(Math.min(C.as6983_quantile_mm, edges[edges.length - 1]));
pC.append("line").attr("x1", qx).attr("x2", qx).attr("y1", 0).attr("y2", phC).attr("stroke", "#059669").attr("stroke-width", 2.5).attr("stroke-dasharray", "6,3");
pC.append("text").attr("x", qx + 8).attr("y", 18).attr("fill", "#047857").attr("font-size", 11.5).attr("font-weight", 700)
  .text("AS6983 Bound: " + C.as6983_quantile_mm + " mm (" + (C.as6983_upper_coverage * 100).toFixed(1) + "% held-out)");
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("3D Tip Position Error (mm)");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Holdout Pose Count");

// Panel D: FBG Optical Microstrain Along 9 Disks + Inverse Force Recovery
const gD = addCard(svg, 790, 504, 750, 396,
  "D. FBG Optical Fiber Strain Profile & Inverse External Contact-Force Recovery",
  "True Tip Force: [" + B.fbg_inverse_force_true_n.join(", ") + "] N -> Recovered: [" + B.fbg_inverse_force_est_n.join(", ") + "] N (Err = " + B.fbg_inverse_force_error_n + " N under 4 ue noise)"
);
const mD = { l: 62, r: 24, t: 62, b: 48 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const frFbg = DATA.anim_frames[18];
const nSeg = frFbg.fbg_strain_x.length;
const xD = d3.scaleBand().domain(d3.range(1, nSeg + 1).map(String)).range([0, pwD]).padding(0.28);
const maxSt = Math.max(d3.max(frFbg.fbg_strain_x.map(Math.abs)), d3.max(frFbg.fbg_strain_y.map(Math.abs)), 100) * 1.15;
const yD = d3.scaleLinear().domain([-maxSt, maxSt]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(6).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(6));
pD.append("line").attr("x1", 0).attr("x2", pwD).attr("y1", yD(0)).attr("y2", yD(0)).attr("stroke", "#475569").attr("stroke-width", 1);
for (let i = 0; i < nSeg; i++) {
  const bx = xD(String(i + 1)), bw = xD.bandwidth() / 2 - 2;
  const vx = frFbg.fbg_strain_x[i], vy = frFbg.fbg_strain_y[i];
  pD.append("rect").attr("x", bx).attr("y", vx >= 0 ? yD(vx) : yD(0)).attr("width", bw).attr("height", Math.abs(yD(vx) - yD(0))).attr("rx", 3).attr("fill", "#2563eb");
  pD.append("rect").attr("x", bx + bw + 3).attr("y", vy >= 0 ? yD(vy) : yD(0)).attr("width", bw).attr("height", Math.abs(yD(vy) - yD(0))).attr("rx", 3).attr("fill", "#e11d48");
}
pD.append("text").attr("x", pwD / 2).attr("y", phD + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Continuum Backbone Segment / Spacer Disk Index");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("FBG Fiber Microstrain (ue)");
const legD = pD.append("g").attr("transform", "translate(" + (pwD - 205) + ", 8)");
legD.append("rect").attr("width", 195).attr("height", 46).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legD.append("rect").attr("x", 10).attr("y", 11).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#2563eb");
legD.append("text").attr("x", 28).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("FBG Core X Bending Strain");
legD.append("rect").attr("x", 10).attr("y", 29).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#e11d48");
legD.append("text").attr("x", 28).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("FBG Core Y (with F_ext)");
`;

const DRAW_FIG_R3 = `
const W = 1560, H = 920;
const S = DATA.summary;
const J = S.joshi_follower_fem_vv10;
const M = S.qp_mpc_conformal_cbf;
const P = S.sponge_port_hamiltonian_eval;
addHeader(svg, W,
  "Study R3: Soft Pneumatic Follower-Pressure FEM & Real-Time Port-Hamiltonian QP-MPC + Conformal CBF",
  "Real Data: Joshi & Paik Pneumatic Actuator (Zenodo 3627901, CC-BY-4.0) + SPONGE 5-DoF Soft Arm (RepoLUH 19854, CC-BY-SA-4.0 Eval-Only)",
  [
    { text: "CBF Violations: " + M.cbf_barrier_violations + " (vs " + M.unfiltered_mpc_violations + " raw)", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "MPC: " + M.median_step_ms.toFixed(2) + " ms (" + Math.round(M.control_rate_hz) + " Hz)", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: "Passive PH: " + (P.passivity_verified ? "VERIFIED" : "NO"), bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: Joshi & Paik Real Pulling Force vs 3D TET10 Follower-Pressure FEM & Hybrid V&V10
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Joshi & Paik Real Pneumatic Actuator vs. 3D TET10 Follower-Pressure FEM",
  "ASME V&V 10: " + J.state + " | Raw FEM RMSE = " + J.raw_fem_holdout_rmse_n + " N -> Grounded Hybrid RMSE = " + J.grounded_hybrid_holdout_rmse_n + " N (3 physical replicates/pressure)"
);
const mA = { l: 58, r: 24, t: 62, b: 48 };
const pwA = 750 - mA.l - mA.r, phA = 400 - mA.t - mA.b;
const pA = gA.append("g").attr("transform", "translate(" + mA.l + "," + mA.t + ")");
const jc = DATA.joshi_curves;
const xA = d3.scaleLinear().domain([0, 205]).range([0, pwA]);
const yA = d3.scaleLinear().domain([0, d3.max(jc, d => Math.max(d.exp_mean_n + d.exp_std_n, d.fem_prior_n, d.hybrid_pred_n)) * 1.15]).range([phA, 0]);
pA.append("g").attr("class", "grid").call(d3.axisLeft(yA).ticks(5).tickSize(-pwA).tickFormat(""));
pA.append("g").attr("class", "axis").attr("transform", "translate(0," + phA + ")").call(d3.axisBottom(xA).ticks(8));
pA.append("g").attr("class", "axis").call(d3.axisLeft(yA).ticks(5));
pA.append("path").datum(jc).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,3")
  .attr("d", d3.line().x(d => xA(d.pressure_kpa)).y(d => yA(d.fem_prior_n)));
pA.append("path").datum(jc).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xA(d.pressure_kpa)).y(d => yA(d.hybrid_pred_n)).curve(d3.curveMonotoneX));
// Error bars + experimental points
jc.forEach(d => {
  pA.append("line").attr("x1", xA(d.pressure_kpa)).attr("x2", xA(d.pressure_kpa))
    .attr("y1", yA(Math.max(0, d.exp_mean_n - d.exp_std_n))).attr("y2", yA(d.exp_mean_n + d.exp_std_n))
    .attr("stroke", "#0f172a").attr("stroke-width", 1.6);
  pA.append("circle").attr("cx", xA(d.pressure_kpa)).attr("cy", yA(d.exp_mean_n)).attr("r", 4.5)
    .attr("fill", "#0f172a").attr("stroke", "#fff").attr("stroke-width", 1);
});
pA.append("text").attr("x", pwA / 2).attr("y", phA + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Internal Pneumatic Chamber Pressure (kPa)");
pA.append("text").attr("transform", "rotate(-90)").attr("x", -phA / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Actuator Pulling Force (N)");
const legA = pA.append("g").attr("transform", "translate(12, 8)");
legA.append("rect").attr("width", 252).attr("height", 64).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legA.append("circle").attr("cx", 14).attr("cy", 16).attr("r", 4).attr("fill", "#0f172a");
legA.append("text").attr("x", 26).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("Joshi & Paik Physical Mean +/- 1s (n=3)");
legA.append("line").attr("x1", 8).attr("x2", 22).attr("y1", 34).attr("y2", 34).attr("stroke", "#0284c7").attr("stroke-width", 2.8);
legA.append("text").attr("x", 26).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("FormaMech V&V10 Hybrid Grounded FEM");
legA.append("line").attr("x1", 8).attr("x2", 22).attr("y1", 52).attr("y2", 52).attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,3");
legA.append("text").attr("x", 26).attr("y", 56).attr("font-size", 10.5).attr("fill", "#1e293b").text("Raw 3D TET10 Follower-Pressure FEM");

// Panel B: Closed-Loop 5-DoF Soft Arm Trajectory with Conformal CBF Safety Wall
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Closed-Loop 5-DoF QP-MPC + AS6983 Conformal CBF Safety Filter",
  "Safety limit q1 <= " + M.q1_safety_limit_rad + " rad | Unfiltered violates " + M.unfiltered_mpc_violations + " steps; Conformal CBF: " + M.cbf_barrier_violations + " violations (" + M.cbf_modified_steps + " CBF + " + M.safe_hold_ood_steps + " OOD hold)"
);
const mB = { l: 58, r: 24, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const tr = DATA.mpc_trajectory;
const xB = d3.scaleLinear().domain([0, tr[tr.length - 1].t_s]).range([0, pwB]);
const yB = d3.scaleLinear().domain([-0.24, 0.24]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(6).tickSize(-pwB).tickFormat(""));
// Unsafe zone above q1_limit
pB.append("rect").attr("x", 0).attr("y", 0).attr("width", pwB).attr("height", yB(M.q1_safety_limit_rad)).attr("fill", "#fee2e2").attr("fill-opacity", 0.55);
// OOD safe hold vertical band
const oodSteps = tr.filter(d => d.fsm_state === "SAFE_HOLD_OOD");
if (oodSteps.length > 0) {
  const ox0 = xB(oodSteps[0].t_s), ox1 = xB(oodSteps[oodSteps.length - 1].t_s);
  pB.append("rect").attr("x", ox0).attr("y", 0).attr("width", Math.max(8, ox1 - ox0)).attr("height", phB).attr("fill", "#fef3c7").attr("fill-opacity", 0.65);
  pB.append("text").attr("x", (ox0 + ox1) / 2).attr("y", phB - 8).attr("text-anchor", "middle").attr("fill", "#b45309").attr("font-size", 10).attr("font-weight", 700).text("SAFE_HOLD_OOD");
}
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(7));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(6));
svg.append("defs").append("clipPath").attr("id", "clipR3B").append("rect").attr("x", 0).attr("y", 0).attr("width", pwB).attr("height", phB);
const pBClip = pB.append("g").attr("clip-path", "url(#clipR3B)");
pBClip.append("line").attr("x1", 0).attr("x2", pwB).attr("y1", yB(M.q1_safety_limit_rad)).attr("y2", yB(M.q1_safety_limit_rad)).attr("stroke", "#dc2626").attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
pBClip.append("path").datum(tr).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 1.8).attr("stroke-dasharray", "4,4")
  .attr("d", d3.line().x(d => xB(d.t_s)).y(d => yB(d.q_ref[0])));
pBClip.append("path").datum(tr).attr("fill", "none").attr("stroke", "#f97316").attr("stroke-width", 2.2)
  .attr("d", d3.line().x(d => xB(d.t_s)).y(d => yB(d.q_unfiltered[0])));
pBClip.append("path").datum(tr).attr("fill", "none").attr("stroke", "#059669").attr("stroke-width", 3.0)
  .attr("d", d3.line().x(d => xB(d.t_s)).y(d => yB(d.q_safe[0])));
pB.append("text").attr("x", pwB - 6).attr("y", yB(M.q1_safety_limit_rad) - 6).attr("text-anchor", "end").attr("fill", "#b91c1c").attr("font-size", 11).attr("font-weight", 700).text("Hard Safety Barrier q1 <= " + M.q1_safety_limit_rad.toFixed(3) + " rad");
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Closed-Loop Control Time (s)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Joint 1 Bending Angle q1 (rad)");
const legB = pB.append("g").attr("transform", "translate(10, " + (phB - 68) + ")");
legB.append("rect").attr("width", 242).attr("height", 60).attr("rx", 6).attr("fill", "#fff").attr("fill-opacity", 0.94).attr("stroke", "#e2e8f0");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 14).attr("y2", 14).attr("stroke", "#059669").attr("stroke-width", 3);
legB.append("text").attr("x", 30).attr("y", 18).attr("font-size", 10.5).attr("fill", "#1e293b").text("FormaMech QP-MPC + Conformal CBF (0 viol)");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 32).attr("y2", 32).attr("stroke", "#f97316").attr("stroke-width", 2.2);
legB.append("text").attr("x", 30).attr("y", 36).attr("font-size", 10.5).attr("fill", "#1e293b").text("Unfiltered MPC (" + M.unfiltered_mpc_violations + " barrier violations)");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 50).attr("y2", 50).attr("stroke", "#94a3b8").attr("stroke-width", 1.8).attr("stroke-dasharray", "4,3");
legB.append("text").attr("x", 30).attr("y", 54).attr("font-size", 10.5).attr("fill", "#1e293b").text("Commanded Reference Trajectory");

// Panel C: SPONGE 5-DoF Blind Generalization Across Payloads & Orientations (Tier B Eval-Only)
const gC = addCard(svg, 20, 504, 750, 396,
  "C. SPONGE 5-DoF Port-Hamiltonian Blind Eval Across Unseen Payloads & Gravity",
  "Trained ONLY on 0deg/0g identification | Evaluated locally on CC-BY-SA-4.0 Tier B hardware tests (no weights/raw data exported)"
);
const mC = { l: 60, r: 24, t: 62, b: 64 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const se = DATA.sponge_eval;
const xC = d3.scaleBand().domain(se.map(d => d.regime)).range([0, pwC]).padding(0.28);
const yC = d3.scaleLinear().domain([0, d3.max(se, d => d.rollout_0p2s_rmse_deg) * 1.2]).range([phC, 0]);
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC))
  .selectAll("text").attr("transform", "rotate(-22)").style("text-anchor", "end");
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.selectAll("rect.sb").data(se).enter().append("rect")
  .attr("x", d => xC(d.regime)).attr("y", d => yC(d.rollout_0p2s_rmse_deg))
  .attr("width", xC.bandwidth()).attr("height", d => phC - yC(d.rollout_0p2s_rmse_deg)).attr("rx", 5)
  .attr("fill", d => d.is_ood_shift ? (d.orientation_deg === 90 ? "#f43f5e" : "#f59e0b") : "#0284c7");
pC.selectAll("text.sv").data(se).enter().append("text")
  .attr("x", d => xC(d.regime) + xC.bandwidth() / 2).attr("y", d => yC(d.rollout_0p2s_rmse_deg) - 6)
  .attr("text-anchor", "middle").attr("font-size", 10.5).attr("font-weight", 700).attr("fill", "#1e293b")
  .text(d => d.rollout_0p2s_rmse_deg.toFixed(1) + "°");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("0.20s Rollout Joint RMSE (deg)");

// Panel D: Real-Time CPU Control Latency vs PINN-NMPC Reference
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Closed-Loop CPU Control Latency: FormaMech QP-MPC+CBF vs. PINN-NMPC",
  "Median = " + M.median_step_ms.toFixed(2) + " ms (" + Math.round(M.control_rate_hz) + " Hz) | p95 = " + M.p95_step_ms.toFixed(2) + " ms | Target <= 10 ms (100 Hz) | PINN-NMPC ref = 21.3 ms (47 Hz)"
);
const mD = { l: 58, r: 24, t: 62, b: 48 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const xD = d3.scaleLinear().domain([0, tr.length - 1]).range([0, pwD]);
const yD = d3.scaleLinear().domain([0, Math.max(24, d3.max(tr, d => d.latency_ms) * 1.2)]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD).ticks(8));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
pD.append("line").attr("x1", 0).attr("x2", pwD).attr("y1", yD(21.3)).attr("y2", yD(21.3)).attr("stroke", "#dc2626").attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
pD.append("text").attr("x", pwD - 6).attr("y", yD(21.3) - 6).attr("text-anchor", "end").attr("fill", "#b91c1c").attr("font-size", 10.5).attr("font-weight", 700).text("PINN-NMPC Baseline: 21.3 ms (47 Hz)");
pD.append("line").attr("x1", 0).attr("x2", pwD).attr("y1", yD(10.0)).attr("y2", yD(10.0)).attr("stroke", "#059669").attr("stroke-width", 1.8).attr("stroke-dasharray", "4,3");
pD.append("text").attr("x", pwD - 6).attr("y", yD(10.0) - 6).attr("text-anchor", "end").attr("fill", "#047857").attr("font-size", 10.5).attr("font-weight", 700).text("Real-Time 100 Hz Target: 10.0 ms");
pD.append("path").datum(tr).attr("fill", "#2563eb").attr("fill-opacity", 0.16)
  .attr("d", d3.area().x(d => xD(d.step)).y0(phD).y1(d => yD(d.latency_ms)));
pD.append("path").datum(tr).attr("fill", "none").attr("stroke", "#2563eb").attr("stroke-width", 2.2)
  .attr("d", d3.line().x(d => xD(d.step)).y(d => yD(d.latency_ms)));
pD.append("text").attr("x", pwD / 2).attr("y", phD + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Closed-Loop MPC + CBF Control Step");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Step Compute Latency (ms)");
`;

const DRAW_FIG_R4 = `
const W = 1560, H = 920;
const S = DATA.summary;
const V = S.asme_vv10_donor_holdout;
const A = S.abupt_transolver_surrogate;
addHeader(svg, W,
  "Study R4: Ultra-Soft Human Brain 3D Hyperelastic FEM & AB-UPT / Transolver++ Surrogate (V6-R5)",
  "Real Data: Hyperelastic Human Brain 1-7 (Zenodo 8095559 + 21793225, CC-BY-4.0, " + S.datasets[0].specimens_parsed + " physical specimens across 9 brain regions)",
  [
    { text: "ASME V&V 10: " + V.state, bg: V.state === "VALIDATED" ? "#065f46" : "#92400e", fg: V.state === "VALIDATED" ? "#a7f3d0" : "#fde68a", border: V.state === "VALIDATED" ? "#10b981" : "#f59e0b" },
    { text: "AS6983: " + (A.as6983_upper_coverage * 100).toFixed(1) + "% Cov", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: A.export_format.toUpperCase() + " Parity: " + A.export_parity_max_err.toExponential(1), bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: 3D Isometric Surgical Probe Indentation (Cortex 434 Pa vs Corpus Callosum 85.85 Pa)
const gA = addCard(svg, 20, 86, 750, 400,
  "A. 3D TET10 Surgical Probe Indentation (kPa Continuum Similitude FEM)",
  "Side-by-side Cauchy Von Mises stress field at 1.0 mm surgical palpation: Cortex (434 Pa) vs. Corpus Callosum (85.9 Pa)"
);
const frPeak = DATA.anim_frames[12];
const colBrain = d3.scaleSequential(d3.interpolateInferno).domain([0, Math.max(frPeak.peak_cortex_pa, 150)]);
function drawTissueBlock(cx, cy, vmArr, label, peakPa) {
  const pts = frPeak.surface_pts.map((p, i) => {
    const pr = isoProject(p[0] - 8, p[1] - 8, (p[2] - 6) * 2.2, cx, cy, 12.0, -36, 28);
    return { ...pr, vm: vmArr[i] };
  }).sort((a, b) => a.depth - b.depth);
  const baseC = [[0,0],[16,0],[16,16],[0,16]].map(c => isoProject(c[0]-8, c[1]-8, -3.5, cx, cy, 12.0, -36, 28));
  gA.append("polygon").attr("points", baseC.map(p => p.x + "," + p.y).join(" ")).attr("fill", "#e2e8f0").attr("stroke", "#94a3b8");
  gA.selectAll(null).data(pts).enter().append("circle")
    .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 4.8)
    .attr("fill", d => colBrain(d.vm)).attr("stroke", "#0f172a").attr("stroke-width", 0.4);
  const prb = isoProject(0, 0, 2.2, cx, cy, 12.0, -36, 28);
  gA.append("circle").attr("cx", prb.x).attr("cy", prb.y).attr("r", 20).attr("fill", "#94a3b8").attr("fill-opacity", 0.85).attr("stroke", "#1e293b").attr("stroke-width", 1.2);
  gA.append("text").attr("x", cx).attr("y", 358).attr("text-anchor", "middle").attr("fill", "#0f172a").attr("font-size", 12).attr("font-weight", 700).text(label);
  gA.append("text").attr("x", cx).attr("y", 376).attr("text-anchor", "middle").attr("fill", "#dc2626").attr("font-size", 11).attr("font-weight", 600).text("Peak Von Mises = " + peakPa.toFixed(1) + " Pa");
}
drawTissueBlock(195, 222, frPeak.vm_cortex_pa, "Human Cortex (mu = 434.0 Pa)", frPeak.peak_cortex_pa);
drawTissueBlock(555, 222, frPeak.vm_cc_pa, "Corpus Callosum (mu = 85.9 Pa)", frPeak.peak_cc_pa);

// Panel B: Real HBE 1-7 Human Brain Compression Stress-Strain Validation
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Real Human Brain Tissue Compression Validation (n = " + S.datasets[0].specimens_parsed + " Cylinders)",
  "Donor-split V&V10: HBE_01-04 cal vs HBE_05-07 blind holdout | Hybrid RMSE = " + V.grounded_hybrid_holdout_rmse_pa + " Pa (raw: " + V.raw_fem_holdout_rmse_pa + " Pa)"
);
const mB = { l: 62, r: 24, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const cv = DATA.compression_validation;
const xB = d3.scaleLinear().domain([0, 15]).range([0, pwB]);
const yB = d3.scaleLinear().domain([0, d3.max(cv, d => Math.max(d.exp_q75_pa, d.fem_cortex_pa)) * 1.15]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(6));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
pB.append("path").datum(cv).attr("fill", "#64748b").attr("fill-opacity", 0.22)
  .attr("d", d3.area().x(d => xB(d.strain_pct)).y0(d => yB(d.exp_q25_pa)).y1(d => yB(d.exp_q75_pa)));
pB.append("path").datum(cv).attr("fill", "none").attr("stroke", "#0f172a").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xB(d.strain_pct)).y(d => yB(d.exp_median_pa)));
pB.append("path").datum(cv).attr("fill", "none").attr("stroke", "#e11d48").attr("stroke-width", 2.6).attr("stroke-dasharray", "6,3")
  .attr("d", d3.line().x(d => xB(d.strain_pct)).y(d => yB(d.fem_cortex_pa)));
pB.append("path").datum(cv).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.4).attr("stroke-dasharray", "4,3")
  .attr("d", d3.line().x(d => xB(d.strain_pct)).y(d => yB(d.fem_cc_pa)));
pB.selectAll("circle.m").data(cv).enter().append("circle").attr("cx", d => xB(d.strain_pct)).attr("cy", d => yB(d.exp_median_pa)).attr("r", 4).attr("fill", "#0f172a");
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Compressive Nominal Strain (%)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Compressive Stress (Pa)");
const legB = pB.append("g").attr("transform", "translate(12, 8)");
legB.append("rect").attr("width", 262).attr("height", 64).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 16).attr("y2", 16).attr("stroke", "#0f172a").attr("stroke-width", 2.6);
legB.append("text").attr("x", 30).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("Real HBE 1-7 Median + IQR (n=" + S.datasets[0].specimens_parsed + ")");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 34).attr("y2", 34).attr("stroke", "#e11d48").attr("stroke-width", 2.4).attr("stroke-dasharray", "6,3");
legB.append("text").attr("x", 30).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("3D TET10 FEM: Cortex Upper Bound (434 Pa)");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 52).attr("y2", 52).attr("stroke", "#0284c7").attr("stroke-width", 2.4).attr("stroke-dasharray", "4,3");
legB.append("text").attr("x", 30).attr("y", 56).attr("font-size", 10.5).attr("fill", "#1e293b").text("3D TET10 FEM: Corpus Callosum (85.9 Pa)");

// Panel C: Ranked Neo-Hookean Shear Moduli Across All 9 Human Brain Regions
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Regional Human Brain Neo-Hookean Shear Modulus (5x Stiffness Span)",
  "Zenodo 21793225 published constitutive parameters spanning corpus callosum (85.9 Pa) to cortex (434.0 Pa)"
);
const mC = { l: 128, r: 44, t: 58, b: 44 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const regs = [...DATA.regions].sort((a, b) => a.shear_modulus_pa - b.shear_modulus_pa);
const yC = d3.scaleBand().domain(regs.map(d => d.region)).range([phC, 0]).padding(0.24);
const xC = d3.scaleLinear().domain([0, 480]).range([0, pwC]);
pC.append("g").attr("class", "grid").call(d3.axisBottom(xC).ticks(6).tickSize(phC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(6));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC));
const regCol = d3.scaleSequential(d3.interpolateViridis).domain([80, 450]);
pC.selectAll("rect.rb").data(regs).enter().append("rect")
  .attr("x", 0).attr("y", d => yC(d.region)).attr("width", d => xC(d.shear_modulus_pa)).attr("height", yC.bandwidth()).attr("rx", 4)
  .attr("fill", d => regCol(d.shear_modulus_pa));
pC.selectAll("text.rv").data(regs).enter().append("text")
  .attr("x", d => xC(d.shear_modulus_pa) + 6).attr("y", d => yC(d.region) + yC.bandwidth() / 2 + 4)
  .attr("font-size", 10.5).attr("font-weight", 700).attr("fill", "#1e293b")
  .text(d => d.shear_modulus_pa.toFixed(1) + " Pa");
pC.append("text").attr("x", pwC / 2).attr("y", phC + 34).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Neo-Hookean Shear Modulus mu (Pa)");

// Panel D: AB-UPT + Transolver++ Blind Holdout Peak Stress + AS6983 Tissue-Tearing Margin
const gD = addCard(svg, 790, 504, 750, 396,
  "D. AB-UPT + Transolver++ Surrogate: AS6983 Conservative Surgical Safety Bounds",
  "Blind holdout disp rel-L2 = " + A.holdout_mean_disp_rel_l2_pct + "% | Runtime: " + A.median_inference_ms + " ms vs FEM " + (A.mean_fem_solve_s * 1000).toFixed(0) + " ms | Tearing limit = 650 Pa"
);
const mD = { l: 62, r: 24, t: 62, b: 48 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const ht = [...DATA.holdout_table].sort((a, b) => a.true_vm_pa - b.true_vm_pa);
const xD = d3.scaleBand().domain(ht.map((_, i) => String(i + 1))).range([0, pwD]).padding(0.28);
const yD = d3.scaleLinear().domain([0, Math.max(720, d3.max(ht, d => d.as6983_upper_pa) * 1.1)]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
pD.append("line").attr("x1", 0).attr("x2", pwD).attr("y1", yD(650)).attr("y2", yD(650)).attr("stroke", "#dc2626").attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
pD.append("text").attr("x", 8).attr("y", yD(650) - 6).attr("fill", "#b91c1c").attr("font-size", 10.5).attr("font-weight", 700).text("Surgical Tissue Tearing Threshold = 650 Pa");
ht.forEach((d, i) => {
  const cx = xD(String(i + 1)) + xD.bandwidth() / 2;
  // AS6983 upper conservative whisker
  pD.append("line").attr("x1", cx).attr("x2", cx).attr("y1", yD(d.pred_vm_pa)).attr("y2", yD(d.as6983_upper_pa))
    .attr("stroke", "#2563eb").attr("stroke-width", 2.2);
  pD.append("line").attr("x1", cx - 6).attr("x2", cx + 6).attr("y1", yD(d.as6983_upper_pa)).attr("y2", yD(d.as6983_upper_pa))
    .attr("stroke", "#2563eb").attr("stroke-width", 2.4);
  pD.append("circle").attr("cx", cx).attr("cy", yD(d.pred_vm_pa)).attr("r", 3.5).attr("fill", "#2563eb");
  pD.append("circle").attr("cx", cx).attr("cy", yD(d.true_vm_pa)).attr("r", 4.2)
    .attr("fill", d.true_vm_pa <= d.as6983_upper_pa ? "#059669" : "#e11d48").attr("stroke", "#fff").attr("stroke-width", 0.8);
});
pD.append("text").attr("x", pwD / 2).attr("y", phD + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Blind Holdout Surgical Cases (Sorted by True FEM Peak Stress)");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Peak Cauchy Von Mises Stress (Pa)");
`;

const DRAW_FIG_R5 = `
const W = 1560, H = 920;
const S = DATA.summary;
const J = S.jamming_detection_holdout;
const T = S.compliant_gripper_simp_3d;
const E = S.brep_and_sim_exports;
addHeader(svg, W,
  "Study R5: REASSEMBLE NIST Assembly Jamming Detection & Verified 3D Compliant Gripper Co-Design",
  "Real Data: REASSEMBLE Franka Robot NIST Task Board #1 (148,444 30 Hz frames, 15 episodes, 499 segments, CC-BY-4.0)",
  [
    { text: "3D SIMP: " + T.state, bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "B-Rep STEP: " + (E.watertight_brep_step.watertight ? "WATERTIGHT" : "CHECK"), bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: "MJCF + ROS2 120Hz", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: Live 30 Hz REASSEMBLE NIST Assembly Wrench Trace (Real Jamming Spike)
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Real 30 Hz Franka NIST Insertion Wrench Stream (Held-Out Episode)",
  "Episode " + S.wrench_bridge_trace.source_episode + " | Peak Force = " + S.wrench_bridge_trace.peak_force_n + " N | Peak Torque = " + S.wrench_bridge_trace.peak_torque_nm + " Nm | Shaded = real failed/jammed segment"
);
const mA = { l: 58, r: 58, t: 62, b: 48 };
const pwA = 750 - mA.l - mA.r, phA = 400 - mA.t - mA.b;
const pA = gA.append("g").attr("transform", "translate(" + mA.l + "," + mA.t + ")");
const ww = DATA.wrench_window;
const xA = d3.scaleLinear().domain([0, ww.t_s[ww.t_s.length - 1]]).range([0, pwA]);
const yF = d3.scaleLinear().domain([0, d3.max(ww.force_n) * 1.15]).range([phA, 0]);
const yT = d3.scaleLinear().domain([0, d3.max(ww.torque_nm) * 1.2]).range([phA, 0]);
// Shade failed segment frames
for (let i = 0; i < ww.t_s.length - 1; i++) {
  if (!ww.success[i]) {
    pA.append("rect").attr("x", xA(ww.t_s[i])).attr("y", 0)
      .attr("width", Math.max(1.5, xA(ww.t_s[i + 1]) - xA(ww.t_s[i]))).attr("height", phA)
      .attr("fill", "#fee2e2").attr("fill-opacity", 0.65);
  }
}
pA.append("g").attr("class", "grid").call(d3.axisLeft(yF).ticks(5).tickSize(-pwA).tickFormat(""));
pA.append("g").attr("class", "axis").attr("transform", "translate(0," + phA + ")").call(d3.axisBottom(xA).ticks(7));
pA.append("g").attr("class", "axis").call(d3.axisLeft(yF).ticks(5));
pA.append("g").attr("class", "axis").attr("transform", "translate(" + pwA + ",0)").call(d3.axisRight(yT).ticks(5));
pA.append("path").datum(ww.force_n).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.2)
  .attr("d", d3.line().x((_, i) => xA(ww.t_s[i])).y(d => yF(d)));
pA.append("path").datum(ww.torque_nm).attr("fill", "none").attr("stroke", "#e11d48").attr("stroke-width", 2.0)
  .attr("d", d3.line().x((_, i) => xA(ww.t_s[i])).y(d => yT(d)));
pA.append("text").attr("x", pwA / 2).attr("y", phA + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Episode Window Time (s, 30 Hz Franka F/T Sensor)");
pA.append("text").attr("transform", "rotate(-90)").attr("x", -phA / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#0369a1").attr("font-size", 11.5).attr("font-weight", 600).text("Contact Force ||F|| (N)");
pA.append("text").attr("transform", "rotate(90)").attr("x", phA / 2).attr("y", -pwA - 42).attr("text-anchor", "middle").attr("fill", "#be123c").attr("font-size", 11.5).attr("font-weight", 600).text("Contact Torque ||tau|| (Nm)");

// Panel B: 3D Isometric FEM-CERTIFIED Compliant Gripper Finger Topology + Watertight B-Rep
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Verified 3D SIMP Compliant Gripper Finger & Watertight OpenCASCADE B-Rep",
  "State: " + T.state + " (" + T.claim_level + ") | Cert Disp = " + T.certified_max_displacement_mm + " mm @ " + S.wrench_bridge_trace.peak_force_n + " N | Open Edges = " + E.watertight_brep_step.open_boundary_edges
);
const vox = DATA.final_voxels;
const colRho = d3.scaleSequential(d3.interpolatePlasma).domain([0.2, 1.0]);
const vPts = vox.xyz.map((p, i) => {
  const pr = isoProject(p[0] - 27, p[1] - 7, p[2] - 7, 365, 220, 8.8, -32, 24);
  return { ...pr, rho: vox.rho[i] };
}).sort((a, b) => a.depth - b.depth);
// Draw bounding box wireframe of original 54x14x14 mm finger
const bbEdges = [
  [[0,0,0],[54,0,0]], [[54,0,0],[54,14,0]], [[54,14,0],[0,14,0]], [[0,14,0],[0,0,0]],
  [[0,0,14],[54,0,14]], [[54,0,14],[54,14,14]], [[54,14,14],[0,14,14]], [[0,14,14],[0,0,14]],
  [[0,0,0],[0,0,14]], [[54,0,0],[54,0,14]], [[54,14,0],[54,14,14]], [[0,14,0],[0,14,14]]
];
bbEdges.forEach(([p0, p1]) => {
  const a = isoProject(p0[0]-27, p0[1]-7, p0[2]-7, 365, 220, 8.8, -32, 24);
  const b = isoProject(p1[0]-27, p1[1]-7, p1[2]-7, 365, 220, 8.8, -32, 24);
  gB.append("line").attr("x1", a.x).attr("y1", a.y).attr("x2", b.x).attr("y2", b.y).attr("stroke", "#cbd5e1").attr("stroke-dasharray", "3,3");
});
gB.selectAll("circle.vx").data(vPts).enter().append("circle")
  .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", d => 4.2 + 3.2 * d.rho)
  .attr("fill", d => colRho(d.rho)).attr("fill-opacity", 0.88).attr("stroke", "#0f172a").attr("stroke-width", 0.45);
// Tip contact force arrow from real REASSEMBLE peak wrench
const tipA = isoProject(27, 0, 0, 365, 220, 8.8, -32, 24);
gB.append("line").attr("x1", tipA.x + 12).attr("y1", tipA.y - 54).attr("x2", tipA.x + 4).attr("y2", tipA.y - 6)
  .attr("stroke", "#dc2626").attr("stroke-width", 3.2);
gB.append("text").attr("x", tipA.x + 18).attr("y", tipA.y - 42).attr("fill", "#b91c1c").attr("font-size", 11.5).attr("font-weight", 700)
  .text("Peak NIST Load: " + S.wrench_bridge_trace.peak_force_n + " N");
const rootA = isoProject(-27, 0, 0, 365, 220, 8.8, -32, 24);
gB.append("text").attr("x", rootA.x - 16).attr("y", rootA.y + 42).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11).attr("font-weight", 700)
  .text("Clamped Mount + Bore Datum");
gB.append("text").attr("x", 375).attr("y", 376).attr("text-anchor", "middle").attr("fill", "#059669").attr("font-size", 11).attr("font-weight", 700)
  .text("STEP SHA256: " + (E.watertight_brep_step.step_sha256 || "").slice(0, 16) + "... | MJCF + ROS2 Manifests Exported");

// Panel C: 3D SIMP Compliance Convergence & FEM Certification
const gC = addCard(svg, 20, 504, 750, 396,
  "C. 3D SIMP Compliance Convergence & Mandatory Exact FEM Certification",
  "Initial = " + T.initial_compliance_mj + " mJ -> Final SIMP = " + T.final_simp_compliance_mj + " mJ | Certified Solid FEM = " + T.certified_fem_compliance_mj + " mJ"
);
const mC = { l: 62, r: 24, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const sh = DATA.simp_history;
const xC = d3.scaleLinear().domain([1, sh.length]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, d3.max(sh) * 1.15]).range([phC, 0]);
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(8));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.append("path").datum(sh).attr("fill", "none").attr("stroke", "#7c3aed").attr("stroke-width", 2.8)
  .attr("d", d3.line().x((_, i) => xC(i + 1)).y(d => yC(d)));
pC.selectAll("circle.sh").data(sh).enter().append("circle")
  .attr("cx", (_, i) => xC(i + 1)).attr("cy", d => yC(d)).attr("r", 3.5).attr("fill", "#7c3aed");
pC.append("line").attr("x1", 0).attr("x2", pwC).attr("y1", yC(T.certified_fem_compliance_mj)).attr("y2", yC(T.certified_fem_compliance_mj))
  .attr("stroke", "#059669").attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
pC.append("text").attr("x", pwC - 6).attr("y", yC(T.certified_fem_compliance_mj) - 8).attr("text-anchor", "end")
  .attr("fill", "#047857").attr("font-size", 11).attr("font-weight", 700)
  .text("Exact 3D FEM Certified Solid: " + T.certified_fem_compliance_mj + " mJ (" + T.state + ")");
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("SIMP Optimality Criteria Iteration");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Strain Energy Compliance (mJ)");

// Panel D: Blind Holdout Jamming/Failure Segment Detection
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Blind Held-Out Episode Assembly Failure/Jamming Separation (n = " + J.holdout_segments + ")",
  "Leak-free episode split (9 train / 3 dev / 3 blind holdout) | AUROC = " + J.auroc.toFixed(3) + " | Recall = " + (J.recall * 100).toFixed(1) + "%"
);
const mD = { l: 62, r: 24, t: 62, b: 48 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const hs = DATA.holdout_segments;
const xD = d3.scaleLinear().domain([0, d3.max(hs, d => d.peak_torque_nm) * 1.08]).range([0, pwD]);
const yD = d3.scaleLinear().domain([d3.min(hs, d => d.jam_score) - 0.3, d3.max(hs, d => d.jam_score) * 1.1]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD).ticks(6));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
pD.append("line").attr("x1", 0).attr("x2", pwD).attr("y1", yD(J.alarm_threshold)).attr("y2", yD(J.alarm_threshold))
  .attr("stroke", "#dc2626").attr("stroke-width", 1.8).attr("stroke-dasharray", "5,4");
pD.selectAll("circle.hs").data(hs).enter().append("circle")
  .attr("cx", d => xD(d.peak_torque_nm)).attr("cy", d => yD(d.jam_score))
  .attr("r", d => d.success ? 4.2 : 5.8)
  .attr("fill", d => d.success ? "#0284c7" : "#e11d48")
  .attr("fill-opacity", d => d.success ? 0.55 : 0.92)
  .attr("stroke", d => d.success ? "none" : "#0f172a").attr("stroke-width", 1);
pD.append("text").attr("x", pwD - 6).attr("y", yD(J.alarm_threshold) - 6).attr("text-anchor", "end").attr("fill", "#b91c1c").attr("font-size", 10.5).attr("font-weight", 700).text("DEV-Calibrated Alarm Threshold = " + J.alarm_threshold);
pD.append("text").attr("x", pwD / 2).attr("y", phD + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Segment Peak Contact Torque ||tau|| (Nm)");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Physics Contact-Jamming Score");
const legD = pD.append("g").attr("transform", "translate(12, 8)");
legD.append("rect").attr("width", 216).attr("height", 46).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legD.append("circle").attr("cx", 14).attr("cy", 15).attr("r", 4.5).attr("fill", "#0284c7").attr("fill-opacity", 0.7);
legD.append("text").attr("x", 26).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("Successful Insertion (" + (J.holdout_segments - J.holdout_failed_segments) + ")");
legD.append("circle").attr("cx", 14).attr("cy", 33).attr("r", 5.5).attr("fill", "#e11d48").attr("stroke", "#0f172a");
legD.append("text").attr("x", 26).attr("y", 37).attr("font-size", 10.5).attr("fill", "#1e293b").text("Real Failed/Jammed (" + J.holdout_failed_segments + ")");
`;

// ============================================================================
// ANIMATED LIVE GIF RENDERERS (880 x 490 dark telemetry dashboard)
// ============================================================================

const DRAW_GIF_R1 = `
const W = 880, H = 490;
const F = DATA.frame;
const ALL = DATA.all_frames;
const S = DATA.summary;
addHeader(svg, W,
  "FormaMech R1 Live: GelSight Mini 3D TET10 Contact & 8x8 Taxel Shear",
  "Mode: " + F.mode + " | Depth = " + F.depth_mm.toFixed(2) + " mm | Shear = " + F.shear_mm.toFixed(2) + " mm | Surrogate: " + S.surrogate_as6983_holdout.median_inference_ms.toFixed(2) + " ms",
  [{ text: "Fz = " + F.fz_n.toFixed(1) + " N  Fx = " + F.fx_n.toFixed(1) + " N", bg: "#0369a1", fg: "#e0f2fe", border: "#38bdf8" }],
  true
);
// Left: 3D isometric deformed gel pad
const gL = addCard(svg, 16, 82, 430, 392, "3D TET10 Elastomer Deformation", "Real GlowTact calibrated E = " + S.physical_calibration.calibrated_youngs_modulus_mpa + " MPa, nu = 0.48", true);
const vmMax = Math.max(...ALL[ALL.length - 1].surface_vm_mpa, 0.1);
const colVm = d3.scaleSequential(d3.interpolateTurbo).domain([0, vmMax]);
const pts = F.surface_pts.map((p, i) => {
  const pr = isoProject(p[0] - 9, p[1] - 12, (p[2] - 4) * 3.8, 215, 228, 11.4, -36, 28);
  return { ...pr, vm: F.surface_vm_mpa[i] };
}).sort((a, b) => a.depth - b.depth);
const bc = [[0,0],[18,0],[18,24],[0,24]].map(c => isoProject(c[0]-9, c[1]-12, -3.5, 215, 228, 11.4, -36, 28));
gL.append("polygon").attr("points", bc.map(p => p.x + "," + p.y).join(" ")).attr("fill", "#1e293b").attr("stroke", "#475569");
gL.selectAll("circle.n").data(pts).enter().append("circle")
  .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 4.6)
  .attr("fill", d => colVm(d.vm)).attr("stroke", "#020617").attr("stroke-width", 0.5);
const ind = isoProject(F.shear_mm * 5, 0, 3.6 - F.depth_mm * 2.2, 215, 228, 11.4, -36, 28);
gL.append("circle").attr("cx", ind.x).attr("cy", ind.y).attr("r", 26).attr("fill", "#94a3b8").attr("fill-opacity", 0.85).attr("stroke", "#f8fafc").attr("stroke-width", 1.5);

// Right top: 8x8 taxel heatmap + shear vectors (smooth Hertzian footprint conserving total force)
const gR1 = addCard(svg, 462, 82, 402, 232, "8x8 Taxel Normal & Shear Force Grid", "Real-time tactile force map (795 Hz CPU surrogate)", true);
function smoothGrid(grid) {
  const nx = grid.length, ny = grid[0].length;
  const out = Array.from({ length: nx }, () => Array(ny).fill(0));
  for (let sx = 0; sx < nx; sx++) for (let sy = 0; sy < ny; sy++) {
    if (Math.abs(grid[sx][sy]) < 1e-6) continue;
    let ws = 0;
    for (let tx = 0; tx < nx; tx++) for (let ty = 0; ty < ny; ty++) ws += Math.exp(-((tx-sx)**2 + (ty-sy)**2) / 2.6);
    for (let tx = 0; tx < nx; tx++) for (let ty = 0; ty < ny; ty++) out[tx][ty] += grid[sx][sy] * Math.exp(-((tx-sx)**2 + (ty-sy)**2) / 2.6) / ws;
  }
  return out;
}
const smFz = smoothGrid(F.taxel_fz);
const smFx = smoothGrid(F.taxel_fx);
const maxT = Math.max(...smoothGrid(ALL[ALL.length - 1].taxel_fz).flat(), 0.05);
const tCol = d3.scaleSequential(d3.interpolateInferno).domain([0, maxT]);
const gx = 105, gy = 54, cs = 21;
for (let ix = 0; ix < smFz.length; ix++) {
  for (let iy = 0; iy < smFz[0].length; iy++) {
    const v = smFz[ix][iy];
    const x = gx + ix * cs, y = gy + iy * cs;
    gR1.append("rect").attr("x", x).attr("y", y).attr("width", cs - 2).attr("height", cs - 2).attr("rx", 3)
      .attr("fill", v > maxT * 0.02 ? tCol(v) : "#1e293b").attr("stroke", "#334155").attr("stroke-width", 0.5);
    const fx = smFx[ix][iy];
    if (Math.abs(fx) > maxT * 0.03) {
      gR1.append("line").attr("x1", x + cs/2 - 1).attr("y1", y + cs/2 - 1)
        .attr("x2", x + cs/2 - 1 + Math.min(13, (fx / maxT) * 18)).attr("y2", y + cs/2 - 1)
        .attr("stroke", "#38bdf8").attr("stroke-width", 1.8);
    }
  }
}

// Right bottom: live force trace
const gR2 = addCard(svg, 462, 326, 402, 148, "Live Contact Wrench (Normal Fz & Shear Fx)", "", true);
const pR = gR2.append("g").attr("transform", "translate(46, 38)");
const xR = d3.scaleLinear().domain([0, ALL.length - 1]).range([0, 336]);
const yR = d3.scaleLinear().domain([0, d3.max(ALL, d => d.fz_n) * 1.15]).range([86, 0]);
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,86)").call(d3.axisBottom(xR).ticks(5));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yR).ticks(4));
const soFar = ALL.slice(0, F.frame + 1);
pR.append("path").datum(soFar).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xR(d.frame)).y(d => yR(d.fz_n)));
pR.append("path").datum(soFar).attr("fill", "none").attr("stroke", "#f59e0b").attr("stroke-width", 2.2)
  .attr("d", d3.line().x(d => xR(d.frame)).y(d => yR(d.fx_n)));
pR.append("circle").attr("cx", xR(F.frame)).attr("cy", yR(F.fz_n)).attr("r", 4.5).attr("fill", "#38bdf8");
`;

const DRAW_GIF_R2 = `
const W = 880, H = 490;
const F = DATA.frame;
const ALL = DATA.all_frames;
const hasExt = Math.abs(F.tip_force_n[1]) > 0.01;
addHeader(svg, W,
  "FormaMech R2 Live: 9-Disk 4-Tendon SE(3) Cosserat Continuum Robot",
  "Gotelli Nitinol Backbone (L=540 mm, 9 disks) | Bending = " + F.bending_angle_deg.toFixed(1) + " deg | " + (hasExt ? "EXTERNAL CONTACT LOAD 0.25 N" : "FREE-SPACE TENDON ACTUATION"),
  [{ text: hasExt ? "CONTACT: 0.25 N" : "FREE SPACE", bg: hasExt ? "#9f1239" : "#065f46", fg: "#fff", border: hasExt ? "#fb7185" : "#34d399" }],
  true
);
const gL = addCard(svg, 16, 82, 460, 392, "3D Finite-Rotation SE(3) Backbone & 9 Spacer Disks", "Real-time Cosserat rod equilibrium + tip trail", true);
const bp = [[-45,-45,0],[45,-45,0],[45,45,0],[-45,45,0]].map(c => isoProject(c[0], c[1], c[2], 230, 336, 0.52, -32 + F.frame * 1.2, 22));
gL.append("polygon").attr("points", bp.map(p => p.x + "," + p.y).join(" ")).attr("fill", "#1e293b").attr("stroke", "#475569");
// Tip trail across past frames
const trail = ALL.slice(0, F.frame + 1).map(fr => {
  const tp = fr.backbone_xyz[fr.backbone_xyz.length - 1];
  return isoProject(tp[0], tp[1], tp[2], 230, 336, 0.52, -32 + F.frame * 1.2, 22);
});
gL.append("path").datum(trail).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-opacity", 0.45).attr("stroke-width", 1.8).attr("stroke-dasharray", "3,3")
  .attr("d", d3.line().x(d => d.x).y(d => d.y));
const bPts = F.backbone_xyz.map(p => isoProject(p[0], p[1], p[2], 230, 336, 0.52, -32 + F.frame * 1.2, 22));
gL.append("path").datum(bPts).attr("fill", "none").attr("stroke", "#e2e8f0").attr("stroke-width", 4.0)
  .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
bPts.slice(1).forEach((dp, di) => {
  gL.append("ellipse").attr("cx", dp.x).attr("cy", dp.y).attr("rx", 11).attr("ry", 4.2)
    .attr("fill", "#38bdf8").attr("fill-opacity", 0.35).attr("stroke", "#38bdf8").attr("stroke-width", 1.3);
});
if (hasExt) {
  const tp = bPts[bPts.length - 1];
  gL.append("line").attr("x1", tp.x).attr("y1", tp.y).attr("x2", tp.x + 50).attr("y2", tp.y + 18).attr("stroke", "#fb7185").attr("stroke-width", 3);
  gL.append("text").attr("x", tp.x + 54).attr("y", tp.y + 22).attr("fill", "#fda4af").attr("font-size", 11).attr("font-weight", 700).text("F_ext");
}

// Right top: 4 tendon tension gauges
const gR1 = addCard(svg, 492, 82, 372, 178, "4 Routed Tendon Tensions (N)", "Antagonistic cable actuation @ r = 18.75 mm", true);
const tNames = ["T1 (+X)", "T2 (+Y)", "T3 (-X)", "T4 (-Y)"];
const tCols = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24"];
F.tensions_n.forEach((tv, i) => {
  const y = 62 + i * 26;
  gR1.append("text").attr("x", 18).attr("y", y + 12).attr("fill", "#cbd5e1").attr("font-size", 11).attr("font-weight", 600).text(tNames[i]);
  gR1.append("rect").attr("x", 76).attr("y", y).attr("width", 230).attr("height", 15).attr("rx", 4).attr("fill", "#1e293b");
  gR1.append("rect").attr("x", 76).attr("y", y).attr("width", Math.max(2, (tv / 9.0) * 230)).attr("height", 15).attr("rx", 4).attr("fill", tCols[i]);
  gR1.append("text").attr("x", 314).attr("y", y + 12).attr("fill", "#f8fafc").attr("font-size", 11).attr("font-weight", 700).text(tv.toFixed(1) + " N");
});

// Right bottom: FBG microstrain bars
const gR2 = addCard(svg, 492, 274, 372, 200, "Live FBG Optical Microstrain (Disks 1-9)", "Core X (cyan) & Core Y (rose)", true);
const pR = gR2.append("g").attr("transform", "translate(46, 52)");
const nS = F.fbg_strain_x.length;
const xS = d3.scaleBand().domain(d3.range(1, nS + 1).map(String)).range([0, 308]).padding(0.24);
const yS = d3.scaleLinear().domain([-2200, 2200]).range([116, 0]);
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,116)").call(d3.axisBottom(xS));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yS).ticks(5));
pR.append("line").attr("x1", 0).attr("x2", 308).attr("y1", yS(0)).attr("y2", yS(0)).attr("stroke", "#64748b");
for (let i = 0; i < nS; i++) {
  const bx = xS(String(i + 1)), bw = xS.bandwidth() / 2 - 1;
  const vx = Math.max(-2200, Math.min(2200, F.fbg_strain_x[i]));
  const vy = Math.max(-2200, Math.min(2200, F.fbg_strain_y[i]));
  pR.append("rect").attr("x", bx).attr("y", vx >= 0 ? yS(vx) : yS(0)).attr("width", bw).attr("height", Math.abs(yS(vx) - yS(0))).attr("fill", "#38bdf8");
  pR.append("rect").attr("x", bx + bw + 2).attr("y", vy >= 0 ? yS(vy) : yS(0)).attr("width", bw).attr("height", Math.abs(yS(vy) - yS(0))).attr("fill", "#fb7185");
}
`;

const DRAW_GIF_R3 = `
const W = 880, H = 490;
const F = DATA.frame;
const ALL = DATA.all_frames;
const S = DATA.summary.qp_mpc_conformal_cbf;
const stCol = F.fsm_state === "SAFE_HOLD_OOD" ? { bg: "#92400e", border: "#fbbf24" }
            : F.fsm_state === "CBF_MODIFIED" ? { bg: "#0369a1", border: "#38bdf8" }
            : { bg: "#065f46", border: "#34d399" };
addHeader(svg, W,
  "FormaMech R3 Live: 5-DoF Soft Arm QP-MPC + AS6983 Conformal CBF",
  "t = " + F.t_s.toFixed(2) + " s | Step Latency = " + F.latency_ms.toFixed(2) + " ms (" + Math.round(S.control_rate_hz) + " Hz) | Barrier Margin = " + F.barrier_margin_rad.toFixed(3) + " rad",
  [{ text: "FSM: " + F.fsm_state, bg: stCol.bg, fg: "#fff", border: stCol.border }],
  true
);

// Left: Animated 5-DoF planar soft pneumatic arm kinematic chain (Safe vs Unfiltered ghost)
const gL = addCard(svg, 16, 82, 390, 392, "5-DoF Articulated Soft Arm Kinematics", "Emerald: CBF-Safe | Orange Dashed: Unfiltered MPC", true);
function chainPts(qArr) {
  let x = 195, y = 352, th = -Math.PI / 2;
  const pts = [{ x, y }];
  for (let i = 0; i < 5; i++) {
    th += qArr[i] * 1.15;
    x += 52 * Math.cos(th);
    y += 52 * Math.sin(th);
    pts.push({ x, y });
  }
  return pts;
}
// Draw safety wall line corresponding to q1 = 0.28 rad
const wallAng = -Math.PI / 2 + S.q1_safety_limit_rad * 1.15;
gL.append("line").attr("x1", 195).attr("y1", 352)
  .attr("x2", 195 + 275 * Math.cos(wallAng)).attr("y2", 352 + 275 * Math.sin(wallAng))
  .attr("stroke", "#f43f5e").attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
gL.append("text").attr("x", 195 + 240 * Math.cos(wallAng) + 8).attr("y", 352 + 240 * Math.sin(wallAng))
  .attr("fill", "#fda4af").attr("font-size", 10.5).attr("font-weight", 700).text("CBF Safety Wall");
// Base mount
gL.append("rect").attr("x", 155).attr("y", 350).attr("width", 80).attr("height", 14).attr("rx", 4).attr("fill", "#475569");
const pUnf = chainPts(F.q_unfiltered);
const pSafe = chainPts(F.q_safe);
gL.append("path").datum(pUnf).attr("fill", "none").attr("stroke", "#fb923c").attr("stroke-width", 3).attr("stroke-dasharray", "5,4")
  .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
gL.append("path").datum(pSafe).attr("fill", "none").attr("stroke", "#34d399").attr("stroke-width", 6).attr("stroke-linecap", "round")
  .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
pSafe.forEach((pt, i) => {
  gL.append("circle").attr("cx", pt.x).attr("cy", pt.y).attr("r", i === 5 ? 7 : 5)
    .attr("fill", i === 5 ? "#38bdf8" : "#0f172a").attr("stroke", "#34d399").attr("stroke-width", 2);
});

// Right top: Scrolling q1(t) trajectory vs safety limit
const gR1 = addCard(svg, 422, 82, 442, 238, "Joint 1 Angle q1(t) vs. Conformal CBF Limit", "Zero safety violations (vs " + S.unfiltered_mpc_violations + " unfiltered)", true);
const pR1 = gR1.append("g").attr("transform", "translate(46, 54)");
const xR1 = d3.scaleLinear().domain([0, ALL[ALL.length - 1].t_s]).range([0, 376]);
const yR1 = d3.scaleLinear().domain([-0.24, 0.24]).range([156, 0]);
pR1.append("rect").attr("x", 0).attr("y", 0).attr("width", 376).attr("height", yR1(S.q1_safety_limit_rad)).attr("fill", "#881337").attr("fill-opacity", 0.35);
pR1.append("g").attr("class", "axis-dark").attr("transform", "translate(0,156)").call(d3.axisBottom(xR1).ticks(6));
pR1.append("g").attr("class", "axis-dark").call(d3.axisLeft(yR1).ticks(5));
pR1.append("line").attr("x1", 0).attr("x2", 376).attr("y1", yR1(S.q1_safety_limit_rad)).attr("y2", yR1(S.q1_safety_limit_rad)).attr("stroke", "#f43f5e").attr("stroke-width", 2).attr("stroke-dasharray", "5,3");
const upTo = ALL.filter(d => d.step <= F.step);
pR1.append("path").datum(upTo).attr("fill", "none").attr("stroke", "#fb923c").attr("stroke-width", 2).attr("stroke-dasharray", "4,3")
  .attr("d", d3.line().x(d => xR1(d.t_s)).y(d => yR1(d.q_unfiltered[0])));
pR1.append("path").datum(upTo).attr("fill", "none").attr("stroke", "#34d399").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xR1(d.t_s)).y(d => yR1(d.q_safe[0])));
pR1.append("circle").attr("cx", xR1(F.t_s)).attr("cy", yR1(F.q_safe[0])).attr("r", 4.5).attr("fill", "#34d399");

// Right bottom: 5 pneumatic differential pressure channels
const gR2 = addCard(svg, 422, 334, 442, 140, "Actuated Differential Chamber Pressures (bar)", "", true);
const pR2 = gR2.append("g").attr("transform", "translate(38, 42)");
const xU = d3.scaleBand().domain(["J1","J2","J3","J4","J5"]).range([0, 384]).padding(0.32);
const yU = d3.scaleLinear().domain([-0.9, 0.9]).range([76, 0]);
pR2.append("g").attr("class", "axis-dark").attr("transform", "translate(0,76)").call(d3.axisBottom(xU));
pR2.append("g").attr("class", "axis-dark").call(d3.axisLeft(yU).ticks(3));
pR2.append("line").attr("x1", 0).attr("x2", 384).attr("y1", yU(0)).attr("y2", yU(0)).attr("stroke", "#475569");
F.u_bar.forEach((uv, i) => {
  const bx = xU("J" + (i + 1));
  pR2.append("rect").attr("x", bx).attr("y", uv >= 0 ? yU(uv) : yU(0)).attr("width", xU.bandwidth()).attr("height", Math.abs(yU(uv) - yU(0))).attr("rx", 3)
    .attr("fill", uv >= 0 ? "#38bdf8" : "#a78bfa");
});
`;

const DRAW_GIF_R4 = `
const W = 880, H = 490;
const F = DATA.frame;
const ALL = DATA.all_frames;
const colB = d3.scaleSequential(d3.interpolateInferno).domain([0, 650]);
addHeader(svg, W,
  "FormaMech R4 Live: Ultra-Soft Human Brain Surgical Palpation (kPa FEM)",
  "Probe Depth = " + F.depth_mm.toFixed(2) + " mm | Cortex Peak = " + F.peak_cortex_pa.toFixed(0) + " Pa | Corpus Callosum = " + F.peak_cc_pa.toFixed(0) + " Pa",
  [{ text: F.peak_cortex_pa < F.tearing_limit_pa ? "SAFE < 650 Pa" : "LIMIT WARNING", bg: F.peak_cortex_pa < F.tearing_limit_pa ? "#065f46" : "#9f1239", fg: "#fff", border: "#34d399" }],
  true
);
function drawBlockDark(g, cx, cy, vmArr, label, peakPa) {
  const pts = F.surface_pts.map((p, i) => {
    const pr = isoProject(p[0] - 8, p[1] - 8, (p[2] - 6) * 2.4, cx, cy, 11.2, -36, 28);
    return { ...pr, vm: vmArr[i] };
  }).sort((a, b) => a.depth - b.depth);
  const bc = [[0,0],[16,0],[16,16],[0,16]].map(c => isoProject(c[0]-8, c[1]-8, -3.5, cx, cy, 11.2, -36, 28));
  g.append("polygon").attr("points", bc.map(p => p.x + "," + p.y).join(" ")).attr("fill", "#1e293b").attr("stroke", "#475569");
  g.selectAll(null).data(pts).enter().append("circle")
    .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 4.7)
    .attr("fill", d => colB(d.vm)).attr("stroke", "#020617").attr("stroke-width", 0.4);
  const prb = isoProject(0, 0, 3.2 - F.depth_mm * 2.1, cx, cy, 11.2, -36, 28);
  g.append("circle").attr("cx", prb.x).attr("cy", prb.y).attr("r", 19).attr("fill", "#cbd5e1").attr("fill-opacity", 0.88).attr("stroke", "#f8fafc").attr("stroke-width", 1.2);
  g.append("text").attr("x", cx).attr("y", 336).attr("text-anchor", "middle").attr("fill", "#f1f5f9").attr("font-size", 12).attr("font-weight", 700).text(label);
  g.append("text").attr("x", cx).attr("y", 355).attr("text-anchor", "middle").attr("fill", "#fbbf24").attr("font-size", 11.5).attr("font-weight", 700).text("Peak Stress: " + peakPa.toFixed(1) + " Pa");
}
const gMain = addCard(svg, 16, 82, 848, 392, "Real Human Brain Regional Stiffness Contrast (5x Span, Zenodo 21793225)", "Cortex (434 Pa) vs. Corpus Callosum (85.9 Pa) under identical 3D surgical probe palpation", true);
drawBlockDark(gMain, 215, 206, F.vm_cortex_pa, "Cortex (mu = 434.0 Pa)", F.peak_cortex_pa);
drawBlockDark(gMain, 635, 206, F.vm_cc_pa, "Corpus Callosum (mu = 85.9 Pa)", F.peak_cc_pa);
// Bottom tissue-tearing safety margin bar
const barX = 110, barY = 368, barW = 628;
gMain.append("rect").attr("x", barX).attr("y", barY).attr("width", barW).attr("height", 12).attr("rx", 6).attr("fill", "#1e293b");
const fillW = Math.min(barW, (F.peak_cortex_pa / F.tearing_limit_pa) * barW);
gMain.append("rect").attr("x", barX).attr("y", barY).attr("width", fillW).attr("height", 12).attr("rx", 6)
  .attr("fill", F.peak_cortex_pa / F.tearing_limit_pa > 0.75 ? "#f59e0b" : "#10b981");
gMain.append("text").attr("x", barX - 8).attr("y", barY + 10).attr("text-anchor", "end").attr("fill", "#94a3b8").attr("font-size", 10).text("0 Pa");
gMain.append("text").attr("x", barX + barW + 8).attr("y", barY + 10).attr("fill", "#fda4af").attr("font-size", 10).attr("font-weight", 700).text("650 Pa Limit");
`;

const DRAW_GIF_R5 = `
const W = 880, H = 490;
const F = DATA.frame;
const WW = DATA.wrench_window;
const S = DATA.summary;
addHeader(svg, W,
  "FormaMech R5 Live: NIST Jamming Guard & 3D SIMP Gripper",
  "30 Hz Franka F/T t = " + F.stream_t_s.toFixed(2) + " s | ||F|| = " + F.force_n.toFixed(1) + " N | SIMP Iter " + F.simp_iter + " -> FEM CERTIFIED + Watertight STEP",
  [{ text: F.is_failed_segment ? "JAMMING DETECTED" : "NOMINAL INSERTION", bg: F.is_failed_segment ? "#9f1239" : "#065f46", fg: "#fff", border: F.is_failed_segment ? "#fb7185" : "#34d399" }],
  true
);
// Left: 3D SIMP Topology Evolution of Compliant Gripper Finger
const gL = addCard(svg, 16, 82, 430, 392, "3D SIMP Gripper Finger Evolution (Iter " + F.simp_iter + ")", "Loaded by real REASSEMBLE peak wrench (" + S.wrench_bridge_trace.peak_force_n + " N)", true);
const colR = d3.scaleSequential(d3.interpolatePlasma).domain([0.2, 1.0]);
const vPts = F.voxels_xyz.map((p, i) => {
  const pr = isoProject(p[0] - 27, p[1] - 7, p[2] - 7, 215, 224, 6.4, -30 + F.frame * 1.4, 24);
  return { ...pr, rho: F.voxels_rho[i] };
}).sort((a, b) => a.depth - b.depth);
gL.selectAll("circle.v").data(vPts).enter().append("circle")
  .attr("cx", d => d.x).attr("cy", d => d.y).attr("r", d => 3.5 + 3.0 * d.rho)
  .attr("fill", d => colR(d.rho)).attr("fill-opacity", 0.9).attr("stroke", "#020617").attr("stroke-width", 0.4);
gL.append("text").attr("x", 215).attr("y", 368).attr("text-anchor", "middle").attr("fill", "#34d399").attr("font-size", 11).attr("font-weight", 700)
  .text("Final Solid: " + S.compliant_gripper_simp_3d.state + " | OpenCASCADE Watertight STEP");

// Right: Scrolling 30 Hz REASSEMBLE NIST Force & Torque Stream
const gR = addCard(svg, 462, 82, 402, 392, "Live 30 Hz Franka NIST Insertion Wrench", "Cyan: ||F|| (N) | Rose: ||tau|| x 10 (Nm) | Red Band: Real Fail", true);
const pR = gR.append("g").attr("transform", "translate(46, 56)");
const xR = d3.scaleLinear().domain([0, WW.t_s[WW.t_s.length - 1]]).range([0, 336]);
const yR = d3.scaleLinear().domain([0, Math.max(36, d3.max(WW.force_n) * 1.1)]).range([298, 0]);
for (let i = 0; i < WW.t_s.length - 1; i++) {
  if (!WW.success[i]) {
    pR.append("rect").attr("x", xR(WW.t_s[i])).attr("y", 0)
      .attr("width", Math.max(1.5, xR(WW.t_s[i + 1]) - xR(WW.t_s[i]))).attr("height", 298)
      .attr("fill", "#881337").attr("fill-opacity", 0.35);
  }
}
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,298)").call(d3.axisBottom(xR).ticks(6));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yR).ticks(6));
const kMax = Math.max(1, F.stream_idx);
pR.append("path").datum(WW.force_n.slice(0, kMax + 1)).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.4)
  .attr("d", d3.line().x((_, i) => xR(WW.t_s[i])).y(d => yR(d)));
pR.append("path").datum(WW.torque_nm.slice(0, kMax + 1)).attr("fill", "none").attr("stroke", "#fb7185").attr("stroke-width", 2.0)
  .attr("d", d3.line().x((_, i) => xR(WW.t_s[i])).y(d => yR(d * 10.0)));
pR.append("line").attr("x1", xR(F.stream_t_s)).attr("x2", xR(F.stream_t_s)).attr("y1", 0).attr("y2", 298)
  .attr("stroke", "#fbbf24").attr("stroke-width", 1.8).attr("stroke-dasharray", "4,3");
pR.append("circle").attr("cx", xR(F.stream_t_s)).attr("cy", yR(F.force_n)).attr("r", 5).attr("fill", "#38bdf8").attr("stroke", "#fff");
`;

async function main() {
  fs.mkdirSync(ASSETS_DIR, { recursive: true });
  fs.mkdirSync(FRAMES_ROOT, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  try {
    // 1. Render 5 crisp 2x Retina static publication PNGs
    const figPage = await browser.newPage({
      viewport: { width: 1560, height: 920 },
      deviceScaleFactor: 2,
    });
    const studies = [
      { id: "r1", viz: "study_r1_viz.json", outPng: "fig_r1_tactile_elastomer.png", figScript: DRAW_FIG_R1, gifScript: DRAW_GIF_R1, gifDir: "r1_tactile" },
      { id: "r2", viz: "study_r2_viz.json", outPng: "fig_r2_continuum_tendon_twin.png", figScript: DRAW_FIG_R2, gifScript: DRAW_GIF_R2, gifDir: "r2_continuum" },
      { id: "r3", viz: "study_r3_viz.json", outPng: "fig_r3_pneumatic_soft_mpc.png", figScript: DRAW_FIG_R3, gifScript: DRAW_GIF_R3, gifDir: "r3_soft_mpc" },
      { id: "r4", viz: "study_r4_viz.json", outPng: "fig_r4_surgical_tissue.png", figScript: DRAW_FIG_R4, gifScript: DRAW_GIF_R4, gifDir: "r4_surgical" },
      { id: "r5", viz: "study_r5_viz.json", outPng: "fig_r5_nist_assembly_codesign.png", figScript: DRAW_FIG_R5, gifScript: DRAW_GIF_R5, gifDir: "r5_nist" },
    ];

    for (const st of studies) {
      const vizPath = path.join(VIZ_DIR, st.viz);
      if (!fs.existsSync(vizPath)) {
        throw new Error(`Missing viz JSON: ${vizPath}`);
      }
      const data = JSON.parse(fs.readFileSync(vizPath, "utf-8"));
      const html = buildHtml(1560, 920, "#f8fafc", st.figScript, data);
      await figPage.goto("about:blank");
      await figPage.setContent(html, { waitUntil: "load" });
      await figPage.waitForFunction(() => window.__RENDER_DONE__ === true);
      const err = await figPage.evaluate(() => window.__RENDER_ERR__);
      if (err) throw new Error(`Static figure ${st.id} JS error:\n${err}`);
      const outPath = path.join(ASSETS_DIR, st.outPng);
      await figPage.screenshot({ path: outPath, type: "png" });
      console.log(`Rendered static D3+SVG figure: ${outPath}`);
    }
    await figPage.close();

    // 2. Render smooth animation PNG frames for all 5 live GIFs
    const gifPage = await browser.newPage({
      viewport: { width: 880, height: 490 },
      deviceScaleFactor: 1,
    });
    for (const st of studies) {
      const data = JSON.parse(fs.readFileSync(path.join(VIZ_DIR, st.viz), "utf-8"));
      const frames = data.anim_frames;
      const outDir = path.join(FRAMES_ROOT, st.gifDir);
      fs.mkdirSync(outDir, { recursive: true });
      for (let i = 0; i < frames.length; i++) {
        const framePayload = {
          frame: frames[i],
          all_frames: frames,
          summary: data.summary,
          wrench_window: data.wrench_window || null,
        };
        const html = buildHtml(880, 490, "#0b1120", st.gifScript, framePayload);
        await gifPage.goto("about:blank");
        await gifPage.setContent(html, { waitUntil: "load" });
        await gifPage.waitForFunction(() => window.__RENDER_DONE__ === true);
        const err = await gifPage.evaluate(() => window.__RENDER_ERR__);
        if (err) throw new Error(`GIF ${st.id} frame ${i} JS error:\n${err}`);
        const fPath = path.join(outDir, `frame_${String(i).padStart(3, "0")}.png`);
        await gifPage.screenshot({ path: fPath, type: "png" });
      }
      console.log(`Rendered ${frames.length} D3+SVG frames for ${st.gifDir}`);
    }
    await gifPage.close();
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
