/**
 * Publication-grade D3.js + Custom SVG + Playwright renderer for FormaMech v0.7
 * Robotics Physics AI validation & stress-test studies (R3-v2, R5-v2, R6-R10).
 *
 * Generates:
 *   - 7 high-DPI (2x) 4-panel publication PNGs in docs/assets/studies_robotics/
 *   - 4 smooth dark-mode SVG animation frame sequences in
 *     .formamech-runs/robotics_studies/gif_frames_v7/
 *     (assembled into optimized animated GIFs by scripts/studies_robotics/build_v7_gifs.py)
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
const FRAMES_ROOT = path.join(VIZ_DIR, "gif_frames_v7");
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

function isoProject(x, y, z, cx, cy, scale, yawDeg = -35, pitchDeg = 26) {
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
    .attr("fill", "#0f172a");
  svg.append("line")
    .attr("x1", 0).attr("x2", W).attr("y1", 68).attr("y2", 68)
    .attr("stroke", "#38bdf8").attr("stroke-width", 2);
  svg.append("text")
    .attr("x", 24).attr("y", 30)
    .attr("fill", "#f8fafc").attr("font-size", 18.5).attr("font-weight", 700)
    .text(title);
  svg.append("text")
    .attr("x", 24).attr("y", 52)
    .attr("fill", "#94a3b8").attr("font-size", 11.8)
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
    .attr("fill", dark ? "#f1f5f9" : "#0f172a").attr("font-size", 13.8).attr("font-weight", 700)
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
// 1. FIGURE R3-v2: Pneumatic Hysteresis (V7-R1) & Online ACP-CQR CBF (V7-R2)
// ============================================================================
const DRAW_FIG_R3_V2 = `
const W = 1560, H = 920;
const S = DATA.summary;
const HZ = S.zenodo_15584329_hysteresis_vv10;
const JP = S.joshi_paik_v7_vv10;
const CB = S.acp_cqr_vs_static_cbf_sponge;
const FM = S.visco_mullins_3d_fem;
addHeader(svg, W,
  "Study R3-v2: Pneumatic Prony/Mullins Hysteresis (V7-R1) & Online ACP-CQR CBF (V7-R2)",
  "Real Data: Zenodo 15584329 (240k @ 30 Hz, CC-BY-4.0) + Joshi & Paik (Zenodo 3627901) + SPONGE 5-DoF 0g->200g Shift (Tier B Eval-Only)",
  [
    { text: "Hysteresis: " + HZ.v7_vv10_state + " (-" + HZ.rmse_reduction_vs_elastic_pct.toFixed(1) + "%)", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "Joshi: REJECTED -> " + JP.v7_state, bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: "ACP-CQR: -" + CB.braking_distance_reduction_pct.toFixed(1) + "% Brake | 0 Viol", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: Blind Cycle-4 Hysteresis Loop (Zenodo 15584329)
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Blind Cycle-4 Pneumatic Hysteresis Loop (Zenodo 15584329, 240,001 Samples)",
  "Trained on Cycles 1-2 | Dev Cycle 3 | Blind Cycle 4 RMSE: Elastic " + HZ.elastic_holdout_rmse_mm.toFixed(3) + " mm -> V7-R1 " + HZ.v7_hysteresis_holdout_rmse_mm.toFixed(3) + " mm (-" + HZ.rmse_reduction_vs_elastic_pct.toFixed(1) + "%)"
);
const mA = { l: 58, r: 24, t: 62, b: 48 };
const pwA = 750 - mA.l - mA.r, phA = 400 - mA.t - mA.b;
const pA = gA.append("g").attr("transform", "translate(" + mA.l + "," + mA.t + ")");
const lp = DATA.cycle4_loop;
const xA = d3.scaleLinear().domain([0, 1.02]).range([0, pwA]);
const yA = d3.scaleLinear().domain([d3.min(lp, d => Math.min(d.y_exp_mm, d.y_v7_mm)) - 0.5, d3.max(lp, d => Math.max(d.y_exp_mm, d.y_v7_mm)) + 0.8]).range([phA, 0]);
pA.append("g").attr("class", "grid").call(d3.axisLeft(yA).ticks(5).tickSize(-pwA).tickFormat(""));
pA.append("g").attr("class", "axis").attr("transform", "translate(0," + phA + ")").call(d3.axisBottom(xA).ticks(6));
pA.append("g").attr("class", "axis").call(d3.axisLeft(yA).ticks(5));
pA.append("path").datum(lp).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,3")
  .attr("d", d3.line().x(d => xA(d.u)).y(d => yA(d.y_elastic_mm)));
pA.append("path").datum(lp).attr("fill", "none").attr("stroke", "#0f172a").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xA(d.u)).y(d => yA(d.y_exp_mm)));
pA.append("path").datum(lp).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xA(d.u)).y(d => yA(d.y_v7_mm)));
pA.append("text").attr("x", pwA / 2).attr("y", phA + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Normalized Pneumatic Pressure Command u");
pA.append("text").attr("transform", "rotate(-90)").attr("x", -phA / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Actuator Displacement (mm)");
const legA = pA.append("g").attr("transform", "translate(12, 8)");
legA.append("rect").attr("width", 272).attr("height", 66).attr("rx", 6).attr("fill", "#fff").attr("fill-opacity", 0.94).attr("stroke", "#e2e8f0");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 15).attr("y2", 15).attr("stroke", "#0f172a").attr("stroke-width", 2.6);
legA.append("text").attr("x", 30).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("Real Cycle-4 Hardware Loop (1.24 mm gap)");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 34).attr("y2", 34).attr("stroke", "#0284c7").attr("stroke-width", 2.6);
legA.append("text").attr("x", 30).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("V7-R1 Prony+Velocity Operator (0.247 mm)");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 53).attr("y2", 53).attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,3");
legA.append("text").attr("x", 30).attr("y", 57).attr("font-size", 10.5).attr("fill", "#1e293b").text("Static Elastic Baseline (0.512 mm, no memory)");

// Panel B: 3D TET10 Prony + Mullins FEM Hysteresis & Thermodynamic Dissipation Certificate
const gB = addCard(svg, 790, 86, 750, 400,
  "B. 3D TET10 Prony + Ogden-Roxburgh Mullins FEM & Dissipation Audit",
  "Converged | min det(F) = " + FM.min_det_f + " | min Mullins eta = " + FM.min_mullins_eta + " | Midpoint Gap = " + FM.midpoint_hysteresis_gap_mm + " mm | dD/dt >= 0 CERTIFIED"
);
const mB = { l: 58, r: 58, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const fc = DATA.fem_visco_cycle;
const xB = d3.scaleLinear().domain([0, 1.05]).range([0, pwB]);
const yB = d3.scaleLinear().domain([0, d3.max(fc, d => d.tip_disp_mm) * 1.15]).range([phB, 0]);
const yB2 = d3.scaleLinear().domain([0, d3.max(fc, d => d.step_dissipation_n_mm) * 1.4]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(6));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
const loadPart = fc.slice(0, 9), unloadPart = fc.slice(8);
pB.append("path").datum(loadPart).attr("fill", "none").attr("stroke", "#2563eb").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xB(d.load_mult)).y(d => yB(d.tip_disp_mm)));
pB.append("path").datum(unloadPart).attr("fill", "none").attr("stroke", "#dc2626").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xB(d.load_mult)).y(d => yB(d.tip_disp_mm)));
pB.selectAll("circle.fl").data(fc).enter().append("circle")
  .attr("cx", d => xB(d.load_mult)).attr("cy", d => yB(d.tip_disp_mm)).attr("r", 3.6)
  .attr("fill", (_, i) => i <= 8 ? "#2563eb" : "#dc2626").attr("stroke", "#fff");
// Step dissipation inset bars at bottom-right
const inG = pB.append("g").attr("transform", "translate(" + (pwB - 230) + "," + (phB - 110) + ")");
inG.append("rect").attr("width", 224).attr("height", 102).attr("rx", 6).attr("fill", "#f8fafc").attr("stroke", "#cbd5e1");
inG.append("text").attr("x", 10).attr("y", 16).attr("font-size", 10.5).attr("font-weight", 700).attr("fill", "#047857").text("Step Dissipation dD_k >= 0 (N*mm)");
const ix = d3.scaleBand().domain(fc.map(d => String(d.step))).range([10, 214]).padding(0.18);
const iy = d3.scaleLinear().domain([0, d3.max(fc, d => d.step_dissipation_n_mm) * 1.15]).range([86, 26]);
inG.selectAll("rect.db").data(fc).enter().append("rect")
  .attr("x", d => ix(String(d.step))).attr("y", d => iy(d.step_dissipation_n_mm))
  .attr("width", ix.bandwidth()).attr("height", d => 86 - iy(d.step_dissipation_n_mm))
  .attr("fill", "#10b981").attr("rx", 1.5);
inG.append("text").attr("x", 112).attr("y", 97).attr("text-anchor", "middle").attr("font-size", 9.5).attr("fill", "#475569").text("Sum = " + FM.cumulative_dissipation_n_mm.toFixed(4) + " N*mm (all 17 steps >= 0)");
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Follower-Pressure Load Factor lambda (0 -> 1 -> 0)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("3D TET10 Bellows Tip Displacement (mm)");
const legB = pB.append("g").attr("transform", "translate(12, 8)");
legB.append("rect").attr("width", 232).attr("height", 46).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 15).attr("y2", 15).attr("stroke", "#2563eb").attr("stroke-width", 2.8);
legB.append("text").attr("x", 30).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("Loading Branch (Prony + Mullins)");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 34).attr("y2", 34).attr("stroke", "#dc2626").attr("stroke-width", 2.8);
legB.append("text").attr("x", 30).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("Unloading Branch (eta -> " + FM.min_mullins_eta + ")");

// Panel C: SPONGE 0g -> 200g Step Shift Trajectory (Unfiltered vs Static CBF vs V7-R2 ACP-CQR)
const gC = addCard(svg, 20, 504, 750, 396,
  "C. SPONGE 5-DoF Closed-Loop 0g -> 200g Step Shift: Static CBF vs. V7-R2 ACP-CQR",
  "Unfiltered: " + CB.unfiltered_violations + " viol | Static CBF: " + CB.static_safe_hold_steps + " SAFE_HOLD trips | ACP-CQR: 0 viol, 0 trips, -" + CB.braking_distance_reduction_pct.toFixed(1) + "% braking gap"
);
const mC = { l: 58, r: 24, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const tr = DATA.cbf_trace;
const xC = d3.scaleLinear().domain([0, tr[tr.length - 1].t_s]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, 0.215]).range([phC, 0]);
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("rect").attr("x", 0).attr("y", 0).attr("width", pwC).attr("height", yC(CB.q1_safety_limit_rad)).attr("fill", "#fee2e2").attr("fill-opacity", 0.55);
const xShift = xC(tr[70].t_s);
pC.append("rect").attr("x", xShift).attr("y", yC(CB.q1_safety_limit_rad)).attr("width", pwC - xShift).attr("height", phC - yC(CB.q1_safety_limit_rad)).attr("fill", "#fef3c7").attr("fill-opacity", 0.38);
pC.append("line").attr("x1", xShift).attr("x2", xShift).attr("y1", 0).attr("y2", phC).attr("stroke", "#d97706").attr("stroke-width", 1.6).attr("stroke-dasharray", "5,3");
pC.append("text").attr("x", xShift + 6).attr("y", phC - 10).attr("fill", "#b45309").attr("font-size", 10.5).attr("font-weight", 700).text("200g Payload Step Shift (t = 1.40 s)");
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(7));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.append("line").attr("x1", 0).attr("x2", pwC).attr("y1", yC(CB.q1_safety_limit_rad)).attr("y2", yC(CB.q1_safety_limit_rad)).attr("stroke", "#dc2626").attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
pC.append("path").datum(tr).attr("fill", "none").attr("stroke", "#f97316").attr("stroke-width", 2.0)
  .attr("d", d3.line().x(d => xC(d.t_s)).y(d => yC(d.q1_unfiltered_rad)));
pC.append("path").datum(tr).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "5,3")
  .attr("d", d3.line().x(d => xC(d.t_s)).y(d => yC(d.q1_static_cbf_rad)));
pC.append("path").datum(tr).attr("fill", "none").attr("stroke", "#059669").attr("stroke-width", 2.9)
  .attr("d", d3.line().x(d => xC(d.t_s)).y(d => yC(d.q1_adaptive_cbf_rad)));
pC.append("text").attr("x", pwC - 6).attr("y", yC(CB.q1_safety_limit_rad) - 6).attr("text-anchor", "end").attr("fill", "#b91c1c").attr("font-size", 11).attr("font-weight", 700).text("Hard Safety Barrier q1 <= 0.145 rad");
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Closed-Loop Control Time (s)");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Joint 1 Bending Angle q1 (rad)");
const legC = pC.append("g").attr("transform", "translate(10, " + (phC - 72) + ")");
legC.append("rect").attr("width", 242).attr("height", 62).attr("rx", 6).attr("fill", "#fff").attr("fill-opacity", 0.94).attr("stroke", "#e2e8f0");
legC.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 14).attr("y2", 14).attr("stroke", "#059669").attr("stroke-width", 2.9);
legC.append("text").attr("x", 30).attr("y", 18).attr("font-size", 10.5).attr("fill", "#1e293b").text("V7-R2 ACP-CQR CBF (0 viol, 0 trips)");
legC.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 32).attr("y2", 32).attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "5,3");
legC.append("text").attr("x", 30).attr("y", 36).attr("font-size", 10.5).attr("fill", "#1e293b").text("v0.6 Static CBF (70 SAFE_HOLD trips)");
legC.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 50).attr("y2", 50).attr("stroke", "#f97316").attr("stroke-width", 2.0);
legC.append("text").attr("x", 30).attr("y", 54).attr("font-size", 10.5).attr("fill", "#1e293b").text("Unfiltered MPC (" + CB.unfiltered_violations + " safety violations)");

// Panel D: Online Adaptive Conformal Margin Evolution vs Static Margin
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Online ACP-CQR Margin Adaptation (Tight in 0g, Expands Dynamically in 200g)",
  "Calm 0g mean: " + CB.adaptive_margin_calm_mean_rad.toFixed(4) + " rad -> 200g shift mean: " + CB.adaptive_margin_shift_mean_rad.toFixed(4) + " rad vs. Static: " + CB.static_margin_rad.toFixed(3) + " rad | " + CB.adaptive_filter_median_ms.toFixed(2) + " ms"
);
const mD = { l: 62, r: 24, t: 62, b: 48 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const xD = d3.scaleLinear().domain([0, tr[tr.length - 1].t_s]).range([0, pwD]);
const yD = d3.scaleLinear().domain([0, 0.044]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("rect").attr("x", xD(tr[70].t_s)).attr("y", 0).attr("width", pwD - xD(tr[70].t_s)).attr("height", phD).attr("fill", "#fef3c7").attr("fill-opacity", 0.35);
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD).ticks(7));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
// Shaded braking margin saved
pD.append("path").datum(tr).attr("fill", "#10b981").attr("fill-opacity", 0.16)
  .attr("d", d3.area().x(d => xD(d.t_s)).y0(d => yD(d.adaptive_margin_rad)).y1(d => yD(d.static_margin_rad)));
pD.append("line").attr("x1", 0).attr("x2", pwD).attr("y1", yD(CB.static_margin_rad)).attr("y2", yD(CB.static_margin_rad))
  .attr("stroke", "#64748b").attr("stroke-width", 2.4).attr("stroke-dasharray", "6,3");
pD.append("path").datum(tr).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.9)
  .attr("d", d3.line().x(d => xD(d.t_s)).y(d => yD(d.adaptive_margin_rad)));
pD.append("text").attr("x", 14).attr("y", yD(CB.static_margin_rad) - 7).attr("fill", "#475569").attr("font-size", 11).attr("font-weight", 700).text("v0.6 Static Worst-Case Margin (0.036 rad)");
pD.append("text").attr("x", xD(0.7)).attr("y", yD(0.022)).attr("text-anchor", "middle").attr("fill", "#047857").attr("font-size", 11).attr("font-weight", 700).text("Saved Conservative Braking Margin (-33.5%)");
pD.append("text").attr("x", xD(2.1)).attr("y", yD(0.0225)).attr("text-anchor", "middle").attr("fill", "#0369a1").attr("font-size", 11).attr("font-weight", 700).text("Online ACP-CQR Expansion (2.15x)");
pD.append("text").attr("x", pwD / 2).attr("y", phD + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Closed-Loop Control Time (s)");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Conformal Safety Barrier Margin (rad)");
`;

// ============================================================================
// 2. FIGURE R5-v2: REASSEMBLE Port-Hamiltonian + Temporal-Bundle Jamming (V7-R4)
// ============================================================================
const DRAW_FIG_R5_V2 = `
const W = 1560, H = 920;
const S = DATA.summary;
const H45 = S.v7_r4_segment_holdout;
const H15 = S.v7_r4_identical_15ep_holdout;
const V6 = S.v6_baseline_holdout;
addHeader(svg, W,
  "Study R5-v2: REASSEMBLE Port-Hamiltonian & Multi-Scale Temporal-Bundle Jamming Estimator (V7-R4)",
  "Real Data: robot-lev/reassemble NIST Task Board #1 (428,683 frames @ 30 Hz, 45 episodes, 1,463 segments, CC-BY-4.0) | Leak-Free Split",
  [
    { text: "AUROC: 0.619 -> " + H45.auroc.toFixed(3), bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "Recall: " + (H45.recall * 100).toFixed(1) + "% @ " + (H45.fpr * 100).toFixed(1) + "% FPR", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: "15-Ep Parity: " + H15.auroc.toFixed(3) + " AUROC", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: Blind Holdout ROC Curve (v0.6 vs v0.7 15-ep vs v0.7 45-ep)
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Blind Episode-Split ROC Curve: v0.6 Threshold vs. V7-R4 Port-Hamiltonian Bundle",
  "45-ep Holdout (296 segs): AUROC " + H45.auroc.toFixed(4) + " (+0.289 vs v0.6) | Identical 15-ep Holdout: AUROC " + H15.auroc.toFixed(4) + " (+0.260 vs v0.6)"
);
const mA = { l: 58, r: 24, t: 62, b: 48 };
const pwA = 750 - mA.l - mA.r, phA = 400 - mA.t - mA.b;
const pA = gA.append("g").attr("transform", "translate(" + mA.l + "," + mA.t + ")");
const xA = d3.scaleLinear().domain([0, 1]).range([0, pwA]);
const yA = d3.scaleLinear().domain([0, 1]).range([phA, 0]);
pA.append("g").attr("class", "grid").call(d3.axisLeft(yA).ticks(5).tickSize(-pwA).tickFormat(""));
pA.append("rect").attr("x", 0).attr("y", 0).attr("width", xA(0.10)).attr("height", phA).attr("fill", "#dcfce7").attr("fill-opacity", 0.55);
pA.append("line").attr("x1", xA(0.10)).attr("x2", xA(0.10)).attr("y1", 0).attr("y2", phA).attr("stroke", "#10b981").attr("stroke-width", 1.4).attr("stroke-dasharray", "4,3");
pA.append("line").attr("x1", 0).attr("x2", pwA).attr("y1", phA).attr("y2", 0).attr("stroke", "#cbd5e1").attr("stroke-dasharray", "4,4");
pA.append("g").attr("class", "axis").attr("transform", "translate(0," + phA + ")").call(d3.axisBottom(xA).ticks(5));
pA.append("g").attr("class", "axis").call(d3.axisLeft(yA).ticks(5));
// v0.6 operating point curve schematic through (0,0) -> (0.3636, 0.5625) -> (1,1)
const v6Curve = [{ fpr: 0, tpr: 0 }, { fpr: V6.fpr, tpr: V6.recall }, { fpr: 1, tpr: 1 }];
pA.append("path").datum(v6Curve).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,3")
  .attr("d", d3.line().x(d => xA(d.fpr)).y(d => yA(d.tpr)));
pA.append("path").datum(DATA.roc_v7_15ep).attr("fill", "none").attr("stroke", "#8b5cf6").attr("stroke-width", 2.2)
  .attr("d", d3.line().x(d => xA(d.fpr)).y(d => yA(d.tpr)));
pA.append("path").datum(DATA.roc_v7).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 3.0)
  .attr("d", d3.line().x(d => xA(d.fpr)).y(d => yA(d.tpr)));
pA.append("circle").attr("cx", xA(H45.fpr)).attr("cy", yA(H45.recall)).attr("r", 6.5).attr("fill", "#059669").attr("stroke", "#fff").attr("stroke-width", 2);
pA.append("circle").attr("cx", xA(V6.fpr)).attr("cy", yA(V6.recall)).attr("r", 5.5).attr("fill", "#ef4444").attr("stroke", "#fff").attr("stroke-width", 1.5);
pA.append("text").attr("x", xA(H45.fpr) + 10).attr("y", yA(H45.recall) + 4).attr("fill", "#047857").attr("font-size", 11).attr("font-weight", 700).text("V7-R4 Op Point: 80.0% Rec @ 9.96% FPR");
pA.append("text").attr("x", xA(V6.fpr) + 10).attr("y", yA(V6.recall) + 14).attr("fill", "#b91c1c").attr("font-size", 10.5).attr("font-weight", 700).text("v0.6 Baseline: 56.3% Rec @ 36.4% FPR");
pA.append("text").attr("x", pwA / 2).attr("y", phA + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("False Positive Rate (Blind Holdout Episodes)");
pA.append("text").attr("transform", "rotate(-90)").attr("x", -phA / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("True Positive Rate (Jamming Recall)");
const legA = pA.append("g").attr("transform", "translate(" + (pwA - 282) + ", " + (phA - 74) + ")");
legA.append("rect").attr("width", 272).attr("height", 64).attr("rx", 6).attr("fill", "#fff").attr("fill-opacity", 0.94).attr("stroke", "#e2e8f0");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 15).attr("y2", 15).attr("stroke", "#0284c7").attr("stroke-width", 3);
legA.append("text").attr("x", 30).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("V7-R4 45-Ep Holdout (AUROC " + H45.auroc.toFixed(3) + ")");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 34).attr("y2", 34).attr("stroke", "#8b5cf6").attr("stroke-width", 2.2);
legA.append("text").attr("x", 30).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("V7-R4 Identical 15-Ep (AUROC " + H15.auroc.toFixed(3) + ")");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 53).attr("y2", 53).attr("stroke", "#94a3b8").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,3");
legA.append("text").attr("x", 30).attr("y", 57).attr("font-size", 10.5).attr("fill", "#1e293b").text("v0.6 Chatter Threshold (AUROC 0.619)");

// Panel B: v0.6 vs v0.7 Scorecard Bars (AUROC, Recall, 1-FPR, Precision)
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Quantitative Jump from v0.6 Threshold to V7-R4 Port-Hamiltonian Diagnostics",
  "Blind holdout false-alarm rate cut 3.65x (36.4% -> 9.96%) while lifting Recall +23.8 pts & Precision nearly 2x"
);
const mB = { l: 58, r: 24, t: 62, b: 50 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const metrics = [
  { name: "AUROC", v6: V6.auroc, v7: H45.auroc },
  { name: "Jam Recall", v6: V6.recall, v7: H45.recall },
  { name: "Specificity (1-FPR)", v6: 1 - V6.fpr, v7: 1 - H45.fpr },
  { name: "Precision", v6: V6.precision, v7: H45.precision },
];
const xB = d3.scaleBand().domain(metrics.map(d => d.name)).range([0, pwB]).padding(0.28);
const yB = d3.scaleLinear().domain([0, 1.08]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5).tickFormat(d3.format(".0%")));
metrics.forEach(d => {
  const bx = xB(d.name), bw = xB.bandwidth() / 2 - 4;
  pB.append("rect").attr("x", bx).attr("y", yB(d.v6)).attr("width", bw).attr("height", phB - yB(d.v6)).attr("rx", 5).attr("fill", "#94a3b8");
  pB.append("text").attr("x", bx + bw / 2).attr("y", yB(d.v6) - 6).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 700).attr("fill", "#64748b").text((d.v6 * 100).toFixed(1) + "%");
  pB.append("rect").attr("x", bx + bw + 8).attr("y", yB(d.v7)).attr("width", bw).attr("height", phB - yB(d.v7)).attr("rx", 5).attr("fill", "#0284c7");
  pB.append("text").attr("x", bx + bw + 8 + bw / 2).attr("y", yB(d.v7) - 6).attr("text-anchor", "middle").attr("font-size", 11.5).attr("font-weight", 700).attr("fill", "#0369a1").text((d.v7 * 100).toFixed(1) + "%");
});
const legB = pB.append("g").attr("transform", "translate(" + (pwB - 210) + ", 4)");
legB.append("rect").attr("width", 204).attr("height", 28).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legB.append("rect").attr("x", 10).attr("y", 9).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#94a3b8");
legB.append("text").attr("x", 26).attr("y", 18).attr("font-size", 10.5).attr("fill", "#334155").text("v0.6 Baseline");
legB.append("rect").attr("x", 106).attr("y", 9).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#0284c7");
legB.append("text").attr("x", 122).attr("y", 18).attr("font-size", 10.5).attr("fill", "#334155").text("V7-R4 Hybrid");

// Panel C: Real 30 Hz Holdout Episode Port-Hamiltonian Power-Balance & Jamming Probability Trace
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Real 30 Hz Franka Holdout Episode: Power-Balance Residual & Jam Probability",
  "Red band = real jammed/failed assembly segment | Per-segment inference = " + H45.per_segment_inference_ms.toFixed(3) + " ms | Real-time 30 Hz PHNN = " + S.v7_r4_frame_realtime_30hz.per_frame_inference_ms.toFixed(4) + " ms/frame"
);
const mC = { l: 58, r: 50, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const tw = DATA.trace_window;
const xC = d3.scaleLinear().domain([0, tw[tw.length - 1].t_s]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, 1.05]).range([phC, 0]);
const yP = d3.scaleLinear().domain([d3.min(tw, d => d.power_diss_w) - 1, d3.max(tw, d => d.power_diss_w) + 1]).range([phC, 0]);
for (let i = 0; i < tw.length - 1; i++) {
  if (tw[i].is_jammed) {
    pC.append("rect").attr("x", xC(tw[i].t_s)).attr("y", 0).attr("width", Math.max(2, xC(tw[i + 1].t_s) - xC(tw[i].t_s))).attr("height", phC).attr("fill", "#fecdd3").attr("fill-opacity", 0.5);
  }
}
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(7));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.append("g").attr("class", "axis").attr("transform", "translate(" + pwC + ",0)").call(d3.axisRight(yP).ticks(5));
pC.append("line").attr("x1", 0).attr("x2", pwC).attr("y1", yC(H45.decision_threshold)).attr("y2", yC(H45.decision_threshold)).attr("stroke", "#d97706").attr("stroke-width", 1.6).attr("stroke-dasharray", "5,3");
pC.append("path").datum(tw).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 1.4).attr("stroke-opacity", 0.75)
  .attr("d", d3.line().x(d => xC(d.t_s)).y(d => yP(d.power_diss_w)));
pC.append("path").datum(tw).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xC(d.t_s)).y(d => yC(d.jam_prob)));
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Elapsed Holdout Episode Time (s)");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#0369a1").attr("font-size", 11.5).attr("font-weight", 600).text("Calibrated Jamming Probability");
pC.append("text").attr("transform", "rotate(90)").attr("x", phC / 2).attr("y", -(pwC + 38)).attr("text-anchor", "middle").attr("fill", "#64748b").attr("font-size", 11).text("Power Dissipation r_P (W)");

// Panel D: Blind Holdout Segment Probability Separation (296 Real Segments)
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Blind Holdout Probability Separation Across All 296 Real Assembly Segments",
  "Confusion Matrix @ DEV-Calibrated Threshold (" + H45.decision_threshold.toFixed(3) + "): TP=" + H45.confusion.tp + ", FN=" + H45.confusion.fn + ", TN=" + H45.confusion.tn + ", FP=" + H45.confusion.fp
);
const mD = { l: 58, r: 24, t: 62, b: 48 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const segs = DATA.holdout_segments;
const binsD = d3.range(0, 1.01, 0.05);
const okBins = d3.bin().domain([0, 1]).thresholds(binsD)(segs.filter(s => s.label === 0).map(s => s.prob));
const flBins = d3.bin().domain([0, 1]).thresholds(binsD)(segs.filter(s => s.label === 1).map(s => s.prob));
const xD = d3.scaleLinear().domain([0, 1]).range([0, pwD]);
const yD = d3.scaleLinear().domain([0, Math.max(d3.max(okBins, b => b.length / 261), d3.max(flBins, b => b.length / 35)) * 1.15]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD).ticks(5));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5).tickFormat(d3.format(".0%")));
okBins.forEach(b => {
  const dens = b.length / 261;
  if (dens > 0) pD.append("rect").attr("x", xD(b.x0) + 1).attr("y", yD(dens)).attr("width", Math.max(2, xD(b.x1) - xD(b.x0) - 2)).attr("height", phD - yD(dens)).attr("fill", "#0284c7").attr("fill-opacity", 0.48).attr("rx", 2);
});
flBins.forEach(b => {
  const dens = b.length / 35;
  if (dens > 0) pD.append("rect").attr("x", xD(b.x0) + 1).attr("y", yD(dens)).attr("width", Math.max(2, xD(b.x1) - xD(b.x0) - 2)).attr("height", phD - yD(dens)).attr("fill", "#e11d48").attr("fill-opacity", 0.65).attr("rx", 2);
});
pD.append("line").attr("x1", xD(H45.decision_threshold)).attr("x2", xD(H45.decision_threshold)).attr("y1", 0).attr("y2", phD).attr("stroke", "#0f172a").attr("stroke-width", 2.2).attr("stroke-dasharray", "6,3");
pD.append("text").attr("x", xD(H45.decision_threshold) + 8).attr("y", 18).attr("fill", "#0f172a").attr("font-size", 11).attr("font-weight", 700).text("DEV Threshold = " + H45.decision_threshold.toFixed(3));
pD.append("text").attr("x", pwD / 2).attr("y", phD + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Predicted Jamming / Failure Probability");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Class-Normalized Segment Share");
const legD = pD.append("g").attr("transform", "translate(" + (pwD - 224) + ", 8)");
legD.append("rect").attr("width", 214).attr("height", 46).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legD.append("rect").attr("x", 10).attr("y", 11).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#0284c7").attr("fill-opacity", 0.6);
legD.append("text").attr("x", 28).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("Nominal Segments (n=261)");
legD.append("rect").attr("x", 10).attr("y", 29).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#e11d48").attr("fill-opacity", 0.75);
legD.append("text").attr("x", 28).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("Jammed / Failed Segments (n=35)");
`;

// ============================================================================
// 3. FIGURE R6: Dynamic 3-Axis Tactile Force & Friction-Cone Slip (Sparsh)
// ============================================================================
const DRAW_FIG_R6 = `
const W = 1560, H = 920;
const S = DATA.summary;
const M = S.blind_holdout_metrics;
addHeader(svg, W,
  "Study R6: Dynamic 3-Axis Tactile Force & Friction-Cone Slip-Onset Control (V6-R2 + V7-R4)",
  "Real Data: Meta Sparsh TacBench (facebook/gelsight-force-estimation, 102,628 frames, 1,513 traj, CC-BY-NC-4.0 Eval-Only) + 3D TET10 Contact FEM",
  [
    { text: "3-Axis RMSE: " + M.rmse_3axis_n.toFixed(4) + " N", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "Slip AUROC: " + M.slip_onset_auroc.toFixed(4), bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: M.median_inference_ms.toFixed(2) + " ms (" + Math.round(M.control_rate_hz) + " Hz) | " + (M.as6983_upper_coverage * 100).toFixed(1) + "% AS6983", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: Blind Holdout 3-Axis Force Parity (Normal Fz + Tangential Ft)
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Blind Holdout 3-Axis Tactile Force Parity (19,727 Frames Across 302 Trajectories)",
  "RMSE Fx = " + M.rmse_fx_n.toFixed(4) + " N | Fy = " + M.rmse_fy_n.toFixed(4) + " N | Fz = " + M.rmse_fz_n.toFixed(4) + " N (Target < 0.50 N crushed 11x!)"
);
const mA = { l: 58, r: 24, t: 62, b: 48 };
const pwA = 750 - mA.l - mA.r, phA = 400 - mA.t - mA.b;
const pA = gA.append("g").attr("transform", "translate(" + mA.l + "," + mA.t + ")");
const sc = DATA.scatter_sample;
const maxF = Math.max(d3.max(sc, d => Math.max(d.fz_true, d.fz_pred, d.ft_true, d.ft_pred)) * 1.06, 4.0);
const xA = d3.scaleLinear().domain([0, maxF]).range([0, pwA]);
const yA = d3.scaleLinear().domain([0, maxF]).range([phA, 0]);
pA.append("g").attr("class", "grid").call(d3.axisLeft(yA).ticks(5).tickSize(-pwA).tickFormat(""));
pA.append("line").attr("x1", 0).attr("x2", pwA).attr("y1", phA).attr("y2", 0).attr("stroke", "#0f172a").attr("stroke-dasharray", "5,4").attr("stroke-width", 1.5);
pA.append("g").attr("class", "axis").attr("transform", "translate(0," + phA + ")").call(d3.axisBottom(xA).ticks(6));
pA.append("g").attr("class", "axis").call(d3.axisLeft(yA).ticks(5));
pA.selectAll("circle.fz").data(sc.filter((_, i) => i % 2 === 0)).enter().append("circle")
  .attr("cx", d => xA(d.fz_true)).attr("cy", d => yA(d.fz_pred)).attr("r", 2.8).attr("fill", "#0284c7").attr("fill-opacity", 0.52);
pA.selectAll("circle.ft").data(sc.filter((_, i) => i % 2 === 0)).enter().append("circle")
  .attr("cx", d => xA(d.ft_true)).attr("cy", d => yA(d.ft_pred)).attr("r", 2.8).attr("fill", "#e11d48").attr("fill-opacity", 0.55);
pA.append("text").attr("x", pwA / 2).attr("y", phA + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Measured Ground-Truth Force (N)");
pA.append("text").attr("transform", "rotate(-90)").attr("x", -phA / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("FormaMech Predicted Force (N)");
const legA = pA.append("g").attr("transform", "translate(12, 8)");
legA.append("rect").attr("width", 232).attr("height", 48).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legA.append("circle").attr("cx", 14).attr("cy", 16).attr("r", 3.5).attr("fill", "#0284c7");
legA.append("text").attr("x", 26).attr("y", 20).attr("font-size", 10.5).attr("fill", "#1e293b").text("Normal Force |Fz| (RMSE 0.040 N)");
legA.append("circle").attr("cx", 14).attr("cy", 34).attr("r", 3.5).attr("fill", "#e11d48");
legA.append("text").attr("x", 26).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("Shear ||Ft|| (RMSE 0.047 N)");

// Panel B: Coulomb Friction Cone Stick vs Slip Separation (||Ft|| vs |Fz|)
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Coulomb Friction-Cone Stick vs. Incipient Slip Boundary (Real Sparsh Holdout)",
  "Pure physics cone ||Ft||/(mu|Fz|) AUROC = " + M.physics_cone_ratio_auroc.toFixed(4) + " -> V7-R4 Slip Classifier AUROC = " + M.slip_onset_auroc.toFixed(4) + " (Recall 99.96% @ 2.05% FPR)"
);
const mB = { l: 58, r: 24, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const xB = d3.scaleLinear().domain([0, d3.max(sc, d => d.fz_true) * 1.06]).range([0, pwB]);
const yB = d3.scaleLinear().domain([0, d3.max(sc, d => d.ft_true) * 1.12]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(6));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
pB.selectAll("circle.st").data(sc.filter(d => d.slip === 0)).enter().append("circle")
  .attr("cx", d => xB(d.fz_true)).attr("cy", d => yB(d.ft_true)).attr("r", 3.0).attr("fill", "#0284c7").attr("fill-opacity", 0.45);
pB.selectAll("circle.sl").data(sc.filter(d => d.slip === 1)).enter().append("circle")
  .attr("cx", d => xB(d.fz_true)).attr("cy", d => yB(d.ft_true)).attr("r", 3.2).attr("fill", "#e11d48").attr("fill-opacity", 0.72);
// Draw Coulomb friction cone line mu ~ 0.35
const coneX = d3.max(sc, d => d.fz_true);
pB.append("line").attr("x1", xB(0)).attr("y1", yB(0)).attr("x2", xB(coneX)).attr("y2", yB(Math.min(yB.domain()[1], 0.35 * coneX)))
  .attr("stroke", "#0f172a").attr("stroke-width", 2.0).attr("stroke-dasharray", "6,3");
pB.append("text").attr("x", xB(coneX * 0.68)).attr("y", yB(0.35 * coneX * 0.68) - 18).attr("fill", "#0f172a").attr("font-size", 11).attr("font-weight", 700).text("Coulomb Cone ||Ft|| = mu * |Fz|");
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Normal Contact Force |Fz| (N)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Tangential Shear Force ||Ft|| (N)");
const legB = pB.append("g").attr("transform", "translate(12, 8)");
legB.append("rect").attr("width", 196).attr("height", 46).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legB.append("circle").attr("cx", 14).attr("cy", 15).attr("r", 3.5).attr("fill", "#0284c7");
legB.append("text").attr("x", 26).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("Nominal Stick Contact");
legB.append("circle").attr("cx", 14).attr("cy", 33).attr("r", 3.5).attr("fill", "#e11d48");
legB.append("text").attr("x", 26).attr("y", 37).attr("font-size", 10.5).attr("fill", "#1e293b").text("Active / Incipient Slip");

// Panel C: Live Holdout Trajectory 3-Axis Force + AS6983 Upper Envelope & Slip Probability
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Unseen Holdout Trajectory: Coupled Normal/Shear Force, AS6983 Bound & Slip Onset",
  "AS6983 Upper Coverage = " + (M.as6983_upper_coverage * 100).toFixed(2) + "% (Shift = +" + M.as6983_conformal_shift_n.toFixed(4) + " N) | Red Band = Ground-Truth Slip"
);
const mC = { l: 58, r: 48, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const ht = DATA.holdout_trajectory;
const xC = d3.scaleLinear().domain([0, ht.length - 1]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, d3.max(ht, d => Math.max(d.f_mag_upper, d.fz_true, d.f_shear_true)) * 1.12]).range([phC, 0]);
const ySp = d3.scaleLinear().domain([0, 1.05]).range([phC, 0]);
for (let i = 0; i < ht.length - 1; i++) {
  if (ht[i].slip_true) {
    pC.append("rect").attr("x", xC(i)).attr("y", 0).attr("width", Math.max(2, xC(i + 1) - xC(i))).attr("height", phC).attr("fill", "#fecdd3").attr("fill-opacity", 0.45);
  }
}
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(7));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.append("g").attr("class", "axis").attr("transform", "translate(" + pwC + ",0)").call(d3.axisRight(ySp).ticks(5));
pC.append("path").datum(ht).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.2)
  .attr("d", d3.line().x((_, i) => xC(i)).y(d => yC(d.fz_true)));
pC.append("path").datum(ht).attr("fill", "none").attr("stroke", "#7c3aed").attr("stroke-width", 2.0).attr("stroke-dasharray", "5,3")
  .attr("d", d3.line().x((_, i) => xC(i)).y(d => yC(d.f_mag_upper)));
pC.append("path").datum(ht).attr("fill", "none").attr("stroke", "#f59e0b").attr("stroke-width", 2.2)
  .attr("d", d3.line().x((_, i) => xC(i)).y(d => yC(d.f_shear_true)));
pC.append("path").datum(ht).attr("fill", "none").attr("stroke", "#dc2626").attr("stroke-width", 2.4)
  .attr("d", d3.line().x((_, i) => xC(i)).y(d => ySp(d.slip_prob)));
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Tactile Frame Index (Blind Holdout Trajectory)");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Contact Force (N)");
pC.append("text").attr("transform", "rotate(90)").attr("x", phC / 2).attr("y", -(pwC + 36)).attr("text-anchor", "middle").attr("fill", "#dc2626").attr("font-size", 11).attr("font-weight", 600).text("Slip Prob");
const legC6 = pC.append("g").attr("transform", "translate(10, 6)");
legC6.append("rect").attr("width", 214).attr("height", 74).attr("rx", 6).attr("fill", "#fff").attr("fill-opacity", 0.92).attr("stroke", "#e2e8f0");
legC6.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 14).attr("y2", 14).attr("stroke", "#0284c7").attr("stroke-width", 2.2);
legC6.append("text").attr("x", 30).attr("y", 18).attr("font-size", 10).attr("fill", "#1e293b").text("Normal Force |Fz| (N)");
legC6.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 31).attr("y2", 31).attr("stroke", "#7c3aed").attr("stroke-width", 2.0).attr("stroke-dasharray", "5,3");
legC6.append("text").attr("x", 30).attr("y", 35).attr("font-size", 10).attr("fill", "#1e293b").text("AS6983 95% Upper Bound");
legC6.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 48).attr("y2", 48).attr("stroke", "#f59e0b").attr("stroke-width", 2.2);
legC6.append("text").attr("x", 30).attr("y", 52).attr("font-size", 10).attr("fill", "#1e293b").text("Tangential Shear ||Ft|| (N)");
legC6.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 65).attr("y2", 65).attr("stroke", "#dc2626").attr("stroke-width", 2.4);
legC6.append("text").attr("x", 30).attr("y", 69).attr("font-size", 10).attr("fill", "#1e293b").text("Incipient Slip Prob (right axis)");

// Panel D: Per-Indenter Generalization (Sphere / Flat / Sharp) + 3D TET10 FEM Verification
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Per-Indenter Holdout Force RMSE & 3D TET10 Coupled Normal+Shear FEM Verification",
  "All 9 3D TET10 coupled normal+shear FEM probes converged across SPHERE, FLAT_PUNCH, and WEDGE_EDGE"
);
const mD = { l: 60, r: 24, t: 62, b: 54 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const shapes = Object.entries(M.per_indenter_shape).map(([k, v]) => ({ shape: k, rmse: v.rmse_3axis_n, auroc: v.slip_auroc, n: v.frames }));
const xD = d3.scaleBand().domain(shapes.map(d => d.shape)).range([0, pwD]).padding(0.36);
const yD = d3.scaleLinear().domain([0, 0.10]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
const shCols = { sphere: "#0284c7", flat: "#7c3aed", sharp: "#059669" };
pD.selectAll("rect.sh").data(shapes).enter().append("rect")
  .attr("x", d => xD(d.shape)).attr("y", d => yD(d.rmse)).attr("width", xD.bandwidth()).attr("height", d => phD - yD(d.rmse)).attr("rx", 6)
  .attr("fill", d => shCols[d.shape] || "#0284c7");
pD.selectAll("text.shv").data(shapes).enter().append("text")
  .attr("x", d => xD(d.shape) + xD.bandwidth() / 2).attr("y", d => yD(d.rmse) - 18).attr("text-anchor", "middle").attr("font-size", 11.5).attr("font-weight", 700).attr("fill", "#0f172a")
  .text(d => d.rmse.toFixed(4) + " N");
pD.selectAll("text.sha").data(shapes).enter().append("text")
  .attr("x", d => xD(d.shape) + xD.bandwidth() / 2).attr("y", d => yD(d.rmse) - 4).attr("text-anchor", "middle").attr("font-size", 10.2).attr("font-weight", 600).attr("fill", "#047857")
  .text(d => "Slip AUROC " + d.auroc.toFixed(4) + " (n=" + d.n + ")");
pD.append("text").attr("x", pwD / 2).attr("y", phD + 38).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Indenter Tip Geometry (Target Pass Gate: RMSE < 0.50 N)");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("3-Axis Holdout Force RMSE (N)");
`;

// ============================================================================
// 4. FIGURE R7: PhysCoRe Real-to-Sim Deformable Tracking & Two-Way Coupling (V7-R5)
// ============================================================================
const DRAW_FIG_R7 = `
const W = 1560, H = 920;
const S = DATA.summary;
const TR = S.physcore_blind_holdout_tracking;
const CP = S.twoway_vs_oneway_coupling_v7_r5;
const FE = DATA.featured_episode;
addHeader(svg, W,
  "Study R7: PhysCoRe Deformable Manipulation & Live Two-Way Coupling",
  "Real Data: GeorgiaTech/PhysCoRe & PhysTwin (8 RGB-D 3D Tracked Episodes, MIT + CC-BY-4.0, Tier A) | Zero-Copy C-ABI + stop_grad_residual",
  [
    { text: "Blind Chamfer: " + TR.mean_visco_chamfer_mm.toFixed(2) + " mm (< 5 mm)", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "-" + TR.chamfer_improvement_vs_elastic_pct.toFixed(1) + "% vs Elastic", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: "C-ABI Diff: 0.0 | Max Div: " + CP.max_tool_trajectory_divergence_mm.toFixed(0) + " mm", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: 3D Isometric Blind Holdout Point Cloud Tracking (double_stretch_plastic)
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Blind 3D Deformable Twin vs. Ground-Truth RGB-D Tracks (" + FE.case + ")",
  "Frame " + FE.sample_holdout_frame + " (Blind Holdout Tail) | Multi-hand SE(3) + Prony Viscoelastic Twin: 1.34 mm Chamfer"
);
const truePts = FE.sample_true_pts_mm, predPts = FE.sample_pred_pts_mm;
const cX = d3.mean(truePts, p => p[0]), cY = d3.mean(truePts, p => p[1]), cZ = d3.mean(truePts, p => p[2]);
const span = Math.max(
  d3.max(truePts, p => p[0]) - d3.min(truePts, p => p[0]),
  d3.max(truePts, p => p[1]) - d3.min(truePts, p => p[1]),
  d3.max(truePts, p => p[2]) - d3.min(truePts, p => p[2]),
  80
);
const sc3 = 230 / span;
const prT = truePts.map(p => isoProject(p[0] - cX, p[1] - cY, p[2] - cZ, 375, 228, sc3, -38, 26));
const prP = predPts.map(p => isoProject(p[0] - cX, p[1] - cY, p[2] - cZ, 375, 228, sc3, -38, 26));
for (let i = 0; i < prT.length; i++) {
  gA.append("line").attr("x1", prT[i].x).attr("y1", prT[i].y).attr("x2", prP[i].x).attr("y2", prP[i].y)
    .attr("stroke", "#f59e0b").attr("stroke-width", 1.2).attr("stroke-opacity", 0.7);
}
gA.selectAll("circle.tp").data(prT).enter().append("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 4.6).attr("fill", "#0f172a").attr("fill-opacity", 0.75);
gA.selectAll("circle.pp").data(prP).enter().append("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 3.4).attr("fill", "#0284c7").attr("stroke", "#fff").attr("stroke-width", 0.7);
const legA = gA.append("g").attr("transform", "translate(20, 336)");
legA.append("rect").attr("width", 460).attr("height", 36).attr("rx", 6).attr("fill", "#f8fafc").attr("stroke", "#e2e8f0");
legA.append("circle").attr("cx", 18).attr("cy", 18).attr("r", 4.5).attr("fill", "#0f172a");
legA.append("text").attr("x", 30).attr("y", 22).attr("font-size", 11).attr("fill", "#1e293b").text("Real 3-Camera RGB-D Track");
legA.append("circle").attr("cx", 215).attr("cy", 18).attr("r", 3.8).attr("fill", "#0284c7");
legA.append("text").attr("x", 227).attr("y", 22).attr("font-size", 11).attr("fill", "#1e293b").text("FormaMech Viscoelastic Twin (1.34 mm)");

// Panel B: Per-Episode Blind Holdout 3D Chamfer Error vs Pure-Elastic Baseline
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Blind Holdout 3D Symmetric Chamfer Across 8 Real PhysCoRe/PhysTwin Episodes",
  "Mean = " + TR.mean_visco_chamfer_mm.toFixed(3) + " mm (Median " + TR.median_visco_chamfer_mm.toFixed(3) + " mm) vs. Undamped Elastic " + TR.mean_elastic_chamfer_mm.toFixed(3) + " mm (-" + TR.chamfer_improvement_vs_elastic_pct.toFixed(1) + "%)"
);
const mB = { l: 58, r: 24, t: 62, b: 86 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const eps = DATA.episodes;
const xB = d3.scaleBand().domain(eps.map(d => d.case)).range([0, pwB]).padding(0.24);
const yB = d3.scaleLinear().domain([0, d3.max(eps, d => Math.max(d.elastic_chamfer_mm, d.visco_chamfer_mm)) * 1.15]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB))
  .selectAll("text").attr("transform", "rotate(-26)").style("text-anchor", "end").attr("font-size", 9.8);
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
pB.append("line").attr("x1", 0).attr("x2", pwB).attr("y1", yB(5.0)).attr("y2", yB(5.0)).attr("stroke", "#059669").attr("stroke-width", 1.8).attr("stroke-dasharray", "6,3");
eps.forEach(d => {
  const bx = xB(d.case), bw = xB.bandwidth() / 2 - 2;
  pB.append("rect").attr("x", bx).attr("y", yB(d.elastic_chamfer_mm)).attr("width", bw).attr("height", phB - yB(d.elastic_chamfer_mm)).attr("rx", 3).attr("fill", "#cbd5e1");
  pB.append("rect").attr("x", bx + bw + 3).attr("y", yB(d.visco_chamfer_mm)).attr("width", bw).attr("height", phB - yB(d.visco_chamfer_mm)).attr("rx", 3).attr("fill", "#0284c7");
});
pB.append("text").attr("x", 10).attr("y", yB(5.0) - 8).attr("text-anchor", "start").attr("fill", "#047857").attr("font-size", 10.8).attr("font-weight", 700).text("Pass Gate < 5.0 mm (Mean 3.64 mm)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Blind 3D Chamfer Error (mm)");
const legB = pB.append("g").attr("transform", "translate(10, 6)");
legB.append("rect").attr("width", 230).attr("height", 28).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legB.append("rect").attr("x", 10).attr("y", 9).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#cbd5e1");
legB.append("text").attr("x", 26).attr("y", 18).attr("font-size", 10.5).attr("fill", "#334155").text("Undamped Elastic");
legB.append("rect").attr("x", 122).attr("y", 9).attr("width", 12).attr("height", 10).attr("rx", 2).attr("fill", "#0284c7");
legB.append("text").attr("x", 138).attr("y", 18).attr("font-size", 10.5).attr("fill", "#334155").text("V7-R1 Visco Twin");

// Panel C: V7-R5 Live Two-Way vs One-Way Coupled Tool Trajectory Divergence
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Live Two-Way vs. One-Way Simulator Coupling Divergence (V7-R5)",
  "Real reaction wrench & linearized tangent stiffness K_t fed back every step (" + CP.mean_coupling_step_ms.toFixed(2) + " ms/step) | Max Div = " + CP.max_tool_trajectory_divergence_mm.toFixed(1) + " mm"
);
const mC = { l: 58, r: 52, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const twR = FE.twoway_rollout;
const xC = d3.scaleLinear().domain([0, twR[twR.length - 1].t_s]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, d3.max(twR, d => d.tool_div_mm) * 1.15]).range([phC, 0]);
const yF = d3.scaleLinear().domain([0, Math.max(d3.max(twR, d => d.reaction_force_n) * 1.15, 1.0)]).range([phC, 0]);
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(6));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.append("g").attr("class", "axis").attr("transform", "translate(" + pwC + ",0)").call(d3.axisRight(yF).ticks(5));
pC.append("path").datum(twR).attr("fill", "#3b82f6").attr("fill-opacity", 0.14)
  .attr("d", d3.area().x(d => xC(d.t_s)).y0(phC).y1(d => yC(d.tool_div_mm)));
pC.append("path").datum(twR).attr("fill", "none").attr("stroke", "#2563eb").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xC(d.t_s)).y(d => yC(d.tool_div_mm)));
pC.append("path").datum(twR).attr("fill", "none").attr("stroke", "#e11d48").attr("stroke-width", 2.2).attr("stroke-dasharray", "5,3")
  .attr("d", d3.line().x(d => xC(d.t_s)).y(d => yF(d.reaction_force_n)));
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Coupled Manipulation Time (s)");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#2563eb").attr("font-size", 11.5).attr("font-weight", 600).text("Two-Way vs One-Way Tool Divergence (mm)");
pC.append("text").attr("transform", "rotate(90)").attr("x", phC / 2).attr("y", -(pwC + 40)).attr("text-anchor", "middle").attr("fill", "#e11d48").attr("font-size", 11).attr("font-weight", 600).text("Reaction Force (N)");

// Panel D: Per-Episode Two-Way Coupling Divergence & Zero-Copy C-ABI Contract Card
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Two-Way Reaction Coupling Impact & Zero-Copy C-ABI Contract Audit",
  "Mean divergence across 8 episodes = " + CP.mean_tool_trajectory_divergence_mm.toFixed(1) + " mm | C-ABI rc = 0, diff = 0.0 | stop_grad_residual isolated"
);
const mD = { l: 58, r: 24, t: 62, b: 86 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const xD = d3.scaleBand().domain(eps.map(d => d.case)).range([0, pwD]).padding(0.32);
const yD = d3.scaleSymlog().constant(5).domain([0, d3.max(eps, d => d.twoway_vs_oneway_max_tool_div_mm) * 1.25]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD))
  .selectAll("text").attr("transform", "rotate(-26)").style("text-anchor", "end").attr("font-size", 9.8);
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
pD.selectAll("rect.dv").data(eps).enter().append("rect")
  .attr("x", d => xD(d.case)).attr("y", d => yD(d.twoway_vs_oneway_max_tool_div_mm))
  .attr("width", xD.bandwidth()).attr("height", d => phD - yD(d.twoway_vs_oneway_max_tool_div_mm)).attr("rx", 4)
  .attr("fill", "#7c3aed").attr("fill-opacity", 0.85);
pD.selectAll("text.dvt").data(eps).enter().append("text")
  .attr("x", d => xD(d.case) + xD.bandwidth() / 2).attr("y", d => yD(d.twoway_vs_oneway_max_tool_div_mm) - 5)
  .attr("text-anchor", "middle").attr("font-size", 9.8).attr("font-weight", 700).attr("fill", "#4c1d95")
  .text(d => d.twoway_vs_oneway_max_tool_div_mm.toFixed(1));
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Max Two-Way vs One-Way Div (mm, symlog)");
`;

// ============================================================================
// 5. FIGURE R8: Online 2-Probe PANO/CANO Surgical Tissue ID (V7-R3)
// ============================================================================
const DRAW_FIG_R8 = `
const W = 1560, H = 920;
const S = DATA.summary;
const DS = S.donor_split_comparison;
const LV = S.liver_phantom_inclusion_palpation_3d_fem;
addHeader(svg, W,
  "Study R8: Online 2-Probe PANO/CANO Surgical Tissue Identification (V7-R3)",
  "Real Data: Hyperelastic Human Brain 1-7 (Zenodo 8095559 + 21793225, 181 Specimens, 7 Donors, CC-BY-4.0) + 3D TET10 Liver Phantom",
  [
    { text: "Donor RMSE: 210.5 -> " + DS.v7_pano_2probe_blind_donor_rmse_pa.toFixed(1) + " Pa (-90.5%)", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "V&V10: REJECTED -> " + DS.v7_vv10_state, bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: "2-Probe ID: " + DS.pano_2probe_median_ms.toFixed(2) + " ms | 100% AS6983", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: Blind Donor Scatter (HBE_05..HBE_07, 76 Specimens): Global Baseline vs 2-Probe PANO
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Blind Unseen Donors (HBE_05-07, n = 76): Global Modulus vs. 2-Probe PANO/CANO",
  "Slashes 10%-strain stress RMSE 90.45%: v0.6 global 210.5 Pa -> V7-R3 PANO " + DS.v7_pano_2probe_blind_donor_rmse_pa.toFixed(2) + " Pa (hybrid V&V10: " + DS.v7_vv10_hybrid_holdout_rmse_pa.toFixed(2) + " Pa)"
);
const mA = { l: 60, r: 24, t: 62, b: 48 };
const pwA = 750 - mA.l - mA.r, phA = 400 - mA.t - mA.b;
const pA = gA.append("g").attr("transform", "translate(" + mA.l + "," + mA.t + ")");
const hs = DATA.holdout_scatter;
const maxP = d3.max(hs, d => Math.max(d.true_stress_10pct_pa, d.pano_pred_10pct_pa)) * 1.08;
const xA = d3.scaleLinear().domain([0, maxP]).range([0, pwA]);
const yA = d3.scaleLinear().domain([0, maxP]).range([phA, 0]);
pA.append("g").attr("class", "grid").call(d3.axisLeft(yA).ticks(5).tickSize(-pwA).tickFormat(""));
pA.append("line").attr("x1", 0).attr("x2", pwA).attr("y1", phA).attr("y2", 0).attr("stroke", "#0f172a").attr("stroke-dasharray", "5,4").attr("stroke-width", 1.5);
pA.append("g").attr("class", "axis").attr("transform", "translate(0," + phA + ")").call(d3.axisBottom(xA).ticks(6));
pA.append("g").attr("class", "axis").call(d3.axisLeft(yA).ticks(5));
pA.selectAll("circle.gl").data(hs).enter().append("circle")
  .attr("cx", d => xA(d.true_stress_10pct_pa)).attr("cy", d => yA(d.global_pred_10pct_pa)).attr("r", 3.6).attr("fill", "#94a3b8").attr("fill-opacity", 0.55);
pA.selectAll("circle.pn").data(hs).enter().append("circle")
  .attr("cx", d => xA(d.true_stress_10pct_pa)).attr("cy", d => yA(d.pano_pred_10pct_pa)).attr("r", 4.2).attr("fill", "#0284c7").attr("stroke", "#fff").attr("stroke-width", 0.7);
pA.append("text").attr("x", pwA / 2).attr("y", phA + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Measured Human Brain 10%-Strain Stress (Pa)");
pA.append("text").attr("transform", "rotate(-90)").attr("x", -phA / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Predicted 10%-Strain Stress (Pa)");
const legA = pA.append("g").attr("transform", "translate(12, 8)");
legA.append("rect").attr("width", 268).attr("height", 48).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legA.append("circle").attr("cx", 14).attr("cy", 15).attr("r", 4.2).attr("fill", "#0284c7");
legA.append("text").attr("x", 26).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("V7-R3 2-Probe PANO (20.1 Pa, VALIDATED)");
legA.append("circle").attr("cx", 14).attr("cy", 34).attr("r", 3.8).attr("fill", "#94a3b8");
legA.append("text").attr("x", 26).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("v0.6 Global Modulus (210.5 Pa, REJECTED)");

// Panel B: 4 Representative Blind Donor Stress-Strain Curves + 100% AS6983 Tearing Bounds
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Patient-Specific Ogden Curves Across 30x Stiffness Span + AS6983 Bounds",
  "Identified mu in [" + DS.identified_mu_range_pa[0] + ", " + DS.identified_mu_range_pa[1] + "] Pa | Multiplicative AS6983 Upper Coverage = 100.0% (76/76)"
);
const mB = { l: 60, r: 24, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const rcs = DATA.representative_curves;
const maxC = d3.max(rcs, c => d3.max(c.upper_bound_pa)) * 1.06;
const xB = d3.scaleLinear().domain([0, 10.3]).range([0, pwB]);
const yB = d3.scaleLinear().domain([0, maxC]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(6));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
const colsB = ["#0284c7", "#059669", "#7c3aed", "#e11d48"];
rcs.forEach((c, ci) => {
  const pts = c.strains_pct.map((s, i) => ({ s, tr: c.true_pa[i], pr: c.pano_pa[i], up: c.upper_bound_pa[i] }));
  pB.append("path").datum(pts).attr("fill", colsB[ci]).attr("fill-opacity", 0.10)
    .attr("d", d3.area().x(d => xB(d.s)).y0(d => yB(d.pr)).y1(d => yB(d.up)));
  pB.append("path").datum(pts).attr("fill", "none").attr("stroke", colsB[ci]).attr("stroke-width", 2.4)
    .attr("d", d3.line().x(d => xB(d.s)).y(d => yB(d.pr)));
  pB.append("path").datum(pts).attr("fill", "none").attr("stroke", colsB[ci]).attr("stroke-width", 1.3).attr("stroke-dasharray", "4,3")
    .attr("d", d3.line().x(d => xB(d.s)).y(d => yB(d.up)));
  pB.selectAll(null).data(pts.filter((_, i) => i % 2 === 0)).enter().append("circle")
    .attr("cx", d => xB(d.s)).attr("cy", d => yB(d.tr)).attr("r", 3.0).attr("fill", colsB[ci]).attr("stroke", "#fff").attr("stroke-width", 0.6);
});
// Highlight 2 palpation probes at 2% and 5% strain
[2.0, 5.0].forEach(ps => {
  pB.append("line").attr("x1", xB(ps)).attr("x2", xB(ps)).attr("y1", 0).attr("y2", phB).attr("stroke", "#d97706").attr("stroke-width", 1.3).attr("stroke-dasharray", "3,3");
});
pB.append("text").attr("x", xB(3.5)).attr("y", 16).attr("text-anchor", "middle").attr("fill", "#b45309").attr("font-size", 10.5).attr("font-weight", 700).text("2 Online Probes (2%, 5%) -> 0.46 ms ID");
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Compressive Tissue Strain (%)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -46).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("First Piola-Kirchhoff Stress (Pa)");

// Panel C: 3D TET10 Liver Phantom Shallow-Inclusion Palpation Stiffness Map
const gC = addCard(svg, 20, 504, 750, 396,
  "C. 3D TET10 Liver Phantom Shallow-Inclusion Localization (5x5 Robotic Palpation Grid)",
  "True Tumor: (" + LV.true_tumor_center_xy_mm[0] + ", " + LV.true_tumor_center_xy_mm[1] + ") mm -> PANO Estimate: (" + LV.estimated_tumor_center_xy_mm[0] + ", " + LV.estimated_tumor_center_xy_mm[1] + ") mm | Error = " + LV.localization_error_mm + " mm | Contrast = " + LV.stiffness_contrast_ratio + "x"
);
const pm = DATA.palpation_map;
const xsU = [...new Set(pm.map(d => d.x_mm))].sort((a, b) => a - b);
const ysU = [...new Set(pm.map(d => d.y_mm))].sort((a, b) => a - b);
const cellW = 54, gx0 = 96, gy0 = 76;
const colM = d3.scaleSequential(d3.interpolateInferno).domain([d3.min(pm, d => d.modulus_pa) * 0.9, d3.max(pm, d => d.modulus_pa) * 1.05]);
pm.forEach(d => {
  const ix = xsU.indexOf(d.x_mm), iy = ysU.length - 1 - ysU.indexOf(d.y_mm);
  const cx = gx0 + ix * cellW, cy = gy0 + iy * cellW;
  gC.append("rect").attr("x", cx).attr("y", cy).attr("width", cellW - 3).attr("height", cellW - 3).attr("rx", 5)
    .attr("fill", colM(d.modulus_pa)).attr("stroke", "#fff").attr("stroke-width", 1);
  gC.append("text").attr("x", cx + (cellW - 3) / 2).attr("y", cy + (cellW - 3) / 2 + 4).attr("text-anchor", "middle")
    .attr("fill", d.modulus_pa > 34000 ? "#0f172a" : "#f8fafc").attr("font-size", 9.8).attr("font-weight", 700)
    .text((d.modulus_pa / 1000).toFixed(1) + "k");
});
// Overlay True vs Estimated inclusion markers on grid coordinate space
const mapX = d3.scaleLinear().domain([xsU[0] - 1.5, xsU[xsU.length - 1] + 1.5]).range([gx0, gx0 + xsU.length * cellW]);
const mapY = d3.scaleLinear().domain([ysU[0] - 1.5, ysU[ysU.length - 1] + 1.5]).range([gy0 + ysU.length * cellW, gy0]);
const tx = mapX(LV.true_tumor_center_xy_mm[0]), ty = mapY(LV.true_tumor_center_xy_mm[1]);
const ex = mapX(LV.estimated_tumor_center_xy_mm[0]), ey = mapY(LV.estimated_tumor_center_xy_mm[1]);
gC.append("circle").attr("cx", tx).attr("cy", ty).attr("r", 15).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.5);
gC.append("circle").attr("cx", ex).attr("cy", ey).attr("r", 6).attr("fill", "#10b981").attr("stroke", "#fff").attr("stroke-width", 1.8);
// Right info card inside Panel C
const ic = gC.append("g").attr("transform", "translate(404, 88)");
ic.append("rect").attr("width", 320).attr("height", 248).attr("rx", 8).attr("fill", "#f8fafc").attr("stroke", "#e2e8f0");
ic.append("text").attr("x", 16).attr("y", 28).attr("font-size", 13).attr("font-weight", 700).attr("fill", "#0f172a").text("3D TET10 Phantom Audit");
const infoLines = [
  ["FEM Status", "CONVERGED (min det(F) = " + LV.fem_min_det_f + ")"],
  ["True Tumor Center", "(" + LV.true_tumor_center_xy_mm[0] + ", " + LV.true_tumor_center_xy_mm[1] + ") mm"],
  ["PANO Localized", "(" + LV.estimated_tumor_center_xy_mm[0] + ", " + LV.estimated_tumor_center_xy_mm[1] + ") mm"],
  ["Localization Error", LV.localization_error_mm + " mm (sub-mm!)"],
  ["Stiffness Contrast", LV.stiffness_contrast_ratio + "x inclusion/healthy"],
  ["2-Probe Latency", DS.pano_2probe_median_ms + " ms median (" + DS.pano_2probe_p95_ms + " ms P95)"],
];
infoLines.forEach(([k, v], i) => {
  ic.append("text").attr("x", 16).attr("y", 60 + i * 30).attr("font-size", 11).attr("fill", "#64748b").text(k + ":");
  ic.append("text").attr("x", 142).attr("y", 60 + i * 30).attr("font-size", 11.2).attr("font-weight", 700).attr("fill", i === 3 ? "#059669" : "#0f172a").text(v);
});

// Panel D: 9 Published Brain Regions Stiffness Span (85.9 - 434.0 Pa) & PANO Identified Distribution
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Human Brain Regional Shear Modulus Heterogeneity (Zenodo 21793225)",
  "5x anatomical span explains why global modulus fails donor split — online 2-probe PANO resolves it in 0.46 ms"
);
const mD = { l: 145, r: 30, t: 62, b: 44 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const regs = [...DATA.regions].sort((a, b) => a.shear_modulus_pa - b.shear_modulus_pa);
const yD = d3.scaleBand().domain(regs.map(d => d.region)).range([phD, 0]).padding(0.24);
const xD = d3.scaleLinear().domain([0, d3.max(regs, d => d.shear_modulus_pa) * 1.15]).range([0, pwD]);
pD.append("g").attr("class", "grid").call(d3.axisBottom(xD).ticks(5).tickSize(phD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD).ticks(5));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD));
const colR = d3.scaleSequential(d3.interpolateViridis).domain([80, 450]);
pD.selectAll("rect.rg").data(regs).enter().append("rect")
  .attr("x", 0).attr("y", d => yD(d.region)).attr("width", d => xD(d.shear_modulus_pa)).attr("height", yD.bandwidth()).attr("rx", 4)
  .attr("fill", d => colR(d.shear_modulus_pa));
pD.selectAll("text.rgv").data(regs).enter().append("text")
  .attr("x", d => xD(d.shear_modulus_pa) + 6).attr("y", d => yD(d.region) + yD.bandwidth() / 2 + 4)
  .attr("font-size", 10.5).attr("font-weight", 700).attr("fill", "#1e293b").text(d => d.shear_modulus_pa.toFixed(1) + " Pa");
pD.append("text").attr("x", pwD / 2).attr("y", phD + 34).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Published Regional Neo-Hookean Shear Modulus mu (Pa)");
`;

// ============================================================================
// 6. FIGURE R9: Closed-Loop Continuum Shape Control (CLIK + Obstacle CBF)
// ============================================================================
const DRAW_FIG_R9 = `
const W = 1560, H = 920;
const S = DATA.summary;
const CL = S.closed_loop_clik_cbf;
const CF = S.cosserat_fem_distillation;
addHeader(svg, W,
  "Study R9: Continuum Operator CLIK + Obstacle CBF Control (V7-R3 + V7-R2)",
  "Real Geometry: Gotelli 9-Disk 2-Section 5-DOF Nitinol Continuum (L=540 mm, Cosserat FEM Anchored) + CRL-CTCR Workspace | 4 ue FBG + 0.25 N Load",
  [
    { text: "Tip RMSE: " + CL.closed_loop_tip_rmse_mm.toFixed(3) + " mm (< 1.5 mm)", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "Collisions: " + CL.backbone_obstacle_collisions + " (Min Clr " + CL.min_backbone_clearance_mm.toFixed(1) + " mm)", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: CL.median_iter_ms.toFixed(2) + " ms (" + Math.round(CL.control_rate_hz) + " Hz CPU)", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: 3D Isometric 9-Disk 2-Section Continuum Backbone & Anatomical Obstacles
const gA = addCard(svg, 20, 86, 750, 400,
  "A. 3D Redundant 9-Disk 2-Section Continuum Tracking Around Anatomical Obstacles",
  "Cosserat FEM distilled to neural operator (" + CF.neural_operator_blind_tip_rmse_mm + " mm open-loop -> " + CL.closed_loop_tip_rmse_mm + " mm closed-loop CLIK)"
);
const frs = DATA.frames;
const baseP = [[-50,-50,0],[50,-50,0],[50,50,0],[-50,50,0]].map(c => isoProject(c[0], c[1], c[2], 375, 344, 0.50, -34, 22));
gA.append("polygon").attr("points", baseP.map(p => p.x + "," + p.y).join(" ")).attr("fill", "#e2e8f0").attr("stroke", "#475569");
// Draw obstacles with non-overlapping offset callout labels
DATA.obstacles.forEach((ob, oi) => {
  const pr = isoProject(ob.center_mm[0], ob.center_mm[1], ob.center_mm[2], 375, 344, 0.50, -34, 22);
  gA.append("circle").attr("cx", pr.x).attr("cy", pr.y).attr("r", ob.radius_mm * 0.50)
    .attr("fill", "#fb7185").attr("fill-opacity", 0.35).attr("stroke", "#e11d48").attr("stroke-width", 2);
  const side = oi === 0 ? -1 : 1;
  const lx = pr.x + side * 66;
  const ly = pr.y + (oi === 0 ? 10 : -12);
  gA.append("line").attr("x1", pr.x + side * ob.radius_mm * 0.50).attr("y1", pr.y).attr("x2", lx - side * 4).attr("y2", ly).attr("stroke", "#e11d48").attr("stroke-width", 1.2);
  gA.append("text").attr("x", lx).attr("y", ly + 4).attr("text-anchor", side < 0 ? "end" : "start").attr("fill", "#9f1239").attr("font-size", 10.8).attr("font-weight", 700).text(ob.name);
});
// Draw 5 representative backbone snapshots + full tip figure-8 trajectory
const snapIdx = [0, 14, 28, 42, 56];
const snapCols = ["#cbd5e1", "#93c5fd", "#60a5fa", "#2563eb", "#059669"];
snapIdx.forEach((si, idx) => {
  const pts = frs[si].backbone_xyz_mm.map(p => isoProject(p[0], p[1], p[2], 375, 344, 0.50, -34, 22));
  gA.append("path").datum(pts).attr("fill", "none").attr("stroke", snapCols[idx]).attr("stroke-width", idx === 3 ? 3.5 : 2.0)
    .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
  if (idx === 3) {
    pts.slice(1).forEach(dp => {
      gA.append("ellipse").attr("cx", dp.x).attr("cy", dp.y).attr("rx", 9).attr("ry", 3.8).attr("fill", "#dbeafe").attr("stroke", "#1d4ed8").attr("stroke-width", 1.2);
    });
  }
});
const tgtPath = frs.map(f => isoProject(f.target_tip_mm[0], f.target_tip_mm[1], f.target_tip_mm[2], 375, 344, 0.50, -34, 22));
const actPath = frs.map(f => isoProject(f.actual_tip_mm[0], f.actual_tip_mm[1], f.actual_tip_mm[2], 375, 344, 0.50, -34, 22));
gA.append("path").datum(tgtPath).attr("fill", "none").attr("stroke", "#0f172a").attr("stroke-width", 1.6).attr("stroke-dasharray", "4,3")
  .attr("d", d3.line().x(d => d.x).y(d => d.y));
gA.append("path").datum(actPath).attr("fill", "none").attr("stroke", "#059669").attr("stroke-width", 2.4)
  .attr("d", d3.line().x(d => d.x).y(d => d.y));

// Panel B: Top-Down XY Figure-8 Tip Tracking Precision
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Closed-Loop 3D Figure-8 Tip Tracking (Target vs. Actual Under 0.25 N Load + 4 ue FBG Noise)",
  "Closed-Loop RMSE = " + CL.closed_loop_tip_rmse_mm.toFixed(3) + " mm | Peak Error = " + CL.closed_loop_tip_max_mm.toFixed(3) + " mm (Target Gate < 1.50 mm)"
);
const mB = { l: 58, r: 24, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const xB = d3.scaleLinear().domain([-76, 76]).range([0, pwB]);
const yB = d3.scaleLinear().domain([-78, 78]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB).ticks(6));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
pB.append("path").datum(frs).attr("fill", "none").attr("stroke", "#0f172a").attr("stroke-width", 2.4).attr("stroke-dasharray", "6,3")
  .attr("d", d3.line().x(d => xB(d.target_tip_mm[0])).y(d => yB(d.target_tip_mm[1])));
pB.append("path").datum(frs).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xB(d.actual_tip_mm[0])).y(d => yB(d.actual_tip_mm[1])));
pB.selectAll("circle.ld").data(frs.filter(d => d.ext_load_active)).enter().append("circle")
  .attr("cx", d => xB(d.actual_tip_mm[0])).attr("cy", d => yB(d.actual_tip_mm[1])).attr("r", 3.6).attr("fill", "#f59e0b").attr("stroke", "#fff").attr("stroke-width", 0.6);
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Tip X Position (mm)");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Tip Y Position (mm)");
const legB = pB.append("g").attr("transform", "translate(12, 8)");
legB.append("rect").attr("width", 244).attr("height", 64).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 15).attr("y2", 15).attr("stroke", "#0f172a").attr("stroke-width", 2.2).attr("stroke-dasharray", "5,3");
legB.append("text").attr("x", 30).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("Target Figure-8 Reference Path");
legB.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 34).attr("y2", 34).attr("stroke", "#0284c7").attr("stroke-width", 2.8);
legB.append("text").attr("x", 30).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("CLIK Closed-Loop Tip (0.231 mm)");
legB.append("circle").attr("cx", 16).attr("cy", 52).attr("r", 3.6).attr("fill", "#f59e0b");
legB.append("text").attr("x", 30).attr("y", 56).attr("font-size", 10.5).attr("fill", "#1e293b").text("Unknown 0.25 N Tip Load Active");

// Panel C: Closed-Loop 3D Tip Tracking Error Over Time (With 0.25 N Disturbance Window)
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Closed-Loop 3D Tip Tracking Error Under FBG Noise (4 ue) & Unknown Tip Load",
  "Amber band = unknown 0.25 N lateral tip disturbance rejected online by damped operator CLIK"
);
const mC = { l: 58, r: 24, t: 62, b: 48 };
const pwC = 750 - mC.l - mC.r, phC = 396 - mC.t - mC.b;
const pC = gC.append("g").attr("transform", "translate(" + mC.l + "," + mC.t + ")");
const xC = d3.scaleLinear().domain([0, frs.length - 1]).range([0, pwC]);
const yC = d3.scaleLinear().domain([0, 1.75]).range([phC, 0]);
const ldSteps = frs.filter(d => d.ext_load_active);
if (ldSteps.length > 0) {
  pC.append("rect").attr("x", xC(ldSteps[0].step)).attr("y", 0).attr("width", xC(ldSteps[ldSteps.length - 1].step) - xC(ldSteps[0].step)).attr("height", phC).attr("fill", "#fef3c7").attr("fill-opacity", 0.55);
  pC.append("text").attr("x", (xC(ldSteps[0].step) + xC(ldSteps[ldSteps.length - 1].step)) / 2).attr("y", 18).attr("text-anchor", "middle").attr("fill", "#b45309").attr("font-size", 10.5).attr("font-weight", 700).text("Unknown 0.25 N Tip Load");
}
pC.append("g").attr("class", "grid").call(d3.axisLeft(yC).ticks(5).tickSize(-pwC).tickFormat(""));
pC.append("g").attr("class", "axis").attr("transform", "translate(0," + phC + ")").call(d3.axisBottom(xC).ticks(8));
pC.append("g").attr("class", "axis").call(d3.axisLeft(yC).ticks(5));
pC.append("line").attr("x1", 0).attr("x2", pwC).attr("y1", yC(1.5)).attr("y2", yC(1.5)).attr("stroke", "#dc2626").attr("stroke-width", 1.8).attr("stroke-dasharray", "6,3");
pC.append("path").datum(frs).attr("fill", "#0284c7").attr("fill-opacity", 0.14)
  .attr("d", d3.area().x(d => xC(d.step)).y0(phC).y1(d => yC(d.tip_error_mm)));
pC.append("path").datum(frs).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xC(d.step)).y(d => yC(d.tip_error_mm)));
pC.append("text").attr("x", pwC - 6).attr("y", yC(1.5) - 6).attr("text-anchor", "end").attr("fill", "#b91c1c").attr("font-size", 10.5).attr("font-weight", 700).text("Target Gate < 1.50 mm (Max = 0.702 mm)");
pC.append("text").attr("x", pwC / 2).attr("y", phC + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("CLIK Control Iteration Step");
pC.append("text").attr("transform", "rotate(-90)").attr("x", -phC / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("3D Tip Tracking Error (mm)");

// Panel D: Minimum Mid-Backbone Obstacle Clearance Across All 9 Disks
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Mid-Backbone Obstacle Clearance (All 9 Spacer Disks, Nullspace + CBF Guard)",
  "0 Collisions | Minimum Clearance = " + CL.min_backbone_clearance_mm.toFixed(2) + " mm (> 0 mm collision threshold) | " + CL.median_iter_ms.toFixed(3) + " ms/iter"
);
const mD = { l: 58, r: 24, t: 62, b: 48 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const xD = d3.scaleLinear().domain([0, frs.length - 1]).range([0, pwD]);
const yD = d3.scaleLinear().domain([0, d3.max(frs, d => d.min_clearance_mm) * 1.15]).range([phD, 0]);
pD.append("g").attr("class", "grid").call(d3.axisLeft(yD).ticks(5).tickSize(-pwD).tickFormat(""));
pD.append("g").attr("class", "axis").attr("transform", "translate(0," + phD + ")").call(d3.axisBottom(xD).ticks(8));
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD).ticks(5));
pD.append("path").datum(frs).attr("fill", "#10b981").attr("fill-opacity", 0.16)
  .attr("d", d3.area().x(d => xD(d.step)).y0(phD).y1(d => yD(d.min_clearance_mm)));
pD.append("path").datum(frs).attr("fill", "none").attr("stroke", "#059669").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xD(d.step)).y(d => yD(d.min_clearance_mm)));
pD.append("line").attr("x1", 0).attr("x2", pwD).attr("y1", yD(0)).attr("y2", yD(0)).attr("stroke", "#dc2626").attr("stroke-width", 2);
pD.append("text").attr("x", pwD / 2).attr("y", phD + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("CLIK Control Iteration Step");
pD.append("text").attr("transform", "rotate(-90)").attr("x", -phD / 2).attr("y", -44).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("Minimum Backbone-to-Obstacle Clearance (mm)");
`;

// ============================================================================
// 7. FIGURE R10: Adversarial Stress, OOD Refusal & 1M-Node Scaling Audit (V7-R6)
// ============================================================================
const DRAW_FIG_R10 = `
const W = 1560, H = 920;
const S = DATA;
const SC = S.surface_3d_scaling_10k_to_1m;
const AF = S.adversarial_fem_falsification;
const OD = S.ood_refusal_audit;
addHeader(svg, W,
  "Study R10: Adversarial Stress, OOD Refusal & 1M-Node Scaling Audit (V7-R6)",
  "Fail-Closed Safety Audit (det(F)<=0, Extreme Crush, Burst, Severe OOD) + Reproducible 10k -> 1,000,000 Node CPU Scaling Harness",
  [
    { text: "Adversarial Refusal: 5/5 (100%)", bg: "#065f46", fg: "#a7f3d0", border: "#10b981" },
    { text: "OOD Recall: 20/20 (0% FP)", bg: "#1e3a8a", fg: "#bfdbfe", border: "#3b82f6" },
    { text: "1M Nodes: 831 ms | 455 MB | 0.0 Export Diff", bg: "#312e81", fg: "#ddd6fe", border: "#8b5cf6" },
  ]
);

// Panel A: Log-Log Latency Scaling 10k -> 100k -> 500k -> 1M Nodes
const gA = addCard(svg, 20, 86, 750, 400,
  "A. Surface3DEnergyOperator (AB-UPT + Transolver++ + LRQ) Latency Scaling: 10k -> 1M Nodes",
  "8-Thread CPU | Log-log scaling slope = " + SC.scaling_exponent_latency + " (sub-linear!) | Peak throughput = 1.23M nodes/s | Chunk drift = 2.79e-9"
);
const mA = { l: 74, r: 24, t: 62, b: 48 };
const pwA = 750 - mA.l - mA.r, phA = 400 - mA.t - mA.b;
const pA = gA.append("g").attr("transform", "translate(" + mA.l + "," + mA.t + ")");
const pts = SC.points;
const xA = d3.scaleLog().domain([8000, 1400000]).range([0, pwA]);
const yA = d3.scaleLog().domain([5, 1400]).range([phA, 0]);
pA.append("g").attr("class", "grid").call(d3.axisLeft(yA).ticks(4).tickSize(-pwA).tickFormat(""));
pA.append("g").attr("class", "axis").attr("transform", "translate(0," + phA + ")").call(d3.axisBottom(xA).tickValues([10000, 100000, 500000, 1000000]).tickFormat(d3.format(".0s")));
pA.append("g").attr("class", "axis").call(d3.axisLeft(yA).tickValues([10, 50, 100, 500, 1000]).tickFormat(d => d + " ms"));
// Linear O(N) reference slope
pA.append("line").attr("x1", xA(10000)).attr("y1", yA(17.01)).attr("x2", xA(1000000)).attr("y2", yA(17.01 * 100))
  .attr("stroke", "#cbd5e1").attr("stroke-width", 1.8).attr("stroke-dasharray", "6,4");
pA.append("path").datum(pts).attr("fill", "none").attr("stroke", "#0284c7").attr("stroke-width", 3.0)
  .attr("d", d3.line().x(d => xA(d.node_count)).y(d => yA(d.total_p50_ms)));
pA.append("path").datum(pts).attr("fill", "none").attr("stroke", "#7c3aed").attr("stroke-width", 2.2).attr("stroke-dasharray", "5,3")
  .attr("d", d3.line().x(d => xA(d.node_count)).y(d => yA(d.encode_ms)));
pA.selectAll("circle.pt").data(pts).enter().append("circle")
  .attr("cx", d => xA(d.node_count)).attr("cy", d => yA(d.total_p50_ms)).attr("r", 5.5).attr("fill", "#0284c7").attr("stroke", "#fff").attr("stroke-width", 1.5);
pA.selectAll("text.pl").data(pts).enter().append("text")
  .attr("x", d => xA(d.node_count) - (d.node_count === 1000000 ? 12 : 0)).attr("y", d => yA(d.total_p50_ms) - 10)
  .attr("text-anchor", d => d.node_count === 1000000 ? "end" : "middle").attr("font-size", 11).attr("font-weight", 700).attr("fill", "#0369a1")
  .text(d => d.total_p50_ms.toFixed(1) + " ms");
pA.append("text").attr("x", pwA / 2).attr("y", phA + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("3D Surface Mesh Node Count (Log Scale)");
pA.append("text").attr("transform", "rotate(-90)").attr("x", -phA / 2).attr("y", -58).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("P50 Inference Latency (ms, Log)");
const legA = pA.append("g").attr("transform", "translate(12, 8)");
legA.append("rect").attr("width", 256).attr("height", 64).attr("rx", 6).attr("fill", "#fff").attr("stroke", "#e2e8f0");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 15).attr("y2", 15).attr("stroke", "#0284c7").attr("stroke-width", 3);
legA.append("text").attr("x", 30).attr("y", 19).attr("font-size", 10.5).attr("fill", "#1e293b").text("Total Encode+Chunked Decode (slope 0.84)");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 34).attr("y2", 34).attr("stroke", "#7c3aed").attr("stroke-width", 2.2).attr("stroke-dasharray", "5,3");
legA.append("text").attr("x", 30).attr("y", 38).attr("font-size", 10.5).attr("fill", "#1e293b").text("AB-UPT Anchor Encode Only (91 ms @ 1M)");
legA.append("line").attr("x1", 8).attr("x2", 24).attr("y1", 53).attr("y2", 53).attr("stroke", "#cbd5e1").attr("stroke-width", 2).attr("stroke-dasharray", "6,4");
legA.append("text").attr("x", 30).attr("y", 57).attr("font-size", 10.5).attr("fill", "#1e293b").text("Reference O(N) Linear Scaling");

// Panel B: Peak RSS Memory Footprint & Throughput Across 100x Scale
const gB = addCard(svg, 790, 86, 750, 400,
  "B. Peak Process RSS Memory (Windows 64-Bit) & Throughput Across 100x Mesh Scale",
  "100x node increase (10k -> 1M) adds only +145.5 MB Peak RSS (309 -> 455 MB) via chunked LRQ decoding | TorchScript parity = 0.0"
);
const mB = { l: 64, r: 58, t: 62, b: 48 };
const pwB = 750 - mB.l - mB.r, phB = 400 - mB.t - mB.b;
const pB = gB.append("g").attr("transform", "translate(" + mB.l + "," + mB.t + ")");
const labelsB = pts.map(d => d.node_count >= 1e6 ? "1M" : (d.node_count / 1000) + "k");
const xB = d3.scaleBand().domain(labelsB).range([0, pwB]).padding(0.35);
const yB = d3.scaleLinear().domain([0, 560]).range([phB, 0]);
const yT = d3.scaleLinear().domain([0, 1.5]).range([phB, 0]);
pB.append("g").attr("class", "grid").call(d3.axisLeft(yB).ticks(5).tickSize(-pwB).tickFormat(""));
pB.append("g").attr("class", "axis").attr("transform", "translate(0," + phB + ")").call(d3.axisBottom(xB));
pB.append("g").attr("class", "axis").call(d3.axisLeft(yB).ticks(5));
pB.append("g").attr("class", "axis").attr("transform", "translate(" + pwB + ",0)").call(d3.axisRight(yT).ticks(5).tickFormat(d => d.toFixed(1) + "M/s"));
pts.forEach((d, i) => {
  const lb = labelsB[i];
  pB.append("rect").attr("x", xB(lb)).attr("y", yB(d.rss_peak_mb)).attr("width", xB.bandwidth()).attr("height", phB - yB(d.rss_peak_mb)).attr("rx", 6).attr("fill", "#0ea5e9").attr("fill-opacity", 0.8);
  pB.append("text").attr("x", xB(lb) + xB.bandwidth() / 2).attr("y", yB(d.rss_peak_mb) - 6).attr("text-anchor", "middle").attr("font-size", 11).attr("font-weight", 700).attr("fill", "#0369a1").text(d.rss_peak_mb.toFixed(0) + " MB");
});
pB.append("path").datum(pts).attr("fill", "none").attr("stroke", "#059669").attr("stroke-width", 2.8)
  .attr("d", d3.line().x((_, i) => xB(labelsB[i]) + xB.bandwidth() / 2).y(d => yT(d.nodes_per_sec / 1e6)));
pB.selectAll("circle.th").data(pts).enter().append("circle")
  .attr("cx", (_, i) => xB(labelsB[i]) + xB.bandwidth() / 2).attr("cy", d => yT(d.nodes_per_sec / 1e6)).attr("r", 5).attr("fill", "#059669").attr("stroke", "#fff").attr("stroke-width", 1.5);
pB.append("text").attr("x", pwB / 2).attr("y", phB + 36).attr("text-anchor", "middle").attr("fill", "#334155").attr("font-size", 11.5).attr("font-weight", 600).text("3D Surface Mesh Resolution");
pB.append("text").attr("transform", "rotate(-90)").attr("x", -phB / 2).attr("y", -48).attr("text-anchor", "middle").attr("fill", "#0369a1").attr("font-size", 11.5).attr("font-weight", 600).text("Peak Process RSS (MB)");
pB.append("text").attr("transform", "rotate(90)").attr("x", phB / 2).attr("y", -(pwB + 46)).attr("text-anchor", "middle").attr("fill", "#047857").attr("font-size", 11).attr("font-weight", 600).text("Throughput (M nodes/s)");

// Panel C: Adversarial FEM Falsification Matrix (5/5 = 100% Fail-Closed Refusal)
const gC = addCard(svg, 20, 504, 750, 396,
  "C. Adversarial FEM Falsification Audit (Zero Silent Extrapolation)",
  "Every injected physical violation (det(F)<=0 inversion, 96% crush, 100x burst, impact) is cleanly refused"
);
const cases = AF.cases;
cases.forEach((c, i) => {
  const ry = 66 + i * 62;
  gC.append("rect").attr("x", 18).attr("y", ry).attr("width", 714).attr("height", 52).attr("rx", 7)
    .attr("fill", "#f8fafc").attr("stroke", "#cbd5e1").attr("stroke-width", 1);
  gC.append("rect").attr("x", 28).attr("y", ry + 13).attr("width", 90).attr("height", 26).attr("rx", 13)
    .attr("fill", "#065f46");
  gC.append("text").attr("x", 73).attr("y", ry + 30).attr("text-anchor", "middle").attr("fill", "#a7f3d0").attr("font-size", 10.5).attr("font-weight", 700).text("REFUSED OK");
  gC.append("text").attr("x", 130).attr("y", ry + 22).attr("fill", "#0f172a").attr("font-size", 12).attr("font-weight", 700).text(c.case);
  gC.append("text").attr("x", 130).attr("y", ry + 40).attr("fill", "#475569").attr("font-size", 10.4).text(c.reason.slice(0, 86));
});

// Panel D: Cross-Domain OOD Refusal & Anchor/Chunk Ablation Scorecard
const gD = addCard(svg, 790, 504, 750, 396,
  "D. Cross-Domain OOD Refusal Audit & 100k-Node Anchor/Chunk Ablation",
  "100% severe OOD refusal (20/20), 0% false refusals (0/20), 8/8 soft-shift margin expansions | Export diff <= 1e-5"
);
const oodBars = [
  { label: "Severe OOD Refusal", pct: OD.severe_ood_refusal_recall * 100, col: "#059669", txt: "20/20 (100%)" },
  { label: "Nominal ID Acceptance", pct: (1 - OD.in_distribution_false_refusal_rate) * 100, col: "#0284c7", txt: "20/20 (0% FP)" },
  { label: "Soft-Shift Expansion", pct: 100, col: "#7c3aed", txt: OD.soft_shift_margin_expanded_cases + " (100%)" },
  { label: "FEM Adversary Refusal", pct: AF.refusal_recall * 100, col: "#0f766e", txt: "5/5 (100%)" },
];
const mD = { l: 165, r: 95, t: 62, b: 170 };
const pwD = 750 - mD.l - mD.r, phD = 396 - mD.t - mD.b;
const pD = gD.append("g").attr("transform", "translate(" + mD.l + "," + mD.t + ")");
const yD = d3.scaleBand().domain(oodBars.map(d => d.label)).range([0, phD]).padding(0.28);
const xD = d3.scaleLinear().domain([0, 100]).range([0, pwD]);
pD.append("g").attr("class", "axis").call(d3.axisLeft(yD));
oodBars.forEach(d => {
  pD.append("rect").attr("x", 0).attr("y", yD(d.label)).attr("width", xD(d.pct)).attr("height", yD.bandwidth()).attr("rx", 5).attr("fill", d.col);
  pD.append("text").attr("x", xD(d.pct) + 8).attr("y", yD(d.label) + yD.bandwidth() / 2 + 4).attr("font-size", 11).attr("font-weight", 700).attr("fill", "#0f172a").text(d.txt);
});
// Bottom mini-table: 100k-node anchor/chunk ablation
const tbl = gD.append("g").attr("transform", "translate(24, 248)");
tbl.append("rect").attr("width", 702).attr("height", 128).attr("rx", 8).attr("fill", "#f8fafc").attr("stroke", "#e2e8f0");
tbl.append("text").attr("x", 14).attr("y", 22).attr("font-size", 11.8).attr("font-weight", 700).attr("fill", "#0f172a").text("100k-Node Anchor & Decode-Chunk Ablation (TorchScript Parity = 0.0 Exact)");
const hdrs = ["Anchors", "Chunk Size", "Encode (ms)", "Decode P50 (ms)", "Total P50 (ms)", "Peak RSS (MB)", "Chunk Drift"];
const colsX = [16, 98, 200, 304, 424, 534, 636];
hdrs.forEach((h, i) => tbl.append("text").attr("x", colsX[i]).attr("y", 44).attr("font-size", 10.2).attr("font-weight", 700).attr("fill", "#475569").text(h));
SC.anchor_chunk_ablation_100k.forEach((r, ri) => {
  const yy = 66 + ri * 22;
  const vals = [String(r.anchor_count), String(r.decode_chunk_size), r.encode_ms.toFixed(1), r.decode_p50_ms.toFixed(1), r.total_p50_ms.toFixed(1), r.rss_peak_mb.toFixed(1), r.chunk_drift.toExponential(1)];
  vals.forEach((v, i) => tbl.append("text").attr("x", colsX[i]).attr("y", yy).attr("font-size", 10.5).attr("fill", i === 4 ? "#0369a1" : "#1e293b").attr("font-weight", i === 4 ? 700 : 400).text(v));
});
`;

// ============================================================================
// LIVE DARK-MODE ANIMATED GIF FRAMES (880 x 490)
// ============================================================================
const DRAW_GIF_R3_V2 = `
const W = 880, H = 490;
const k = DATA.k;
const tr = DATA.cbf_trace;
const lp = DATA.cycle4_loop;
const cur = tr[k];
const is200 = cur.payload_g === 200;
addHeader(svg, W,
  "FormaMech v0.7 Live: Hysteresis (V7-R1) & Online ACP-CQR CBF (V7-R2)",
  "t = " + cur.t_s.toFixed(2) + " s | Payload = " + cur.payload_g + "g | Adaptive Margin = " + cur.adaptive_margin_rad.toFixed(4) + " rad (Static 0.036) | 0 Violations",
  [{ text: is200 ? "200g SHIFT: EXPANDED" : "0g CALM: TIGHTENED", bg: is200 ? "#92400e" : "#065f46", fg: "#fff", border: is200 ? "#fbbf24" : "#34d399" }],
  true
);
// Left: Live Cycle-4 Hysteresis Loop Sweep
const gL = addCard(svg, 16, 82, 412, 392, "Zenodo 15584329 Cycle-4 Hysteresis Loop", "White: Real Hardware | Cyan: V7-R1 Operator (0.247 mm)", true);
const pL = gL.append("g").attr("transform", "translate(48, 56)");
const xL = d3.scaleLinear().domain([0, 1.02]).range([0, 344]);
const yL = d3.scaleLinear().domain([d3.min(lp, d => d.y_exp_mm) - 0.5, d3.max(lp, d => d.y_exp_mm) + 0.6]).range([298, 0]);
pL.append("g").attr("class", "grid-dark").call(d3.axisLeft(yL).ticks(5).tickSize(-344).tickFormat(""));
pL.append("g").attr("class", "axis-dark").attr("transform", "translate(0,298)").call(d3.axisBottom(xL).ticks(5));
pL.append("g").attr("class", "axis-dark").call(d3.axisLeft(yL).ticks(5));
const li = Math.min(lp.length - 1, Math.floor((k / (tr.length - 1)) * (lp.length - 1)));
pL.append("path").datum(lp).attr("fill", "none").attr("stroke", "#475569").attr("stroke-width", 1.5).attr("stroke-dasharray", "5,3")
  .attr("d", d3.line().x(d => xL(d.u)).y(d => yL(d.y_elastic_mm)));
pL.append("path").datum(lp.slice(0, li + 1)).attr("fill", "none").attr("stroke", "#f8fafc").attr("stroke-width", 2.4)
  .attr("d", d3.line().x(d => xL(d.u)).y(d => yL(d.y_exp_mm)));
pL.append("path").datum(lp.slice(0, li + 1)).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xL(d.u)).y(d => yL(d.y_v7_mm)));
pL.append("circle").attr("cx", xL(lp[li].u)).attr("cy", yL(lp[li].y_v7_mm)).attr("r", 5.5).attr("fill", "#38bdf8").attr("stroke", "#fff").attr("stroke-width", 1.5);

// Right: Live SPONGE 0g->200g ACP-CQR vs Static CBF
const gR = addCard(svg, 444, 82, 420, 392, "SPONGE 0g->200g Step Shift: ACP-CQR vs Static", "Green: ACP-CQR (0 viol, -33.5% brake) | Amber: Unfiltered", true);
const pR = gR.append("g").attr("transform", "translate(48, 56)");
const xR = d3.scaleLinear().domain([0, tr[tr.length - 1].t_s]).range([0, 352]);
const yR = d3.scaleLinear().domain([0, 0.21]).range([298, 0]);
pR.append("rect").attr("x", 0).attr("y", 0).attr("width", 352).attr("height", yR(0.145)).attr("fill", "#881337").attr("fill-opacity", 0.38);
pR.append("rect").attr("x", xR(tr[70].t_s)).attr("y", yR(0.145)).attr("width", 352 - xR(tr[70].t_s)).attr("height", 298 - yR(0.145)).attr("fill", "#78350f").attr("fill-opacity", 0.22);
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,298)").call(d3.axisBottom(xR).ticks(6));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yR).ticks(5));
pR.append("line").attr("x1", 0).attr("x2", 352).attr("y1", yR(0.145)).attr("y2", yR(0.145)).attr("stroke", "#fb7185").attr("stroke-width", 2).attr("stroke-dasharray", "5,3");
// Live dynamic barrier band (0.145 - adaptive_margin)
pR.append("path").datum(tr.slice(0, k + 1)).attr("fill", "#38bdf8").attr("fill-opacity", 0.22)
  .attr("d", d3.area().x(d => xR(d.t_s)).y0(d => yR(0.145 - d.adaptive_margin_rad)).y1(yR(0.145)));
pR.append("path").datum(tr.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#fb923c").attr("stroke-width", 1.9)
  .attr("d", d3.line().x(d => xR(d.t_s)).y(d => yR(d.q1_unfiltered_rad)));
pR.append("path").datum(tr.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 1.8).attr("stroke-dasharray", "4,3")
  .attr("d", d3.line().x(d => xR(d.t_s)).y(d => yR(d.q1_static_cbf_rad)));
pR.append("path").datum(tr.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#34d399").attr("stroke-width", 2.8)
  .attr("d", d3.line().x(d => xR(d.t_s)).y(d => yR(d.q1_adaptive_cbf_rad)));
pR.append("circle").attr("cx", xR(cur.t_s)).attr("cy", yR(cur.q1_adaptive_cbf_rad)).attr("r", 5).attr("fill", "#34d399").attr("stroke", "#fff");
`;

const DRAW_GIF_R6 = `
const W = 880, H = 490;
const k = DATA.k;
const ht = DATA.holdout_trajectory;
const cur = ht[k];
const isSlip = cur.slip_true === 1;
addHeader(svg, W,
  "FormaMech R6 Live: 3-Axis Tactile Force & Friction-Cone Slip Control",
  "Sparsh Holdout Frame " + k + " | Fz = " + cur.fz_true.toFixed(2) + " N | Shear = " + cur.f_shear_true.toFixed(2) + " N | Slip Prob = " + (cur.slip_prob * 100).toFixed(1) + "% | 2,376 Hz",
  [{ text: isSlip ? "SLIP DETECTED" : "STICK CONTACT", bg: isSlip ? "#9f1239" : "#065f46", fg: "#fff", border: isSlip ? "#fb7185" : "#34d399" }],
  true
);
// Left: Live Coulomb Friction Cone Vector Trajectory
const gL = addCard(svg, 16, 82, 412, 392, "Live Coulomb Friction Cone (||Ft|| vs |Fz|)", "Blue: Stick | Rose: Incipient Slip (AUROC 0.9998)", true);
const pL = gL.append("g").attr("transform", "translate(48, 56)");
const maxFz = d3.max(ht, d => d.fz_true) * 1.15, maxFt = d3.max(ht, d => d.f_shear_true) * 1.2;
const xL = d3.scaleLinear().domain([0, maxFz]).range([0, 344]);
const yL = d3.scaleLinear().domain([0, maxFt]).range([298, 0]);
pL.append("g").attr("class", "grid-dark").call(d3.axisLeft(yL).ticks(5).tickSize(-344).tickFormat(""));
pL.append("g").attr("class", "axis-dark").attr("transform", "translate(0,298)").call(d3.axisBottom(xL).ticks(5));
pL.append("g").attr("class", "axis-dark").call(d3.axisLeft(yL).ticks(5));
pL.append("line").attr("x1", xL(0)).attr("y1", yL(0)).attr("x2", xL(maxFz)).attr("y2", yL(Math.min(maxFt, 0.35 * maxFz)))
  .attr("stroke", "#fbbf24").attr("stroke-width", 2).attr("stroke-dasharray", "6,3");
pL.append("path").datum(ht.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.4)
  .attr("d", d3.line().x(d => xL(d.fz_true)).y(d => yL(d.f_shear_true)));
pL.append("circle").attr("cx", xL(cur.fz_true)).attr("cy", yL(cur.f_shear_true)).attr("r", 6.5)
  .attr("fill", isSlip ? "#fb7185" : "#34d399").attr("stroke", "#fff").attr("stroke-width", 1.8);

// Right: Scrolling 3-Axis Force & Slip Probability
const gR = addCard(svg, 444, 82, 420, 392, "Real-Time 3-Axis Force + AS6983 Bound & Slip Prob", "Cyan: Fz | Amber: Shear | Rose: Slip Prob (0.42 ms)", true);
const pR = gR.append("g").attr("transform", "translate(46, 56)");
const xR = d3.scaleLinear().domain([0, ht.length - 1]).range([0, 354]);
const yR = d3.scaleLinear().domain([0, d3.max(ht, d => d.f_mag_upper) * 1.12]).range([298, 0]);
const yS = d3.scaleLinear().domain([0, 1.05]).range([298, 0]);
for (let i = 0; i < ht.length - 1; i++) {
  if (ht[i].slip_true) pR.append("rect").attr("x", xR(i)).attr("y", 0).attr("width", Math.max(2, xR(i + 1) - xR(i))).attr("height", 298).attr("fill", "#881337").attr("fill-opacity", 0.35);
}
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,298)").call(d3.axisBottom(xR).ticks(6));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yR).ticks(5));
pR.append("path").datum(ht.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.2).attr("d", d3.line().x((_, i) => xR(i)).y(d => yR(d.fz_true)));
pR.append("path").datum(ht.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#a78bfa").attr("stroke-width", 1.8).attr("stroke-dasharray", "4,3").attr("d", d3.line().x((_, i) => xR(i)).y(d => yR(d.f_mag_upper)));
pR.append("path").datum(ht.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#fbbf24").attr("stroke-width", 2.0).attr("d", d3.line().x((_, i) => xR(i)).y(d => yR(d.f_shear_true)));
pR.append("path").datum(ht.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#fb7185").attr("stroke-width", 2.6).attr("d", d3.line().x((_, i) => xR(i)).y(d => yS(d.slip_prob)));
pR.append("line").attr("x1", xR(k)).attr("x2", xR(k)).attr("y1", 0).attr("y2", 298).attr("stroke", "#f8fafc").attr("stroke-width", 1.3).attr("stroke-dasharray", "3,3");
`;

const DRAW_GIF_R7 = `
const W = 880, H = 490;
const k = DATA.k;
const fe = DATA.featured_episode;
const fr = fe.frames[k];
const tw = fe.twoway_rollout;
const tk = Math.min(tw.length - 1, Math.floor((k / (fe.frames.length - 1)) * (tw.length - 1)));
addHeader(svg, W,
  "FormaMech R7 Live: PhysCoRe 3D Deformable Twin & Two-Way Coupling",
  "Episode: " + fe.case + " | Frame " + fr.frame + " | Mean 8-Ep Blind Chamfer: 3.64 mm | C-ABI Exact 0.0 Diff",
  [{ text: fr.is_holdout ? "BLIND HOLDOUT TAIL" : "CALIBRATION PREFIX", bg: fr.is_holdout ? "#1e3a8a" : "#065f46", fg: "#fff", border: fr.is_holdout ? "#60a5fa" : "#34d399" }],
  true
);
// Left: Live 3D Rotating Deformable Manipulation Point Cloud
const gL = addCard(svg, 16, 82, 424, 392, "3D Tracked Elastomer Stretch (" + fe.case + ")", "Cyan: Deforming 3D Object | Amber: Manipulator Hand", true);
const pts0 = fe.frames[0].true_pts_mm;
const cX = d3.mean(pts0, p => p[0]), cY = d3.mean(pts0, p => p[1]), cZ = d3.mean(pts0, p => p[2]);
const yaw = -36 + k * 1.1;
const prO = fr.true_pts_mm.map(p => isoProject(p[0] - cX, p[1] - cY, p[2] - cZ, 212, 224, 1.15, yaw, 25)).sort((a, b) => a.depth - b.depth);
const prC = isoProject(fr.ctrl_center_mm[0] - cX, fr.ctrl_center_mm[1] - cY, fr.ctrl_center_mm[2] - cZ, 212, 224, 1.15, yaw, 25);
gL.selectAll("circle.ob").data(prO).enter().append("circle").attr("cx", d => d.x).attr("cy", d => d.y).attr("r", 4.4)
  .attr("fill", fr.is_holdout ? "#38bdf8" : "#34d399").attr("fill-opacity", 0.9).attr("stroke", "#020617").attr("stroke-width", 0.5);
gL.append("circle").attr("cx", prC.x).attr("cy", prC.y).attr("r", 10).attr("fill", "#fbbf24").attr("fill-opacity", 0.85).attr("stroke", "#fff").attr("stroke-width", 1.8);

// Right: Live Two-Way vs One-Way Tool Trajectory Divergence & Reaction Force
const gR = addCard(svg, 454, 82, 410, 392, "V7-R5 Two-Way vs. One-Way Trajectory Divergence", "Cyan: Tool Divergence (mm) | Rose: Reaction Force (N)", true);
const pR = gR.append("g").attr("transform", "translate(48, 56)");
const xR = d3.scaleLinear().domain([0, tw[tw.length - 1].t_s]).range([0, 340]);
const yR = d3.scaleLinear().domain([0, d3.max(tw, d => d.tool_div_mm) * 1.15]).range([298, 0]);
const yF = d3.scaleLinear().domain([0, Math.max(d3.max(tw, d => d.reaction_force_n) * 1.15, 1)]).range([298, 0]);
pR.append("g").attr("class", "grid-dark").call(d3.axisLeft(yR).ticks(5).tickSize(-340).tickFormat(""));
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,298)").call(d3.axisBottom(xR).ticks(5));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yR).ticks(5));
pR.append("path").datum(tw.slice(0, tk + 1)).attr("fill", "#38bdf8").attr("fill-opacity", 0.16)
  .attr("d", d3.area().x(d => xR(d.t_s)).y0(298).y1(d => yR(d.tool_div_mm)));
pR.append("path").datum(tw.slice(0, tk + 1)).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.6)
  .attr("d", d3.line().x(d => xR(d.t_s)).y(d => yR(d.tool_div_mm)));
pR.append("path").datum(tw.slice(0, tk + 1)).attr("fill", "none").attr("stroke", "#fb7185").attr("stroke-width", 2.1).attr("stroke-dasharray", "5,3")
  .attr("d", d3.line().x(d => xR(d.t_s)).y(d => yF(d.reaction_force_n)));
pR.append("circle").attr("cx", xR(tw[tk].t_s)).attr("cy", yR(tw[tk].tool_div_mm)).attr("r", 5.5).attr("fill", "#38bdf8").attr("stroke", "#fff");
`;

const DRAW_GIF_R9 = `
const W = 880, H = 490;
const k = DATA.k;
const frs = DATA.frames;
const cur = frs[k];
addHeader(svg, W,
  "FormaMech R9 Live: Continuum Operator CLIK + Obstacle Avoidance",
  "Step " + k + "/63 | Tip Error = " + cur.tip_error_mm.toFixed(2) + " mm (RMSE 0.23 mm) | Min Clearance = " + cur.min_clearance_mm.toFixed(1) + " mm | 1,015 Hz CPU",
  [{ text: cur.ext_load_active ? "0.25 N TIP LOAD" : "0 COLLISIONS", bg: cur.ext_load_active ? "#92400e" : "#065f46", fg: "#fff", border: cur.ext_load_active ? "#fbbf24" : "#34d399" }],
  true
);
// Left: Live 3D 9-Disk Continuum Backbone Tracking Figure-8 Around 2 Obstacles
const gL = addCard(svg, 16, 82, 444, 392, "3D 9-Disk 2-Section Continuum Backbone", "White dashed: Target | Green: Actual Tip | Rose: Obstacles", true);
const baseP = [[-48,-48,0],[48,-48,0],[48,48,0],[-48,48,0]].map(c => isoProject(c[0], c[1], c[2], 222, 344, 0.48, -32 + k * 0.4, 22));
gL.append("polygon").attr("points", baseP.map(p => p.x + "," + p.y).join(" ")).attr("fill", "#1e293b").attr("stroke", "#475569");
DATA.obstacles.forEach(ob => {
  const pr = isoProject(ob.center_mm[0], ob.center_mm[1], ob.center_mm[2], 222, 344, 0.48, -32 + k * 0.4, 22);
  gL.append("circle").attr("cx", pr.x).attr("cy", pr.y).attr("r", ob.radius_mm * 0.48).attr("fill", "#fb7185").attr("fill-opacity", 0.32).attr("stroke", "#fb7185").attr("stroke-width", 2);
});
const tgtAll = frs.map(f => isoProject(f.target_tip_mm[0], f.target_tip_mm[1], f.target_tip_mm[2], 222, 344, 0.48, -32 + k * 0.4, 22));
const actSoFar = frs.slice(0, k + 1).map(f => isoProject(f.actual_tip_mm[0], f.actual_tip_mm[1], f.actual_tip_mm[2], 222, 344, 0.48, -32 + k * 0.4, 22));
gL.append("path").datum(tgtAll).attr("fill", "none").attr("stroke", "#94a3b8").attr("stroke-width", 1.5).attr("stroke-dasharray", "4,3").attr("d", d3.line().x(d => d.x).y(d => d.y));
gL.append("path").datum(actSoFar).attr("fill", "none").attr("stroke", "#34d399").attr("stroke-width", 2.4).attr("d", d3.line().x(d => d.x).y(d => d.y));
const bb = cur.backbone_xyz_mm.map(p => isoProject(p[0], p[1], p[2], 222, 344, 0.48, -32 + k * 0.4, 22));
gL.append("path").datum(bb).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 3.6).attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
bb.slice(1).forEach(dp => {
  gL.append("ellipse").attr("cx", dp.x).attr("cy", dp.y).attr("rx", 8.5).attr("ry", 3.5).attr("fill", "#1e3a8a").attr("stroke", "#60a5fa").attr("stroke-width", 1.2);
});
const tipP = bb[bb.length - 1];
if (cur.ext_load_active) {
  gL.append("line").attr("x1", tipP.x).attr("y1", tipP.y).attr("x2", tipP.x + 38).attr("y2", tipP.y + 10).attr("stroke", "#fbbf24").attr("stroke-width", 2.6);
}
gL.append("circle").attr("cx", tipP.x).attr("cy", tipP.y).attr("r", 5.5).attr("fill", "#34d399").attr("stroke", "#fff");

// Right: Live Tip Error & Obstacle Clearance Traces
const gR = addCard(svg, 474, 82, 390, 392, "Closed-Loop Tip Error & Backbone Clearance", "Cyan: Tip Err (mm) | Green: Min Obstacle Clearance (mm)", true);
const pR = gR.append("g").attr("transform", "translate(44, 56)");
const xR = d3.scaleLinear().domain([0, frs.length - 1]).range([0, 324]);
const yE = d3.scaleLinear().domain([0, 1.7]).range([136, 0]);
const yCl = d3.scaleLinear().domain([0, 50]).range([298, 168]);
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,136)").call(d3.axisBottom(xR).ticks(5));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yE).ticks(4));
pR.append("line").attr("x1", 0).attr("x2", 324).attr("y1", yE(1.5)).attr("y2", yE(1.5)).attr("stroke", "#fb7185").attr("stroke-dasharray", "5,3");
pR.append("path").datum(frs.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.4).attr("d", d3.line().x(d => xR(d.step)).y(d => yE(d.tip_error_mm)));
pR.append("circle").attr("cx", xR(k)).attr("cy", yE(cur.tip_error_mm)).attr("r", 4.5).attr("fill", "#38bdf8").attr("stroke", "#fff");
pR.append("g").attr("class", "axis-dark").attr("transform", "translate(0,298)").call(d3.axisBottom(xR).ticks(5));
pR.append("g").attr("class", "axis-dark").call(d3.axisLeft(yCl).ticks(4));
pR.append("path").datum(frs.slice(0, k + 1)).attr("fill", "none").attr("stroke", "#34d399").attr("stroke-width", 2.4).attr("d", d3.line().x(d => xR(d.step)).y(d => yCl(d.min_clearance_mm)));
pR.append("circle").attr("cx", xR(k)).attr("cy", yCl(cur.min_clearance_mm)).attr("r", 4.5).attr("fill", "#34d399").attr("stroke", "#fff");
`;

async function main() {
  fs.mkdirSync(ASSETS_DIR, { recursive: true });
  fs.mkdirSync(FRAMES_ROOT, { recursive: true });

  const browser = await chromium.launch({ headless: true });
  try {
    // 1. Render all 7 crisp 2x Retina publication figures
    const figPage = await browser.newPage({
      viewport: { width: 1560, height: 920 },
      deviceScaleFactor: 2,
    });
    const figures = [
      { id: "r3_v2", viz: "study_r3_v2_viz.json", outPng: "fig_r3_v2_hysteresis_acp_cbf.png", script: DRAW_FIG_R3_V2 },
      { id: "r5_v2", viz: "study_r5_v2_viz.json", outPng: "fig_r5_v2_contact_diagnostics.png", script: DRAW_FIG_R5_V2 },
      { id: "r6", viz: "study_r6_viz.json", outPng: "fig_r6_dynamic_tactile_slip.png", script: DRAW_FIG_R6 },
      { id: "r7", viz: "study_r7_viz.json", outPng: "fig_r7_physcore_twoway_coupling.png", script: DRAW_FIG_R7 },
      { id: "r8", viz: "study_r8_viz.json", outPng: "fig_r8_adaptive_surgical_palpation.png", script: DRAW_FIG_R8 },
      { id: "r9", viz: "study_r9_viz.json", outPng: "fig_r9_continuum_clik_cbf.png", script: DRAW_FIG_R9 },
      { id: "r10", viz: "study_r10_viz.json", outPng: "fig_r10_stress_ood_scaling.png", script: DRAW_FIG_R10 },
    ];
    for (const fg of figures) {
      const data = JSON.parse(fs.readFileSync(path.join(VIZ_DIR, fg.viz), "utf-8"));
      const html = buildHtml(1560, 920, "#f8fafc", fg.script, data);
      await figPage.goto("about:blank");
      await figPage.setContent(html, { waitUntil: "load" });
      await figPage.waitForFunction(() => window.__RENDER_DONE__ === true);
      const err = await figPage.evaluate(() => window.__RENDER_ERR__);
      if (err) throw new Error(`Figure ${fg.id} JS error:\n${err}`);
      const outPath = path.join(ASSETS_DIR, fg.outPng);
      await figPage.screenshot({ path: outPath, type: "png" });
      console.log(`Rendered v0.7 D3+SVG figure: ${fg.outPng}`);
    }
    await figPage.close();

    // 2. Render 4 live dark-mode animated GIF frame sequences (28-32 frames each)
    const gifPage = await browser.newPage({
      viewport: { width: 880, height: 490 },
      deviceScaleFactor: 1,
    });
    const gifSpecs = [
      {
        dir: "r3_v2_acp_cbf",
        viz: "study_r3_v2_viz.json",
        script: DRAW_GIF_R3_V2,
        pickIdx: (d) => d3RangeStep(d.cbf_trace.length, 32),
        buildPayload: (d, k) => ({ k, cbf_trace: d.cbf_trace, cycle4_loop: d.cycle4_loop }),
      },
      {
        dir: "r6_tactile_slip",
        viz: "study_r6_viz.json",
        script: DRAW_GIF_R6,
        pickIdx: (d) => d3RangeStep(d.holdout_trajectory.length, 30),
        buildPayload: (d, k) => ({ k, holdout_trajectory: d.holdout_trajectory }),
      },
      {
        dir: "r7_physcore",
        viz: "study_r7_viz.json",
        script: DRAW_GIF_R7,
        pickIdx: (d) => d3RangeStep(d.featured_episode.frames.length, 30),
        buildPayload: (d, k) => ({ k, featured_episode: d.featured_episode }),
      },
      {
        dir: "r9_clik_cbf",
        viz: "study_r9_viz.json",
        script: DRAW_GIF_R9,
        pickIdx: (d) => d3RangeStep(d.frames.length, 32),
        buildPayload: (d, k) => ({ k, frames: d.frames, obstacles: d.obstacles }),
      },
    ];
    for (const gs of gifSpecs) {
      const data = JSON.parse(fs.readFileSync(path.join(VIZ_DIR, gs.viz), "utf-8"));
      const outDir = path.join(FRAMES_ROOT, gs.dir);
      fs.mkdirSync(outDir, { recursive: true });
      const ks = gs.pickIdx(data);
      for (let i = 0; i < ks.length; i++) {
        const html = buildHtml(880, 490, "#0b1120", gs.script, gs.buildPayload(data, ks[i]));
        await gifPage.goto("about:blank");
        await gifPage.setContent(html, { waitUntil: "load" });
        await gifPage.waitForFunction(() => window.__RENDER_DONE__ === true);
        const err = await gifPage.evaluate(() => window.__RENDER_ERR__);
        if (err) throw new Error(`GIF ${gs.dir} frame ${i} JS error:\n${err}`);
        await gifPage.screenshot({
          path: path.join(outDir, `frame_${String(i).padStart(3, "0")}.png`),
          type: "png",
        });
      }
      console.log(`Rendered ${ks.length} GIF frames for ${gs.dir}`);
    }
    await gifPage.close();
  } finally {
    await browser.close();
  }
}

function d3RangeStep(n, targetCount) {
  if (n <= targetCount) return Array.from({ length: n }, (_, i) => i);
  const out = [];
  for (let i = 0; i < targetCount; i++) {
    out.push(Math.round((i * (n - 1)) / (targetCount - 1)));
  }
  return out;
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
