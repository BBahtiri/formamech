/**
 * Broad-public, intuitive LinkedIn GIF renderer (D3.js + custom SVG + Playwright)
 * for FormaMech v0.6 Robotics Physics AI validation studies (R1-R5).
 *
 * Renders 100% real verified study data from `.formamech-runs/robotics_studies/study_r*_viz.json`
 * as clear, mobile-readable 16:9 (960x540) visual stories with continuous shaded 3D meshes,
 * isometric 3D shaded solid blocks, plain-English telemetry callouts, and zero jargon overload.
 *
 * Outputs PNG frames to `.formamech-runs/robotics_studies/linkedin_frames/` for:
 *   1. hero_5in1         (all 5 studies in one 12s LinkedIn showcase reel)
 *   2. r3_safety_shield  (standalone soft arm safety wall: Standard AI vs FormaMech Shield)
 *   3. r1_robot_touch    (standalone 3D shaded silicone fingertip pressing & sliding)
 *   4. r4_brain_palpation (standalone 3D shaded human brain Cortex vs 5x softer Corpus Callosum)
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
const FRAMES_ROOT = path.join(VIZ_DIR, "linkedin_frames");
const D3_SOURCE = fs.readFileSync(D3_UMD_PATH, "utf-8");

const W = 960;
const H = 540;

function buildHtml(drawScript, payload) {
  const safeJson = JSON.stringify(payload).replace(/<\/script>/gi, "<\\/script>");
  return `<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  html, body {
    width: ${W}px; height: ${H}px; overflow: hidden;
    background: #0b1120;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
    -webkit-font-smoothing: antialiased;
  }
  svg { display: block; width: ${W}px; height: ${H}px; }
  .axis-dk path, .axis-dk line { stroke: #334155; stroke-width: 1.1; }
  .axis-dk text { fill: #94a3b8; font-size: 11px; font-weight: 500; }
</style>
<script>${D3_SOURCE}</script>
</head>
<body>
<svg id="canvas" viewBox="0 0 ${W} ${H}" width="${W}" height="${H}"></svg>
<script>
window.__DONE__ = false;
window.__ERR__ = null;
try {
(() => {
const DATA = ${safeJson};
const svg = d3.select("#canvas");
const W = ${W}, H = ${H};

// Background subtle radial depth
const defs = svg.append("defs");
const bgGrad = defs.append("linearGradient").attr("id", "bgG").attr("x1", "0%").attr("y1", "0%").attr("x2", "100%").attr("y2", "100%");
bgGrad.append("stop").attr("offset", "0%").attr("stop-color", "#0f172a");
bgGrad.append("stop").attr("offset", "100%").attr("stop-color", "#090d1a");
svg.append("rect").attr("width", W).attr("height", H).attr("fill", "url(#bgG)");

// Metallic sphere probe gradient
const metalG = defs.append("radialGradient").attr("id", "metalSph").attr("cx", "32%").attr("cy", "28%");
metalG.append("stop").attr("offset", "0%").attr("stop-color", "#f8fafc");
metalG.append("stop").attr("offset", "45%").attr("stop-color", "#94a3b8");
metalG.append("stop").attr("offset", "85%").attr("stop-color", "#334155");
metalG.append("stop").attr("offset", "100%").attr("stop-color", "#0f172a");

// Soft red hazard hatch pattern
const pat = defs.append("pattern").attr("id", "dangerHatch").attr("width", 12).attr("height", 12).attr("patternTransform", "rotate(45)").attr("patternUnits", "userSpaceOnUse");
pat.append("rect").attr("width", 12).attr("height", 12).attr("fill", "#7f1d1d").attr("fill-opacity", 0.28);
pat.append("line").attr("x1", 0).attr("y1", 0).attr("x2", 0).attr("y2", 12).attr("stroke", "#f43f5e").attr("stroke-opacity", 0.35).attr("stroke-width", 3);

function iso(x, y, z, cx, cy, sc, yawDeg = -36, pitchDeg = 28) {
  const yaw = (yawDeg * Math.PI) / 180;
  const pitch = (pitchDeg * Math.PI) / 180;
  const rx = x * Math.cos(yaw) - y * Math.sin(yaw);
  const ry = x * Math.sin(yaw) + y * Math.cos(yaw);
  const px = rx;
  const py = ry * Math.sin(pitch) - z * Math.cos(pitch);
  const depth = ry * Math.cos(pitch) + z * Math.sin(pitch);
  return { x: cx + px * sc, y: cy + py * sc, depth };
}

function drawChrome(tag, headline, subline, pillText, pillBg, pillBorder, sceneIdx = -1) {
  // Top header bar
  svg.append("rect").attr("x", 0).attr("y", 0).attr("width", W).attr("height", 80).attr("fill", "#0b1120").attr("fill-opacity", 0.92);
  svg.append("line").attr("x1", 0).attr("x2", W).attr("y1", 80).attr("y2", 80).attr("stroke", "#1e293b").attr("stroke-width", 1.5);
  // Category pill on top-left
  const tw = Math.max(88, tag.length * 6.8 + 18);
  svg.append("rect").attr("x", 24).attr("y", 12).attr("width", tw).attr("height", 21).attr("rx", 10.5)
    .attr("fill", "#1e293b").attr("stroke", "#38bdf8").attr("stroke-width", 1);
  svg.append("text").attr("x", 24 + tw / 2).attr("y", 26).attr("text-anchor", "middle")
    .attr("fill", "#38bdf8").attr("font-size", 10.5).attr("font-weight", 800).attr("letter-spacing", "0.5px").text(tag);
  // Headline + subline (compact width guaranteed clear of right status pill)
  svg.append("text").attr("x", 24 + tw + 12).attr("y", 28)
    .attr("fill", "#f8fafc").attr("font-size", 18.5).attr("font-weight", 800).text(headline);
  svg.append("text").attr("x", 24).attr("y", 59)
    .attr("fill", "#cbd5e1").attr("font-size", 12.2).attr("font-weight", 500).text(subline);
  // Right status pill
  if (pillText) {
    const pw = Math.max(132, pillText.length * 7.4 + 24);
    svg.append("rect").attr("x", W - 24 - pw).attr("y", 19).attr("width", pw).attr("height", 34).attr("rx", 17)
      .attr("fill", pillBg).attr("stroke", pillBorder).attr("stroke-width", 1.6);
    svg.append("text").attr("x", W - 24 - pw / 2).attr("y", 40.5).attr("text-anchor", "middle")
      .attr("fill", "#ffffff").attr("font-size", 12).attr("font-weight", 800).text(pillText);
  }
  // Bottom brand / scene progress strip
  svg.append("rect").attr("x", 0).attr("y", H - 36).attr("width", W).attr("height", 36).attr("fill", "#0b1120");
  svg.append("line").attr("x1", 0).attr("x2", W).attr("y1", H - 36).attr("y2", H - 36).attr("stroke", "#1e293b").attr("stroke-width", 1);
  svg.append("text").attr("x", 24).attr("y", H - 13).attr("fill", "#64748b").attr("font-size", 11.5).attr("font-weight", 700)
    .text("FormaMech v0.6 (MIT) · Validated on 5 Real Robotics Datasets");
  if (sceneIdx >= 0) {
    const labels = ["1 Touch", "2 Continuum", "3 Safe Arm", "4 Brain Tissue", "5 Assembly"];
    let sx = W - 24;
    for (let i = labels.length - 1; i >= 0; i--) {
      const active = i === sceneIdx;
      const bw = labels[i].length * 6.5 + 16;
      sx -= bw;
      svg.append("rect").attr("x", sx).attr("y", H - 28).attr("width", bw).attr("height", 20).attr("rx", 10)
        .attr("fill", active ? "#0284c7" : "#1e293b").attr("stroke", active ? "#38bdf8" : "#334155");
      svg.append("text").attr("x", sx + bw / 2).attr("y", H - 14).attr("text-anchor", "middle")
        .attr("fill", active ? "#ffffff" : "#94a3b8").attr("font-size", 10).attr("font-weight", 700).text(labels[i]);
      sx -= 6;
    }
  }
}

function drawCard(x, y, w, h, title, sub = "") {
  const g = svg.append("g").attr("transform", "translate(" + x + "," + y + ")");
  g.append("rect").attr("width", w).attr("height", h).attr("rx", 14)
    .attr("fill", "#111827").attr("stroke", "#1e293b").attr("stroke-width", 1.4);
  if (title) {
    g.append("text").attr("x", 18).attr("y", 28).attr("fill", "#f1f5f9").attr("font-size", 14.5).attr("font-weight", 700).text(title);
  }
  if (sub) {
    g.append("text").attr("x", 18).attr("y", 46).attr("fill", "#94a3b8").attr("font-size", 11.5).text(sub);
  }
  return g;
}

function drawStatTile(g, x, y, w, h, label, value, accent = "#38bdf8", sub = "") {
  g.append("rect").attr("x", x).attr("y", y).attr("width", w).attr("height", h).attr("rx", 10)
    .attr("fill", "#0f172a").attr("stroke", "#1e293b").attr("stroke-width", 1.2);
  g.append("rect").attr("x", x).attr("y", y + 8).attr("width", 4).attr("height", h - 16).attr("rx", 2).attr("fill", accent);
  g.append("text").attr("x", x + 14).attr("y", y + 20).attr("fill", "#94a3b8").attr("font-size", 10.5).attr("font-weight", 700).text(label);
  g.append("text").attr("x", x + 14).attr("y", y + 44).attr("fill", "#f8fafc").attr("font-size", 21).attr("font-weight", 800).text(value);
  if (sub) {
    g.append("text").attr("x", x + 14).attr("y", y + 60).attr("fill", accent).attr("font-size", 10.5).attr("font-weight", 600).text(sub);
  }
}

// Reconstruct a smooth 2D structured grid from TET10 top surface nodes and draw a continuous shaded 3D deformable block!
function drawShaded3DBlock(g, pts, vals, nxG, nyG, lx, ly, zTop, zExag, cx, cy, sc, colorScale, sideBaseFill, sideStroke, yaw = -36, pitch = 29) {
  // Snap each node to nearest (ix, iy) on [0..nxG-1] x [0..nyG-1]
  const grid = Array.from({ length: nxG }, () => Array(nyG).fill(null));
  for (let k = 0; k < pts.length; k++) {
    const p = pts[k];
    const ix = Math.max(0, Math.min(nxG - 1, Math.round((p[0] / lx) * (nxG - 1))));
    const iy = Math.max(0, Math.min(nyG - 1, Math.round((p[1] / ly) * (nyG - 1))));
    grid[ix][iy] = { x: p[0] - lx / 2, y: p[1] - ly / 2, z: (p[2] - zTop) * zExag, v: vals[k] };
  }
  // Fill any rare missing slot from nearest neighbor
  for (let ix = 0; ix < nxG; ix++) {
    for (let iy = 0; iy < nyG; iy++) {
      if (!grid[ix][iy]) {
        grid[ix][iy] = { x: (ix / (nxG - 1) - 0.5) * lx, y: (iy / (nyG - 1) - 0.5) * ly, z: 0, v: 0 };
      }
    }
  }
  // Draw front-facing side walls of the solid elastomer/tissue block (y = +ly/2 and x = +lx/2 for yaw=-36)
  const zBot = -3.8;
  // Side 1: iy = nyG - 1 (front-left)
  for (let ix = 0; ix < nxG - 1; ix++) {
    const a = grid[ix][nyG - 1], b = grid[ix + 1][nyG - 1];
    const p0 = iso(a.x, a.y, a.z, cx, cy, sc, yaw, pitch);
    const p1 = iso(b.x, b.y, b.z, cx, cy, sc, yaw, pitch);
    const p2 = iso(b.x, b.y, zBot, cx, cy, sc, yaw, pitch);
    const p3 = iso(a.x, a.y, zBot, cx, cy, sc, yaw, pitch);
    g.append("polygon").attr("points", [p0,p1,p2,p3].map(q => q.x.toFixed(1)+","+q.y.toFixed(1)).join(" "))
      .attr("fill", sideBaseFill).attr("fill-opacity", 0.92).attr("stroke", sideStroke).attr("stroke-width", 0.5);
  }
  // Side 2: ix = nxG - 1 (front-right)
  for (let iy = 0; iy < nyG - 1; iy++) {
    const a = grid[nxG - 1][iy], b = grid[nxG - 1][iy + 1];
    const p0 = iso(a.x, a.y, a.z, cx, cy, sc, yaw, pitch);
    const p1 = iso(b.x, b.y, b.z, cx, cy, sc, yaw, pitch);
    const p2 = iso(b.x, b.y, zBot, cx, cy, sc, yaw, pitch);
    const p3 = iso(a.x, a.y, zBot, cx, cy, sc, yaw, pitch);
    g.append("polygon").attr("points", [p0,p1,p2,p3].map(q => q.x.toFixed(1)+","+q.y.toFixed(1)).join(" "))
      .attr("fill", d3.color(sideBaseFill).darker(0.45)).attr("fill-opacity", 0.95).attr("stroke", sideStroke).attr("stroke-width", 0.5);
  }
  // Upsample top surface 2x bilinearly for silky-smooth dimple contours
  const rxN = (nxG - 1) * 2, ryN = (nyG - 1) * 2;
  function sample(u, v) {
    const fx = u * (nxG - 1), fy = v * (nyG - 1);
    const i0 = Math.min(nxG - 2, Math.floor(fx)), j0 = Math.min(nyG - 2, Math.floor(fy));
    const tx = fx - i0, ty = fy - j0;
    const c00 = grid[i0][j0], c10 = grid[i0+1][j0], c01 = grid[i0][j0+1], c11 = grid[i0+1][j0+1];
    const bilerp = (k) => (1-tx)*(1-ty)*c00[k] + tx*(1-ty)*c10[k] + (1-tx)*ty*c01[k] + tx*ty*c11[k];
    return { x: bilerp("x"), y: bilerp("y"), z: bilerp("z"), v: bilerp("v") };
  }
  const quads = [];
  for (let i = 0; i < rxN; i++) {
    for (let j = 0; j < ryN; j++) {
      const q00 = sample(i / rxN, j / ryN);
      const q10 = sample((i + 1) / rxN, j / ryN);
      const q11 = sample((i + 1) / rxN, (j + 1) / ryN);
      const q01 = sample(i / rxN, (j + 1) / ryN);
      const pr = [q00, q10, q11, q01].map(q => iso(q.x, q.y, q.z, cx, cy, sc, yaw, pitch));
      const avgV = 0.25 * (q00.v + q10.v + q11.v + q01.v);
      const avgD = 0.25 * (pr[0].depth + pr[1].depth + pr[2].depth + pr[3].depth);
      quads.push({ pr, avgV, avgD });
    }
  }
  quads.sort((a, b) => a.avgD - b.avgD);
  quads.forEach(q => {
    g.append("polygon")
      .attr("points", q.pr.map(p => p.x.toFixed(1) + "," + p.y.toFixed(1)).join(" "))
      .attr("fill", colorScale(q.avgV))
      .attr("stroke", "#0f172a").attr("stroke-opacity", 0.28).attr("stroke-width", 0.55);
  });
}

// Draw a 3D isometric shaded cube voxel (3 faces lit top/left/right)
function drawIsoVoxel(g, x, y, z, sz, cx, cy, sc, baseHex, yaw, pitch) {
  const h = sz * 0.5;
  const pts = [
    [-h,-h, h], [ h,-h, h], [ h, h, h], [-h, h, h], // top 0..3
    [-h,-h,-h], [ h,-h,-h], [ h, h,-h], [-h, h,-h], // bot 4..7
  ].map(([dx,dy,dz]) => iso(x+dx, y+dy, z+dz, cx, cy, sc, yaw, pitch));
  const cTop = d3.color(baseHex).brighter(0.35);
  const cLeft = d3.color(baseHex);
  const cRight = d3.color(baseHex).darker(0.55);
  const faces = [
    { idx: [0,1,2,3], fill: cTop },
    { idx: [3,2,6,7], fill: cLeft },
    { idx: [1,5,6,2], fill: cRight },
  ];
  faces.forEach(f => {
    g.append("polygon")
      .attr("points", f.idx.map(k => pts[k].x.toFixed(1)+","+pts[k].y.toFixed(1)).join(" "))
      .attr("fill", f.fill).attr("stroke", "#0f172a").attr("stroke-opacity", 0.35).attr("stroke-width", 0.45);
  });
}

// =========================================================================
// SCENE 1: ROBOT SENSE OF TOUCH (Study R1 - GelSight Mini + 3,997 CNC Probes)
// =========================================================================
function renderSceneR1(D, sceneIdx) {
  const F = D.frame, ALL = D.all_frames, S = D.summary;
  const isShear = F.shear_mm > 0.01;
  drawChrome(
    sceneIdx >= 0 ? "1 / 5 · TACTILE AI" : "STUDY R1 · ROBOT TOUCH",
    "Giving Robots a Real-Time Sense of Touch & Slip",
    "Real GelSight Mini silicone sensor (3,997 CNC physical probes) · 3D physics -> 796x/sec AI with 95% safety bounds",
    isShear ? "SLIP & SHEAR DETECTED" : "PRESSING INTO GEL: " + F.fz_n.toFixed(1) + " N",
    isShear ? "#92400e" : "#0369a1",
    isShear ? "#fbbf24" : "#38bdf8",
    sceneIdx
  );
  // Left card: Continuous shaded 3D silicone fingertip pad + descending glossy probe
  const gL = drawCard(24, 96, 488, 392, "3D Squishy Silicone Fingertip (Real 3D FEM)",
    isShear ? "Object sliding sideways -> shear force = " + F.fx_n.toFixed(1) + " N" : "Depth = " + F.depth_mm.toFixed(2) + " mm -> silicone bulges smoothly around tip");
  const vmMax = Math.max(...ALL[ALL.length - 1].surface_vm_mpa, 0.12);
  const colGel = d3.scaleSequential(t => d3.interpolateTurbo(0.12 + 0.85 * t)).domain([0, vmMax]);
  drawShaded3DBlock(gL, F.surface_pts, F.surface_vm_mpa, 7, 9, 18, 24, 4.0, 3.6, 244, 245, 11.4, colGel, "#1e3a5f", "#38bdf8", -38, 29);
  // Descending metallic sphere probe + shaft
  const probeZ = 3.2 - F.depth_mm * 2.1;
  const prC = iso(F.shear_mm * 6.0, 0, probeZ, 244, 245, 11.4, -38, 29);
  gL.append("line").attr("x1", prC.x).attr("y1", prC.y - 88).attr("x2", prC.x).attr("y2", prC.y - 26)
    .attr("stroke", "#94a3b8").attr("stroke-width", 10).attr("stroke-linecap", "round");
  gL.append("circle").attr("cx", prC.x).attr("cy", prC.y).attr("r", 30)
    .attr("fill", "url(#metalSph)").attr("stroke", "#e2e8f0").attr("stroke-width", 1.4);
  if (isShear) {
    // Clear sideways slide arrow
    gL.append("line").attr("x1", prC.x + 36).attr("y1", prC.y).attr("x2", prC.x + 92).attr("y2", prC.y)
      .attr("stroke", "#fbbf24").attr("stroke-width", 4).attr("stroke-linecap", "round");
    gL.append("polygon").attr("points", (prC.x+96)+","+prC.y+" "+(prC.x+82)+","+(prC.y-8)+" "+(prC.x+82)+","+(prC.y+8)).attr("fill", "#fbbf24");
    gL.append("text").attr("x", prC.x + 66).attr("y", prC.y - 14).attr("text-anchor", "middle").attr("fill", "#fde68a").attr("font-size", 11.5).attr("font-weight", 800).text("SLIP!");
  }
  // Colorbar caption
  gL.append("text").attr("x", 244).attr("y", 374).attr("text-anchor", "middle").attr("fill", "#94a3b8").attr("font-size", 11)
    .text("Color = Internal Silicone Stress (Blue = Relaxed -> Red = High Contact Pressure)");

  // Right top: Glowing 8x8 Robot Touch Pad
  const gR1 = drawCard(528, 96, 408, 224, "What the Robot Feels (8x8 Touch Pad)", "Live pressure hotspot + white slip direction arrows");
  function smoothGrid(grid) {
    const nx = grid.length, ny = grid[0].length;
    const out = Array.from({ length: nx }, () => Array(ny).fill(0));
    for (let sx = 0; sx < nx; sx++) for (let sy = 0; sy < ny; sy++) {
      if (Math.abs(grid[sx][sy]) < 1e-6) continue;
      let ws = 0;
      for (let tx = 0; tx < nx; tx++) for (let ty = 0; ty < ny; ty++) ws += Math.exp(-((tx-sx)**2 + (ty-sy)**2) / 2.7);
      for (let tx = 0; tx < nx; tx++) for (let ty = 0; ty < ny; ty++) out[tx][ty] += grid[sx][sy] * Math.exp(-((tx-sx)**2 + (ty-sy)**2) / 2.7) / ws;
    }
    return out;
  }
  const smFz = smoothGrid(F.taxel_fz), smFx = smoothGrid(F.taxel_fx);
  const maxT = Math.max(...smoothGrid(ALL[ALL.length - 1].taxel_fz).flat(), 0.05);
  const tCol = d3.scaleSequential(d3.interpolateInferno).domain([0, maxT]);
  const cs = 19, gx = 32, gy = 58;
  for (let ix = 0; ix < 8; ix++) {
    for (let iy = 0; iy < 8; iy++) {
      const v = smFz[ix][iy], fx = smFx[ix][iy];
      const x = gx + ix * cs, y = gy + iy * cs;
      gR1.append("rect").attr("x", x).attr("y", y).attr("width", cs - 2).attr("height", cs - 2).attr("rx", 3.5)
        .attr("fill", v > maxT * 0.02 ? tCol(v) : "#1e293b").attr("stroke", "#334155").attr("stroke-width", 0.5);
      if (Math.abs(fx) > maxT * 0.03) {
        const ln = Math.min(12, (fx / maxT) * 16);
        gR1.append("line").attr("x1", x + cs/2 - 1).attr("y1", y + cs/2 - 1).attr("x2", x + cs/2 - 1 + ln).attr("y2", y + cs/2 - 1)
          .attr("stroke", "#ffffff").attr("stroke-width", 2).attr("stroke-linecap", "round");
      }
    }
  }
  // Mini force readout beside 8x8 pad
  drawStatTile(gR1, 204, 62, 186, 68, "TOUCH SPEED", Math.round(S.surrogate_as6983_holdout.inference_hz) + " Hz", "#38bdf8", "1,493x faster than 3D FEM");
  drawStatTile(gR1, 204, 140, 186, 68, "LAB TESTS SAVED", S.active_test_reduction.tests_saved_pct + "% fewer", "#34d399", "Active AI test planner");

  // Right bottom: Live Real-Time Force Curve + Safety Bound
  const gR2 = drawCard(528, 334, 408, 154, "Real-Time Touch Force (Press -> Slide)", "");
  const p = gR2.append("g").attr("transform", "translate(46, 40)");
  const xS = d3.scaleLinear().domain([0, ALL.length - 1]).range([0, 340]);
  const yS = d3.scaleLinear().domain([0, d3.max(ALL, d => d.fz_n) * 1.18]).range([86, 0]);
  p.append("g").attr("class", "axis-dk").attr("transform", "translate(0,86)").call(d3.axisBottom(xS).ticks(4).tickFormat(""));
  p.append("g").attr("class", "axis-dk").call(d3.axisLeft(yS).ticks(3).tickFormat(d => d + " N"));
  const soFar = ALL.slice(0, F.frame + 1);
  p.append("path").datum(soFar).attr("fill", "#38bdf8").attr("fill-opacity", 0.14)
    .attr("d", d3.area().x(d => xS(d.frame)).y0(86).y1(d => yS(d.fz_n)));
  p.append("path").datum(soFar).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 3)
    .attr("d", d3.line().x(d => xS(d.frame)).y(d => yS(d.fz_n)));
  p.append("path").datum(soFar).attr("fill", "none").attr("stroke", "#fbbf24").attr("stroke-width", 2.5)
    .attr("d", d3.line().x(d => xS(d.frame)).y(d => yS(d.fx_n)));
  p.append("circle").attr("cx", xS(F.frame)).attr("cy", yS(F.fz_n)).attr("r", 5.5).attr("fill", "#38bdf8").attr("stroke", "#fff").attr("stroke-width", 1.5);
  p.append("text").attr("x", 10).attr("y", 12).attr("text-anchor", "start").attr("fill", "#38bdf8").attr("font-size", 11).attr("font-weight", 700).text("Normal Press: " + F.fz_n.toFixed(1) + " N");
  p.append("text").attr("x", 10).attr("y", 28).attr("text-anchor", "start").attr("fill", "#fbbf24").attr("font-size", 11).attr("font-weight", 700).text("Sideways Slip: " + F.fx_n.toFixed(1) + " N");
}

// =========================================================================
// SCENE 2: FLEXIBLE SURGICAL CATHETER / CONTINUUM ROBOT (Study R2)
// =========================================================================
function renderSceneR2(D, sceneIdx) {
  const F = D.frame, ALL = D.all_frames, S = D.summary;
  const hasContact = Math.abs(F.tip_force_n[1]) > 0.01;
  drawChrome(
    sceneIdx >= 0 ? "2 / 5 · CONTINUUM" : "STUDY R2 · SURGICAL SPINE",
    "Flexible Cable-Driven Robot: 0.71 mm Precision",
    "100,000 real EM-tracked poses (MIT) + 9-disk Nitinol spine · Optical fiber contact sensing",
    hasContact ? "TOUCH FELT: 0.25 N!" : "BENDING: " + F.bending_angle_deg.toFixed(0) + " deg",
    hasContact ? "#9f1239" : "#065f46",
    hasContact ? "#fb7185" : "#34d399",
    sceneIdx
  );
  // Left: Smooth 3D flexible tendon-driven robot with glowing cables & contact pulse
  const gL = drawCard(24, 96, 488, 392, "Flexible 9-Disk Continuum Spine (54 cm Nitinol)", "4 internal cables steer the tip smoothly in 3D space");
  const yaw = -32 + F.frame * 1.4;
  const bp = [[-50,-50,0],[50,-50,0],[50,50,0],[-50,50,0]].map(c => iso(c[0], c[1], c[2], 244, 344, 0.54, yaw, 22));
  gL.append("polygon").attr("points", bp.map(p => p.x+","+p.y).join(" ")).attr("fill", "#1e293b").attr("stroke", "#475569").attr("stroke-width", 1.5);
  // Glowing tip trajectory ribbon
  const trail = ALL.slice(0, F.frame + 1).map(fr => {
    const tp = fr.backbone_xyz[fr.backbone_xyz.length - 1];
    return iso(tp[0], tp[1], tp[2], 244, 344, 0.54, yaw, 22);
  });
  gL.append("path").datum(trail).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-opacity", 0.45).attr("stroke-width", 2.2).attr("stroke-dasharray", "4,4")
    .attr("d", d3.line().x(d => d.x).y(d => d.y));
  const bPts = F.backbone_xyz.map(p => iso(p[0], p[1], p[2], 244, 344, 0.54, yaw, 22));
  // Draw pulled outer tendon cables along left/right of disks
  const leftCable = bPts.map(p => ({ x: p.x - 11, y: p.y }));
  const rightCable = bPts.map(p => ({ x: p.x + 11, y: p.y }));
  gL.append("path").datum(leftCable).attr("fill", "none").attr("stroke", "#a78bfa").attr("stroke-opacity", 0.65).attr("stroke-width", 1.6).attr("d", d3.line().x(d=>d.x).y(d=>d.y).curve(d3.curveCatmullRom));
  gL.append("path").datum(rightCable).attr("fill", "none").attr("stroke", "#fbbf24").attr("stroke-opacity", 0.65).attr("stroke-width", 1.6).attr("d", d3.line().x(d=>d.x).y(d=>d.y).curve(d3.curveCatmullRom));
  // Main metallic backbone
  gL.append("path").datum(bPts).attr("fill", "none").attr("stroke", "#f8fafc").attr("stroke-width", 5.5).attr("stroke-linecap", "round")
    .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
  // 9 spacer disks
  bPts.slice(1).forEach((dp, i) => {
    gL.append("ellipse").attr("cx", dp.x).attr("cy", dp.y).attr("rx", 13).attr("ry", 5)
      .attr("fill", i === 8 ? "#38bdf8" : "#0ea5e9").attr("fill-opacity", i === 8 ? 0.85 : 0.35)
      .attr("stroke", "#7dd3fc").attr("stroke-width", 1.5);
  });
  if (hasContact) {
    const tp = bPts[bPts.length - 1];
    gL.append("circle").attr("cx", tp.x).attr("cy", tp.y).attr("r", 22).attr("fill", "#f43f5e").attr("fill-opacity", 0.25).attr("stroke", "#fb7185").attr("stroke-width", 2);
    gL.append("line").attr("x1", tp.x + 74).attr("y1", tp.y + 18).attr("x2", tp.x + 16).attr("y2", tp.y + 4)
      .attr("stroke", "#fb7185").attr("stroke-width", 4).attr("stroke-linecap", "round");
    gL.append("polygon").attr("points", (tp.x+10)+","+(tp.y+2)+" "+(tp.x+24)+","+(tp.y-4)+" "+(tp.x+21)+","+(tp.y+12)).attr("fill", "#fb7185");
    gL.append("rect").attr("x", tp.x + 76).attr("y", tp.y - 2).attr("width", 142).attr("height", 36).attr("rx", 8).attr("fill", "#881337").attr("stroke", "#fb7185");
    gL.append("text").attr("x", tp.x + 147).attr("y", tp.y + 14).attr("text-anchor", "middle").attr("fill", "#fff").attr("font-size", 11).attr("font-weight", 800).text("OBSTACLE TOUCH!");
    gL.append("text").attr("x", tp.x + 147).attr("y", tp.y + 28).attr("text-anchor", "middle").attr("fill", "#fecdd3").attr("font-size", 10).attr("font-weight", 600).text("Recovered to 0.0003 N");
  }

  // Right: Plain-English KPI tiles + live cable pulls
  const gR = drawCard(528, 96, 408, 392, "Real-World Accuracy & Live Cable Pull", "5,000 blind unseen robot poses (zero data leakage)");
  drawStatTile(gR, 18, 60, 180, 74, "AVERAGE TIP ERROR", S.ctcr_real_data_holdout.tip_mean_error_mm.toFixed(2) + " mm", "#34d399", "Sub-millimeter on real arm");
  drawStatTile(gR, 210, 60, 180, 74, "PREDICTION SPEED", Math.round(S.ctcr_real_data_holdout.inference_hz) + " Hz", "#38bdf8", "0.53 ms on standard CPU");
  drawStatTile(gR, 18, 146, 180, 74, "CERTIFIED BOUND", S.ctcr_real_data_holdout.as6983_quantile_mm.toFixed(2) + " mm", "#a78bfa", "AS6983 safety envelope");
  drawStatTile(gR, 210, 146, 180, 74, "FORCE SENSING ERR", "0.0003 N", "#fb7185", "From optical fiber strain");

  gR.append("text").attr("x", 18).attr("y", 248).attr("fill", "#e2e8f0").attr("font-size", 13).attr("font-weight", 700).text("Live Cable Pulling Forces (4 Tendons)");
  const names = ["Cable 1 (Right)", "Cable 2 (Front)", "Cable 3 (Left)", "Cable 4 (Back)"];
  const cols = ["#38bdf8", "#a78bfa", "#34d399", "#fbbf24"];
  F.tensions_n.forEach((tv, i) => {
    const y = 264 + i * 28;
    gR.append("text").attr("x", 18).attr("y", y + 14).attr("fill", "#cbd5e1").attr("font-size", 11.5).attr("font-weight", 600).text(names[i]);
    gR.append("rect").attr("x", 118).attr("y", y + 2).attr("width", 214).attr("height", 16).attr("rx", 8).attr("fill", "#1e293b");
    gR.append("rect").attr("x", 118).attr("y", y + 2).attr("width", Math.max(4, (tv / 8.5) * 214)).attr("height", 16).attr("rx", 8).attr("fill", cols[i]);
    gR.append("text").attr("x", 342).attr("y", y + 14).attr("fill", "#f8fafc").attr("font-size", 11.5).attr("font-weight", 700).text(tv.toFixed(1) + " N");
  });
}

// =========================================================================
// SCENE 3: SOFT PNEUMATIC ROBOT ARM SAFETY SHIELD (Study R3 - THE CROWD-PLEASER!)
// =========================================================================
function renderSceneR3(D, sceneIdx) {
  const F = D.frame, ALL = D.all_frames, M = D.summary.qp_mpc_conformal_cbf;
  const isOod = F.fsm_state === "SAFE_HOLD_OOD";
  const isCbf = F.fsm_state === "CBF_MODIFIED";
  const pill = isOod ? { t: "200g SHOCK -> SAFE HOLD", bg: "#92400e", bd: "#fbbf24" }
             : isCbf ? { t: "SHIELD BRAKING!", bg: "#0369a1", bd: "#38bdf8" }
             : { t: "SAFE · 0 BREACHES", bg: "#065f46", bd: "#34d399" };
  drawChrome(
    sceneIdx >= 0 ? "3 / 5 · SAFE CONTROL" : "STUDY R3 · SAFETY SHIELD",
    "Stopping a Soft Arm Before the Red Safety Wall",
    "Real 5-joint soft arm · Standard AI breaches wall 24x vs. FormaMech Safety Shield (0x)",
    pill.t, pill.bg, pill.bd, sceneIdx
  );

  // Left card: Big unmistakable animated 5-segment soft robot arm vs glowing RED NO-GO ZONE
  const gL = drawCard(24, 96, 462, 392, "Live Comparison: Standard AI vs. Physics Shield",
    "Orange dashed arm breaches safety zone · Emerald arm brakes automatically");

  // Compute exact wall angle & shade entire right region as RED NO-GO ZONE!
  const baseX = 185, baseY = 352;
  const angScale = 1.65; // slightly amplified visually so wall clearance is unmistakable on mobile!
  const wallAng = -Math.PI / 2 + M.q1_safety_limit_rad * angScale;
  const wallX2 = baseX + 290 * Math.cos(wallAng);
  const wallY2 = baseY + 290 * Math.sin(wallAng);
  // Shaded red hazard wedge to the right of the safety wall
  gL.append("polygon")
    .attr("points", baseX+","+baseY+" "+wallX2.toFixed(1)+","+wallY2.toFixed(1)+" 446,62 446,352")
    .attr("fill", "url(#dangerHatch)");
  gL.append("line").attr("x1", baseX).attr("y1", baseY).attr("x2", wallX2).attr("y2", wallY2)
    .attr("stroke", "#f43f5e").attr("stroke-width", 3).attr("stroke-dasharray", "8,5");
  gL.append("rect").attr("x", 308).attr("y", 84).attr("width", 124).attr("height", 44).attr("rx", 8)
    .attr("fill", "#881337").attr("fill-opacity", 0.88).attr("stroke", "#fb7185");
  gL.append("text").attr("x", 370).attr("y", 103).attr("text-anchor", "middle").attr("fill", "#fff").attr("font-size", 11.5).attr("font-weight", 800).text("NO-GO SAFETY");
  gL.append("text").attr("x", 370).attr("y", 119).attr("text-anchor", "middle").attr("fill", "#fecdd3").attr("font-size", 10.5).attr("font-weight", 700).text("ZONE (HUMAN)");

  function armPts(qArr) {
    let x = baseX, y = baseY, th = -Math.PI / 2;
    const pts = [{ x, y }];
    for (let i = 0; i < 5; i++) {
      th += qArr[i] * (i === 0 ? angScale : 0.75);
      x += 50 * Math.cos(th);
      y += 50 * Math.sin(th);
      pts.push({ x, y });
    }
    return pts;
  }
  const pUnf = armPts(F.q_unfiltered);
  const pSafe = armPts(F.q_safe);
  const unfBreach = F.q_unfiltered[0] > M.q1_safety_limit_rad;

  // Base pedestal
  gL.append("rect").attr("x", baseX - 48).attr("y", baseY - 2).attr("width", 96).attr("height", 18).attr("rx", 5).attr("fill", "#334155").attr("stroke", "#64748b");
  // Unfiltered orange ghost arm
  gL.append("path").datum(pUnf).attr("fill", "none").attr("stroke", unfBreach ? "#fb7185" : "#fb923c").attr("stroke-width", 4).attr("stroke-dasharray", "7,4")
    .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
  if (unfBreach) {
    const tu = pUnf[5];
    gL.append("circle").attr("cx", tu.x).attr("cy", tu.y).attr("r", 15).attr("fill", "#f43f5e").attr("fill-opacity", 0.35).attr("stroke", "#fb7185").attr("stroke-width", 2);
    gL.append("text").attr("x", tu.x + 14).attr("y", tu.y + 4).attr("fill", "#fda4af").attr("font-size", 11).attr("font-weight", 800).text("BREACH!");
  }
  // Safe FormaMech pneumatic bellows arm
  gL.append("path").datum(pSafe).attr("fill", "none").attr("stroke", isOod ? "#fbbf24" : "#34d399").attr("stroke-width", 10).attr("stroke-linecap", "round")
    .attr("d", d3.line().x(d => d.x).y(d => d.y).curve(d3.curveCatmullRom));
  pSafe.forEach((pt, i) => {
    gL.append("circle").attr("cx", pt.x).attr("cy", pt.y).attr("r", i === 5 ? 9 : 6)
      .attr("fill", i === 5 ? (isOod ? "#fbbf24" : "#38bdf8") : "#0f172a")
      .attr("stroke", isOod ? "#fde68a" : "#34d399").attr("stroke-width", 2.2);
  });
  if (isOod) {
    // Show sudden 200g payload weight badge on tip
    const ts = pSafe[5];
    gL.append("rect").attr("x", ts.x - 42).attr("y", ts.y - 44).attr("width", 84).attr("height", 26).attr("rx", 6).attr("fill", "#92400e").attr("stroke", "#fbbf24");
    gL.append("text").attr("x", ts.x).attr("y", ts.y - 27).attr("text-anchor", "middle").attr("fill", "#fff").attr("font-size", 10.5).attr("font-weight", 800).text("+200g SHOCK");
  }

  // Right top: Crystal-clear live Safety Margin Timeline
  const gR1 = drawCard(502, 96, 434, 240, "Arm Angle vs. Hard Safety Wall", "Green stays strictly below red wall · Orange crosses 24 times");
  const p = gR1.append("g").attr("transform", "translate(48, 56)");
  const xS = d3.scaleLinear().domain([0, ALL[ALL.length - 1].t_s]).range([0, 366]);
  const yS = d3.scaleLinear().domain([-0.14, 0.22]).range([156, 0]);
  p.append("rect").attr("x", 0).attr("y", 0).attr("width", 366).attr("height", yS(M.q1_safety_limit_rad)).attr("fill", "#881337").attr("fill-opacity", 0.42);
  p.append("g").attr("class", "axis-dk").attr("transform", "translate(0,156)").call(d3.axisBottom(xS).ticks(5).tickFormat(d => d.toFixed(1) + "s"));
  p.append("g").attr("class", "axis-dk").call(d3.axisLeft(yS).ticks(4));
  p.append("line").attr("x1", 0).attr("x2", 366).attr("y1", yS(M.q1_safety_limit_rad)).attr("y2", yS(M.q1_safety_limit_rad)).attr("stroke", "#f43f5e").attr("stroke-width", 2.4).attr("stroke-dasharray", "6,3");
  p.append("text").attr("x", 362).attr("y", yS(M.q1_safety_limit_rad) - 6).attr("text-anchor", "end").attr("fill", "#fda4af").attr("font-size", 10.5).attr("font-weight", 800).text("SAFETY LIMIT");
  const upTo = ALL.filter(d => d.step <= F.step);
  p.append("path").datum(upTo).attr("fill", "none").attr("stroke", "#fb923c").attr("stroke-width", 2.5).attr("stroke-dasharray", "5,3")
    .attr("d", d3.line().x(d => xS(d.t_s)).y(d => yS(d.q_unfiltered[0])));
  p.append("path").datum(upTo).attr("fill", "none").attr("stroke", "#34d399").attr("stroke-width", 3.4)
    .attr("d", d3.line().x(d => xS(d.t_s)).y(d => yS(d.q_safe[0])));
  p.append("circle").attr("cx", xS(F.t_s)).attr("cy", yS(F.q_safe[0])).attr("r", 5.5).attr("fill", "#34d399").attr("stroke", "#fff").attr("stroke-width", 1.5);

  // Right bottom: Big intuitive headline KPIs
  const gR2 = drawCard(502, 348, 434, 140, "", "");
  drawStatTile(gR2, 14, 14, 196, 112, "SAFETY BREACHES", "0 vs. 24", "#34d399", "FormaMech vs. Standard MPC");
  drawStatTile(gR2, 224, 14, 196, 112, "DECISION SPEED", Math.round(M.control_rate_hz) + " Hz", "#38bdf8", "0.47 ms/step (45x faster)");
}

// =========================================================================
// SCENE 4: ULTRA-SOFT HUMAN BRAIN TISSUE PALPATION (Study R4)
// =========================================================================
function renderSceneR4(D, sceneIdx) {
  const F = D.frame, S = D.summary;
  const pctLimit = Math.min(100, (F.peak_cortex_pa / F.tearing_limit_pa) * 100);
  drawChrome(
    sceneIdx >= 0 ? "4 / 5 · BRAIN TISSUE" : "STUDY R4 · BRAIN PHYSICS",
    "Palpating Real Human Brain Tissue Safely",
    "182 human brain specimens · Cortex (434 Pa) is 5x stiffer than Corpus Callosum (86 Pa)",
    "SAFE: " + (100 - pctLimit).toFixed(0) + "% MARGIN",
    "#065f46", "#34d399", sceneIdx
  );

  // Main upper area: Two continuous shaded 3D human brain tissue cushions side-by-side
  const gM = drawCard(24, 96, 912, 286, "Identical " + F.depth_mm.toFixed(2) + " mm Surgical Probe Palpation — 5x Stiffness Contrast Across Brain Regions",
    "Continuous 3D finite-strain hyperelastic mesh (real Zenodo 8095559 / 21793225 human brain parameters)");
  const colBrain = d3.scaleSequential(t => d3.interpolateInferno(0.16 + 0.82 * t)).domain([0, 420]);

  // Left block: Human Cortex (434 Pa)
  drawShaded3DBlock(gM, F.surface_pts, F.vm_cortex_pa, 7, 7, 16, 16, 6.0, 2.3, 225, 178, 10.6, colBrain, "#4c1d3d", "#f472b6", -36, 29);
  const prL = iso(0, 0, 2.8 - F.depth_mm * 2.0, 225, 178, 10.6, -36, 29);
  gM.append("line").attr("x1", prL.x).attr("y1", prL.y - 68).attr("x2", prL.x).attr("y2", prL.y - 18).attr("stroke", "#cbd5e1").attr("stroke-width", 7).attr("stroke-linecap", "round");
  gM.append("circle").attr("cx", prL.x).attr("cy", prL.y).attr("r", 20).attr("fill", "url(#metalSph)").attr("stroke", "#f8fafc").attr("stroke-width", 1.2);
  gM.append("text").attr("x", 225).attr("y", 252).attr("text-anchor", "middle").attr("fill", "#f8fafc").attr("font-size", 14).attr("font-weight", 800).text("Outer Brain Cortex (434 Pa)");
  gM.append("text").attr("x", 225).attr("y", 271).attr("text-anchor", "middle").attr("fill", "#fbbf24").attr("font-size", 12.5).attr("font-weight", 700).text("Peak Tissue Stress: " + F.peak_cortex_pa.toFixed(0) + " Pa");

  // Right block: Corpus Callosum (85.9 Pa - 5x softer!)
  drawShaded3DBlock(gM, F.surface_pts, F.vm_cc_pa, 7, 7, 16, 16, 6.0, 2.3, 685, 178, 10.6, colBrain, "#1e293b", "#64748b", -36, 29);
  const prR = iso(0, 0, 2.8 - F.depth_mm * 2.0, 685, 178, 10.6, -36, 29);
  gM.append("line").attr("x1", prR.x).attr("y1", prR.y - 68).attr("x2", prR.x).attr("y2", prR.y - 18).attr("stroke", "#cbd5e1").attr("stroke-width", 7).attr("stroke-linecap", "round");
  gM.append("circle").attr("cx", prR.x).attr("cy", prR.y).attr("r", 20).attr("fill", "url(#metalSph)").attr("stroke", "#f8fafc").attr("stroke-width", 1.2);
  gM.append("text").attr("x", 685).attr("y", 252).attr("text-anchor", "middle").attr("fill", "#f8fafc").attr("font-size", 14).attr("font-weight", 800).text("Inner Corpus Callosum (86 Pa · 5x Softer)");
  gM.append("text").attr("x", 685).attr("y", 271).attr("text-anchor", "middle").attr("fill", "#38bdf8").attr("font-size", 12.5).attr("font-weight", 700).text("Peak Tissue Stress: " + F.peak_cc_pa.toFixed(0) + " Pa");

  // Center VS badge
  gM.append("rect").attr("x", 412).attr("y", 148).attr("width", 88).attr("height", 32).attr("rx", 16).attr("fill", "#1e293b").attr("stroke", "#f59e0b");
  gM.append("text").attr("x", 456).attr("y", 169).attr("text-anchor", "middle").attr("fill", "#fde68a").attr("font-size", 12).attr("font-weight", 800).text("5x RATIO");

  // Bottom bar: Tissue Safety Meter + 3 crisp stat tiles
  const gB = drawCard(24, 394, 912, 94, "", "");
  gB.append("text").attr("x", 18).attr("y", 25).attr("fill", "#f1f5f9").attr("font-size", 12.5).attr("font-weight", 700)
    .text("Live Tissue-Tearing Safety Gauge (AS6983 Conservative Bound)");
  const bx = 18, by = 40, bw = 480;
  gB.append("rect").attr("x", bx).attr("y", by).attr("width", bw).attr("height", 22).attr("rx", 11).attr("fill", "#1e293b");
  const curW = Math.min(bw, (F.peak_cortex_pa / F.tearing_limit_pa) * bw);
  const cshW = Math.min(bw, ((F.peak_cortex_pa + 32) / F.tearing_limit_pa) * bw);
  gB.append("rect").attr("x", bx).attr("y", by).attr("width", cshW).attr("height", 22).attr("rx", 11).attr("fill", "#38bdf8").attr("fill-opacity", 0.35);
  gB.append("rect").attr("x", bx).attr("y", by).attr("width", curW).attr("height", 22).attr("rx", 11).attr("fill", pctLimit > 70 ? "#f59e0b" : "#10b981");
  gB.append("text").attr("x", bx).attr("y", by + 38).attr("fill", "#94a3b8").attr("font-size", 10.5).text("0 Pa (No Touch)");
  gB.append("text").attr("x", bx + bw).attr("y", by + 38).attr("text-anchor", "end").attr("fill", "#fda4af").attr("font-size", 10.5).attr("font-weight", 700).text("650 Pa Tissue Tearing Limit");

  drawStatTile(gB, 520, 14, 184, 66, "AI PREDICTION", S.abupt_transolver_surrogate.median_inference_ms.toFixed(1) + " ms", "#38bdf8", "346x faster than FEM");
  drawStatTile(gB, 714, 14, 184, 66, "AS6983 SAFETY", "100% covered", "#34d399", "0.0 TorchScript parity");
}

// =========================================================================
// SCENE 5: ROBOT ASSEMBLY JAMMING GUARD & 3D GRIPPER CO-DESIGN (Study R5)
// =========================================================================
function renderSceneR5(D, sceneIdx) {
  const F = D.frame, WW = D.wrench_window, S = D.summary;
  const jammed = F.is_failed_segment;
  drawChrome(
    sceneIdx >= 0 ? "5 / 5 · ASSEMBLY" : "STUDY R5 · ASSEMBLY & CAD",
    "Catching Jammed Parts & Sculpting 3D Grippers",
    "148,444 real 30 Hz Franka robot frames (NIST Board #1) -> live jam alert + 3D STEP CAD",
    jammed ? "JAM DETECTED!" : "INSERTION OK",
    jammed ? "#9f1239" : "#065f46",
    jammed ? "#fb7185" : "#34d399",
    sceneIdx
  );

  // Left card: Intuitive Peg-in-Hole visual schematic + real 30 Hz Franka wrist force/torque stream
  const gL = drawCard(24, 96, 444, 392, "1. Live Assembly Monitor (Real Franka Robot)",
    "30 Hz wrist sensor · Red zone = real failed/jammed insertion");
  // Simple, unmistakable robot gripper + peg + NIST hole visual at top of card
  const gx = 222, gy = 112;
  // Metal NIST task board block with slot
  gL.append("rect").attr("x", 92).attr("y", gy + 26).attr("width", 108).attr("height", 36).attr("rx", 4).attr("fill", "#334155").attr("stroke", "#64748b");
  gL.append("rect").attr("x", 244).attr("y", gy + 26).attr("width", 108).attr("height", 36).attr("rx", 4).attr("fill", "#334155").attr("stroke", "#64748b");
  gL.append("rect").attr("x", 200).attr("y", gy + 50).attr("width", 44).attr("height", 12).attr("fill", "#1e293b");
  // Peg & gripper fingers (straight when nominal, tilted & glowing red when jammed!)
  const pegTilt = jammed ? 11 : 0;
  const pegDrop = jammed ? 4 : Math.min(20, (F.frame / 10) * 20);
  const pegG = gL.append("g").attr("transform", "translate(" + gx + "," + (gy + pegDrop) + ") rotate(" + pegTilt + ")");
  // Gripper fingers
  pegG.append("rect").attr("x", -28).attr("y", -38).attr("width", 10).attr("height", 34).attr("rx", 3).attr("fill", "#38bdf8");
  pegG.append("rect").attr("x", 18).attr("y", -38).attr("width", 10).attr("height", 34).attr("rx", 3).attr("fill", "#38bdf8");
  pegG.append("rect").attr("x", -34).attr("y", -48).attr("width", 68).attr("height", 12).attr("rx", 4).attr("fill", "#64748b");
  // Peg
  pegG.append("rect").attr("x", -15).attr("y", -28).attr("width", 30).attr("height", 64).attr("rx", 4)
    .attr("fill", jammed ? "#f43f5e" : "#34d399").attr("stroke", "#f8fafc").attr("stroke-width", 1.5);
  if (jammed) {
    gL.append("circle").attr("cx", gx + 20).attr("cy", gy + 28).attr("r", 13).attr("fill", "#f43f5e").attr("fill-opacity", 0.35).attr("stroke", "#fb7185").attr("stroke-width", 2);
    gL.append("text").attr("x", gx + 120).attr("y", gy + 14).attr("text-anchor", "middle").attr("fill", "#fda4af").attr("font-size", 11.5).attr("font-weight", 800).text("PEG WEDGED!");
  } else {
    gL.append("text").attr("x", gx + 120).attr("y", gy + 14).attr("text-anchor", "middle").attr("fill", "#6ee7b7").attr("font-size", 11.5).attr("font-weight", 800).text("ALIGNED OK");
  }

  // Real 30 Hz Franka force & torque chart below schematic
  const p = gL.append("g").attr("transform", "translate(46, 206)");
  const xS = d3.scaleLinear().domain([0, WW.t_s[WW.t_s.length - 1]]).range([0, 376]);
  const yS = d3.scaleLinear().domain([0, 22]).range([144, 0]);
  for (let i = 0; i < WW.t_s.length - 1; i++) {
    if (!WW.success[i]) {
      p.append("rect").attr("x", xS(WW.t_s[i])).attr("y", 0).attr("width", Math.max(2, xS(WW.t_s[i+1]) - xS(WW.t_s[i]))).attr("height", 144)
        .attr("fill", "#881337").attr("fill-opacity", 0.38);
    }
  }
  p.append("g").attr("class", "axis-dk").attr("transform", "translate(0,144)").call(d3.axisBottom(xS).ticks(5).tickFormat(d => d + "s"));
  p.append("g").attr("class", "axis-dk").call(d3.axisLeft(yS).ticks(4).tickFormat(d => d + " N"));
  const km = Math.max(1, F.stream_idx);
  p.append("path").datum(WW.force_n.slice(0, km + 1)).attr("fill", "none").attr("stroke", "#38bdf8").attr("stroke-width", 2.6)
    .attr("d", d3.line().x((_, i) => xS(WW.t_s[i])).y(d => yS(d)));
  p.append("path").datum(WW.torque_nm.slice(0, km + 1)).attr("fill", "none").attr("stroke", "#fb7185").attr("stroke-width", 2.2)
    .attr("d", d3.line().x((_, i) => xS(WW.t_s[i])).y(d => yS(d * 35.0)));
  p.append("line").attr("x1", xS(F.stream_t_s)).attr("x2", xS(F.stream_t_s)).attr("y1", 0).attr("y2", 144).attr("stroke", "#fbbf24").attr("stroke-width", 2).attr("stroke-dasharray", "4,3");
  p.append("text").attr("x", 372).attr("y", 14).attr("text-anchor", "end").attr("fill", "#38bdf8").attr("font-size", 10.5).attr("font-weight", 700).text("Wrist Force: " + F.force_n.toFixed(1) + " N");
  p.append("text").attr("x", 372).attr("y", 29).attr("text-anchor", "end").attr("fill", "#fb7185").attr("font-size", 10.5).attr("font-weight", 700).text("Wrist Torque: " + F.torque_nm.toFixed(2) + " Nm");

  // Right card: 3D Shaded Isometric Sculpted Compliant Gripper Finger + CAD Certification Stamp
  const gR = drawCard(484, 96, 452, 392, "2. AI-Sculpted 3D Robot Finger (Iter " + F.simp_iter + " / 18)",
    "Loaded by real 35.3 N NIST peak force -> Watertight 3D STEP CAD");
  const colR = d3.scaleSequential(d3.interpolatePlasma).domain([0.18, 1.0]);
  const yaw = -30 + F.frame * 1.1;
  // Sort 3D solid voxels back-to-front and render as true 3-face shaded isometric blocks!
  const voxList = F.voxels_xyz.map((pt, i) => {
    const x = pt[0] - 27, y = pt[1] - 7, z = pt[2] - 7;
    const pr = iso(x, y, z, 226, 204, 6.1, yaw, 25);
    return { x, y, z, rho: F.voxels_rho[i], depth: pr.depth };
  }).sort((a, b) => a.depth - b.depth);
  voxList.forEach(v => {
    drawIsoVoxel(gR, v.x, v.y, v.z, 5.2 + 2.4 * v.rho, 226, 204, 6.1, colR(v.rho), yaw, 25);
  });
  // Tip force arrow
  const tipP = iso(27, 0, 5, 226, 204, 6.1, yaw, 25);
  gR.append("line").attr("x1", tipP.x + 8).attr("y1", tipP.y - 46).attr("x2", tipP.x + 2).attr("y2", tipP.y - 6)
    .attr("stroke", "#fb7185").attr("stroke-width", 3.5).attr("stroke-linecap", "round");
  gR.append("text").attr("x", tipP.x + 16).attr("y", tipP.y - 34).attr("fill", "#fda4af").attr("font-size", 11).attr("font-weight", 800).text("35.3 N Load");

  // Bottom badges on right card
  drawStatTile(gR, 18, 304, 202, 72, "EXACT 3D FEM CHECK", "CERTIFIED", "#34d399", "Tip flex = 0.104 mm");
  drawStatTile(gR, 232, 304, 202, 72, "EXPORTS READY", "STEP + ROS 2", "#38bdf8", "Watertight CAD & MuJoCo");
}

${drawScript}
})();
window.__DONE__ = true;
} catch (err) {
  window.__ERR__ = String(err && err.stack ? err.stack : err);
  window.__DONE__ = true;
}
</script>
</body>
</html>`;
}

async function renderSequence(page, outDir, frames, scriptBody) {
  fs.mkdirSync(outDir, { recursive: true });
  // Clean stale frames if count changed
  for (const fn of fs.readdirSync(outDir)) {
    if (fn.endsWith(".png")) fs.unlinkSync(path.join(outDir, fn));
  }
  for (let i = 0; i < frames.length; i++) {
    const html = buildHtml(scriptBody, frames[i]);
    await page.goto("about:blank");
    await page.setContent(html, { waitUntil: "load" });
    await page.waitForFunction(() => window.__DONE__ === true);
    const err = await page.evaluate(() => window.__ERR__);
    if (err) throw new Error(`Render error in ${outDir} frame ${i}:\n${err}`);
    await page.screenshot({ path: path.join(outDir, `frame_${String(i).padStart(3, "0")}.png`), type: "png" });
  }
  console.log(`Rendered ${frames.length} LinkedIn frames -> ${outDir}`);
}

async function main() {
  fs.mkdirSync(FRAMES_ROOT, { recursive: true });
  const r = {};
  for (let i = 1; i <= 5; i++) {
    r[i] = JSON.parse(fs.readFileSync(path.join(VIZ_DIR, `study_r${i}_viz.json`), "utf-8"));
  }

  const browser = await chromium.launch({ headless: true });
  try {
    const page = await browser.newPage({ viewport: { width: W, height: H }, deviceScaleFactor: 1 });

    // 1. Standalone LinkedIn GIF #1 (Most intuitive): Soft Arm Safety Shield (30 frames)
    await renderSequence(
      page,
      path.join(FRAMES_ROOT, "r3_safety_shield"),
      r[3].anim_frames.map(f => ({ frame: f, all_frames: r[3].anim_frames, summary: r[3].summary })),
      "renderSceneR3(DATA, -1);"
    );

    // 2. Standalone LinkedIn GIF #2: Squishy 3D Silicone Fingertip Touch & Slip (24 frames)
    await renderSequence(
      page,
      path.join(FRAMES_ROOT, "r1_robot_touch"),
      r[1].anim_frames.map(f => ({ frame: f, all_frames: r[1].anim_frames, summary: r[1].summary })),
      "renderSceneR1(DATA, -1);"
    );

    // 3. Standalone LinkedIn GIF #3: Human Brain 3D Surgical Palpation (24 frames)
    await renderSequence(
      page,
      path.join(FRAMES_ROOT, "r4_brain_palpation"),
      r[4].anim_frames.map(f => ({ frame: f, all_frames: r[4].anim_frames, summary: r[4].summary })),
      "renderSceneR4(DATA, -1);"
    );

    // 4. Hero 5-in-1 LinkedIn Highlight Reel (5 scenes x 16 sampled frames = 80 smooth frames!)
    const pick = (arr, n) => Array.from({ length: n }, (_, k) => arr[Math.round((k / (n - 1)) * (arr.length - 1))]);
    const heroPayloads = [
      ...pick(r[1].anim_frames, 16).map(f => ({ scene: 1, frame: f, all_frames: r[1].anim_frames, summary: r[1].summary })),
      ...pick(r[2].anim_frames, 16).map(f => ({ scene: 2, frame: f, all_frames: r[2].anim_frames, summary: r[2].summary })),
      ...pick(r[3].anim_frames, 18).map(f => ({ scene: 3, frame: f, all_frames: r[3].anim_frames, summary: r[3].summary })),
      ...pick(r[4].anim_frames, 16).map(f => ({ scene: 4, frame: f, all_frames: r[4].anim_frames, summary: r[4].summary })),
      ...pick(r[5].anim_frames, 16).map(f => ({ scene: 5, frame: f, all_frames: r[5].anim_frames, summary: r[5].summary, wrench_window: r[5].wrench_window })),
    ];
    await renderSequence(
      page,
      path.join(FRAMES_ROOT, "hero_5in1"),
      heroPayloads,
      `if (DATA.scene === 1) renderSceneR1(DATA, 0);
       else if (DATA.scene === 2) renderSceneR2(DATA, 1);
       else if (DATA.scene === 3) renderSceneR3(DATA, 2);
       else if (DATA.scene === 4) renderSceneR4(DATA, 3);
       else renderSceneR5(DATA, 4);`
    );

    await page.close();
  } finally {
    await browser.close();
  }
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
