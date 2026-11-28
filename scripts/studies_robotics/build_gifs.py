"""Assemble Playwright-rendered D3.js + custom SVG PNG frames into optimized GIFs.

Reads frames from gitignored `.formamech-runs/robotics_studies/gif_frames/<study>/`
and writes compact, crisp, adaptive-palette animated GIFs into
`docs/assets/studies_robotics/`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMES_ROOT = REPO_ROOT / ".formamech-runs" / "robotics_studies" / "gif_frames"
ASSETS_DIR = REPO_ROOT / "docs" / "assets" / "studies_robotics"

GIF_SPECS: tuple[tuple[str, str, int], ...] = (
    ("r1_tactile", "study_r1_tactile_live.gif", 130),
    ("r2_continuum", "study_r2_continuum_live.gif", 120),
    ("r3_soft_mpc", "study_r3_soft_mpc_cbf_live.gif", 125),
    ("r4_surgical", "study_r4_surgical_tissue_live.gif", 130),
    ("r5_nist", "study_r5_nist_jamming_codesign_live.gif", 135),
)


def build_gif(frame_dir: Path, out_gif: Path, duration_ms: int) -> None:
    pngs = sorted(frame_dir.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")
    frames: list[Image.Image] = []
    for fp in pngs:
        with Image.open(fp) as im:
            rgb = im.convert("RGB")
            # Adaptive 256-color palette preserves crisp anti-aliased SVG vector text
            pal = rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            frames.append(pal)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_kb = out_gif.stat().st_size / 1024.0
    print(f"Built {out_gif.name}: {len(frames)} frames, {size_kb:.1f} KB")


def main() -> None:
    for folder, gif_name, dur in GIF_SPECS:
        build_gif(FRAMES_ROOT / folder, ASSETS_DIR / gif_name, dur)


if __name__ == "__main__":
    main()
