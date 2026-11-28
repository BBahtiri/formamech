"""Assemble Playwright-rendered LinkedIn D3.js + SVG frames into crisp GIFs.

Writes broad-public LinkedIn GIFs into `docs/assets/studies_robotics/linkedin/`.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
FRAMES_ROOT = REPO_ROOT / ".formamech-runs" / "robotics_studies" / "linkedin_frames"
OUT_DIR = REPO_ROOT / "docs" / "assets" / "studies_robotics" / "linkedin"

LINKEDIN_GIFS: tuple[tuple[str, str, int], ...] = (
    ("hero_5in1", "linkedin_hero_5in1.gif", 150),
    ("r3_safety_shield", "linkedin_r3_safety_shield.gif", 125),
    ("r1_robot_touch", "linkedin_r1_robot_touch.gif", 130),
    ("r4_brain_palpation", "linkedin_r4_brain_palpation.gif", 130),
)


def build_gif(frame_dir: Path, out_gif: Path, duration_ms: int) -> None:
    pngs = sorted(frame_dir.glob("frame_*.png"))
    if not pngs:
        raise FileNotFoundError(f"No PNG frames found in {frame_dir}")
    frames: list[Image.Image] = []
    for fp in pngs:
        with Image.open(fp) as im:
            rgb = im.convert("RGB")
            pal = rgb.convert("P", palette=Image.Palette.ADAPTIVE, colors=256)
            frames.append(pal)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    # Slightly longer hold on the final frame of standalone loops for easy reading
    durations = [duration_ms] * len(frames)
    if len(durations) <= 32:
        durations[-1] = duration_ms * 4
    frames[0].save(
        out_gif,
        save_all=True,
        append_images=frames[1:],
        duration=durations,
        loop=0,
        optimize=True,
        disposal=2,
    )
    size_mb = out_gif.stat().st_size / (1024.0 * 1024.0)
    print(f"Built {out_gif.name}: {len(frames)} frames, {size_mb:.2f} MB")


def main() -> None:
    for folder, gif_name, dur in LINKEDIN_GIFS:
        build_gif(FRAMES_ROOT / folder, OUT_DIR / gif_name, dur)


if __name__ == "__main__":
    main()
