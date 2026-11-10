"""Record demo/move-the-map.gif: drive the built page with the gstack `browse` headless browser through a short
script (no edit -> full edits to a few positions -> halfway), screenshot each state, and stitch a GIF.

Usage: uv run python demo/make_gif.py --scene 0 --targets 3 7 11 --halfway 7"""

import argparse
import json
import subprocess
import time
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
B = str(Path.home() / ".claude/skills/browse/dist/browse")
TMP = Path("/tmp/move_the_map_frames")  # browse only writes screenshots under /tmp


def b(*args):
    return subprocess.run([B, *args], capture_output=True, text=True, check=True).stdout


def set_state(scene, target, alpha):
    b("js", f"state.scene={scene}; state.target={target}; state.alpha={alpha}; "
            f"document.getElementById('alpha').value={alpha}; render(); 1")


def shot(name):
    p = TMP / f"{name}.png"
    b("screenshot", str(p))
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--page", default=str(ROOT / "demo" / "move-the-map.html"))
    ap.add_argument("--scene", type=int, default=0)
    ap.add_argument("--targets", type=int, nargs="+", required=True)
    ap.add_argument("--halfway", type=int, required=True, help="target index shown at alpha = 1/2")
    ap.add_argument("--out", default=str(ROOT / "demo" / "move-the-map.gif"))
    ap.add_argument("--width", type=int, default=980)
    ap.add_argument("--height", type=int, default=760)
    ap.add_argument("--crop", type=int, nargs=2, default=None, metavar=("TOP", "BOTTOM"),
                    help="keep only this vertical band of the page (pixels), e.g. the map and the answers")
    a = ap.parse_args()
    TMP.mkdir(exist_ok=True)
    b("viewport", f"{a.width}x{a.height}")
    b("goto", f"file://{a.page}")
    b("js", "document.documentElement.setAttribute('data-theme','light'); "
            "document.querySelector('footer').style.visibility='hidden'; 1")  # the post carries the stats

    # (scene, target, alpha, hold seconds)
    script = [(a.scene, a.targets[0], 0, 2.2)]
    script += [(a.scene, t, 4, 2.6) for t in a.targets]
    script += [(a.scene, a.halfway, 4, 1.4), (a.scene, a.halfway, 2, 4.0), (a.scene, a.halfway, 4, 1.6)]
    frames, durations = [], []
    for i, (sc, t, al, hold) in enumerate(script):
        set_state(sc, t, al)
        # two in-between frames catch the dots gliding (CSS transition 0.45 s), then the settled frame
        for k, wait in enumerate((0.12, 0.15)):
            time.sleep(wait)
            frames.append(shot(f"f{i:02d}_{k}")); durations.append(120)
        time.sleep(0.4)
        frames.append(shot(f"f{i:02d}_s")); durations.append(int(hold * 1000))
    imgs = [Image.open(p).convert("RGB") for p in frames]
    if a.crop:
        imgs = [im.crop((0, a.crop[0], im.width, a.crop[1])) for im in imgs]
    w = 900
    imgs = [im.resize((w, int(im.height * w / im.width)), Image.LANCZOS) for im in imgs]
    # one shared palette from all frames: per-frame median cut drops the small saturated marks (dots, ticks, crosses)
    sheet = Image.new("RGB", (w, sum(im.height for im in imgs)))
    y = 0
    for im in imgs:
        sheet.paste(im, (0, y)); y += im.height
    palette = sheet.quantize(colors=256, method=Image.Quantize.FASTOCTREE)
    pal = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in imgs]
    pal[0].save(a.out, save_all=True, append_images=pal[1:], duration=durations, loop=0, optimize=True)
    print(f"{a.out}: {len(pal)} frames, {Path(a.out).stat().st_size / 1e6:.1f} MB")


if __name__ == "__main__":
    main()
