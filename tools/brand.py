"""Renders the Auricle brand files into demo/public/brand/ from one geometry:
a bare serif-weight A whose crossbar is a sine wave, followed by the word
set in Instrument Serif (converted to paths, so nothing depends on fonts).

    ../VAD-train/.venv/bin/python tools/brand.py path/to/InstrumentSerif.ttf
"""
import sys
from pathlib import Path

import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath

OUT = Path("demo/public/brand")
OUT.mkdir(parents=True, exist_ok=True)
FP = FontProperties(fname=sys.argv[1])

INK, PAPER, GREEN, BLUE, GOLD = "#e6e8ee", "#0f1115", "#5ad1a1", "#5ab0ff", "#f4c95d"


def glyph(text, size, x_left, baseline):
    """Text as an SVG path, left edge at x_left, baseline at y=baseline."""
    tp = TextPath((0, 0), text, size=size, prop=FP)
    x0 = tp.vertices[:, 0].min()
    d = []
    for verts, code in tp.iter_segments():
        pts = np.array(verts).reshape(-1, 2)
        pts[:, 0] += x_left - x0
        pts[:, 1] = baseline - pts[:, 1]
        op = {MPath.MOVETO: "M", MPath.LINETO: "L", MPath.CURVE3: "Q", MPath.CURVE4: "C", MPath.CLOSEPOLY: "Z"}[code]
        d.append(op + ("" if code == MPath.CLOSEPOLY else " " + " ".join(f"{x:.2f} {y:.2f}" for x, y in pts)))
    return " ".join(d), tp.vertices[:, 0].max() - x0


def a_mark(x, y, h, ink, wave, kind="wave", sw=None):
    """The A: apex at (x + w/2, y), feet at y + h. kind: wave | flat-then-wave | two-tone."""
    w = h * 0.88
    sw = sw or h * 0.11
    cx, cy = x + w / 2, y + h * 0.72
    legs = f'<path d="M{x:.1f} {y + h:.1f} L{cx:.1f} {y:.1f} L{x + w:.1f} {y + h:.1f}" fill="none" stroke="{ink}" stroke-width="{sw:.2f}" stroke-linejoin="round" stroke-linecap="round"/>'
    amp = h * 0.16
    x0, x1 = x + w * 0.19, x + w * 0.81
    if kind == "flat-then-wave":
        m = x0 + (x1 - x0) * 0.38
        q = (x1 - m) / 4
        bar = f"M{x0:.1f} {cy:.1f} L{m:.1f} {cy:.1f} C{m + q * 0.5:.1f} {cy:.1f} {m + q * 0.5:.1f} {cy - amp:.1f} {m + q:.1f} {cy - amp:.1f} S{m + 2 * q:.1f} {cy + amp:.1f} {m + 2 * q:.1f} {cy + amp:.1f} S{m + 3 * q:.1f} {cy - amp:.1f} {m + 3 * q:.1f} {cy - amp:.1f} S{x1:.1f} {cy:.1f} {x1:.1f} {cy:.1f}"
    else:
        q = (x1 - x0) / 4
        bar = f"M{x0:.1f} {cy:.1f} C{x0 + q * 0.5:.1f} {cy:.1f} {x0 + q * 0.5:.1f} {cy - amp:.1f} {x0 + q:.1f} {cy - amp:.1f} S{x0 + 2 * q:.1f} {cy + amp:.1f} {x0 + 2 * q:.1f} {cy + amp:.1f} S{x0 + 3 * q:.1f} {cy - amp:.1f} {x0 + 3 * q:.1f} {cy - amp:.1f} S{x1:.1f} {cy:.1f} {x1:.1f} {cy:.1f}"
    crossbar = f'<path d="{bar}" fill="none" stroke="{wave}" stroke-width="{sw * 0.82:.2f}" stroke-linecap="round"/>'
    return legs + crossbar, w


def svg(w, h, body, title):
    return f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-label="{title}">\n  <title>{title}</title>\n  {body}\n</svg>\n'


def write(name, content):
    (OUT / name).write_text(content)
    print("wrote", OUT / name)


# Bare mark, 64 x 64, transparent: the A alone.
for name, ink, wave, kind in [
    ("mark.svg", INK, GREEN, "wave"),
    ("mark-light.svg", PAPER, "#1f9a6b", "wave"),
    ("mark-mono.svg", INK, INK, "wave"),
    ("mark-mono-light.svg", PAPER, PAPER, "wave"),
]:
    body, _ = a_mark(8, 7, 50, ink, wave, kind)
    write(name, svg(64, 64, body, "Auricle"))

# The five variants, as tiles (for app icons, favicons, avatars).
VARIANTS = [
    ("tile-green-wave.svg", PAPER, INK, GREEN, "wave"),         # 1 default: dark tile, white A, green wave
    ("tile-green.svg", GREEN, PAPER, PAPER, "wave"),            # 2 green tile, dark A
    ("tile-blue-wave.svg", PAPER, INK, BLUE, "wave"),           # 3 dark tile, blue wave
    ("tile-gold-wave.svg", PAPER, INK, GOLD, "wave"),           # 4 dark tile, gold wave
    ("tile-silence-speech.svg", PAPER, INK, GREEN, "flat-then-wave"),  # 5 crossbar: silence, then speech
    ("tile-paper.svg", "#f5f6f8", PAPER, "#1f9a6b", "wave"),    # light tile for light surfaces
]
for name, bg, ink, wave, kind in VARIANTS:
    body, _ = a_mark(11, 10, 44, ink, wave, kind)
    write(name, svg(64, 64, f'<rect width="64" height="64" rx="14" fill="{bg}"/>' + body, "Auricle"))

# Wordmark: A then "uricle" in Instrument Serif, cap height matched.
H = 44
for name, ink, wave in [("logo.svg", INK, GREEN), ("logo-light.svg", PAPER, "#1f9a6b"), ("logo-mono.svg", INK, INK)]:
    a, aw = a_mark(2, 10, H, ink, wave)
    text, tw = glyph("uricle", 60, 2 + aw + 1, 10 + H)
    total = int(2 + aw + 1 + tw + 4)
    write(name, svg(total, 64, a + f'<path d="{text}" fill="{ink}"/>', "Auricle"))

# Favicon: variant 1 tile.
Path("demo/public/favicon.svg").write_text((OUT / "tile-green-wave.svg").read_text())
print("favicon <- tile-green-wave.svg")
