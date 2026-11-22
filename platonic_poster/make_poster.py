"""Illusory Platonic solids on a polka-dot grid — A2 poster generator.

A white page with a regular grid of black discs. Wireframes of the five
Platonic solids are drawn in a bright colour, but the lines are clipped to
the discs, so they only show inside the black "holes". Every vertex of every
solid sits exactly on a disc centre; the eye fills in the missing edges
across the white, so the solids appear as illusory figures (after Bradley &
Petry's illusory Necker cube). Orthographic projection keeps each solid
depth-ambiguous, so they flip like a Necker cube.

Outputs: vector SVG + vector PDF at exact A2 size, and a 300 dpi PNG.

  python make_poster.py                       # default cyan
  python make_poster.py --colour "#ff2bd6" --seed 7 --out poster_magenta
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

import solids

A2_W, A2_H = 420.0, 594.0  # mm, portrait (1 : sqrt 2)

TYPES = ["tetrahedron", "cube", "octahedron", "icosahedron", "dodecahedron"]
# target projected extent (in grid steps) for each solid
BASE_EXTENT = {"tetrahedron": 6.5, "cube": 7.0, "octahedron": 7.5,
               "icosahedron": 9.0, "dodecahedron": 9.5}


def _solve(job):
    name, extent, seed = job
    rng = np.random.default_rng(seed)
    return solids.best_view(name, extent, rng)


def layout(cols, rows, cell_cols, cell_rows, rng):
    """One solid per cell; types arranged so neighbours always differ."""
    cw, ch = cols / cell_cols, rows / cell_rows
    jobs, centres = [], []
    for r in range(cell_rows):
        for c in range(cell_cols):
            name = TYPES[(c + 2 * r) % len(TYPES)]
            ext = BASE_EXTENT[name] * rng.uniform(0.85, 1.08)
            ext = min(ext, cw - 1.5, ch - 1.5)
            cx = (c + 0.5) * cw - 0.5 + rng.uniform(-0.6, 0.6)
            cy = (r + 0.5) * ch - 0.5 + rng.uniform(-0.6, 0.6)
            jobs.append((name, ext, int(rng.integers(1 << 31))))
            centres.append((cx, cy))
    return jobs, centres


def build_svg(cols, rows, pitch, ox, oy, shapes, colour, dot_ratio, line_ratio,
              dot_colour="#000000", paper="#ffffff"):
    r = dot_ratio * pitch / 2
    lw = line_ratio * pitch
    X = lambda i: ox + i * pitch
    Y = lambda j: oy + j * pitch
    circles = "\n".join(
        f'<circle cx="{X(i):.4f}" cy="{Y(j):.4f}" r="{r:.4f}"/>'
        for j in range(rows) for i in range(cols))
    lines = []
    for sh in shapes:
        pts = sh["points"]
        for a, b in sh["edges"]:
            (x1, y1), (x2, y2) = pts[a], pts[b]
            lines.append(f'<line x1="{X(x1):.4f}" y1="{Y(y1):.4f}" '
                         f'x2="{X(x2):.4f}" y2="{Y(y2):.4f}"/>')
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{A2_W}mm" height="{A2_H}mm"
     viewBox="0 0 {A2_W} {A2_H}">
  <title>Illusory Platonic Solids</title>
  <defs>
    <clipPath id="holes">
{circles}
    </clipPath>
  </defs>
  <rect width="{A2_W}" height="{A2_H}" fill="{paper}"/>
  <g fill="{dot_colour}">
{circles}
  </g>
  <g clip-path="url(#holes)" stroke="{colour}" stroke-width="{lw:.4f}"
     stroke-linecap="round" fill="none">
{chr(10).join(lines)}
  </g>
</svg>
'''


def write_pdf(path, cols, rows, pitch, ox, oy, shapes, colour, dot_ratio,
              line_ratio, dot_colour="#000000", paper="#ffffff"):
    """Vector PDF at exact A2 with a single clip path (compact, print-ready)."""
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas

    r = dot_ratio * pitch / 2
    X = lambda i: (ox + i * pitch) * mm
    Y = lambda j: (A2_H - (oy + j * pitch)) * mm      # PDF y axis points up
    c = canvas.Canvas(path, pagesize=(A2_W * mm, A2_H * mm))
    c.setTitle("Illusory Platonic Solids")
    c.setFillColor(HexColor(paper))
    c.rect(0, 0, A2_W * mm, A2_H * mm, stroke=0, fill=1)

    def holes():
        p = c.beginPath()
        for j in range(rows):
            for i in range(cols):
                p.circle(X(i), Y(j), r * mm)
        return p

    c.setFillColor(HexColor(dot_colour))
    c.drawPath(holes(), stroke=0, fill=1)
    c.saveState()
    c.clipPath(holes(), stroke=0, fill=0)
    c.setStrokeColor(HexColor(colour))
    c.setLineWidth(line_ratio * pitch * mm)
    c.setLineCap(1)
    for sh in shapes:
        pts = sh["points"]
        for a, b in sh["edges"]:
            (x1, y1), (x2, y2) = pts[a], pts[b]
            c.line(X(x1), Y(y1), X(x2), Y(y2))
    c.restoreState()
    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=44, help="dots across")
    ap.add_argument("--cells", type=int, nargs=2, default=(4, 6),
                    metavar=("ACROSS", "DOWN"), help="solids across / down")
    ap.add_argument("--colour", default="#00d2ff", help="line colour")
    ap.add_argument("--dot", type=float, default=0.74, help="disc diameter / pitch")
    ap.add_argument("--line", type=float, default=0.095, help="line width / pitch")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="platonic_poster")
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()

    cols = args.cols
    pitch = A2_W / (cols + 1)                 # one pitch of margin each side
    rows = int(A2_H / pitch) - 1
    ox = pitch
    oy = (A2_H - (rows - 1) * pitch) / 2

    rng = np.random.default_rng(args.seed)
    jobs, centres = layout(cols, rows, *args.cells, rng)
    with Pool() as pool:
        views = pool.map(_solve, jobs)

    shapes = []
    for v, (cx, cy) in zip(views, centres):
        pts = v["points"] + np.array([int(round(cx)), int(round(cy))])
        # nudge (never distort) the solid so it stays >= 1 dot inside the grid
        lo, hi = pts.min(0), pts.max(0)
        shift = np.maximum(1 - lo, 0) - np.maximum(hi - np.array([cols - 2, rows - 2]), 0)
        pts = pts + shift
        shapes.append(dict(points=pts, edges=v["edges"]))
        print(f'{v["name"]:13s} at ({cx:5.1f},{cy:5.1f})  fit rms '
              f'{np.sqrt(v["err"]):.3f} grid steps')

    svg = build_svg(cols, rows, pitch, ox, oy, shapes, args.colour, args.dot, args.line)
    svg_path = args.out + ".svg"
    with open(svg_path, "w") as f:
        f.write(svg)
    print("wrote", svg_path, f"({cols}x{rows} dots, pitch {pitch:.2f} mm)")

    write_pdf(args.out + ".pdf", cols, rows, pitch, ox, oy, shapes,
              args.colour, args.dot, args.line)
    print("wrote", args.out + ".pdf")

    import cairosvg
    if not args.no_png:
        w_px = round(A2_W / 25.4 * args.dpi)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=args.out + ".png",
                         output_width=w_px, dpi=args.dpi)
        try:
            from PIL import Image
            Image.MAX_IMAGE_PIXELS = None
            im = Image.open(args.out + ".png")
            im.convert("RGB").save(args.out + ".png", dpi=(args.dpi, args.dpi),
                                   optimize=True)
            print("wrote", args.out + ".png", im.size, f"@ {args.dpi} dpi")
        except ImportError:
            print("wrote", args.out + ".png")


if __name__ == "__main__":
    main()
