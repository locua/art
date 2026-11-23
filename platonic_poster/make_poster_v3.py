"""Illusory Platonic solids, v3: every solid is a geometrically exact
orthographic projection (perfectly regular -- the condition for Necker-type
depth reversal). Corners are no longer forced onto dot centres; instead the
rotation, size and position of each solid are chosen so that as many
corners as possible fall inside the dots.

Same layout as v2: fewer, bigger dots, bolder lines.

Each row is one solid seen in three successive turns about a shared,
slightly tilted axis. Reading along a row makes it look like one object
spinning, and alternate rows spin in opposite directions. Because the
projection is orthographic, each frame is also depth-ambiguous (Necker
reversal), so the apparent direction of spin can flip as you watch.

  python make_poster_v3.py
  python make_poster_v3.py --only cube --frames 4 --rows 6 --out output/v3_cubes/necker_cubes_A2
  python make_poster_v3.py --colour "#ff2bd6" --seed 4 --out output/v3/magenta
"""
import argparse
import os
from multiprocessing import Pool

import numpy as np

import solids_v2
import solids_v3
from make_poster import A2_W, A2_H, build_svg, write_pdf

# (solid, projected extent in dots, row height in dots)  -- top to bottom:
# 4, 6, 8, 12 and 20 faces
ROWS = [("tetrahedron", 6.5, 8), ("cube", 7.0, 9), ("octahedron", 7.5, 10),
        ("icosahedron", 9.0, 12), ("dodecahedron", 9.5, 12)]


def _solve(job):
    name, extent, frames, step, direction, seed = job
    rng = np.random.default_rng(seed)
    for attempt in range(4):          # grow a little if no clean fit exists
        try:
            return solids_v3.exact_sequence(name, extent + 0.5 * attempt,
                                               frames, step, direction, rng)
        except RuntimeError as e:
            print(e, "- retrying larger")
    raise RuntimeError(f"could not fit {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=36, help="dots across")
    ap.add_argument("--frames", type=int, default=3, help="solids per row")
    ap.add_argument("--step", type=float, default=20, help="degrees turned per frame")
    ap.add_argument("--colour", default="#00d2ff")
    ap.add_argument("--dot", type=float, default=0.80, help="disc diameter / pitch")
    ap.add_argument("--line", type=float, default=0.16, help="line width / pitch")
    ap.add_argument("--seed", type=int, default=5)
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--out", default="output/v3/platonic_poster_v3_A2")
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--only", choices=[r[0] for r in ROWS],
                    help="use a single solid for every row (e.g. --only cube)")
    ap.add_argument("--rows", type=int, default=6, help="number of rows with --only")
    ap.add_argument("--extent", type=float, help="solid size in dots with --only")
    args = ap.parse_args()
    rows_spec = ROWS
    if args.only:
        base_ext = dict((n, e) for n, e, _ in ROWS)[args.only]
        rows_spec = [(args.only, args.extent or base_ext, 1)] * args.rows
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    cols = args.cols
    pitch = A2_W / (cols + 1)
    rows = int(A2_H / pitch) - 1
    ox, oy = pitch, (A2_H - (rows - 1) * pitch) / 2

    heights = np.array([h for _, _, h in rows_spec], float)
    heights *= rows / heights.sum()
    tops = np.concatenate([[0], np.cumsum(heights)[:-1]])

    rng = np.random.default_rng(args.seed)
    jobs = [(name, ext, args.frames, args.step, 1 if r % 2 == 0 else -1,
             int(rng.integers(1 << 31))) for r, (name, ext, _) in enumerate(rows_spec)]
    with Pool() as pool:
        seqs = pool.map(_solve, jobs)

    shapes = []
    cw = cols / args.frames
    for r, seq in enumerate(seqs):
        cy = tops[r] + heights[r] / 2 - 0.5
        for k, f in enumerate(seq):
            cx = (k + 0.5) * cw - 0.5
            pts = f["points"] + np.array([int(round(cx)), int(round(cy))])  # integer shift keeps dot alignment
            lo, hi = pts.min(0), pts.max(0)
            pts = pts + np.maximum(1 - lo, 0) - np.maximum(hi - [cols - 2, rows - 2], 0)
            shapes.append(dict(points=pts, edges=f["edges"]))
            extra = (f'  corners in dots {f["in_dot"]:2d}/{f["n"]:2d}'
                     f'  parallel-edge error {solids_v2.regularity(pts, f["edges"], f["name"]):.1e}')
            print(f'{f["name"]:13s} frame {k}  turn {f["angle"]:6.1f} deg  '
                  f'cost {f["cost"]:.3f}{extra}')

    svg = build_svg(cols, rows, pitch, ox, oy, shapes, args.colour, args.dot, args.line)
    with open(args.out + ".svg", "w") as fh:
        fh.write(svg)
    write_pdf(args.out + ".pdf", cols, rows, pitch, ox, oy, shapes,
              args.colour, args.dot, args.line)
    print(f"wrote {args.out}.svg/.pdf  ({cols}x{rows} dots, pitch {pitch:.2f} mm)")
    if not args.no_png:
        import cairosvg
        from PIL import Image
        Image.MAX_IMAGE_PIXELS = None
        cairosvg.svg2png(bytestring=svg.encode(), write_to=args.out + ".png",
                         output_width=round(A2_W / 25.4 * args.dpi))
        im = Image.open(args.out + ".png").convert("RGB")
        im.save(args.out + ".png", dpi=(args.dpi, args.dpi), optimize=True)
        im.resize((1240, 1754), Image.LANCZOS).save(
            os.path.join(os.path.dirname(args.out), "preview.png"))
        print(f"wrote {args.out}.png {im.size} @ {args.dpi} dpi")


if __name__ == "__main__":
    main()
