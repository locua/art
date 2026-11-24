"""v8 "Necker Tumbler": the Tumbling Field with every block drawn as a full
Necker cube -- all 12 edges, hidden ones included.

Each block's hidden edges are drawn, so every cube carries both readings
(seen from above / from below) at once, and so does the whole field. In a
true isometric view each cube's hidden corner would land on its front corner
(a flat, accidental view), so the three edge vectors are unequal whole
multiples of the hexagonal lattice's row directions: 7, 7 and 5 dots. A true
orthographic view of a cube on this lattice needs the vertical edge to be
1/sqrt 2 of the others (5/7 is within 1%), so the cubes look regular. And it
keeps everything exact:
  * every cube corner sits on a dot centre,
  * every edge runs straight through a row of dot centres,
  * no corner lies on, or near, an edge it doesn't belong to
    (min clearance 2.6 dot spacings, checked below).
The dot lattice keeps its own three-way row ambiguity (Kubovy), along the
same three directions as the edges, coloured with the series greens.

Every third cube of the tumbling layer is drawn (the index-3 sublattice, the
classic 3-colouring of the tumbling-blocks tiling): in that pattern the
cubes' outlines touch but never overlap, so each one stays a distinct Necker.
Dots inside a cube's outline are enlarged and those between cubes shrunk, so
each cube appears as a dark hexagon in the halftone without shading any face
(which would favour one reading).

  python make_poster_v8.py
"""
import argparse
import os

import numpy as np

import necker_field as nf
from make_poster_v4 import A2_W, A2_H, render_svg, render_pdf
from make_poster_v7 import warp, S3


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--abc", type=int, nargs=3, default=(7, 7, 5),
                    help="cube edge lengths along x, y, z in dot steps")
    ap.add_argument("--sub", default="third", choices=["third", "all"],
                    help="third: every third cube (outlines touch, don't overlap); all: full packing")
    ap.add_argument("--pitch", type=float, default=6.0, help="dot spacing, mm")
    ap.add_argument("--whole", action=argparse.BooleanOptionalAction, default=True,
                    help="draw only cubes that fit entirely on the page")
    ap.add_argument("--r", type=float, default=0.38, help="dot radius / pitch")
    ap.add_argument("--overlap", type=float, nargs=4, default=(0.8, 1.2, 1.2, 1.2),
                    metavar=("C0", "C1", "C2", "C3"),
                    help="dot size factor by how many cube silhouettes cover the dot")
    ap.add_argument("--magnify", type=float, default=0.35)
    ap.add_argument("--lens", type=float, default=200, help="lens radius, mm")
    ap.add_argument("--profile", type=float, default=3)
    ap.add_argument("--r-in", type=float, default=0.44, help="dot radius / pitch at lens centre")
    ap.add_argument("--vignette", type=float, default=0.35)
    ap.add_argument("--line", type=float, default=1.6)
    ap.add_argument("--cy", type=float, default=0.5)
    ap.add_argument("--out", default="output/v8/necker_tumbler_A2")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from make_poster_v5 import PALETTES
    g = PALETTES["greens"]["C"][0]
    pal = {"x": g[0], "y": g[1], "z": g[2]}

    abc = tuple(args.abc)
    sub = None if args.sub == "all" else args.sub
    gap, clear, contacts = nf.quality(abc, 6 if sub else 3, sub)
    assert contacts == 0, "a corner lies on an unrelated edge"
    print(f"edges {abc} dots: min corner gap {gap:.2f}, min corner-edge clearance "
          f"{clear:.2f} dot spacings")

    pitch = args.pitch
    centre = np.array([A2_W / 2, A2_H * args.cy])
    lens = args.lens if args.magnify else 0
    m = pitch * 1.6

    # ---- hexagonal dot lattice
    n = int(A2_H / pitch) + 4
    ii, jj = np.meshgrid(np.arange(-2 * n, 2 * n + 1), np.arange(-2 * n, 2 * n + 1), indexing="ij")
    Lp = np.column_stack([ii.ravel(), jj.ravel()]).astype(float)
    P0 = centre + pitch * nf.lat_to_xy(Lp)
    keep = (np.abs(P0[:, 0] - A2_W / 2) < A2_W) & (np.abs(P0[:, 1] - A2_H / 2) < A2_H)
    P0 = P0[keep]
    W, radial, scale, ang, bump = warp(P0, centre, lens, args.magnify, args.profile)
    # halftone by overlap: where neighbouring cubes' silhouettes interpenetrate
    # the dots grow, so each Necker cube's outline emerges from the field
    # without shading any face (which would favour one reading)
    cell = min(abc) * pitch
    radius_cells = int(A2_H / cell) + 2
    positions = nf.cubes_in(radius_cells, sub)

    def corners_page(p):
        L = np.array([nf.lat_of(p + u, abc) for u in nf.UNIT], float)
        return warp(centre + pitch * nf.lat_to_xy(L), centre, lens, args.magnify, args.profile)[0]
    if args.whole:      # keep only cubes that sit entirely on the page
        pad = pitch * 1.2
        positions = [p for p in positions
                     if ((c := corners_page(p))[:, 0].min() > pad and c[:, 0].max() < A2_W - pad and
                         c[:, 1].min() > pad and c[:, 1].max() < A2_H - pad)]
    hulls = [centre + pitch * h for h in nf.cube_hulls(abc, positions)]
    cover = nf.coverage(P0, hulls)
    tone = np.array(args.overlap)[np.clip(cover, 0, 3)]
    rho = np.minimum(pitch * (args.r + (args.r_in - args.r) * bump) * tone, 0.455 * pitch)
    a_r, a_t = rho * radial, rho * scale
    if args.vignette:
        e = np.hypot((P0[:, 0] - A2_W / 2) / (A2_W / 2), (P0[:, 1] - A2_H / 2) / (A2_H / 2)) / S3 * 1.2
        sh = 1 - args.vignette * np.clip(e, 0, 1) ** 2
        a_r, a_t = a_r * sh, a_t * sh
    # fixed margin on dot centres -> a clean, straight-edged field
    inpage = ((W[:, 0] > m) & (W[:, 0] < A2_W - m) & (W[:, 1] > m) & (W[:, 1] < A2_H - m))
    dots = (W[inpage, 0], W[inpage, 1], a_r[inpage], a_t[inpage], ang[inpage])

    # ---- cubes: every edge as unit lattice steps, warped point by point
    E = {}
    for p in positions:
        for u, w in nf.CUBE_EDGES:
            A, B = tuple(p + nf.UNIT[u]), tuple(p + nf.UNIT[w])
            E[tuple(sorted([A, B]))] = "xyz"[int(np.argmax(np.abs(np.array(B) - np.array(A))))]
    E = [(np.array(A), np.array(B), ax) for (A, B), ax in E.items()]

    def to_page(L):
        p = centre + pitch * nf.lat_to_xy(np.asarray(L, float)[None])
        return warp(p, centre, lens, args.magnify, args.profile)[0][0]

    lines = {d: [] for d in "xyz"}
    n_cubes = len(positions)
    for A, B, ax in E:
        La, Lb = nf.lat_of(A, abc), nf.lat_of(B, abc)
        steps = int(np.abs(Lb - La).max())
        pts = [to_page(La + (Lb - La) * t / steps) for t in range(steps + 1)]
        # skip segments entirely outside the page (they'd be invisible anyway)
        for p, q in zip(pts[:-1], pts[1:]):
            if (min(p[0], q[0]) < A2_W + 5 and max(p[0], q[0]) > -5 and
                    min(p[1], q[1]) < A2_H + 5 and max(p[1], q[1]) > -5):
                lines[ax].append((p, q))
    lines = [(pal[d], args.line, lines[d]) for d in "xyz"]

    print(f"{inpage.sum()} dots, {n_cubes} cubes generated, "
          f"{sum(len(l[2]) for l in lines)} line segments on the page")
    svg = render_svg(dots, lines, args.line).replace("<title>Inscribed</title>",
                                                     "<title>Necker Tumbler</title>")
    open(args.out + ".svg", "w").write(svg)
    render_pdf(args.out + ".pdf", dots, lines, args.line, title="Necker Tumbler")
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
    print("wrote", args.out)


if __name__ == "__main__":
    main()
