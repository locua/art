"""v7 "Tumbling Field": reversible cubes (tumbling blocks) on a hexagonal dot
lattice -- two multistable effects locked together.

1. Tumbling blocks. An isometric field of cubes: every hexagon reads as a
   cube seen from above or from below, and regions flip independently.
2. Multistable dot lattice (Kubovy). A hexagonal lattice is the maximally
   ambiguous dot lattice: its three row directions are equally strong, so
   the eye keeps regrouping the dots into rows along one, then another.

They are locked together exactly: the three row directions of the lattice
ARE the three cube-edge directions (isometric x, y, z). Every cube corner
sits on a dot centre and every edge runs straight through a line of dot
centres -- no approximation anywhere. Edges are coloured by direction with
the series greens (x chartreuse, y sea-green, z true green), so when the
eye groups the dots along one direction, that green's edges come forward.

Creative licence:
  * --inner-flip: inside a central disc every hexagon is split the other
    way (cube seen from below instead of above), so a circle appears that
    is drawn only by the orientation of the cubes. Which side is the figure
    and which the hole is itself reversible.
  * halftone relief: dots inside each rhombus are sized by the rhombus's
    orientation (the classic three-tone tumbling-block shading); the same
    rhombus is a top face in one reading and a bottom face in the other, so
    the shading supports both.
  * optional dot globe (--magnify): the whole lattice, lines included, is
    warped through the series lens; corners stay exactly on dot centres.

  python make_poster_v7.py
"""
import argparse
import os

import numpy as np

from make_poster_v4 import A2_W, A2_H, render_svg, render_pdf

S3 = np.sqrt(3)
# isometric axes in page units (y grows DOWN on the page)
DIR = {"z": np.array([0.0, -1.0]),            # up
       "x": np.array([-S3 / 2, 0.5]),         # lower-left
       "y": np.array([S3 / 2, 0.5])}          # lower-right


def lattice_point(i, j, pitch):
    """Lattice point i*Y + j*Z (unit = one dot pitch)."""
    return pitch * (i * DIR["y"] + j * DIR["z"])


def build(pitch, L, centre, cover, inner_r=None):
    """Cube field. Hexagon centres at L-scaled lattice points with
    (I + J) % 3 == 0. Returns segments as lattice-index pairs with a
    direction label, and a per-hexagon split ('above' or 'below')."""
    # neighbours of a hexagon centre, in lattice index steps (i, j) for Y and Z
    nb = {"+y": (1, 0), "-y": (-1, 0), "+z": (0, 1), "-z": (0, -1),
          "+x": (-1, -1), "-x": (1, 1)}            # X = -Y - Z
    ring = ["-x", "+z", "-y", "+x", "-z", "+y"]   # angular order 30, 90, ... 330 deg
    n = int(cover / (L * pitch)) + 3
    segs = {}
    hexes = []
    for I in range(-n, n + 1):
        for J in range(-n, n + 1):
            if (I + J) % 3:
                continue
            c = (I * L, J * L)
            pc = centre + lattice_point(*c, pitch)
            if np.abs(pc - centre).max() > cover:
                continue
            split = "above"
            if inner_r is not None and np.linalg.norm(pc - centre) < inner_r:
                split = "below"
            hexes.append((c, split))
            V = {k: (c[0] + L * v[0], c[1] + L * v[1]) for k, v in nb.items()}
            for a, b in zip(ring, ring[1:] + ring[:1]):            # outline
                key = tuple(sorted([V[a], V[b]]))
                segs[key] = None
            # spokes: "above" = near corner on top -> edges to -x, -y, -z
            spokes = ["-x", "-y", "-z"] if split == "above" else ["+x", "+y", "+z"]
            for s in spokes:
                segs[tuple(sorted([c, V[s]]))] = None
    out = []
    for a, b in segs:
        d = (b[0] - a[0], b[1] - a[1])
        if d[1] == 0:
            lab = "y"
        elif d[0] == 0:
            lab = "z"
        else:
            lab = "x"
        out.append((a, b, lab))
    return out, hexes


def warp(P, centre, lens_r, magnify, profile):
    """Series lens: radial magnification with smooth falloff. Returns warped
    points plus radial / tangential stretch and the radial angle."""
    d = P - centre
    r = np.linalg.norm(d, axis=-1)
    u = np.clip(r / lens_r, 0, 1) if lens_r else np.ones_like(r)
    p = profile
    bump = (1 - u ** p) ** 2 if magnify else np.zeros_like(r)
    scale = 1 + magnify * bump
    radial = 1 + magnify * bump - 2 * p * magnify * u ** p * (1 - u ** p)
    W = centre + d * scale[..., None]
    ang = np.degrees(np.arctan2(d[..., 1], d[..., 0]))
    return W, radial, scale, ang, bump




def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pitch", type=float, default=9.0, help="dot spacing, mm")
    ap.add_argument("--L", type=int, default=5, help="cube edge, in dot steps")
    ap.add_argument("--inner-flip", type=float, default=150.0,
                    help="radius (mm) of the central disc where cubes are split the other way")
    ap.add_argument("--relief", type=float, nargs=3, default=(0.78, 1.0, 1.18),
                    metavar=("FLAT", "LEFT", "RIGHT"), help="dot size factor per rhombus orientation")
    ap.add_argument("--r", type=float, default=0.38, help="base dot radius / pitch")
    ap.add_argument("--magnify", type=float, default=0.35)
    ap.add_argument("--lens", type=float, default=200, help="lens radius, mm")
    ap.add_argument("--profile", type=float, default=3)
    ap.add_argument("--r-in", type=float, default=0.44, help="dot radius / pitch at lens centre")
    ap.add_argument("--vignette", type=float, default=0.35)
    ap.add_argument("--line", type=float, default=1.8)
    ap.add_argument("--cy", type=float, default=0.5)
    ap.add_argument("--out", default="output/v7/tumbling_field_A2")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    from make_poster_v5 import PALETTES
    g = PALETTES["greens"]["C"][0]
    pal = {"x": g[0], "y": g[1], "z": g[2]}

    pitch, L = args.pitch, args.L
    centre = np.array([A2_W / 2, A2_H * args.cy])
    cover = A2_H * 0.75

    # ---- dot lattice (index space), every lattice point
    n = int(cover / pitch) + 4
    ii, jj = np.meshgrid(np.arange(-2 * n, 2 * n + 1), np.arange(-2 * n, 2 * n + 1), indexing="ij")
    ii, jj = ii.ravel(), jj.ravel()
    P0 = centre + ii[:, None] * pitch * DIR["y"] + jj[:, None] * pitch * DIR["z"]
    keep = (np.abs(P0[:, 0] - A2_W / 2) < A2_W) & (np.abs(P0[:, 1] - A2_H / 2) < A2_H)
    ii, jj, P0 = ii[keep], jj[keep], P0[keep]

    segs, hexes = build(pitch, L, centre, cover, args.inner_flip or None)

    # ---- classify every dot: on an edge, or inside which rhombus orientation
    on_edge = set()
    for a, b, lab in segs:
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        for t in range(steps + 1):
            on_edge.add((a[0] + (b[0] - a[0]) * t // steps, a[1] + (b[1] - a[1]) * t // steps))
    # rhombus orientation: find the hexagon (nearest centre) and the sector
    hex_c = np.array([c for c, _ in hexes], float)
    hex_below = np.array([s == "below" for _, s in hexes])
    Hp = centre + hex_c[:, 0:1] * pitch * DIR["y"] + hex_c[:, 1:2] * pitch * DIR["z"]
    tone = np.full(len(P0), -1)
    near = np.concatenate([np.argmin(((P0[s:s + 2000, None] - Hp[None]) ** 2).sum(-1), axis=1)
                           for s in range(0, len(P0), 2000)])
    for k in range(len(P0)):
        if (ii[k], jj[k]) in on_edge:
            continue
        v = P0[k] - Hp[near[k]]
        a = np.degrees(np.arctan2(-v[1], v[0])) % 360            # math angle (y up)
        if hex_below[near[k]]:
            # spokes at 90, 210, 330 -> sectors: 330-90 right, 90-210 left, 210-330 flat
            tone[k] = 0 if 210 <= a < 330 else (1 if 90 <= a < 210 else 2)
        else:
            # spokes at 30, 150, 270 -> sectors: 30-150 flat, 150-270 left, 270-30 right
            tone[k] = 0 if 30 <= a < 150 else (1 if 150 <= a < 270 else 2)

    # ---- sizes, warp
    W, radial, scale, ang, bump = warp(P0, centre, args.lens if args.magnify else 0,
                                       args.magnify, args.profile)
    rho = pitch * (args.r + (args.r_in - args.r) * bump)
    f = np.ones(len(P0))
    for t in range(3):
        f[tone == t] = args.relief[t]
    rho = np.minimum(rho * f, 0.455 * pitch)      # never touch a neighbour
    a_r, a_t = rho * radial, rho * scale
    if args.vignette:
        e = np.hypot((P0[:, 0] - A2_W / 2) / (A2_W / 2), (P0[:, 1] - A2_H / 2) / (A2_H / 2)) / S3 * 1.2
        sh = 1 - args.vignette * np.clip(e, 0, 1) ** 2
        a_r, a_t = a_r * sh, a_t * sh
    big = np.maximum(a_r, a_t)
    m = pitch * 1.4
    inpage = ((W[:, 0] - big > m) & (W[:, 0] + big < A2_W - m) &
              (W[:, 1] - big > m) & (W[:, 1] + big < A2_H - m))
    dots = (W[inpage, 0], W[inpage, 1], a_r[inpage], a_t[inpage], ang[inpage])

    # ---- lines: each edge as straight sub-segments between consecutive warped
    # lattice points, so lines follow the lens and still pass through dot centres
    def page_pt(i, j):
        p = centre + i * pitch * DIR["y"] + j * pitch * DIR["z"]
        return warp(p[None], centre, args.lens if args.magnify else 0, args.magnify,
                    args.profile)[0][0]

    lines = {d: [] for d in "xyz"}
    for a, b, lab in segs:
        steps = max(abs(b[0] - a[0]), abs(b[1] - a[1]))
        pts = [page_pt(a[0] + (b[0] - a[0]) * t / steps, a[1] + (b[1] - a[1]) * t / steps)
               for t in range(steps + 1)]
        lines[lab] += list(zip(pts[:-1], pts[1:]))
    lines = [(pal[d], args.line, lines[d]) for d in "xyz"]

    print(f"{inpage.sum()} dots, {len(hexes)} cubes, {sum(len(l[2]) for l in lines)} line segments")
    svg = render_svg(dots, lines, args.line).replace("<title>Inscribed</title>",
                                                     "<title>Tumbling Field</title>")
    open(args.out + ".svg", "w").write(svg)
    render_pdf(args.out + ".pdf", dots, lines, args.line, title="Tumbling Field")
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
