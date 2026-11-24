"""v6 "Schröder": a reversible staircase over the dot globe.

Geometry (exact):
  * The stair profile is a zigzag of risers and treads that starts AND ends
    with a riser, plus a floor strip before it and a landing after it.
    That profile is centrally symmetric, and so is its extrusion -- the 3D
    object maps onto itself under point reflection through its centre.
  * In a parallel (orthographic) view, point reflection of the object equals
    a 180-degree turn of the picture; and reading the picture with depth
    reversed equals seeing the mirror object. So the depth-reversed reading
    of this drawing is *the same staircase turned upside down*: stairs seen
    from above, or an overhang seen from below -- equally valid, which is
    why Schröder's staircase flips so readily.

Style (consistent with the series):
  * edges coloured by 3D direction, same greens as the green Nested Necker:
    x (treads, floor, landing) chartreuse, y (extrusion) sea-green,
    z (risers) true green
  * equal line weights: weighting by depth would favour one reading
  * optional halftone relief: dots inside the stair's faces are resized by
    face orientation, so the staircase also appears in the dot field

  python make_poster_v6.py
"""
import argparse
import os

import numpy as np

from make_poster_v4 import A2_W, A2_H, dot_field, render_svg, render_pdf
from make_poster_v5 import view_from


# ------------------------------------------------------------------ geometry
def profile(n, run=1.0, rise=1.0, floor=1.5):
    """Centrally symmetric stair profile in the (x, z) plane."""
    pts = [(-floor, 0.0), (0.0, 0.0)]
    for i in range(n):
        pts += [(i * run, (i + 1) * rise), ((i + 1) * run, (i + 1) * rise)]
    pts += [(n * run, (n + 1) * rise), (n * run + floor, (n + 1) * rise)]
    P = np.array(pts)
    c = (P[0] + P[-1]) / 2
    assert np.allclose(np.sort(2 * c - P, axis=0), np.sort(P, axis=0)), "not centrally symmetric"
    return P - c


def staircase(n=6, width=3.2, run=1.0, rise=1.0, floor=1.5):
    """Vertices (m,3), edges [(i, j, direction)], faces [(quad idx, kind)]."""
    P = profile(n, run, rise, floor)
    m = len(P)
    V = np.array([(x, y, z) for y in (-width / 2, width / 2) for x, z in P])
    edges, faces = [], []
    for k in range(m - 1):
        d = "z" if np.isclose(P[k, 0], P[k + 1, 0]) else "x"
        edges += [(k, k + 1, d), (m + k, m + k + 1, d)]
        faces.append(([k, k + 1, m + k + 1, m + k], "riser" if d == "z" else "tread"))
    edges += [(k, m + k, "y") for k in range(m)]
    # sanity: the 3D object is invariant under point reflection
    assert np.allclose(np.sort(-V, axis=0), np.sort(V, axis=0))
    return V, edges, faces


# --------------------------------------------------------------- view search
def score(R, V, edges, min_tilt=16, min_aspect=0.0):
    """Clearance of corners from unrelated edges, no face near edge-on, and a
    compact block (the extrusion crosses the flight instead of running along
    it, as in Schröder's drawing)."""
    tilt = np.degrees(np.arcsin(np.abs(R[2]).min()))        # normals are x, y, z
    if tilt < min_tilt:
        return -1, tilt, 0
    P = V @ R[:2].T
    w, h = np.ptp(P, axis=0)
    if min(w, h) / max(w, h) < min_aspect:
        return -1, tilt, 0
    E = np.array([(i, j) for i, j, _ in edges])
    A, B = P[E[:, 0]], P[E[:, 1]]
    AB = B - A
    t = np.clip(((P[None] - A[:, None]) * AB[:, None]).sum(-1) / (AB ** 2).sum(1)[:, None], 0, 1)
    dist = np.linalg.norm(A[:, None] + t[..., None] * AB[:, None] - P[None], axis=-1)
    mask = np.ones_like(dist, bool)
    mask[np.arange(len(E)), E[:, 0]] = False
    mask[np.arange(len(E)), E[:, 1]] = False
    clear = dist[mask].min()
    return clear + 0.002 * min(tilt, 30), tilt, clear


def best_view(V, edges, azim, elev, rng, tries=6000):
    best = (-np.inf, None)
    for _ in range(tries):
        R = view_from(rng.uniform(*azim), rng.uniform(*elev))
        s = score(R, V, edges)[0]
        if s > best[0]:
            best = (s, R)
    return best[1]


# -------------------------------------------------------------- halftone relief
def point_in_quad(px, py, quad):
    """Vectorised point-in-convex-quad test (quad: 4x2, any winding)."""
    inside = np.ones(px.shape, bool)
    sgn = None
    for a, b in zip(quad, np.roll(quad, -1, axis=0)):
        cr = (b[0] - a[0]) * (py - a[1]) - (b[1] - a[1]) * (px - a[0])
        s = np.sign(cr)
        sgn = s if sgn is None else sgn
        inside &= (s == sgn) | (s == 0)
    return inside


# ---------------------------------------------------------------- palettes
def _palette(name):
    from make_poster_v5 import PALETTES
    outer = PALETTES["greens"]["C"][0]            # [x, y, z] greens of the series
    if name == "greens":
        return {"x": outer[0], "y": outer[1], "z": outer[2]}
    if name == "mono":
        return {k: outer[1] for k in "xyz"}
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--width", type=float, default=3.5, help="stair width, in step units")
    ap.add_argument("--floor", type=float, default=1.5, help="floor / landing depth")
    ap.add_argument("--azim", type=float, nargs=2, default=(-130, -50),
                    help="camera azimuth range; -90 = looking straight at the stair profile")
    ap.add_argument("--elev", type=float, nargs=2, default=(18, 40))
    ap.add_argument("--size", type=float, default=255, help="figure width, mm")
    ap.add_argument("--cy", type=float, default=0.47)
    ap.add_argument("--palette", default="greens", choices=["greens", "mono"])
    ap.add_argument("--relief", type=float, default=0.25,
                    help="halftone relief: risers' dots grow, treads' shrink, by this fraction")
    ap.add_argument("--cols", type=int, default=36)
    ap.add_argument("--lens", type=float, default=1.25,
                    help="lens radius / radius of the figure's bounding circle")
    ap.add_argument("--magnify", type=float, default=0.45)
    ap.add_argument("--profile", type=float, default=3)
    ap.add_argument("--r-out", type=float, default=0.30)
    ap.add_argument("--r-in", type=float, default=0.47)
    ap.add_argument("--vignette", type=float, default=0.4)
    ap.add_argument("--line", type=float, default=1.8)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", default="output/v6/schroder_A2")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    V, edges, faces = staircase(args.steps, args.width, floor=args.floor)
    R = best_view(V, edges, args.azim, args.elev, np.random.default_rng(args.seed))
    s, tilt, clear = score(R, V, edges)
    print(f"view: clearance {clear:.3f} step units, faces >= {tilt:.1f} deg from edge-on")

    P2 = V @ R[:2].T
    span = np.ptp(P2, axis=0)
    k = args.size / span[0]
    centre = np.array([A2_W / 2, A2_H * args.cy])
    mid = (P2.max(0) + P2.min(0)) / 2
    page = centre + (P2 - mid) * k

    pal = _palette(args.palette)
    lines = []
    for d in "yxz":                               # draw order: extrusion under profile
        segs = [(page[i], page[j]) for i, j, dd in edges if dd == d]
        lines.append((pal[d], args.line, segs))

    bound_r = np.linalg.norm(page - centre, axis=1).max()
    lens_r = min(args.lens * bound_r, A2_W / 2 - 8)
    (X, Y, Ar, At, Ang), pitch = dot_field(args.cols, centre, lens_r, args.magnify, args.r_out,
                                           args.r_in, args.vignette, args.profile)
    if args.relief:
        # risers darken (dots grow), treads lighten (dots shrink): halftone
        # shading of the faces. Growth is capped by the distance to the
        # nearest neighbour so no two dots ever touch.
        XY = np.column_stack([X, Y])
        D = np.linalg.norm(XY[:, None] - XY[None], axis=-1)
        np.fill_diagonal(D, np.inf)
        nn = D.min(axis=1)
        cap = 0.45 * nn / np.maximum(Ar, At)
        f = np.ones_like(X)
        for quad, kind in faces:
            inside = point_in_quad(X, Y, page[quad])
            if kind == "riser":
                f[inside] = np.minimum(1 + args.relief, np.maximum(cap[inside], 1.0))
            else:
                f[inside] = 1 - args.relief
        Ar, At = Ar * f, At * f
    dots = (X, Y, Ar, At, Ang)
    print(f"{len(X)} dots, figure {args.size:.0f} x {span[1] * k:.0f} mm")

    svg = render_svg(dots, lines, args.line).replace("<title>Inscribed</title>",
                                                     "<title>Schröder</title>")
    open(args.out + ".svg", "w").write(svg)
    render_pdf(args.out + ".pdf", dots, lines, args.line, title="Schröder")
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
