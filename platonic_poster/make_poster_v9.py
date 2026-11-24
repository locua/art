"""v9 "Cleave": an occlusion-coupled Necker cube -- a new construction.

Idea: the paper is a plane slicing through the centre of a Necker cube.
  * Every part of every edge NEARER than the cube's centre plane is drawn
    only on the white paper: it vanishes under the black dots.
  * Every part FURTHER than the centre plane is visible only inside the
    black dots.
  * Each edge switches mode exactly where it crosses the centre plane.

Why the two reversals should lock together (occlusion / border ownership):
a line that stops exactly at a region's boundary is read as passing BEHIND
that region (or as painted on the region it is visible in).
  * Dots read as HOLES in a white sheet: inside-dot lines are behind the
    sheet, paper lines lie on it -> the paper half is nearer = reading 1.
  * Dots read as DISCS on a white ground: paper lines run behind the discs,
    inside-dot lines lie on them -> the dot half is nearer = reading 2.
So the figure-ground state of the dot field (holes vs discs) and the depth
state of the Necker cube admit only two consistent pairings: flipping one
should flip the other. Dot coverage is set near 50 % black so that neither
figure-ground reading dominates.

The cube is an exact orthographic view; its position, size and angle are
searched so that the near-half corners sit in the white gaps and the
far-half corners sit inside dots, so every junction is visible in its layer.

  python make_poster_v9.py
"""
import argparse
import os

import numpy as np

from make_poster_v4 import A2_W, A2_H, dot_field
from make_poster_v5 import CUBE, CUBE_EDGES, view_from


# ------------------------------------------------------------------ geometry
def split_edges(R, k, centre):
    """Project the cube (half-side 1) with rotation R and scale k (mm per
    unit), centred at `centre`. Returns (near_segments, far_segments,
    corners_page, corner_depths)."""
    V2 = CUBE @ R[:2].T * k + centre
    d = CUBE @ R[2]                                # + = towards the viewer
    near, far = [], []
    for i, j in CUBE_EDGES:
        a, b, da, db = V2[i], V2[j], d[i], d[j]
        if da * db < 0:
            t = da / (da - db)
            m = a + t * (b - a)
            (near if da > 0 else far).append((a, m))
            (near if db > 0 else far).append((m, b))
        else:
            (near if da + db > 0 else far).append((a, b))
    return near, far, V2, d


def clearance(V2):
    """Min distance of a corner from an edge it doesn't belong to (mm)."""
    best = np.inf
    for i, j in CUBE_EDGES:
        a, b = V2[i], V2[j]
        ab = b - a
        for m in range(8):
            if m in (i, j):
                continue
            t = np.clip((V2[m] - a) @ ab / (ab @ ab), 0, 1)
            best = min(best, np.linalg.norm(a + t * ab - V2[m]))
    return best


def corner_fit(V2, depth, dots, lw):
    """Fraction of corners whose junction is fully visible in its own layer:
    far corners well inside a dot, near corners well clear of every dot."""
    X, Y, Ar, At, _ = dots
    P = np.column_stack([X, Y])
    r = np.maximum(Ar, At)
    ok = 0
    worst = np.inf
    for v, dd in zip(V2, depth):
        dist = np.linalg.norm(P - v, axis=1)
        n = np.argmin(dist - r)
        if dd < 0:                                   # far: inside the nearest dot
            n = np.argmin(dist)
            margin = r[n] - dist[n] - 1.5 * lw
        else:                                        # near: outside all dots
            margin = (dist - r).min() - 1.0 * lw
        worst = min(worst, margin)
        ok += margin > 0
    return ok / 8, worst


def visibility(near, far, dots, step=0.6):
    """Per drawn segment set: fraction of each ORIGINAL edge's length visible
    in its own layer. Returns the minimum over the 12 edges (0..1)."""
    X, Y, Ar, At, _ = dots
    P = np.column_stack([X, Y])
    r = np.maximum(Ar, At)

    def inside(pts):
        out = np.zeros(len(pts), bool)
        for s in range(0, len(pts), 512):
            D = np.linalg.norm(pts[s:s + 512, None] - P[None], axis=-1)
            out[s:s + 512] = (D < r[None]).any(axis=1)
        return out
    vis, tot = [], []
    for S, want_inside in ((near, False), (far, True)):
        for a, b in S:
            n = max(2, int(np.linalg.norm(b - a) / step))
            t = np.linspace(0, 1, n)[:, None]
            pts = a + t * (b - a)
            vis.append((inside(pts) == want_inside).mean() * np.linalg.norm(b - a))
            tot.append(np.linalg.norm(b - a))
    return vis, tot


def edge_visibility(R, k, c, dots):
    """Min over the 12 cube edges of the visible fraction (each edge may be
    split into a near and a far piece)."""
    worst = 1.0
    V2 = CUBE @ R[:2].T * k + c
    d = CUBE @ R[2]
    for i, j in CUBE_EDGES:
        near, far = [], []
        a, b, da, db = V2[i], V2[j], d[i], d[j]
        if da * db < 0:
            mpt = a + da / (da - db) * (b - a)
            (near if da > 0 else far).append((a, mpt))
            (near if db > 0 else far).append((mpt, b))
        else:
            (near if da + db > 0 else far).append((a, b))
        vis, tot = visibility(near, far, dots)
        worst = min(worst, sum(vis) / sum(tot))
    return worst


# ------------------------------------------------------------------ rendering
def _ellipse_subpath(x, y, a, b, g):
    c, s = np.cos(np.radians(g)), np.sin(np.radians(g))
    p1 = (x + a * c, y + a * s)
    p2 = (x - a * c, y - a * s)
    return (f"M{p1[0]:.3f} {p1[1]:.3f}A{a:.3f} {b:.3f} {g:.2f} 1 0 {p2[0]:.3f} {p2[1]:.3f}"
            f"A{a:.3f} {b:.3f} {g:.2f} 1 0 {p1[0]:.3f} {p1[1]:.3f}Z")


def render_svg(dots, near, far, colour, lw, corners=((), ()), paper="#ffffff", ink="#000000"):
    X, Y, Ar, At, Ang = dots
    ell = "".join(_ellipse_subpath(*t) for t in zip(X, Y, Ar, At, Ang))
    seg = lambda S: "\n".join(f'<line x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{b[0]:.3f}" '
                              f'y2="{b[1]:.3f}"/>' for a, b in S)
    dot = lambda C: "\n".join(f'<circle cx="{p[0]:.3f}" cy="{p[1]:.3f}" r="{lw / 2:.3f}" '
                               f'fill="{colour}" stroke="none"/>' for p in C)
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{A2_W}mm" height="{A2_H}mm" viewBox="0 0 {A2_W} {A2_H}">
  <title>Cleave</title>
  <defs>
    <clipPath id="holes"><path d="{ell}"/></clipPath>
    <clipPath id="paper"><path clip-rule="evenodd" d="M0 0H{A2_W}V{A2_H}H0Z{ell}"/></clipPath>
  </defs>
  <rect width="{A2_W}" height="{A2_H}" fill="{paper}"/>
  <path d="{ell}" fill="{ink}"/>
  <g stroke="{colour}" stroke-width="{lw:.3f}" stroke-linecap="butt" fill="none">
    <g clip-path="url(#holes)">
{seg(far)}
{dot(corners[1])}
    </g>
    <g clip-path="url(#paper)">
{seg(near)}
{dot(corners[0])}
    </g>
  </g>
</svg>
'''


def render_pdf(path, dots, near, far, colour, lw, corners=((), ()), paper="#ffffff", ink="#000000"):
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    from reportlab.pdfgen.canvas import FILL_EVEN_ODD
    X, Y, Ar, At, Ang = dots
    c = canvas.Canvas(path, pagesize=(A2_W * mm, A2_H * mm))
    c.setTitle("Cleave")
    K = 0.5522847498
    pt = lambda x, y: (x * mm, (A2_H - y) * mm)

    def ellipses(p):
        for x, y, a, b, g in zip(X, Y, Ar, At, Ang):
            cs, sn = np.cos(np.radians(g)), np.sin(np.radians(g))
            P = lambda lx, ly: pt(x + lx * cs - ly * sn, y + lx * sn + ly * cs)
            p.moveTo(*P(a, 0))
            for q in (((a, K * b), (K * a, b), (0, b)), ((-K * a, b), (-a, K * b), (-a, 0)),
                      ((-a, -K * b), (-K * a, -b), (0, -b)), ((K * a, -b), (a, -K * b), (a, 0))):
                p.curveTo(*P(*q[0]), *P(*q[1]), *P(*q[2]))
            p.close()
        return p

    def lines(S, C):
        for a, b in S:
            c.line(*pt(*a), *pt(*b))
        for p in C:
            c.circle(*pt(*p), lw / 2 * mm, stroke=0, fill=1)

    c.setFillColor(HexColor(paper))
    c.rect(0, 0, A2_W * mm, A2_H * mm, stroke=0, fill=1)
    c.setFillColor(HexColor(ink))
    c.drawPath(ellipses(c.beginPath()), stroke=0, fill=1)
    c.setStrokeColor(HexColor(colour))
    c.setLineWidth(lw * mm)
    c.setLineCap(0)
    c.setFillColor(HexColor(colour))
    c.saveState()                                   # far half: inside the dots
    c.clipPath(ellipses(c.beginPath()), stroke=0, fill=0)
    lines(far, corners[1])
    c.restoreState()
    c.saveState()                                   # near half: paper only
    p = c.beginPath()
    p.rect(0, 0, A2_W * mm, A2_H * mm)
    c.clipPath(ellipses(p), stroke=0, fill=0, fillMode=FILL_EVEN_ODD)
    lines(near, corners[0])
    c.restoreState()
    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=18, help="dots across")
    ap.add_argument("--r", type=float, default=0.40,
                    help="dot radius / pitch (0.40 -> ~50%% black, figure-ground balanced)")
    ap.add_argument("--magnify", type=float, default=0.0, help="optional series dot globe")
    ap.add_argument("--size", type=float, default=300, help="cube drawing width, mm")
    ap.add_argument("--cy", type=float, default=0.48)
    ap.add_argument("--azim", type=float, nargs=2, default=(18, 72))
    ap.add_argument("--elev", type=float, nargs=2, default=(16, 38))
    ap.add_argument("--colour", default="#1faf4b",
                    help="one mid green, legible on both black and white")
    ap.add_argument("--line", type=float, default=4.2)
    ap.add_argument("--tries", type=int, default=6000)
    ap.add_argument("--min-vis", type=float, default=0.3,
                    help="every edge must be at least this visible in its own layer")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="output/v9/cleave_A2")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-png", action="store_true")
    ap.add_argument("--control", action="store_true",
                    help="control plate: same cube, every edge seen through the dots "
                         "(plain Bradley-Petry) -- compare to judge the coupling")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    centre0 = np.array([A2_W / 2, A2_H * args.cy])
    dots, pitch = dot_field(args.cols, centre0, 1e-6 if not args.magnify else 260,
                            args.magnify, args.r, args.r, 0.0, 3)
    lw = args.line

    # search view + sub-pitch offset + scale so junctions land in their layer
    # and every edge stays well visible in its own layer
    rng = np.random.default_rng(args.seed)
    cands = []
    for _ in range(args.tries):
        R = view_from(rng.uniform(*args.azim), rng.uniform(*args.elev))
        P = CUBE @ R[:2].T
        span = np.ptp(P, axis=0)[0]
        k = args.size / span * rng.uniform(0.96, 1.04)
        bbox_mid = (P.max(0) + P.min(0)) / 2 * k
        c = centre0 - bbox_mid + rng.uniform(-0.5, 0.5, 2) * pitch
        near, far, V2, depth = split_edges(R, k, c)
        if np.abs(depth).min() < 1e-6:
            continue
        cl = clearance(V2) / pitch
        if cl < 1.2:
            continue
        fit, worst = corner_fit(V2, depth, dots, lw)
        cands.append((fit + 0.02 * min(cl, 4) + 0.01 * min(worst, 3), R, k, c, fit, cl))
    cands.sort(key=lambda t: -t[0])
    best = None
    for sc, R, k, c, fit, cl in cands[:60]:      # costlier visibility test on the top few
        vis = edge_visibility(R, k, c, dots)
        total = sc + 1.5 * min(vis, 0.45)
        if vis >= args.min_vis and (best is None or total > best[0]):
            best = (total, R, k, c, fit, cl, vis)
    if best is None:
        raise SystemExit("no view met --min-vis; lower it or raise --tries")
    _, R, k, c, fit, cl, vis = best
    near, far, V2, depth = split_edges(R, k, c)
    print(f"corners visible in their own layer: {fit * 8:.0f}/8, "
          f"min corner-edge clearance {cl:.1f} dot spacings, "
          f"every edge >= {vis:.0%} visible in its layer, {len(dots[0])} dots, "
          f"black coverage {np.pi * args.r ** 2:.0%}")
    corners = ([v for v, d in zip(V2, depth) if d > 0], [v for v, d in zip(V2, depth) if d < 0])
    if args.control:
        far, near = far + near, []
        corners = ([], list(V2))

    svg = render_svg(dots, near, far, args.colour, lw, corners)
    open(args.out + ".svg", "w").write(svg)
    render_pdf(args.out + ".pdf", dots, near, far, args.colour, lw, corners)
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
