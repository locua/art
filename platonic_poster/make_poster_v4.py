"""v4 "Inscribed": the five Platonic solids nested inside one another as one
exact compound, floating over an op-art dot field that swells toward the
centre.

The chain of inscriptions (all exact):
  dodecahedron  -> its 20 corners include the cube's 8
  cube          -> alternate corners give the tetrahedron
  tetrahedron   -> its 6 edge midpoints are the octahedron's corners
  octahedron    -> golden-ratio points on its edges are the icosahedron's corners
  icosahedron

Everything is one rigid object seen in a single orthographic view, so the
whole compound reverses in depth at once (a 90-edge Necker figure). The dot
grid is magnified and the dots enlarged toward the centre, a bulge in the
spirit of Vasarely; the solids' lines stay perfectly straight, so they read
as floating in front of the swelling surface.

  python make_poster_v4.py
"""
import argparse
import itertools
import os

import numpy as np

from solids import PHI, random_rotation
from solids_v2 import symmetry_axes

A2_W, A2_H = 420.0, 594.0


# ----------------------------------------------------------------- geometry
def _edges(v, length):
    D = np.linalg.norm(v[:, None] - v[None], axis=-1)
    return [(i, j) for i in range(len(v)) for j in range(i + 1, len(v))
            if abs(D[i, j] - length) < 1e-6]


def compound():
    """Return {name: (vertices (n,3), edges)} for the inscribed chain,
    in a common frame (circumradius of the outer three = sqrt 3)."""
    s3 = np.sqrt(3)
    cube = np.array(list(itertools.product((-1, 1), repeat=3)), float)
    dodec = list(cube)
    for a in (-1 / PHI, 1 / PHI):
        for b in (-PHI, PHI):
            dodec += [(0, a, b), (a, b, 0), (b, 0, a)]
    dodec = np.array(dodec, float)
    tet = np.array([[1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    octa = np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]], float)
    a, b = 1 / PHI, 1 / PHI ** 2
    ico = []
    for s1 in (-1, 1):
        for s2 in (-1, 1):
            ico += [(0, s1 * b, s2 * a), (s1 * b, s2 * a, 0), (s2 * a, 0, s1 * b)]
    ico = np.array(ico, float)
    return {
        "dodecahedron": (dodec, _edges(dodec, 2 / PHI)),
        "cube": (cube, _edges(cube, 2)),
        "tetrahedron": (tet, _edges(tet, 2 * np.sqrt(2))),
        "octahedron": (octa, _edges(octa, np.sqrt(2))),
        "icosahedron": (ico, _edges(ico, 2 * b)),
    }


def merged_graph(parts):
    """Unique 3D points + all edges split at any point lying on them, so
    *intended* incidences (shared corners, corners on edges) are explicit
    and only accidental 2D coincidences get penalised."""
    pts = []
    for v, _ in parts.values():
        for p in v:
            if not any(np.linalg.norm(p - q) < 1e-9 for q in pts):
                pts.append(p)
    pts = np.array(pts)
    idx = lambda p: int(np.argmin(np.linalg.norm(pts - p, axis=1)))
    segs = set()
    for v, edges in parts.values():
        for i, j in edges:
            A, B = v[i], v[j]
            AB = B - A
            t = ((pts - A) @ AB) / (AB @ AB)
            on = np.linalg.norm(A + t[:, None] * AB - pts, axis=1) < 1e-9
            chain = sorted([(tt, k) for k, (tt, o) in enumerate(zip(t, on))
                            if o and -1e-9 <= tt <= 1 + 1e-9])
            for (_, k1), (_, k2) in zip(chain, chain[1:]):
                segs.add((min(k1, k2), max(k1, k2)))
    return pts, sorted(segs)


# --------------------------------------------------------------- view search
MIN_FACE_TILT = 13
_diag = np.array(list(itertools.product((-1, 1), repeat=3)), float) / np.sqrt(3)
FACE_NORMALS = np.vstack([np.eye(3),          # cube faces
                          _diag])             # octahedron faces (incl. tetrahedron faces)

def view_score(R, pts, segs, axes):
    """Higher is better: distinct points stay apart in 2D, no segment grazes
    a point it doesn't touch, and the view is far from every symmetry axis."""
    P = pts @ R[:2].T
    n = len(P)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)[np.triu_indices(n, 1)]
    gap = D.min()
    S = np.array(segs)
    A, B = P[S[:, 0]], P[S[:, 1]]
    AB = B - A
    t = np.clip(((P[None] - A[:, None]) * AB[:, None]).sum(-1) / (AB ** 2).sum(1)[:, None], 0, 1)
    dist = np.linalg.norm(A[:, None] + t[..., None] * AB[:, None] - P[None], axis=-1)
    mask = np.ones_like(dist, bool)
    mask[np.arange(len(S)), S[:, 0]] = False
    mask[np.arange(len(S)), S[:, 1]] = False
    clear = dist[mask].min()
    axv, lim = axes
    angs = np.degrees(np.arccos(np.minimum(1.0, np.abs(axv @ R[2]))))
    flat = (angs - lim).min() + 14      # 14 = exactly at the limit
    if (angs < lim).any():              # some member would collapse into a flat pattern
        return -1.0, gap, clear, flat
    # no face of the tetrahedron / cube / octahedron seen edge-on (it would
    # collapse to a line and weaken the 3D reading)
    tilt = np.degrees(np.arcsin(np.minimum(1.0, np.abs(FACE_NORMALS @ R[2])))).min()
    if tilt < MIN_FACE_TILT:
        return -1.0, gap, clear, flat
    flat = min(flat, tilt + 4)
    return min(gap, 1.6 * clear) + 0.01 * min(flat, 25), gap, clear, flat


def best_views(rng, tries, keep):
    parts = compound()
    pts, segs = merged_graph(parts)
    # every member's own special axes: viewing near any of them flattens that solid
    # (5-fold solids have many more axes, so their limit is a little looser)
    # The compound's own symmetry axes (tetrahedral: shared by tetrahedron,
    # cube and octahedron) must be well clear; the 5-fold solids' extra axes
    # get a light guard so no pentagon is seen face-on.
    lims = {"tetrahedron": 14, "dodecahedron": 8, "icosahedron": 8}
    axv = np.vstack([symmetry_axes(n) for n in lims])
    lim = np.concatenate([[lims[n]] * len(symmetry_axes(n)) for n in lims])
    axes = (axv, lim)
    res = []
    for _ in range(tries):
        R = random_rotation(rng)
        res.append((view_score(R, pts, segs, axes)[0], R))
    res.sort(key=lambda t: -t[0])
    seeds = []                                 # keep only mutually distinct views
    for sc, R in res:
        if sc <= 0:
            break
        if all(abs(R[2] @ S[2]) < np.cos(np.radians(15)) for _, S in seeds):
            seeds.append((sc, R))
        if len(seeds) == keep:
            break
    out = []
    for sc, R in seeds:                        # hill-climb each
        for i in range(400):
            ax = rng.normal(size=3)
            ax /= np.linalg.norm(ax)
            ang = rng.normal() * (0.05 * (1 - i / 400) + 0.003)
            K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
            R2 = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K) @ R
            s2 = view_score(R2, pts, segs, axes)[0]
            if s2 > sc:
                sc, R = s2, R2
        out.append((sc, R))
    out.sort(key=lambda t: -t[0])
    return parts, pts, segs, axes, out


# ------------------------------------------------------------------ the field
def dot_field(cols, centre, lens_r, magnify, r_out, r_in, vignette=0.0, profile=2):
    """Square dot grid seen through a smooth magnifying lens.

    Each dot is a disc on the underlying flat sheet, pushed through the same
    radial mapping as its position: stretched around the circle by the
    magnification and squashed along the radius where the lens compresses.
    So dots become radially foreshortened ellipses toward the rim -- like
    discs printed on a sphere -- and, being the image of non-overlapping
    discs, they never overlap. Disc size also grows toward the centre, so the
    middle swells like a dome.

    Returns (x, y, a_radial, a_tangential, angle_deg), pitch.
    """
    pitch = A2_W / (cols + 1)
    nx = cols + 8
    ny = int(A2_H / pitch) + 8
    gx = (np.arange(nx) - (nx - 1) / 2) * pitch + A2_W / 2
    gy = (np.arange(ny) - (ny - 1) / 2) * pitch + A2_H / 2
    X, Y = np.meshgrid(gx, gy)
    dx, dy = X - centre[0], Y - centre[1]
    d = np.hypot(dx, dy)
    u = np.clip(d / lens_r, 0, 1)
    p = profile
    bump = (1 - u ** p) ** 2                      # 1 at centre, 0 at lens edge
    scale = 1 + magnify * bump                    # tangential stretch
    radial = 1 + magnify * bump - 2 * p * magnify * u ** p * (1 - u ** p)   # d(d')/dd
    Xw, Yw = centre[0] + dx * scale, centre[1] + dy * scale
    rho = pitch * (r_out + (r_in - r_out) * bump)  # disc radius on the sheet
    a_r, a_t = rho * radial, rho * scale
    shrink = np.ones_like(X)
    if vignette:
        e = np.hypot((X - A2_W / 2) / (A2_W / 2), (Y - A2_H / 2) / (A2_H / 2)) / np.sqrt(2)
        shrink = 1 - vignette * np.clip(e, 0, 1) ** 2
    ang = np.degrees(np.arctan2(dy, dx))
    big = np.maximum(a_r, a_t)                    # margin test before vignette
    mg = pitch * 0.5
    keep = (Xw - big > mg) & (Xw + big < A2_W - mg) & (Yw - big > mg) & (Yw + big < A2_H - mg)
    return (Xw[keep], Yw[keep], (a_r * shrink)[keep], (a_t * shrink)[keep], ang[keep]), pitch


# ------------------------------------------------------------------- output
def render_svg(dots, lines, lw, paper="#ffffff", ink="#000000"):
    X, Y, Ar, At, Ang = dots
    circ = "\n".join(
        f'<ellipse cx="{x:.3f}" cy="{y:.3f}" rx="{ar:.3f}" ry="{at:.3f}" '
        f'transform="rotate({g:.2f} {x:.3f} {y:.3f})"/>'
        for x, y, ar, at, g in zip(X, Y, Ar, At, Ang))
    groups = []
    for colour, w, segs in lines:
        body = "\n".join(f'<line x1="{a[0]:.3f}" y1="{a[1]:.3f}" x2="{b[0]:.3f}" y2="{b[1]:.3f}"/>'
                         for a, b in segs)
        groups.append(f'<g stroke="{colour}" stroke-width="{w:.3f}">\n{body}\n</g>')
    return f'''<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" width="{A2_W}mm" height="{A2_H}mm" viewBox="0 0 {A2_W} {A2_H}">
  <title>Inscribed</title>
  <defs><clipPath id="holes">
{circ}
  </clipPath></defs>
  <rect width="{A2_W}" height="{A2_H}" fill="{paper}"/>
  <g fill="{ink}">
{circ}
  </g>
  <g clip-path="url(#holes)" stroke-linecap="round" fill="none">
{chr(10).join(groups)}
  </g>
</svg>
'''


def render_pdf(path, dots, lines, lw, paper="#ffffff", ink="#000000"):
    from reportlab.lib.colors import HexColor
    from reportlab.lib.units import mm
    from reportlab.pdfgen import canvas
    X, Y, Ar, At, Ang = dots
    c = canvas.Canvas(path, pagesize=(A2_W * mm, A2_H * mm))
    c.setTitle("Inscribed")
    c.setFillColor(HexColor(paper))
    c.rect(0, 0, A2_W * mm, A2_H * mm, stroke=0, fill=1)

    K = 0.5522847498                       # cubic Bezier quarter-ellipse constant

    def holes():
        p = c.beginPath()
        for x, y, ar, at, g in zip(X, Y, Ar, At, Ang):
            cs, sn = np.cos(np.radians(g)), np.sin(np.radians(g))
            def P(lx, ly):                 # local ellipse coords -> PDF points
                px = x + lx * cs - ly * sn
                py = y + lx * sn + ly * cs
                return px * mm, (A2_H - py) * mm
            p.moveTo(*P(ar, 0))
            for (x1, y1), (x2, y2), (x3, y3) in (
                    ((ar, K * at), (K * ar, at), (0, at)),
                    ((-K * ar, at), (-ar, K * at), (-ar, 0)),
                    ((-ar, -K * at), (-K * ar, -at), (0, -at)),
                    ((K * ar, -at), (ar, -K * at), (ar, 0))):
                p.curveTo(*P(x1, y1), *P(x2, y2), *P(x3, y3))
            p.close()
        return p
    c.setFillColor(HexColor(ink))
    c.drawPath(holes(), stroke=0, fill=1)
    c.saveState()
    c.clipPath(holes(), stroke=0, fill=0)
    c.setLineCap(1)
    for colour, w, segs in lines:
        c.setLineWidth(w * mm)
        c.setStrokeColor(HexColor(colour))
        for a, b in segs:
            c.line(a[0] * mm, (A2_H - a[1]) * mm, b[0] * mm, (A2_H - b[1]) * mm)
    c.restoreState()
    c.showPage()
    c.save()


# line weight relative to --line: front (white) layer heaviest, core finest
WEIGHT = {"tetrahedron": 1.25, "dodecahedron": 1.0, "cube": 1.0,
          "octahedron": 0.8, "icosahedron": 0.8}

PALETTES = {
    "spectrum": {"dodecahedron": "#00c8ff", "cube": "#ff2bd6", "tetrahedron": "#ffe14d",
                 "octahedron": "#6dff4a", "icosahedron": "#ff7a1a"},
    "cyan": {k: "#00d2ff" for k in ("dodecahedron", "cube", "tetrahedron", "octahedron", "icosahedron")},
    # cool outside -> hot inside: reads as concentric depth shells
    "ember": {"dodecahedron": "#00d2ff", "cube": "#00f0b4", "tetrahedron": "#ffe14d",
              "octahedron": "#ff8a1f", "icosahedron": "#ff2b6a"},
    # two depth layers: the tetrahedron in paper-white (seen only as cuts
    # through the dots -> an illusory figure in front of the sheet), the rest
    # in colour (seen through the holes -> behind the sheet)
    "ember-white": {"dodecahedron": "#00d2ff", "cube": "#00f0b4", "tetrahedron": "#ffffff",
                    "octahedron": "#ff8a1f", "icosahedron": "#ff2b6a"},
    # outer cage (dodecahedron + cube share corners) in one cool colour
    "cage": {"dodecahedron": "#00d2ff", "cube": "#00d2ff", "tetrahedron": "#ffffff",
             "octahedron": "#ff8a1f", "icosahedron": "#ff2bb8"},
    "cage-violet": {"dodecahedron": "#00d2ff", "cube": "#8f7bff", "tetrahedron": "#ffffff",
                    "octahedron": "#ff8a1f", "icosahedron": "#ff2bb8"},
    "duo": {"dodecahedron": "#00d2ff", "cube": "#ff2bd6", "tetrahedron": "#00d2ff",
            "octahedron": "#ff2bd6", "icosahedron": "#00d2ff"},
    "cyan-white": {"dodecahedron": "#00d2ff", "cube": "#00d2ff", "tetrahedron": "#ffffff",
                   "octahedron": "#00d2ff", "icosahedron": "#ffffff"},
}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=36)
    ap.add_argument("--size", type=float, default=280, help="compound width, mm")
    ap.add_argument("--cy", type=float, default=0.47, help="centre height (fraction of page)")
    ap.add_argument("--lens", type=float, default=1.3,
                    help="lens radius as a multiple of the circumsphere radius")
    ap.add_argument("--magnify", type=float, default=0.45)
    ap.add_argument("--r-out", type=float, default=0.30, help="dot radius / spacing at edge")
    ap.add_argument("--r-in", type=float, default=0.47, help="dot radius / spacing at centre")
    ap.add_argument("--profile", type=float, default=3,
                    help="lens shape: 2 = soft dome, 3-4 = flatter top, sharper rim")
    ap.add_argument("--vignette", type=float, default=0.4,
                    help="how much dots shrink toward the page corners (0-1)")
    ap.add_argument("--line", type=float, default=1.7, help="line width, mm")
    ap.add_argument("--palette", default="cage", choices=PALETTES)
    ap.add_argument("--view", type=int, default=0, help="which of the best views")
    ap.add_argument("--apex", default="tetrahedron",
                    help="roll the picture so this solid's outermost corner points straight up "
                         "('none' to disable)")
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="output/v4/inscribed_A2")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    rng = np.random.default_rng(args.seed)
    parts, pts, segs, axes, views = best_views(rng, tries=6000, keep=6)
    if not views:
        raise SystemExit("no view satisfies the constraints; loosen MIN_FACE_TILT")
    sc, R = views[min(args.view, len(views) - 1)]
    _, gap, clear, flat = view_score(R, pts, segs, axes)
    print(f"view {args.view}: min point gap {gap:.3f}, min clearance {clear:.3f}, "
          f"{flat:.1f} deg from nearest symmetry axis")

    # roll in the picture plane (doesn't change the view) so a corner of the
    # chosen solid points straight up -> an upright, pyramidal composition
    if args.apex != "none":
        Pa = parts[args.apex][0] @ R[:2].T
        top = Pa[np.argmax(np.linalg.norm(Pa, axis=1))]
        ang = np.arctan2(top[0], -top[1])           # angle from 'up' (page y grows down)
        c_, s_ = np.cos(ang), -np.sin(ang)          # rotate by -ang
        roll = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
        R = roll @ R

    # scale & place
    P_all = np.vstack([v for v, _ in parts.values()]) @ R[:2].T
    span = np.ptp(P_all, axis=0)
    k = args.size / span[0]
    centre = np.array([A2_W / 2, A2_H * args.cy])
    mid = (P_all.max(0) + P_all.min(0)) / 2
    to_page = lambda p: centre + (p - mid) * k

    order = ["dodecahedron", "cube", "tetrahedron", "octahedron", "icosahedron"]
    pal = PALETTES[args.palette]
    lines = []
    for name in order:
        v, edges = parts[name]
        P = to_page(v @ R[:2].T)
        lines.append((pal[name], args.line * WEIGHT[name], [(P[i], P[j]) for i, j in edges]))

    sphere_r = np.sqrt(3) * k                      # circumsphere, projected
    lens_r = min(args.lens * sphere_r, A2_W / 2 - 8)
    dots, pitch = dot_field(args.cols, centre, lens_r, args.magnify,
                            args.r_out, args.r_in, args.vignette, args.profile)
    print(f"{len(dots[0])} dots, base pitch {pitch:.2f} mm, compound {args.size:.0f} x "
          f"{span[1] * k:.0f} mm")

    svg = render_svg(dots, lines, args.line)
    open(args.out + ".svg", "w").write(svg)
    render_pdf(args.out + ".pdf", dots, lines, args.line)
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
