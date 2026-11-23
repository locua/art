"""v5 "Nested Necker": only cubes -- Escher's compound of three cubes,
nested inside itself at shrinking scales -- over the v4 dot globe.

Geometry (all exact):
  * A level is three equal cubes, each turned 45 deg about a different
    coordinate axis (x, y, z). They interpenetrate: the compound M.C. Escher
    put on the towers of "Waterfall".
  * The next level is the same compound scaled by 1/sqrt 2. That is forced:
    sqrt 2 is the side of the largest axis-aligned cube that fits inside the
    common core of the three cubes (checked numerically below).
  * At that scale every corner of every inner cube lies exactly on a face of
    an outer cube *and* pokes out through another outer cube: the levels both
    kiss and interlock.

Every cube is drawn as an exact orthographic wireframe, so each one -- and
the whole nest -- is a Necker figure that can flip in depth.

  python make_poster_v5.py
"""
import argparse
import itertools
import os

import numpy as np

from solids import random_rotation
from make_poster_v4 import A2_W, A2_H, dot_field, render_svg, render_pdf

CUBE = np.array(list(itertools.product((-1, 1), repeat=3)), float)
CUBE_EDGES = [(i, j) for i in range(8) for j in range(i + 1, 8)
              if np.count_nonzero(CUBE[i] != CUBE[j]) == 1]


def rot(axis, ang):
    c, s = np.cos(ang), np.sin(ang)
    i, j = [(1, 2), (2, 0), (0, 1)][axis]
    R = np.eye(3)
    R[i, i], R[i, j], R[j, i], R[j, j] = c, -s, s, c
    return R


def nest(depth):
    """Alternating chain, outermost first, each step scaled by 1/sqrt 2:

        C(1)  >  K(1/sqrt2)  >  C(1/2)  >  K(1/(2 sqrt2))  > ...

    C(h): Escher's compound -- three cubes of half-side h, each turned 45 deg
          about x, y or z. The three interpenetrate.
    K(h): the kernel -- one axis-aligned cube of half-side h, the largest
          cube that fits inside the compound around it. Each of its corners
          touches a face of all three surrounding cubes; and the next compound
          fits exactly inside it, every corner landing on a kernel face.
    Returns [(step, kind, axis, vertices)].
    """
    out = []
    for step in range(depth):
        h = (1 / np.sqrt(2)) ** step
        if step % 2 == 0:
            for a in range(3):
                out.append((step, "C", a, CUBE @ rot(a, np.pi / 4).T * h))
        else:
            out.append((step, "K", -1, CUBE * h))
    return out


def _half_side_and_rot(kind, a, step):
    h = (1 / np.sqrt(2)) ** step
    return h, (rot(a, np.pi / 4) if kind == "C" else np.eye(3))


def verify(cubes):
    """Assert the exact contacts: every corner of step n+1 lies on a face of
    step n (on all three faces when the outer step is a compound), and no
    corner is outside its container -- except in the interlocking compounds."""
    depth = max(s for s, *_ in cubes) + 1
    for n in range(1, depth):
        outer = [_half_side_and_rot(k, a, s) for s, k, a, _ in cubes if s == n - 1]
        for s, k, a, V in cubes:
            if s != n:
                continue
            for p in V:
                q = np.array([np.abs(R.T @ p / h).max() for h, R in outer])
                assert np.all(q <= 1 + 1e-9), "corner outside its container"
                assert np.all(np.isclose(q, 1)) if len(outer) == 3 else np.isclose(q[0], 1), \
                    "corner not touching"
    return True


# --------------------------------------------------------------- view search
# Octahedral symmetry: viewing near a cube axis, body diagonal or face
# diagonal flattens the figure; and no face of any of the 3 orientations
# should be seen edge-on.
_ax = [np.eye(3)[i] for i in range(3)]
_diag = [np.array(p, float) / np.sqrt(3) for p in itertools.product((-1, 1), repeat=3)]
_fd = [np.array(v, float) / np.sqrt(2) for v in
       [(1, 1, 0), (1, -1, 0), (1, 0, 1), (1, 0, -1), (0, 1, 1), (0, 1, -1)]]
SYM_AXES = np.array(_ax + _diag + _fd)
FACE_NORMALS = np.array(_ax + _fd)        # normals of the axis-, and 45deg-turned faces


def graph(cubes):
    pts, segs = [], []
    for *_, V in cubes:
        base = len(pts)
        pts.extend(V)
        segs.extend((base + i, base + j) for i, j in CUBE_EDGES)
    return np.array(pts), segs


def view_score(R, pts, segs, min_axis=14, min_tilt=12):
    v = R[2]
    flat = np.degrees(np.arccos(min(1.0, np.abs(SYM_AXES @ v).max())))
    tilt = np.degrees(np.arcsin(min(1.0, np.abs(FACE_NORMALS @ v).min())))
    if flat < min_axis or tilt < min_tilt:
        return -1.0, 0, 0, flat, tilt
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
    # clearance / spacing dominate; axis and tilt limits are hard constraints above
    return min(gap, 1.6 * clear) + 0.0003 * min(flat, 25) + 0.0003 * min(tilt, 25), gap, clear, flat, tilt


def view_from(azim, elev):
    """Camera looking at the origin from azimuth/elevation (z up); rows of R
    are screen-right, screen-up (negated later for page y), view direction."""
    a, e = np.radians(azim), np.radians(elev)
    d = np.array([np.cos(e) * np.cos(a), np.cos(e) * np.sin(a), np.sin(e)])   # towards camera
    right = np.array([-np.sin(a), np.cos(a), 0.0])
    up = np.cross(d, right)
    return np.array([right, -up, d])            # page y grows downward


def elevation(R):
    return np.degrees(np.arcsin(abs(R[2][2])))


def best_view(pts, segs, rng, tries=8000, lim=(12, 8), elev=None):
    best = (-np.inf, None)
    for _ in range(tries):
        if elev:
            R = view_from(rng.uniform(0, 360), rng.uniform(*elev))
        else:
            R = random_rotation(rng)
        sc = view_score(R, pts, segs, *lim)[0]
        if sc > best[0]:
            best = (sc, R)
    sc, R = best
    for i in range(1500):
        ax = rng.normal(size=3)
        ax /= np.linalg.norm(ax)
        ang = rng.normal() * (0.04 * (1 - i / 1500) + 0.002)
        K = np.array([[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]])
        R2 = (np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K) @ R
        if elev and not (elev[0] <= elevation(R2) <= elev[1]):
            continue
        s2 = view_score(R2, pts, segs, *lim)[0]
        if s2 > sc:
            sc, R = s2, R2
    return R


# colour per step of the chain: outer compound cyan (seen through the holes,
# behind the sheet); kernels in paper-white (visible only as cuts through the
# dots -> illusory Necker cubes in front of the sheet); inner compound hot.
def _search(pts, segs, seed, lim, elev):
    return best_view(pts, segs, np.random.default_rng(seed), tries=8000, lim=lim, elev=elev)


PALETTES = {
    # compounds coloured by turning axis in the print primaries C / M / Y (the
    # same three colours recur at every scale), kernels in paper-white
    "cmy": {"C": ["#00c8ff", "#ff2bb8", "#ffe14d"], "K": "#ffffff"},
    "layers": ["#00d2ff", "#ffffff", "#ff2bb8", "#ffffff", "#ffe14d"],
    "layers-warm": ["#00d2ff", "#ffffff", "#ff8a1f", "#ffffff", "#ffe14d"],
    "cyan-white": ["#00d2ff", "#ffffff", "#00d2ff", "#ffffff", "#00d2ff"],
}
WEIGHT = [1.0, 1.3, 0.85, 1.1, 0.75]   # per step; white kernels heaviest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--depth", type=int, default=3,
                    help="steps in the chain C, K, C, K ... (3 = C K C)")
    ap.add_argument("--palette", default="cmy", choices=PALETTES)
    ap.add_argument("--size", type=float, default=280, help="figure width, mm")
    ap.add_argument("--cy", type=float, default=0.47)
    ap.add_argument("--cols", type=int, default=36)
    ap.add_argument("--lens", type=float, default=1.3)
    ap.add_argument("--magnify", type=float, default=0.45)
    ap.add_argument("--profile", type=float, default=3)
    ap.add_argument("--r-out", type=float, default=0.30)
    ap.add_argument("--r-in", type=float, default=0.47)
    ap.add_argument("--vignette", type=float, default=0.4)
    ap.add_argument("--line", type=float, default=1.7)
    ap.add_argument("--elev", type=float, nargs=2, default=(15, 35), metavar=("MIN", "MAX"),
                    help="restrict the view to this elevation range (deg above the horizon, z up)")
    ap.add_argument("--min-axis", type=float, default=12)
    ap.add_argument("--min-tilt", type=float, default=8)
    ap.add_argument("--upright", default="z", choices=["x", "y", "z", "none"],
                    help="roll so this axis is vertical on the page")
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--out", default="output/v5/nested_necker_A2")
    ap.add_argument("--dpi", type=int, default=300)
    ap.add_argument("--no-png", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    cubes = nest(args.depth)
    verify(cubes)
    pts, segs = graph(cubes)
    # search from several seeds in parallel, keep the cleanest view
    from multiprocessing import Pool
    lim = (args.min_axis, args.min_tilt)
    elev = tuple(args.elev) if args.elev else None
    with Pool() as pool:
        Rs = pool.starmap(_search, [(pts, segs, args.seed * 100 + i, lim, elev) for i in range(4)])
    R = max(Rs, key=lambda R: view_score(R, pts, segs, *lim)[0])
    sc, gap, clear, flat, tilt = view_score(R, pts, segs, *lim)
    print(f"view: min corner gap {gap:.3f}, min clearance {clear:.3f}, "
          f"{flat:.1f} deg from symmetry axes, faces >= {tilt:.1f} deg from edge-on, "
          f"elevation {elevation(R):.1f} deg")

    if args.upright != "none":                  # roll: chosen axis points straight up
        a = R[:2, "xyz".index(args.upright)]
        ang = np.arctan2(a[0], -a[1])
        if a[1] > 0:                            # either end of the axis may go up
            ang = np.arctan2(-a[0], a[1])
        c_, s_ = np.cos(ang), -np.sin(ang)
        R = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]]) @ R

    P_all = pts @ R[:2].T
    span = np.ptp(P_all, axis=0)
    k = args.size / span[0]
    centre = np.array([A2_W / 2, A2_H * args.cy])
    mid = (P_all.max(0) + P_all.min(0)) / 2
    to_page = lambda p: centre + (p - mid) * k

    pal = PALETTES[args.palette]
    lines = []
    for step, kind, axis, V in cubes:
        P = to_page(V @ R[:2].T)
        if isinstance(pal, dict):
            colour = pal["C"][axis] if kind == "C" else pal["K"]
        else:
            colour = pal[step]
        lines.append((colour, args.line * WEIGHT[step],
                      [(P[i], P[j]) for i, j in CUBE_EDGES]))

    sphere_r = np.sqrt(3) * k                  # circumsphere of the outer level
    lens_r = min(args.lens * sphere_r, A2_W / 2 - 8)
    dots, pitch = dot_field(args.cols, centre, lens_r, args.magnify, args.r_out,
                            args.r_in, args.vignette, args.profile)
    print(f"{len(dots[0])} dots, {len(cubes)} cubes, figure {args.size:.0f} x {span[1] * k:.0f} mm")

    svg = render_svg(dots, lines, args.line).replace("<title>Inscribed</title>",
                                                     "<title>Nested Necker</title>")
    open(args.out + ".svg", "w").write(svg)
    render_pdf(args.out + ".pdf", dots, lines, args.line, title="Nested Necker")
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
