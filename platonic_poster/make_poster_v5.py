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


TRIPLE_TURNS = {"zz": (2, 2), "zx": (2, 0)}


def triple(turns="zz"):
    """Three nested Necker cubes. Each is the previous one turned 45 deg about
    one of its own axes and scaled by 1/sqrt 2 -- the largest cube that fits
    inside it in that orientation, so all 8 corners land exactly on the
    faces of the cube around it.  "zz": both turns about the upright axis
    (the inner cube is parallel to the outer one, the middle one sits
    diagonally between); "zx": the second turn is about a horizontal axis.
    Returns [(step, "N", -1, vertices, rotation)]."""
    R = np.eye(3)
    out = [(0, "N", -1, CUBE.copy(), R)]
    for step, a in enumerate(TRIPLE_TURNS[turns], start=1):
        R = R @ rot(a, np.pi / 4)               # turn about the cube's own axis
        h = (1 / np.sqrt(2)) ** step
        out.append((step, "N", -1, CUBE @ R.T * h, R))
    return out


def verify_triple(cubes):
    for (s0, _, _, _, R0), (s1, _, _, V1, _) in zip(cubes, cubes[1:]):
        h0 = (1 / np.sqrt(2)) ** s0
        for p in V1:
            q = np.abs(R0.T @ p / h0).max()
            assert np.isclose(q, 1), "corner not on the outer cube's face"
    return True


def face_normals(cubes):
    """Every cube's three face directions (= its edge directions)."""
    ns = []
    for c in cubes:
        V = c[3]
        for i, j in CUBE_EDGES:
            d = V[j] - V[i]
            d /= np.linalg.norm(d)
            if not any(abs(d @ n) > 1 - 1e-9 for n in ns):
                ns.append(d)
    return np.array(ns)


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
_NORMALS = [FACE_NORMALS]                  # replaced per structure in main()


def graph(cubes):
    pts, segs = [], []
    for c in cubes:
        V = c[3]
        base = len(pts)
        pts.extend(V)
        segs.extend((base + i, base + j) for i, j in CUBE_EDGES)
    return np.array(pts), segs


def view_score(R, pts, segs, min_axis=14, min_tilt=12):
    v = R[2]
    flat = np.degrees(np.arccos(min(1.0, np.abs(SYM_AXES @ v).max())))
    tilt = np.degrees(np.arcsin(min(1.0, np.abs(_NORMALS[0] @ v).min())))
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
def _search(pts, segs, seed, lim, elev, normals):
    _NORMALS[0] = normals                      # (worker processes)
    return best_view(pts, segs, np.random.default_rng(seed), tries=8000, lim=lim, elev=elev)


def _greens():
    """Green palette for the compound chain, built in OKLCH:
      hue       <- which axis a cube is turned about (x chartreuse, z true green,
                   y sea-green; z, the upright anchor, takes the middle hue).
                   The same hue recurs at both scales: each cube's half-size
                   twin is its colour match.
      lightness <- depth of nesting: outer compound in saturated mid-greens,
                   inner compound in lighter tints, glowing toward the core.
                   The three axis greens also differ in lightness, so they
                   stay distinct for red-green colour-blind viewers.
      kernel    <- the OKLab mean of the three outer greens (it touches all
                   three), lifted almost to white so it keeps its illusory,
                   in-front-of-the-sheet quality.
    """
    from colours import from_oklch, max_chroma, hex_to_oklab, oklab_to_hex
    H = [125, 168, 145]                     # axis 0 = x, 1 = y, 2 = z
    Lo = [0.86, 0.80, 0.76]
    Li = [0.92, 0.89, 0.85]
    outer = [from_oklch(L, 0.92 * max_chroma(L, h), h) for L, h in zip(Lo, H)]
    inner = [from_oklch(L, 0.90 * max_chroma(L, h), h) for L, h in zip(Li, H)]
    mix = np.mean([hex_to_oklab(c) for c in outer], axis=0)
    kernel = oklab_to_hex(np.array([0.965, mix[1] * 0.35, mix[2] * 0.35]))
    return {"C": {0: outer, 2: inner, 4: inner}, "K": kernel}


def _green_ramp(n=3):
    """Three nested cubes: hue steps through the green family as each cube
    turns 45 deg; lightness rises inward."""
    from colours import from_oklch, max_chroma
    Ls = np.linspace(0.74, 0.90, n)
    Hs = np.linspace(128, 168, n)
    return [from_oklch(L, 0.92 * max_chroma(L, h), h) for L, h in zip(Ls, Hs)]


def _green_pair():
    """Three nested cubes (turns "zz"): after two 45-degree turns the inner
    cube is parallel to the outer one, so they share a hue (sea-green) and
    differ only in lightness, brightening inward; the middle cube, the one
    turned diagonally between them, takes the complementary chartreuse."""
    from colours import from_oklch, max_chroma
    spec = [(0.78, 165), (0.86, 122), (0.93, 165)]      # outer, middle, inner
    return [from_oklch(L, 0.92 * max_chroma(L, h), h) for L, h in spec]


PALETTES = {
    "greens": _greens(),
    "green-pair": _green_pair(),
    "green-ramp": _green_ramp(),
    # compounds coloured by turning axis in the print primaries C / M / Y (the
    # same three colours recur at every scale), kernels in paper-white
    "cmy": {"C": ["#00c8ff", "#ff2bb8", "#ffe14d"], "K": "#ffffff"},
    "layers": ["#00d2ff", "#ffffff", "#ff2bb8", "#ffffff", "#ffe14d"],
    "layers-warm": ["#00d2ff", "#ffffff", "#ff8a1f", "#ffffff", "#ffe14d"],
    "cyan-white": ["#00d2ff", "#ffffff", "#00d2ff", "#ffffff", "#00d2ff"],
}
WEIGHT = [1.0, 1.3, 0.85, 1.1, 0.75]   # per step; white kernels heaviest
TRIPLE_WEIGHT = [1.25, 1.05, 0.9]      # outer cube boldest, inner finest


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--structure", default="compound", choices=["compound", "triple"],
                    help="compound: Escher compound / kernel chain; triple: three nested cubes")
    ap.add_argument("--turns", default="zz", choices=list(TRIPLE_TURNS),
                    help="(triple) axes of the two 45-degree turns")
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

    if args.structure == "triple":
        cubes = triple(args.turns)
        verify_triple(cubes)
    else:
        cubes = nest(args.depth)
        verify(cubes)
    _NORMALS[0] = face_normals(cubes)
    pts, segs = graph(cubes)
    # search from several seeds in parallel, keep the cleanest view
    from multiprocessing import Pool
    lim = (args.min_axis, args.min_tilt)
    elev = tuple(args.elev) if args.elev else None
    with Pool() as pool:
        Rs = pool.starmap(_search, [(pts, segs, args.seed * 100 + i, lim, elev, _NORMALS[0])
                                    for i in range(4)])
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
    weights = TRIPLE_WEIGHT if args.structure == "triple" else WEIGHT
    for step, kind, axis, V, *_ in cubes:
        P = to_page(V @ R[:2].T)
        if isinstance(pal, dict):
            cmap = pal["C"]
            if kind == "C":
                colour = (cmap[step] if isinstance(cmap, dict) else cmap)[axis]
            else:
                colour = pal["K"]
        else:
            colour = pal[step % len(pal)]
        lines.append((colour, args.line * weights[step],
                      [(P[i], P[j]) for i, j in CUBE_EDGES]))

    sphere_r = np.sqrt(3) * k                  # circumsphere of the outer cube(s)
    lens_r = min(args.lens * sphere_r, A2_W / 2 - 8)
    dots, pitch = dot_field(args.cols, centre, lens_r, args.magnify, args.r_out,
                            args.r_in, args.vignette, args.profile)
    print(f"{len(dots[0])} dots, {len(cubes)} cubes, figure {args.size:.0f} x {span[1] * k:.0f} mm")

    title = "Three Nested Neckers" if args.structure == "triple" else "Nested Necker"
    svg = render_svg(dots, lines, args.line).replace("<title>Inscribed</title>",
                                                     f"<title>{title}</title>")
    open(args.out + ".svg", "w").write(svg)
    render_pdf(args.out + ".pdf", dots, lines, args.line, title=title)
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
