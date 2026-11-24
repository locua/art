"""Geometry for the Necker Tumbler: a layer of unit cubes (i + j + k = 0, the
layer whose isometric shadow is the tumbling-blocks tiling) drawn as full
12-edge wireframes in a parallel projection whose three edge vectors are
whole multiples of the hexagonal lattice's three row directions:

    e_x = a * X,  e_y = b * Y,  e_z = c * Z      (a, b, c dots)

Every 3D lattice point lands on a dot; every edge runs along a lattice row
through dot centres. With a = b = c the view is isometric and each cube's
hidden corner coincides with its front corner (a flat, accidental view);
unequal lengths move it clear. By Pohlke's theorem the drawing is still an
exact parallel projection of true cubes.
"""
import itertools

import numpy as np

S3 = np.sqrt(3)
DIR = {"z": np.array([0.0, -1.0]), "x": np.array([-S3 / 2, 0.5]), "y": np.array([S3 / 2, 0.5])}
# lattice coordinates (i along Y, j along Z); X = -Y - Z
LAT = {"x": np.array([-1, -1]), "y": np.array([1, 0]), "z": np.array([0, 1])}
UNIT = [np.array(v) for v in itertools.product((0, 1), repeat=3)]
CUBE_EDGES = [(p, q) for p in range(8) for q in range(p + 1, 8)
              if np.abs(UNIT[p] - UNIT[q]).sum() == 1]


# sublattice of the tumbling layer: every third cube (index-3 sublattice,
# the classic 3-colouring of the tumbling-blocks tiling). Basis in 3D.
THIRD = (np.array([-3, 0, 3]), np.array([-1, -1, 2]))


def cubes_in(radius_cells, sub=None):
    """Integer cube positions (i, j, k) with i + j + k = 0 inside a hexagon;
    with sub="third", only the index-3 sublattice."""
    out = []
    R = radius_cells
    for i in range(-R, R + 1):
        for j in range(-R, R + 1):
            k = -i - j
            if max(abs(i), abs(j), abs(k)) <= R:
                out.append(np.array([i, j, k]))
    if sub == "third":
        # p is in the sublattice iff (i - j) % 3 == 0  (B1, B2 both satisfy it,
        # and the index of {(i,j,k): i+j+k=0, i-j = 0 mod 3} is 3)
        out = [p for p in out if (p[0] - p[1]) % 3 == 0]
    return out


def lat_of(P3, abc):
    """3D lattice point -> 2D lattice coords (int)."""
    a, b, c = abc
    return P3[0] * a * LAT["x"] + P3[1] * b * LAT["y"] + P3[2] * c * LAT["z"]


def lat_to_xy(L):
    return L[..., 0:1] * DIR["y"] + L[..., 1:2] * DIR["z"]


def field(abc, radius_cells, sub=None):
    """Unique 3D edges of all cubes -> list of (p3, q3, axis)."""
    edges = {}
    for p in cubes_in(radius_cells, sub):
        for u, v in CUBE_EDGES:
            A, B = tuple(p + UNIT[u]), tuple(p + UNIT[v])
            axis = "xyz"[int(np.argmax(np.abs(np.array(B) - np.array(A))))]
            edges[tuple(sorted([A, B]))] = axis
    return [(np.array(A), np.array(B), ax) for (A, B), ax in edges.items()]


def quality(abc, radius_cells=3, sub=None):
    """(min vertex gap, min vertex-edge clearance, collinear overlaps), in dots."""
    E = field(abc, radius_cells, sub)
    V3 = {tuple(x) for A, B, _ in E for x in (A, B)}
    V3 = [np.array(v) for v in V3]
    V2 = np.array([lat_to_xy(lat_of(v, abc).astype(float)) for v in V3])
    D = np.linalg.norm(V2[:, None] - V2[None], axis=-1)
    np.fill_diagonal(D, np.inf)
    gap = D.min()
    idx = {tuple(v): n for n, v in enumerate(V3)}
    S = np.array([(idx[tuple(A)], idx[tuple(B)]) for A, B, _ in E])
    Pa, Pb = V2[S[:, 0]], V2[S[:, 1]]
    AB = Pb - Pa
    t = np.clip(((V2[None] - Pa[:, None]) * AB[:, None]).sum(-1) / (AB ** 2).sum(1)[:, None], 0, 1)
    dist = np.linalg.norm(Pa[:, None] + t[..., None] * AB[:, None] - V2[None], axis=-1)
    mask = np.ones_like(dist, bool)
    mask[np.arange(len(S)), S[:, 0]] = False
    mask[np.arange(len(S)), S[:, 1]] = False
    clear = dist[mask].min()
    return gap, clear, int((dist[mask] < 1e-9).sum())


def _cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def hull(P):
    """Convex hull of 2D points (monotone chain), counter-clockwise."""
    P = sorted(map(tuple, P))

    def half(seq):
        h = []
        for p in seq:
            while len(h) >= 2 and _cross2(h[-2], h[-1], p) <= 0:
                h.pop()
            h.append(p)
        return h
    lo, up = half(P), half(P[::-1])
    return np.array(lo[:-1] + up[:-1])


def cube_hulls(abc, positions):
    """Page-unit (pitch = 1) silhouettes of each cube."""
    return [hull(lat_to_xy(np.array([lat_of(p + u, abc) for u in UNIT], float)))
            for p in positions]


def coverage(G, hulls):
    """How many silhouettes contain each point of G (n, 2)."""
    cnt = np.zeros(len(G), int)
    for h in hulls:
        lo, hi = h.min(0) - 1e-6, h.max(0) + 1e-6
        cand = np.nonzero((G[:, 0] >= lo[0]) & (G[:, 0] <= hi[0]) &
                          (G[:, 1] >= lo[1]) & (G[:, 1] <= hi[1]))[0]
        inside = np.ones(len(cand), bool)
        for a, b in zip(h, np.roll(h, -1, axis=0)):
            inside &= ((b[0] - a[0]) * (G[cand, 1] - a[1]) -
                       (b[1] - a[1]) * (G[cand, 0] - a[0])) >= -1e-9
        cnt[cand[inside]] += 1
    return cnt
