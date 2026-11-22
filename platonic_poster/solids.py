"""Platonic solid geometry + search for grid-snapped orthographic views."""
import itertools
import numpy as np

PHI = (1 + 5 ** 0.5) / 2


def _edges_by_min_dist(v):
    d = np.linalg.norm(v[:, None] - v[None], axis=-1)
    m = d[d > 1e-9].min()
    return [(i, j) for i in range(len(v)) for j in range(i + 1, len(v)) if abs(d[i, j] - m) < 1e-6]


def solid(name):
    if name == "tetrahedron":
        v = [(1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)]
    elif name == "cube":
        v = list(itertools.product((-1, 1), repeat=3))
    elif name == "octahedron":
        v = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    elif name == "icosahedron":
        v = []
        for a in (-1, 1):
            for b in (-PHI, PHI):
                v += [(0, a, b), (a, b, 0), (b, 0, a)]
    elif name == "dodecahedron":
        v = list(itertools.product((-1, 1), repeat=3))
        for a in (-1 / PHI, 1 / PHI):
            for b in (-PHI, PHI):
                v += [(0, a, b), (a, b, 0), (b, 0, a)]
    else:
        raise ValueError(name)
    v = np.array(v, float)
    v /= np.linalg.norm(v, axis=1).max()
    return v, _edges_by_min_dist(v)


def random_rotation(rng):
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _seg_point_dist(a, b, p):
    ab = b - a
    t = np.clip(np.dot(p - a, ab) / np.dot(ab, ab), 0, 1)
    return np.linalg.norm(a + t * ab - p)


def score_view(proj, snapped, edges, min_vertex_gap=1.0, min_edge_clear=0.45):
    """Lower is better. Returns inf for unusable (ambiguous / degenerate) views."""
    n = len(snapped)
    # distinct vertices, not too close
    for i in range(n):
        for j in range(i + 1, n):
            if np.linalg.norm(snapped[i] - snapped[j]) < min_vertex_gap - 1e-9:
                return np.inf
    # no edge passes through (or grazes) a vertex that isn't its own
    for (i, j) in edges:
        for k in range(n):
            if k in (i, j):
                continue
            if _seg_point_dist(snapped[i], snapped[j], snapped[k]) < min_edge_clear:
                return np.inf
    # no two edges at one vertex nearly collinear-overlapping
    adj = {i: [] for i in range(n)}
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)
    for i in range(n):
        for a, b in itertools.combinations(adj[i], 2):
            u, w = snapped[a] - snapped[i], snapped[b] - snapped[i]
            c = np.dot(u, w) / (np.linalg.norm(u) * np.linalg.norm(w))
            if c > np.cos(np.radians(14)):
                return np.inf
    err = np.sum((proj - snapped) ** 2)
    return err


def find_views(name, extent, rng, tries=4000, keep=12):
    """Search random orientations; return best grid-snapped 2D vertex sets.

    extent = approx width of the projected solid in grid units.
    Returns list of (score, int_points (n,2), edges, depth (n,)).
    """
    v, edges = solid(name)
    out = []
    for _ in range(tries):
        R = random_rotation(rng)
        p3 = v @ R.T
        p = p3[:, :2]
        span = (p.max(0) - p.min(0)).max()
        s = extent / span * rng.uniform(0.95, 1.05)
        proj = p * s
        # random sub-grid offset so the snap lattice isn't always centred
        off = rng.uniform(-0.5, 0.5, size=2)
        proj = proj + off
        snapped = np.round(proj)
        sc = score_view(proj, snapped, edges)
        if np.isfinite(sc):
            out.append((sc / len(v), snapped.astype(int), edges, p3[:, 2]))
    out.sort(key=lambda t: t[0])
    return out[:keep]


def _small_rotation(rng, sigma):
    axis = rng.normal(size=3)
    axis /= np.linalg.norm(axis)
    ang = rng.normal() * sigma
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


def best_view(name, extent, rng, tries=3000, refine_steps=900, n_refine=8):
    """Random search followed by hill-climbing refinement of the best few.

    Returns dict(points=int array (n,2), edges, depth, err)."""
    v, edges = solid(name)

    def evaluate(R, s, off):
        p3 = v @ R.T
        proj = p3[:, :2] * s + off
        snapped = np.round(proj)
        return score_view(proj, snapped, edges) / len(v), snapped, p3[:, 2]

    seeds = []
    for _ in range(tries):
        R = random_rotation(rng)
        p = (v @ R.T)[:, :2]
        s = extent / (p.max(0) - p.min(0)).max() * rng.uniform(0.93, 1.07)
        off = rng.uniform(-0.5, 0.5, size=2)
        sc, _, _ = evaluate(R, s, off)
        if np.isfinite(sc):
            seeds.append((sc, R, s, off))
    seeds.sort(key=lambda t: t[0])
    refined = []
    for sc, R, s, off in seeds[:n_refine]:
        for step in range(refine_steps):
            sig = 0.08 * (1 - step / refine_steps) + 0.005
            R2 = _small_rotation(rng, sig) @ R
            s2 = s * (1 + rng.normal() * sig * 0.3)
            off2 = off + rng.normal(size=2) * sig
            # keep extent within +-12 % of target
            p = (v @ R2.T)[:, :2] * s2
            if abs((p.max(0) - p.min(0)).max() / extent - 1) > 0.12:
                continue
            sc2, _, _ = evaluate(R2, s2, off2)
            if sc2 < sc:
                sc, R, s, off = sc2, R2, s2, off2
        refined.append((sc, R, s, off))
    # pick randomly among the good fits so repeated solids get different views
    refined.sort(key=lambda t: t[0])
    good = [t for t in refined if t[0] <= max(1.6 * refined[0][0], 0.01)]
    sc, R, s, off = good[rng.integers(len(good))]
    _, snapped, depth = evaluate(R, s, off)
    pts = snapped.astype(int)
    pts -= np.round((pts.max(0) + pts.min(0)) / 2).astype(int)  # centre on origin
    return dict(points=pts, edges=edges, depth=depth, err=sc, name=name)
