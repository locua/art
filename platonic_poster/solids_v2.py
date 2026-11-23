"""v2 view search: grid-snapped views chosen for illusory strength, and
rotation sequences (one solid turning across a row of the poster)."""
import itertools
import numpy as np

from solids import solid, random_rotation

# Solids whose vertices are integer combinations of a cube's three edge
# vectors. For these we snap the three projected edge vectors (the basis)
# to the grid instead of each vertex, so every vertex is an integer sum of
# integer vectors: parallel edges stay exactly parallel and equal length.
# By Pohlke's theorem the result is an exact parallel projection of the
# *regular* solid -- no warping, only a slight obliqueness of view.
BASIS_SOLIDS = ("tetrahedron", "cube", "octahedron")


def basis_vertices(name):
    v, edges = solid(name)
    raw = np.round(v / np.abs(v).max()).astype(int)     # entries in {-1,0,1}
    if name in ("cube", "tetrahedron"):
        raw = (raw + 1) // 2                            # unit-cube corners
    return raw.astype(float), edges


def regularity(points, edges, name):
    """Largest mismatch (in grid steps) between any two drawn edges that are
    parallel and equal on the real 3D solid. 0 = perfectly consistent."""
    v, _ = solid(name)
    groups = {}
    for a, b in edges:
        d3 = v[b] - v[a]
        key = tuple(np.round(d3 / np.linalg.norm(d3), 6))
        neg = tuple(np.round(-d3 / np.linalg.norm(d3), 6))
        if neg in groups:
            groups[neg].append(points[a] - points[b])
        else:
            groups.setdefault(key, []).append(points[b] - points[a])
    worst = 0.0
    for vecs in groups.values():
        vecs = np.array(vecs, float)
        worst = max(worst, np.abs(vecs - vecs[0]).max())
    return worst


def axis_rotation(axis, ang):
    axis = np.asarray(axis, float)
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(ang) * K + (1 - np.cos(ang)) * K @ K


_DUAL = {"tetrahedron": "tetrahedron", "cube": "octahedron", "octahedron": "cube",
         "icosahedron": "dodecahedron", "dodecahedron": "icosahedron"}


def symmetry_axes(name):
    """Unit directions of all rotation-symmetry axes: through vertices,
    edge midpoints and face centres. Viewing along (or near) one of these
    gives an 'accidental' view that collapses into a flat 2D pattern."""
    v, edges = solid(name)
    d, _ = solid(_DUAL[name])
    if name in ("icosahedron", "dodecahedron"):
        d = d[:, (1, 0, 2)]          # align dual with our coordinates
    if name == "tetrahedron":
        d = -d
    mids = np.array([(v[a] + v[b]) / 2 for a, b in edges])
    ax = np.vstack([v, mids, d])
    return ax / np.linalg.norm(ax, axis=1, keepdims=True)


class ViewScorer:
    """Vectorised scoring of a projected, grid-snapped solid.

    Hard rejects (return inf):
      * two vertices closer than `min_gap` dots
      * an edge passing within `edge_clear` of a vertex it doesn't belong to
        (it would read as a false junction)
      * two edges at a vertex closer than `hard_angle` degrees (junction unreadable)
    Soft costs (what makes the illusion strong):
      * snap error: distortion from forcing vertices onto dot centres
      * junction clarity: every vertex dot should show a crisp Y / arrow / X
        with wide angles between edges -> strongest 3D cue inside each hole
      * spread: vertices well apart, so the figure needs completion across white
    """

    def __init__(self, name, min_gap=1.0, edge_clear=0.5, hard_angle=20, good_angle=40):
        self.v, self.edges = solid(name)
        self.E = np.array(self.edges)
        n = len(self.v)
        inc = np.zeros((len(self.edges), n), bool)
        inc[np.arange(len(self.edges)), self.E[:, 0]] = True
        inc[np.arange(len(self.edges)), self.E[:, 1]] = True
        self.not_inc = ~inc
        adj = {i: [] for i in range(n)}
        for a, b in self.edges:
            adj[a].append(b)
            adj[b].append(a)
        self.trip = np.array([(i, a, b) for i in range(n)
                              for a, b in itertools.combinations(adj[i], 2)])
        self.iu = np.triu_indices(n, 1)
        self.min_gap, self.edge_clear = min_gap, edge_clear
        self.cos_hard = np.cos(np.radians(hard_angle))
        self.cos_good = np.cos(np.radians(good_angle))
        self.axes = symmetry_axes(name)

    def flatness(self, view_dir):
        """0 for a general 3D-looking view; rises towards special axes."""
        ang = np.degrees(np.arccos(np.clip(np.abs(self.axes @ view_dir).max(), -1, 1)))
        if ang < 8:
            return np.inf
        return max(0.0, (16 - ang) / 8) ** 2

    def cost(self, proj, view_dir=None, snap=True):
        S = np.round(proj) if snap else np.asarray(proj, float)
        d = np.linalg.norm(S[:, None] - S[None], axis=-1)[self.iu]
        if d.min() < self.min_gap - 1e-9:
            return np.inf, S
        A, B = S[self.E[:, 0]], S[self.E[:, 1]]
        AB = B - A
        L2 = (AB ** 2).sum(1)
        t = np.clip(((S[None] - A[:, None]) * AB[:, None]).sum(-1) / L2[:, None], 0, 1)
        dist = np.linalg.norm(A[:, None] + t[..., None] * AB[:, None] - S[None], axis=-1)
        if dist[self.not_inc].min() < self.edge_clear:
            return np.inf, S
        u = S[self.trip[:, 1]] - S[self.trip[:, 0]]
        w = S[self.trip[:, 2]] - S[self.trip[:, 0]]
        c = (u * w).sum(1) / (np.linalg.norm(u, axis=1) * np.linalg.norm(w, axis=1))
        if c.max() > self.cos_hard:
            return np.inf, S
        snap = ((proj - S) ** 2).sum(1).mean()
        junction = np.clip(c - self.cos_good, 0, None).mean() * 4
        spread = np.clip(1.6 - d, 0, None).mean()
        flat = 0.0 if view_dir is None else self.flatness(view_dir)
        return snap + 0.6 * junction + 0.5 * spread + 0.4 * flat, S


def rotation_sequence(name, extent, frames, step_deg, direction, rng,
                      candidates=40000, quick=80, refine=1200, enough=16):
    """Find `frames` views of one solid turning about a common axis.

    The axis is near the screen's vertical, tilted toward the viewer
    (a turntable seen slightly from above or below), so a row reads as
    one object spinning. Each frame may deviate a few degrees from the
    ideal step so its vertices can land on dot centres.
    """
    hard = {"icosahedron": 12, "dodecahedron": 13}.get(name, 16)
    sc = ViewScorer(name, edge_clear=0.45, hard_angle=hard)
    use_basis = name in BASIS_SOLIDS
    v = basis_vertices(name)[0] if use_basis else sc.v
    step = np.radians(step_deg) * direction

    def frame_cost(R0, axis, k, delta, s, off):
        R = axis_rotation(axis, k * step + delta) @ R0
        if use_basis:
            M = R[:2] * s                      # true orthographic basis
            B = np.round(M)                    # snapped basis -> exact parallel proj.
            p = v @ B.T
            c, S = sc.cost(p, view_dir=R[2])
            c += 1.5 * ((M - B) ** 2).sum() / 3
        else:
            # centre on a dot or exactly between dots: the solid is centrally
            # symmetric, so opposite vertices then snap symmetrically and every
            # pair of parallel edges stays exactly parallel and equal.
            p = (v @ R.T)[:, :2] * s + np.round(off * 2) / 2
            c, S = sc.cost(p, view_dir=R[2])
        return c + 0.003 * np.degrees(delta) ** 2, S

    def fit_frame(R0, axis, k, s, n, state=None, sig0=0.5):
        best = state or (np.inf, 0.0, s, np.zeros(2))
        for i in range(n):
            if np.isfinite(best[0]):
                sig = sig0 * (1 - i / n) + 0.02
                delta = best[1] + rng.normal() * np.radians(3) * sig
                s2 = best[2] * (1 + rng.normal() * 0.02 * sig)
                off = best[3] + rng.normal(size=2) * sig
            else:
                delta = rng.uniform(-1, 1) * np.radians(4)
                s2 = s * rng.uniform(0.97, 1.03)
                off = rng.uniform(-0.5, 0.5, 2)
            if abs(delta) > np.radians(4.5) or abs(s2 / s - 1) > 0.05:
                continue
            c, _ = frame_cost(R0, axis, k, delta, s2, off)
            if c < best[0]:
                best = (c, delta, s2, off)
        return best

    pool = []
    for _ in range(candidates):
        tilt = np.radians(rng.uniform(12, 30)) * rng.choice([-1, 1])
        lean = np.radians(rng.uniform(-12, 12))
        axis = np.array([np.sin(lean), np.cos(tilt), np.sin(tilt)])
        R0 = random_rotation(rng)
        # common scale: biggest frame fits `extent`
        spans = []
        for k in range(frames):
            p = (v @ (axis_rotation(axis, k * step) @ R0).T)[:, :2]
            spans.append((p.max(0) - p.min(0)).max())
        s = extent / max(spans)
        fits = []
        for k in range(frames):
            # cheap screen on the first frame, full quick fit on the rest
            f = fit_frame(R0, axis, k, s, 20 if k == 0 else quick)
            fits.append(f)
            if not np.isfinite(f[0]):
                break
        total = sum(f[0] for f in fits) if len(fits) == frames else np.inf
        if np.isfinite(total):
            pool.append((total, R0, axis, s, fits))
            if len(pool) >= enough:
                break
    if not pool:
        raise RuntimeError(f"no sequence found for {name} at extent {extent}")
    pool.sort(key=lambda t: t[0])

    best = None
    for total, R0, axis, s, fits in pool[:4]:
        fits = [fit_frame(R0, axis, k, s, refine, state=fits[k], sig0=0.3)
                for k in range(frames)]
        total = sum(f[0] for f in fits)
        if best is None or total < best[0]:
            best = (total, R0, axis, s, fits)

    total, R0, axis, s, fits = best
    out = []
    for k, (c, delta, s2, off) in enumerate(fits):
        _, S = frame_cost(R0, axis, k, delta, s2, off)
        pts = S.astype(int)
        pts -= np.round((pts.max(0) + pts.min(0)) / 2).astype(int)
        out.append(dict(points=pts, edges=sc.edges, cost=c, name=name,
                        angle=np.degrees(k * step + delta)))
    return out


def _small_rotation_like(rng, sigma):
    axis = rng.normal(size=3)
    return axis_rotation(axis, rng.normal() * sigma)
