"""v3: geometrically exact solids (true orthographic projections, no
snapping), placed and oriented so that as many vertices as possible fall
inside the dots.

Necker-type reversal needs a figure that is a *perfect* parallel
projection: no perspective, no warping, all edges drawn. Each solid here
is exactly that. Freedom comes only from rotation, size and position,
which are chosen so that:
  * the view is generic (not along a symmetry axis -> no flat readings),
  * every corner junction is clearly spread (crisp Y / arrow / X shapes),
  * no edge passes near a corner it doesn't belong to (no false junctions),
  * as many corners as possible sit close to a dot centre, so the junction
    is visible inside the black hole.
"""
import numpy as np

from solids import random_rotation
from solids_v2 import ViewScorer, axis_rotation, _small_rotation_like

OFF_GRID = np.stack(np.meshgrid(np.linspace(0, 1, 24, endpoint=False),
                                np.linspace(0, 1, 24, endpoint=False)), -1).reshape(-1, 2)


def hit_cost(pts, offs, good=0.12, bad=0.34):
    """For each candidate offset: mean per-vertex 'miss' in [0, 1].
    0 when a vertex is within `good` of a dot centre (junction fully
    visible), 1 when it's beyond `bad` (junction cut off or in the white)."""
    q = pts[None] + offs[:, None]                  # (G, n, 2)
    d = np.linalg.norm(q - np.round(q), axis=-1)   # distance to nearest dot centre
    return np.clip((d - good) / (bad - good), 0, 1).mean(1)


def best_offset(pts):
    h = hit_cost(pts, OFF_GRID)
    i = np.argmin(h)
    # local polish
    fine = OFF_GRID[i] + np.stack(np.meshgrid(np.linspace(-0.04, 0.04, 9),
                                              np.linspace(-0.04, 0.04, 9)), -1).reshape(-1, 2)
    hf = hit_cost(pts, fine)
    j = np.argmin(hf)
    return hf[j], fine[j]


def exact_sequence(name, extent, frames, step_deg, direction, rng,
                   candidates=30000, keep=5, refine=300, w_hit=1.2, enough=24):
    # (hard junction angle, min vertex gap, edge clearance) -- the 12- and
    # 20-vertex solids pack corners closer in any exact view, so they get a
    # little more latitude
    hard, gap, clear = {"icosahedron": (12, 0.85, 0.4),
                        "dodecahedron": (12, 0.7, 0.4)}.get(name, (16, 1.0, 0.45))
    sc = ViewScorer(name, min_gap=gap, edge_clear=clear, hard_angle=hard)
    v = sc.v
    step = np.radians(step_deg) * direction

    deltas = np.radians(np.array([0, 1, -1, 2, -2, 3, -3, 4, -4, 5, -5], float))

    def evaluate(R0, axis, s):
        # each frame may turn up to +-5 deg off its ideal step to dodge an
        # accidental alignment; geometry stays exact either way
        feas = []
        for k in range(frames):
            opts = []
            for dl in deltas:
                R = axis_rotation(axis, k * step + dl) @ R0
                p = (v @ R.T)[:, :2] * s
                c, _ = sc.cost(p, view_dir=R[2], snap=False)
                if np.isfinite(c):
                    opts.append((p, c + 0.003 * np.degrees(dl) ** 2, R, dl))
            if not opts:
                return np.inf, None
            feas.append(opts)
        total, outs = 0.0, []
        for opts in feas:
            best = None
            for p, c, R, dl in opts[:5]:
                h, off = best_offset(p)
                tot = c + w_hit * h
                if best is None or tot < best[0]:
                    best = (tot, (p + off, c, h, R, dl))
            total += best[0]
            outs.append(best[1])
        return total, outs

    def scale_for(R0, axis):
        spans = []
        for k in range(frames):
            p = (v @ (axis_rotation(axis, k * step) @ R0).T)[:, :2]
            spans.append(np.ptp(p, axis=0).max())
        return extent / max(spans)

    pool = []
    for _ in range(candidates):
        tilt = np.radians(rng.uniform(12, 30)) * rng.choice([-1, 1])
        lean = np.radians(rng.uniform(-12, 12))
        axis = np.array([np.sin(lean), np.cos(tilt), np.sin(tilt)])
        R0 = random_rotation(rng)
        s = scale_for(R0, axis)
        t, outs = evaluate(R0, axis, s)
        if np.isfinite(t):
            pool.append((t, R0, axis, s))
            if len(pool) >= enough:
                break
    if not pool:
        raise RuntimeError(f"no clean views for {name}")
    pool.sort(key=lambda x: x[0])

    best = None
    for t, R0, axis, s in pool[:keep]:
        for i in range(refine):
            sig = 0.06 * (1 - i / refine) + 0.004
            R0b = _small_rotation_like(rng, sig) @ R0
            axb = axis + rng.normal(size=3) * sig * 0.3
            axb /= np.linalg.norm(axb)
            sb = scale_for(R0b, axb) * rng.uniform(0.97, 1.03)
            tb, _ = evaluate(R0b, axb, sb)
            if tb < t:
                t, R0, axis, s = tb, R0b, axb, sb
        if best is None or t < best[0]:
            best = (t, R0, axis, s)

    t, R0, axis, s = best
    _, outs = evaluate(R0, axis, s)
    res = []
    for k, (p, c, h, R, dl) in enumerate(outs):
        q = p - np.round((p.max(0) + p.min(0)) / 2)     # integer shift keeps dot alignment
        d = np.linalg.norm(q - np.round(q), axis=-1)
        res.append(dict(points=q, edges=sc.edges, name=name, cost=c, hit=h,
                        angle=np.degrees(k * step + dl),
                        in_dot=int((d < 0.32).sum()), n=len(q)))
    return res
