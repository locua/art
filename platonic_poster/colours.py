"""Tiny OKLab / OKLCH helpers for choosing and checking palettes."""
import numpy as np


def _srgb_to_lin(c):
    c = np.asarray(c, float)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * c ** (1 / 2.4) - 0.055)


_M1 = np.array([[0.4122214708, 0.5363325363, 0.0514459929],
                [0.2119034982, 0.6806995451, 0.1073969566],
                [0.0883024619, 0.2817188376, 0.6299787005]])
_M2 = np.array([[0.2104542553, 0.7936177850, -0.0040720468],
                [1.9779984951, -2.4285922050, 0.4505937099],
                [0.0259040371, 0.7827717662, -0.8086757660]])


def hex_to_oklab(h):
    rgb = np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) / 255
    lms = _M1 @ _srgb_to_lin(rgb)
    return _M2 @ np.cbrt(lms)


def oklab_to_hex(lab):
    lms = np.linalg.solve(_M2, lab) ** 3
    rgb = _lin_to_srgb(np.linalg.solve(_M1, lms))
    return "#" + "".join(f"{int(round(v * 255)):02x}" for v in rgb)


def oklch(h):
    L, a, b = hex_to_oklab(h)
    return L, np.hypot(a, b), np.degrees(np.arctan2(b, a)) % 360


def from_oklch(L, C, H):
    return oklab_to_hex(np.array([L, C * np.cos(np.radians(H)), C * np.sin(np.radians(H))]))


def in_gamut(L, C, H):
    lab = np.array([L, C * np.cos(np.radians(H)), C * np.sin(np.radians(H))])
    lin = np.linalg.solve(_M1, np.linalg.solve(_M2, lab) ** 3)
    return bool(np.all(lin >= -1e-6) and np.all(lin <= 1 + 1e-6))


def max_chroma(L, H):
    lo, hi = 0.0, 0.4
    for _ in range(40):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if in_gamut(L, mid, H) else (lo, mid)
    return lo
