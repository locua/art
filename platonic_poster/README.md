# Illusory Platonic Solids — A2 poster

A 44 × 62 grid of black dots on white. Wireframes of the five Platonic solids
(tetrahedron, cube, octahedron, icosahedron, dodecahedron) are drawn in bright
cyan, but the lines are **clipped to the black dots**. You only see them inside
the "holes", and your eye fills in the edges across the white. This builds on
Bradley & Petry's illusory Necker cube.

* Every vertex sits exactly on a dot centre. The script searches thousands of 3D
  orientations per solid and refines the best ones, so that the orthographic
  projection snaps onto the dot grid with very little distortion. The fit error
  is under 0.25 of the dot spacing, even for the dodecahedron.
* Views where an edge would run through another vertex, or where vertices or
  edges would overlap, are rejected, so each solid can be read clearly.
* The projection is orthographic (no perspective), so every solid is
  depth-ambiguous and flips like a Necker cube.

## Output (`output/`)

| file | use |
|---|---|
| `platonic_poster_A2.pdf` | **send this to the printer**: vector, exact A2 (420 × 594 mm) |
| `platonic_poster_A2.svg` | vector master for editing (Illustrator / Inkscape) |
| `platonic_poster_A2.png` | 4961 × 7016 px raster at 300 dpi |
| `preview.png` | small preview |

## Regenerate / variations

```bash
pip install numpy cairosvg reportlab pillow
python make_poster.py                                  # default
python make_poster.py --colour "#ff2bd6" --out output/magenta
python make_poster.py --cols 56 --cells 5 7 --seed 3   # denser grid, 35 solids
python make_poster.py --dot 0.65 --line 0.08           # smaller dots, thinner lines
```

Print note: the screen cyan `#00d2ff` is close to process cyan. For maximum
punch, ask the printer for a fluorescent or spot colour. You can also choose a
colour that sits inside the CMYK gamut, such as `#00aeef`, `#ec008c` or `#ffed00`.

---

## v2: bigger dots, bolder lines, views chosen for the illusion

`python make_poster_v2.py` writes to `output/v2/`.

* **36 × 51 dots** (v1 had 44 × 62). The dots are 80 % of the spacing and the
  lines are 16 % of the spacing, about 70 % thicker than v1.
* **Each row is one solid turning.** From top to bottom the rows are the
  tetrahedron, cube, octahedron, icosahedron and dodecahedron (4, 6, 8, 12 and
  20 faces). Each row shows three views of the solid about 20° apart, turning
  about a shared axis that tilts slightly toward the viewer. Reading along the
  row gives a sense of one object spinning, and alternate rows spin in opposite
  directions. Every view is also a Necker-style ambiguous figure, so the
  apparent direction of spin can flip while you look.
* **The angles are chosen to make the illusion stronger.** The search in
  `solids_v2.py` scores each view and keeps only the good ones:
  * **No flat views.** Views within 8° of any symmetry axis are rejected, and
    views within 16° are penalised. Seen down those axes, a solid collapses
    into a flat pattern, such as a square with an X or a hexagon.
  * **Clear junctions in the dots.** The edges meeting at a corner must be well
    spread, so each corner dot shows a crisp Y or arrow. Those junctions are the
    strongest cue that makes the shape pop out in 3D.
  * **No false junctions.** No edge may pass near a corner it isn't part of,
    and corners must sit well apart.
  * **Little distortion.** Every corner still lands exactly on a dot centre.

Options: `--step` (degrees per frame), `--frames`, `--cols`, `--dot`, `--line`,
`--colour` and `--seed`.
