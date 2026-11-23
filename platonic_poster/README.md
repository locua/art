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
  * **No warping.** Every corner still lands exactly on a dot centre, and edges
    that are parallel on the real solid stay exactly parallel and equal on the
    poster:
    * For the **cube, tetrahedron and octahedron**, the script snaps the three
      projected cube-edge directions to the grid, rather than each corner
      separately. Every corner is then a whole-number sum of those directions,
      so the drawing is an exact parallel projection of the regular solid
      (Pohlke's theorem). It is very slightly oblique, the same way the classic
      Necker cube drawing is.
    * For the **icosahedron and dodecahedron**, each solid's centre is fixed on
      a dot or exactly halfway between dots. These solids are symmetric through
      their centre, so opposite corners snap symmetrically and every pair of
      parallel edges stays parallel.
    * The script prints a `parallel-edge error` for every solid, and it is 0
      for all of them.

Options: `--step` (degrees per frame), `--frames`, `--cols`, `--dot`, `--line`,
`--colour` and `--seed`.

---

## v3: geometrically exact solids, tuned for the Necker effect

`python make_poster_v3.py` writes to `output/v3/`. It uses the same dots, lines
and row layout as v2.

Necker-type depth reversal depends on the drawing being a *perfect* parallel
projection: no perspective and no warping, with every edge drawn. In v3 every
solid is an exact orthographic projection of the regular solid. The script
checks this, and the largest mismatch between edges that should be parallel
is about 1e-15, which is floating-point noise.

To make this possible, corners are no longer forced onto dot centres. The only
freedom is each solid's rotation, size and position, and `solids_v3.py`
chooses them to:

* **Land as many corners as possible near a dot centre,** so the junction shows
  inside the hole. The tetrahedron, cube and octahedron get most or all of their
  corners inside dots. The icosahedron gets 50–85 % and the dodecahedron
  60–70 %, and your eye fills in the rest across the white.
* **Keep the views generic and clean**, using the same rules as v2: no flat
  views close to a symmetry axis, wide-angled junctions, and no edge passing
  near a corner it doesn't belong to.
* **Keep the turning rows.** Each frame is an exact 3D rotation of the solid,
  about 20° from the previous frame. A frame may move up to ±5° from its ideal
  angle so it can avoid an accidental alignment.

### v3 variation: Necker cubes only

```bash
python make_poster_v3.py --only cube --frames 4 --rows 6 --extent 6.5 \
    --out output/v3_cubes/necker_cubes_A2
```

This makes 24 exact regular cubes in 6 rows of 4. Each row is one cube turning
in about 20° steps, and alternate rows turn in opposite directions. `--only`
works with any solid, for example `--only dodecahedron --rows 5 --frames 3`.
