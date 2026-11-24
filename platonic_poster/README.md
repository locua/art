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

---

## v4 "Inscribed": my own creative direction

`python make_poster_v4.py` writes to `output/v4/`.

**Form.** Instead of a grid of separate figures, v4 is one monumental object.
The five Platonic solids sit exactly inside one another as a single rigid
compound, and every link in the chain is exact:

| outer | inner | how |
|---|---|---|
| dodecahedron | cube | the cube's 8 corners are 8 of the dodecahedron's 20 |
| cube | tetrahedron | alternate corners of the cube |
| tetrahedron | octahedron | the octahedron's corners are the tetrahedron's edge midpoints |
| octahedron | icosahedron | the icosahedron's corners sit on the octahedron's edges, at the golden-ratio points |

The innermost solid is still about 40 % the size of the outermost, so the
nesting stays readable. It nods to Kepler's *Mysterium Cosmographicum*, where
the Platonic solids nest between spheres.

**Illusions, in layers:**
1. **A sphere.** The dot grid is seen through a magnifying lens. Each dot is a
   disc pushed through the same mapping as its position, so toward the rim the
   dots foreshorten into ellipses, like discs printed on a globe (after
   Vasarely's *Vega* series). The globe is the compound's circumscribing sphere.
   The dots also shrink gently toward the page corners, so the globe rises out
   of a receding field.
2. **Two depth layers.** The tetrahedron is drawn in paper white. It is visible
   only as cuts through the black dots, so your eye continues it across the
   white paper as an illusory figure floating in front of the sheet: the
   original Bradley & Petry mechanism. The other solids are in colour, seen
   through the holes, so they sit behind the sheet. The dodecahedron and cube,
   which share corners, form a cyan cage, and the octahedron and icosahedron
   make a hot orange and magenta core. Line weight follows depth too: the white
   layer is heaviest and the core finest.
3. **Necker reversal.** The whole 90-edge compound is one exact orthographic
   view, so it flips in depth as a single object. A tetrahedron corner is
   rolled to point straight up, so the white pyramid can be read as seen from
   above or from below.
4. **No accidental views.** The view is searched so that no two corners nearly
   coincide in 2D and no edge grazes a corner it doesn't belong to. The
   compound's symmetry axes are kept well clear, and no face of the cube,
   tetrahedron or octahedron is seen edge-on.

The lines stay perfectly straight while the field bulges, which is what makes
the solids float in front of the globe.

Options: `--palette` (cage, ember-white, ember, cyan-white, spectrum),
`--magnify`, `--profile` (lens shape), `--lens`, `--size`, `--vignette` and
`--view`.

---

## v5 "Nested Necker": cubes only, nested and interlocking

`python make_poster_v5.py` writes to `output/v5/`. It uses the same dot globe as
v4, but every figure is a cube: seven of them, at three sizes, in an exact chain
that alternates between two structures:

```
C(1)  ⊃  K(1/√2)  ⊃  C(1/2)          (each step scaled by 1/√2)
```

* **C, Escher's compound.** Three equal cubes, each turned 45° about a
  different axis (x, y or z), passing through one another. This is the solid
  on the towers in M.C. Escher's *Waterfall*.
* **K, the kernel.** One axis-aligned cube, the largest that fits inside the
  compound. Each of its 8 corners touches a face of **all three** surrounding
  cubes at once.
* **The next compound** is half the size and fits exactly inside the kernel:
  every one of its 24 corners lands on a kernel face.

So the cubes interlock within each compound and touch exactly between levels.
`verify()` asserts every one of these contacts each time the poster is
generated.

**Colour.** The three cubes of each compound are coloured by their turning
axis in the print primaries cyan, magenta and yellow, and the same three
colours recur at half size, so the self-similarity reads at a glance. The
kernel cube is paper-white: it shows only as cuts through the dots, so it reads
as an illusory Necker cube floating in front of the sheet, like the original
Bradley & Petry figure.

**View.** The camera looks down on the nest from 15–35° above the horizon,
with one axis upright, as Escher drew it. The yellow cube's vertical edges
anchor the figure. Within that range the search maximises the clearance
between corners and unrelated edges, keeps 12° away from every symmetry axis,
and keeps every face at least 8° from edge-on. Every cube is an exact
orthographic wireframe, so each one flips independently.

`--depth 4` adds the next, tiny white kernel inside. It's tighter: the minimum
clearance drops from 0.026 to 0.016. `--palette layers` colours by level
instead (cyan, white, magenta).

### v5 in greens (`output/v5_green/`)

```bash
python make_poster_v5.py --palette greens --out output/v5_green/nested_necker_green_A2
```

The greens are built in OKLCH (`colours.py`) so that each colour dimension
encodes one relationship:

| colour dimension | meaning |
|---|---|
| **hue** | which axis a cube is turned about: x is chartreuse `#b3e637`, z (the upright anchor) is true green `#33d34b`, y is sea-green `#38dcab` |
| **lightness** | depth: the outer compound is in saturated mid-greens and the half-size compound in lighter tints of the *same* hues, so each cube's twin is its colour match |
| **kernel** | `#dbffd9`, the OKLab average of the three outer greens (it touches all three), lifted almost to white so it keeps its illusory, in-front-of-the-sheet quality |

The three axis greens also differ in lightness, so they stay distinct for
red-green colour-blind viewers.

### Three nested Neckers, in green (`output/v5_triple_green/`)

```bash
python make_poster_v5.py --structure triple --palette green-pair --size 255 \
    --out output/v5_triple_green/three_neckers_green_A2
```

This uses just three cubes. Each one is the previous cube turned 45° about the
upright axis and scaled by 1/√2, which is the largest cube that fits in that
orientation. All 8 corners of each inner cube land exactly on the faces of the
cube around it, and `verify_triple()` checks this. After two 45° turns the
inner cube is parallel to the outer one. **The colours show that:** the outer
and inner cubes share one sea-green hue and differ only in lightness
(`#35d59f` → `#a8fdd8`). The middle cube, turned diagonally between them, takes
the contrasting chartreuse `#bde437`. With so few edges, this view has four
times the clearance of the compound, so each cube flips very cleanly.
`--turns zx` makes the second turn about a horizontal axis instead.

### Four nested Neckers in one orientation, in green (`output/v5_four_green/`)

```bash
python make_poster_v5.py --structure parallel --palette green-depth --size 255 \
    --out output/v5_four_green/four_neckers_green_A2
```

Four concentric cubes in exactly the same orientation. Each is 1/√2 the size
of the one around it (sizes 1, 0.71, 0.5, 0.35), so cubes 1 and 3 are exactly
the parallel outer and inner pair of the three-cube poster. `verify_parallel()`
checks that every face is parallel and that each cube sits strictly inside,
with an equal gap on all six sides.

The "inside" feeling comes from three cues working together:

* **Colour:** one family of green, because there is one orientation. Lightness
  rises inward (`#2dbc55`, `#35d785`, `#3ef3af`, `#b4fee3`), so the core glows
  as if lit from within.
* **Weight:** the lines get thinner with each step inward, as if further away.
* **Shrinking steps:** the gaps between the cubes get smaller toward the
  centre, so it reads as a tunnel receding into the globe.

All four share one viewpoint, so they tend to flip in depth together. The
nest then turns inside out: the innermost cube becomes the nearest.
`--ratio 0.75` gives a gentler, more evenly stepped tunnel, and `--count`
changes the number of cubes.

---

## v6 "Schröder": a reversible staircase

`python make_poster_v6.py` writes to `output/v6/`.

**Why it flips, exactly.** The stair profile starts and ends with a riser,
with a floor strip leading in and a landing leading out. That makes it
centrally symmetric, and the script asserts this. Its extrusion is therefore
symmetric through its centre too. In a parallel view, reflecting the object
through its centre is the same as turning the picture 180°, and reading the
picture with depth reversed is the same as seeing the mirror-image object. So
the reversed reading is *the same staircase upside down*. Stairs seen from
above and an overhang seen from below are equally valid readings, which is
what makes Schröder's figure so unstable.

**Style, kept consistent with the series:**
* Edges are coloured by 3D direction, using the same greens as the green
  Nested Necker: treads, floor and landing (x) are chartreuse `#b3e637`,
  depth edges (y) are sea-green `#38dcab`, and risers (z) are true green
  `#33d34b`.
* All lines have equal weight. Thinning lines with depth would favour one
  reading.
* The view is searched on the profile side, as in Schröder's drawing, with every
  face at least 16° from edge-on and maximum clearance between corners and
  unrelated edges.

**Halftone relief (the creative licence).** Dots inside the stair's faces are
resized by face orientation: tread dots shrink by 25 % and riser dots grow,
capped so that no two dots ever touch (closest gap 1 mm). The staircase appears
in the dot field itself, like a halftone print, even before you notice the
lines. Shading implies a light from above, so it slightly favours the "stairs
seen from above" reading. `schroder_pure_A2` (`--relief 0`) is the unshaded
version, with maximum ambiguity.

Options: `--steps`, `--width`, `--relief`, `--azim`/`--elev` (view ranges),
`--size`, `--seed` and `--palette mono`.

---

## v7 "Tumbling Field": tumbling blocks and multistable dot lattices, combined

`python make_poster_v7.py` writes to `output/v7/`.

Two multistable effects, locked together exactly:

1. **Tumbling blocks.** An isometric field of 143 cubes. Every hexagon reads as
   a cube seen from above or from below, and regions of the field flip
   independently.
2. **A multistable dot lattice** (after Michael Kubovy). A hexagonal dot
   lattice is the most ambiguous lattice possible: its three row directions
   are equally strong, so the eye keeps regrouping the dots into rows along
   one direction, then another.

**The link:** the lattice's three row directions *are* the cubes' three edge
directions. Every cube corner sits exactly on a dot centre, and every edge
runs straight through a line of dot centres, with no approximation. This
finally gives the "corners in the circles" ideal of v1 with zero distortion.
Edges are coloured by direction with the series greens: x is chartreuse, y is
sea-green and z is true green. When the eye groups the dots along one
direction, that green comes forward.

**Creative licence:**
* **Three-tone halftone shading.** Dots inside each rhombus are sized by its
  orientation: the flat diamonds are lightest and the right-leaning ones
  darkest. A flat diamond is a top face in one reading and a bottom face in
  the other, so the shading supports both readings, just as in the
  traditional quilt.
* **Dome or bowl?** The field is warped through the series' dot globe, and
  corners stay exactly on dot centres because the dots and lines go through
  the same mapping. Inside a 150 mm disc under the dome, every hexagon is split
  the other way, so the cubes there are seen from *below*. The bulge says
  "towards you" while the cubes say "from beneath", so the centre reads as a
  dome or as a bowl. That's a third ambiguity on top of the other two.

`tumbling_field_flat_A2` (`--inner-flip 0 --magnify 0`) is the pure, uniform
field. Options include `--L` (cube edge, in dots), `--pitch`, `--relief`,
`--inner-flip`, `--magnify` and `--lens`. The closest dot gap is 0.66 mm.
