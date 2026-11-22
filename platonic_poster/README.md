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
