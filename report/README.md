# PSG Tech tasklet-scheduling project report

`main.tex` follows the supplied LaTeX template's A4 layout, Latin Modern font,
12-point type, one-and-a-half spacing, heading styles, page numbering, front
matter and section sequence. Institutional branding and project content were
adapted for PSG College of Technology.

The title and five students' names/roll numbers come from the supplied Review 1
presentation. Dr. Jayashree L S and the CSE Department were confirmed by the user.
The logo comes from the official institutional website:
https://www.psgtech.edu/images/logo.png

The report distinguishes the Review 1 proposal from implemented behavior, and
distinguishes local verification from physical multi-host or WAN results.
Its committed code/CI baseline is `e0c0507` (4 October 2026). New workspace mT5
work is described separately and is not included in the published 61-test result.
`evidence.json` preserves the Git-history and CI evidence used for the report.

## Compile

Upload `main.tex`, `preamble.tex` and the `figures/` directory to Overleaf, with
`main.tex` as the main document. Select XeLaTeX or pdfLaTeX.

With a local TeX installation:

```bash
pdflatex main.tex
pdflatex main.tex
pdflatex main.tex
```

Or use Tectonic, which resolves the references and contents automatically:

```bash
tectonic main.tex
```

All figures required for compilation are included. `build_figures.py` is optional
and regenerates three vector figures from the stored evidence using Matplotlib:

```bash
python -m pip install matplotlib
python build_figures.py
```

The proposed architecture/recovery diagrams and planned timeline were extracted
from the team's supplied presentation. They are labeled as proposed or planned
in the report. No synthetic speedup or model-quality results were inserted.
