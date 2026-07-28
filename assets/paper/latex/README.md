# Trimwise manuscript source

`main.tex` is the canonical manuscript source. Do not edit `../paper.md`; it is
the archived Markdown draft used for the one-time migration.

The manuscript uses the neutral `article` class until a target venue is chosen.
Its bibliography is `../references.bib`. The two diagrams are committed as
vector PDFs under `figures/`; the v1.2 query-aware result chart is read directly
from `../../../benchmark/reports/evidence-sensitivity-v1-2/normalized_contiguous_vs_budget.png`.

With a standard TeX Live installation, build from this directory:

```bash
latexmk -pdf main.tex
```

For a venue submission, replace the generic document class and front matter
with the venue's official template while keeping the body, bibliography, and
figure paths intact.
