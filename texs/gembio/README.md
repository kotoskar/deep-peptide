# GEM version

Same experiments and the same appendix as `../ai4dd`, retargeted at
**Generative and Experimental Perspectives for Biomolecular Design** (GEM),
NeurIPS 2026. Short paper track: 5 pages excluding references and appendix,
NeurIPS 2026 template, double-blind, non-archival.

## What differs from the AI4DD version

The experiments, the tables, the figures and the appendix are the same. The
framing is not. GEM's topics are inverse design, modelling biomolecular data,
model interpretability, and benchmarks/datasets/oracles, so this version argues
the work from the oracle end:

* a cleavage predictor is an oracle inside a peptide-design loop, and the loop
  can only use it at the resolution the oracle has been measured at;
* the benchmark's ±3 acceptance window is most of a short peptide, so it cannot
  report the property a design loop needs;
* the head's gain is placement alone and the adapter's placement on top of
  coverage, which the tolerance-free gate establishes, and what grows as the
  requirement tightens is placement, the end of the curve a design loop reads.

Title, abstract, introduction, related work, the baseline paragraph and the
conclusion are rewritten. Everything else is inherited.

## Build

    tectonic -X compile main.tex

`tex2md.py` renders `main.md` from the built `main.aux`, as in the AI4DD folder.

## Numbers

Every number is the same as the AI4DD version's and comes from the same runs.
The anonymous release carries the per-cell scores and a script that rebuilds a
superset of both papers' numbers without a GPU.
