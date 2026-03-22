# Paper Completion Plan: BCC Void Analytical Solution

## Current Status

**Done:**
- Sections 1–6 (Introduction through Activation Pressure): complete analytical content
- Section 7 (BCC vs FCC comparison): table done, needs prose and FCC reproduction
- Section 8 (Numerical verification): **COMPLETE** — 4 subsections, 3 figures, Borja (2013) reference added
- Figures: yield surface, sector structure, interior stress field, CPFEM verification, mesh refinement, normalized pattern (all generated)
- CPFEM code: Ultimate Algorithm implemented, sector structure verified (100% match)
- Mesh refinement and E-convergence studies completed
- Paper compiles to 13 pages

**Not done:**
- Section 7 (BCC vs FCC): prose around the comparison table
- Section 9 (Discussion): subsection on ductile fracture models, limitations
- Geometry schematic figure (Fig. 1)
- FCC solution reproduction for side-by-side comparison plots
- Acknowledgements
- Final polish

---

## Remaining To-Do Items

### 1. Complete Section 7 (BCC vs FCC Comparison Prose)

Currently has only the comparison table. Add:
- 2–3 paragraphs discussing yield polygon rotation (arctan(2√2)/2 ≈ 35.3°) and its physical origin in the different slip system geometries
- Why cross-plane pairings arise in BCC but not FCC
- Why the activation pressure ratio is exactly 3/2 (geometric argument from yield polygon dimensions)
- Comparison of sector boundary angles: arctan(√2) vs arctan(2√2)

**Effort: ~1 hour**

### 2. Complete Section 9 (Discussion)

**2a. Ductile fracture models** (~0.5 page)
- How the analytical activation pressure feeds into Keralavarma-Benzerga-type porous crystal plasticity models
- Orientation-dependent void growth resistance as input to homogenized Gurson models
- Comparison with computational yield surfaces: Mbiakop et al. (2015), Paux et al. (2018)

**2b. Limitations and extensions** (~0.5 page)
- Limited to {110}<111> slip; BCC also has {112}<111> and {123}<111> (pencil glide)
- Rigid-ideally plastic: no hardening, no elastic effects
- Equibiaxial loading only; non-equibiaxial loading → Paper 2
- [110] void axis only; other orientations ([100], [111]) are future work
- Temperature-dependent CRSS and non-Schmid effects in BCC not considered

**Effort: ~1.5 hours**

### 3. Generate Missing Figures

**3a. Geometry schematic (Fig. 1)**
- TikZ or matplotlib diagram: cylindrical void, crystal axes [001] and [-110]/√2, loading arrows, polar coordinates (r, θ)

**3b. FCC comparison figures**
- Reproduce FCC yield polygon using the same SymPy framework
- Polish the side-by-side BCC vs FCC yield polygon plot (current version is rough)
- Side-by-side sector structure comparison (optional — adds visual impact)

**Effort: ~2 hours**

### 4. Polish and Finalize

- Verify abstract numbers match final results
- Ensure introduction roadmap matches actual section numbering
- Check all citations resolve (currently 22 entries)
- Fill in acknowledgements
- Full pdflatex+bibtex build; check figure placement, orphaned refs, undefined labels
- Proofread for consistency in notation (τ_CRSS vs τ_Y, etc.)

**Effort: ~1 hour**

---

## Priority Order

| Priority | Item | Effort | Status |
|----------|------|--------|--------|
| ~~1~~ | ~~Write Section 8 (numerical verification)~~ | ~~2–3 hrs~~ | **DONE** |
| ~~2~~ | ~~CPFEM figures for Section 8~~ | ~~1–2 hrs~~ | **DONE** (3 figure files referenced) |
| 3 | Complete Section 7 (BCC vs FCC prose) | 1 hr | TODO |
| 4 | Discussion: ductile fracture models | 1 hr | TODO |
| 5 | Discussion: limitations & extensions | 0.5 hr | TODO |
| 6 | Geometry schematic (Fig. 1) | 1 hr | TODO |
| 7 | FCC comparison figures (polish) | 1 hr | TODO |
| 8 | Polish, acknowledgements, compile | 1 hr | TODO |

**Remaining effort: ~5.5 hours**
