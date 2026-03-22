# Paper Completion Plan: BCC Void Analytical Solution

## Current Status

**Done:**
- Sections 1–6 (Introduction through Activation Pressure): complete analytical content
- Section 7 (BCC vs FCC comparison): table done, needs prose and FCC reproduction
- Figures: yield surface, sector structure, interior stress field (all generated)
- CPFEM code: Ultimate Algorithm implemented, sector structure verified (100% match)
- Mesh refinement and E-convergence studies completed

**Not done:**
- Section 7: numerical verification (currently empty placeholder)
- Section 8: discussion subsection on ductile fracture models
- Geometry schematic figure (Fig. 1)
- FCC solution reproduction for side-by-side plots
- Acknowledgements

---

## To-Do Items

### 1. Write Section 7: Numerical Verification (~3 pages)

The CPFEM results need to be presented carefully, distinguishing what can and cannot be directly compared between elastic-plastic FEM and rigid-plastic analytical solutions.

**1a. Model description** (~0.5 page)
- Annular domain (a < r < b, b/a = 10), P1 triangular elements
- BCC {110}<111> elastic-perfectly plastic constitutive model
- Ultimate Algorithm (Borja 2013) for rate-independent active set identification
- Equibiaxial pressure loading at outer boundary, traction-free void

**1b. Sector structure verification** (~1 page) — THE STRONGEST RESULT
- Show the active slip system map (Fig. from `verify_yielded_zone.py` panel c)
- Table: predicted vs computed active systems in each angular range
- All three sectors verified with 100% match:
  - Sector I (0°–35.26°): systems 5, 12 ✓
  - Sector II (35.26°–54.74°): systems 3, 4 ✓
  - Sector III (54.74°–90°): systems 6, 11 ✓

**1c. Elastic-plastic vs rigid-plastic comparison** (~1 page)
- Explain why stress magnitudes differ: elastic Kirsch concentration at void surface
- Show E-convergence study: results E-independent for E/τ ≥ 10^7
- Show mesh refinement study: error is constant (not a discretization artifact)
- Key insight: rigid-plastic sector structure is a far-field plastic flow pattern; near the void surface, elastic stress concentration dominates

**1d. Non-circular plastic zone** (~0.5 page)
- Show R(θ) from CPFEM vs analytical prediction
- The anisotropic plastic zone shape is confirmed qualitatively
- Four-lobed pattern from σ_rr contour plot

### 2. Generate Missing Figures

**2a. Geometry schematic (Fig. 1)**
- TikZ diagram showing: cylindrical void, crystal axes, plane strain frame, loading
- Or a clean matplotlib figure

**2b. FCC comparison figures**
- Reproduce the FCC yield polygon using the same SymPy framework
- Side-by-side: BCC vs FCC yield polygon (already have rough version, needs polish)
- Side-by-side: BCC vs FCC sector structure
- Side-by-side: BCC vs FCC void surface stress

**2c. CPFEM verification figures (for Section 7)**
- Active slip system map around void (from `verify_yielded_zone.py`)
- σ_rr contour from CPFEM showing 4-lobed pattern
- Elastic-plastic boundary R(θ) comparison
- Possibly: E-convergence and mesh refinement as supplementary

### 3. Complete Section 7 (BCC vs FCC Comparison)

Currently has only the table. Add:
- 2–3 paragraphs of prose discussing each row of the comparison table
- Discussion of the yield polygon rotation (arctan(2√2)/2 ≈ 35.3°) and its physical origin
- Discussion of cross-plane vs same-plane pairings
- Why the activation pressure ratio is exactly 3/2

### 4. Complete Section 8 (Discussion)

**4a. Ductile fracture models** (~0.5 page)
- How this analytical solution feeds into Keralavarma-Benzerga-type porous crystal plasticity models
- The orientation-dependent activation pressure as input to homogenized Gurson models
- Comparison with computational results: Mbiakop et al. (2015), Paux et al. (2018)

**4b. Limitations and extensions** (~0.5 page)
- Limited to {110}<111> slip; BCC also has {112}<111> and {123}<111> (pencil glide)
- Rigid-ideally plastic: no hardening, no elastic effects
- Equibiaxial loading only; non-equibiaxial is Paper 2
- [110] void axis only; other orientations ([100], [111]) are future work
- Temperature-dependent CRSS and non-Schmid effects in BCC not considered

### 5. Polish and Finalize

**5a. Abstract** — verify all numbers match the final results

**5b. Introduction** — ensure the "roadmap" paragraph matches the actual section structure

**5c. References** — check all citations resolve; add any missing references from the numerical section (Borja 2013 for Ultimate Algorithm)

**5d. Acknowledgements** — fill in

**5e. Compile and proofread** — full pdflatex+bibtex build, check for orphaned references, undefined labels, figure placement

---

## Priority Order

| Priority | Item | Effort | Impact |
|----------|------|--------|--------|
| 1 | Write Section 7 (numerical verification) | 2–3 hours | High — reviewers expect this |
| 2 | CPFEM figures for Section 7 | 1–2 hours | High — visual evidence |
| 3 | Complete Section 7 (BCC vs FCC prose) | 1 hour | Medium |
| 4 | Discussion: ductile fracture models | 1 hour | Medium |
| 5 | Discussion: limitations | 0.5 hour | Medium |
| 6 | Geometry schematic | 1 hour | Low — nice to have |
| 7 | FCC reproduction figures | 2 hours | Medium |
| 8 | Polish and compile | 1 hour | Required |

**Estimated total: 10–12 hours of writing/coding work**
