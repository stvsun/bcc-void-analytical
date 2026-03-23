# Cylindrical Void in a BCC Single Crystal (Part IV)

Analytical stress field around a cylindrical void in a rigid-ideally plastic body-centered cubic single crystal, using Rice's (1973) anisotropic slip-line theory. This extends the series by Kysar et al. (2005, FCC) and Gan & Kysar (2007, HCP) to BCC crystals with {110}⟨111⟩ and {112}⟨111⟩ slip.

**Manuscript:** `manuscript/main.tex` (19 pages, Springer svjour3 format, targets IJP/JMPS)

## Main Results

**{110}⟨111⟩ slip (12 systems):**
- Yield polygon: hexagon with vertices (±√6/2, 0), (±√6/4, ±√3) in τ_CRSS units
- 6 angular sectors in [0°, 180°] with boundaries at arctan(2√2)/2 ≈ 35.26° and (π − arctan(2√2))/2 ≈ 54.74°
- Activation pressure: p* = 3√6/4 τ_CRSS ≈ 1.837 τ_CRSS
- BCC/FCC activation pressure ratio: exactly 3/2
- Systems 1,2 on the (110) plane have zero in-plane Schmid factor (anti-plane only)

**Exact interior stress field (Kysar-type Airy stress function solution):**
- Closed-form σ'_ij(r, θ) in each sector via characteristic tracing back to void surface or symmetry axis
- Sector I (sys 5,12): φ = −arctan(2√2)/2, A = 2√3/3, both characteristics reach void
- Sector II (sys 3,4): φ = 0, A = √3, both characteristics reach void
- Sector III (sys 6,11): φ = +arctan(2√2)/2, A = 2√3/3, α-lines reach x₂-axis symmetry boundary (modified Eq. 30 with √2 factor and −√6/3 offset)
- σ_rθ ≠ 0 in the interior — no approximations
- **φ₁ = γ degeneracy:** the characteristic angle equals the sector boundary angle, so no curved sector boundary forms (unlike FCC where 8+ sectors are needed). The 6 primary sectors provide the complete exact field.

**Combined {110}+{112}⟨111⟩ slip (24 systems):**
- Yield polygon: decagon (10 vertices), truncating all 6 hexagonal vertices
- 13 sectors; {112} systems dominate most of the angular range
- Activation pressure drops 4.5% to ≈ 1.754 τ_CRSS; still 43% above FCC
- Equal-CRSS (decagon) and τ^{112} → ∞ (hexagon) bracket the physical range

**CPFEM verification (Ultimate Algorithm, Borja 2013):**
- Sector structure confirmed with 100% match in all angular sectors
- Non-circular plastic zone confirmed (four-lobed pattern)
- Elastic-plastic vs rigid-plastic distinction characterized via mesh refinement and E-convergence studies

## Repository Structure

```
manuscript/           LaTeX paper (svjour3 class, plainnat bibliography)
  main.tex            Main manuscript (19 pages)
  references.bib      22 entries
  svjour3.cls         Springer journal class

src/                  Python scripts (NumPy, SymPy, SciPy, Matplotlib)
  derive_bcc_slip_systems.py      {110}<111> effective in-plane systems
  derive_bcc_combined_slip.py     {110}+{112} combined yield polygon
  exact_stress_field.py           Exact SymPy void surface stress
  sector_solution.py              Numerical sector structure
  combined_sector_solution.py     Combined {110}+{112} sectors
  exact_interior_kysar.py         Exact Kysar-type Airy stress function solution
  extended_sectors.py             Extended sectors analysis (φ₁=γ degeneracy)
  interior_stress_field.py        Leading-order interior field (historical)
  ultimate_algorithm.py           Rate-independent CPFEM (Borja 2013)
  cpfem_bcc_void.py               FEM solver and post-processing
  mesh_refinement_study.py        h-convergence study
  convergence_E_study.py          E → ∞ convergence study
  normalized_pattern_comparison.py   Angular pattern analysis
  verify_yielded_zone.py          Yielded-zone verification
  fig_yield_surface_comparison.py    Publication figure generation
  fig_geometry_schematic.py          Geometry schematic
  fig_stress_sectors_map.py          Sector map figure

figures/              Generated figures (PNG, 200 dpi)

plan_paper1.md        Completion plan for this paper
plan_paper2.md        Plan for Paper 2 (non-equibiaxial loading)
TODO_paper_completion.md   Remaining items
```

## Quick Start

```bash
pip install numpy sympy scipy matplotlib

# Derive the yield polygon and sector structure
python src/derive_bcc_slip_systems.py
python src/exact_stress_field.py

# Run the combined {110}+{112} analysis
python src/derive_bcc_combined_slip.py
python src/combined_sector_solution.py

# Run CPFEM verification
python src/ultimate_algorithm.py

# Compile the manuscript
cd manuscript && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

## References

1. Rice (1973), *J. Mech. Phys. Solids* 21, 63–74.
2. Kysar, Gan & Mendez-Arzuza (2005), *Int. J. Plast.* 21, 1481–1520.
3. Gan & Kysar (2007), *Int. J. Plast.* 23, 592–619.
4. Borja (2013), *Plasticity: Modeling & Computation*, Springer.

## License

MIT
