# Analytical Stress Field Around a Cylindrical Void in a BCC Single Crystal

Analytical stress field around a cylindrical void in a rigid-ideally plastic body-centered cubic single crystal, using Rice's (1973) anisotropic slip-line theory. This extends the FCC solution of Kysar et al. (2005) and the HCP solution of Gan & Kysar (2007) to BCC crystals with {110}⟨111⟩ and {112}⟨111⟩ slip.

**Manuscript:** `manuscript/main.tex` (22 pages, Springer svjour3 format, targets IJP)

## Main Results

**{110}⟨111⟩ slip (12 systems, 3 effective in-plane constraints):**
- Yield polygon: elongated hexagon with vertices (±√6/2, 0), (±√6/4, ±√3)
- 6 angular sectors with boundaries at θ₁ = arctan(2√2)/2 ≈ 35.26° and θ₂ ≈ 54.74°
- Sector I (sys 8,9, face V₄→V₅), Sector II (sys 3,4, face V₅→V₆), Sector III (sys 5,12, face V₆→V₁)
- Activation pressure: p* = 3√6/4 τ_CRSS ≈ 1.837 τ_CRSS (exactly 3/2 times FCC)
- Systems 1,2 have zero in-plane Schmid factor; systems 6,7,10,11 produce redundant constraints

**Exact near-void interior stress field (Kysar-type Airy stress functions):**
- Closed-form σ'_ij(r, θ) in each sector via characteristic tracing
- All three sectors share the same functional form: σ'₁₁ = A ρ sin(φ-θ)/√(1-ρ²sin²(φ-θ)), σ'₂₂ = -A ρ cos(φ-θ)/√(1-ρ²cos²(φ-θ))
- Sector III: the α-line reflection off the x₂-axis and the vertex V₁ condition cancel exactly, giving the same form as Sector I (no √2 factor or offset)
- Valid domain: r < r_crit(θ), where r_crit ≈ 1.2–1.4a

**Secondary sectors (beyond r_crit):**
- Curved boundary C₁: α-line from void at θ₁, parametric: (cosγ + t sinγ, sinγ + t cosγ)
- Curved boundary C₂: vertical β-line from void at θ₂ (x = 1/√3)
- Secondary sector Ia (between θ₁ and C₁): constant-stress sector, σ_m ≈ −1.84 τ_CRSS
- Secondary sector IIIa (between C₂ and θ₂): mirrors Ia by symmetry

**Combined {110}+{112}⟨111⟩ slip (24 systems):**
- Yield polygon: decagon (10 vertices), 13 sectors
- Activation pressure drops 4.5% to ≈ 1.754 τ_CRSS; still 43% above FCC

**CPFE verification:**
- Sector structure confirmed with 100% match in all angular sectors
- Non-circular plastic zone confirmed (four-lobed pattern)

## Repository Structure

```
manuscript/           LaTeX paper (svjour3 class, plainnat bibliography)
  main.tex            Main manuscript (22 pages)
  references.bib      Bibliography
  svjour3.cls         Springer journal class

src/                  Python scripts (NumPy, SymPy, SciPy, Matplotlib)
  derive_bcc_slip_systems.py      {110}<111> effective in-plane systems
  derive_bcc_combined_slip.py     {110}+{112} combined yield polygon
  exact_stress_field.py           Exact SymPy void surface stress
  sector_solution.py              Numerical sector structure
  combined_sector_solution.py     Combined {110}+{112} sectors
  exact_interior_kysar.py         Kysar-type Airy stress function solution
  secondary_sectors_full.py       Secondary sector computation
  secondary_sectors_sympy.py      SymPy symbolic derivation
  complete_sector_map.py          Complete sector map (primary + secondary)
  domain_of_validity.py           r_crit(θ) computation and figure
  sector_III_derivation.py        Sector III formula derivation
  verify_criticisms.py            Characteristic geometry verification
  ultimate_algorithm.py           Rate-independent CPFEM (Borja 2013)
  cpfem_bcc_void.py               FEM solver and post-processing
  fig_yield_surface_comparison.py Publication figure generation
  fig_geometry_schematic.py       Geometry schematic
  fig_stress_sectors_map.py       Sector map figure

figures/              Generated figures (PNG, 200 dpi)
```

## Quick Start

```bash
pip install numpy sympy scipy matplotlib

# Derive the yield polygon and sector structure
python src/derive_bcc_slip_systems.py
python src/exact_stress_field.py

# Generate interior field with secondary sectors
python src/exact_interior_kysar.py
python src/complete_sector_map.py

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
