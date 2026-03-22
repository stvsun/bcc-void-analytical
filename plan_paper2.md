# Paper 2: Orientation-Dependent Void Growth in FCC and BCC Single Crystals Under Non-Equibiaxial Loading

**Target journal:** JMPS or IJP
**Working title:** "Anisotropic void growth rates in FCC and BCC single crystals: analytical solutions under general biaxial loading"

---

## Motivation

Paper 1 reveals that BCC and FCC have the same activation pressure under equibiaxial loading (up to orientation). The key physical differences emerge under **non-equibiaxial** loading, where:

- The rotated BCC yield polygon produces different sector structures than FCC
- Void growth rates become orientation- and stress-state-dependent
- The Lode angle and stress triaxiality affect BCC and FCC differently
- This is the regime relevant to real ductile fracture (crack-tip stress fields are NOT equibiaxial)

This paper would produce the **first analytical Rice-Tracey-equivalent formula for single crystals** — giving void growth rate as a function of stress triaxiality and crystal orientation.

---

## Key Steps

### Step 1: Non-Equibiaxial Formulation

- Far-field stress: σ₁₁ → −p₁, σ₂₂ → −p₂ with p₁ ≠ p₂
- Define stress triaxiality ratio: T = (p₁ + p₂) / |p₁ − p₂|
- Void surface BC remains: σ_rr = σ_rθ = 0 at r = a
- The void surface stress path on the yield polygon now depends on the ratio p₁/p₂

### Step 2: Asymmetric Sector Solution for FCC

- Start with Kysar's FCC yield polygon (known)
- Under non-equibiaxial loading, the 4-fold symmetry breaks to 2-fold (mirror about one axis only)
- The sector boundary angles now depend on p₁/p₂
- Derive the sector structure as a function of T
- The void surface stress traces a different path on the yield polygon for each T
- Compute σ_m(θ; T) in each sector

### Step 3: Asymmetric Sector Solution for BCC

- Same approach using the BCC yield polygon from Paper 1
- The BCC hexagon's rotation means different faces become active at different T values
- Critical finding expected: for certain T ranges, BCC activates different face sequences than FCC, leading to qualitatively different deformation patterns
- Compute σ_m(θ; T) in each sector

### Step 4: Void Growth Rate

- From the velocity field in each sector, compute the radial velocity at the void surface: v_r(r=a, θ)
- The volumetric void growth rate: V̇/V = (2/πa²) ∫₀^π v_r(a, θ) a dθ
- Express as ȧ_eff/a = f(T, crystal structure)
- This is the single-crystal Rice-Tracey formula

### Step 5: Comparison with Isotropic Rice-Tracey

- Classical Rice-Tracey (1969): ȧ/a ∝ exp(3T/2) for isotropic material
- Plot the crystal-specific void growth rate vs T for both FCC and BCC
- Identify the triaxiality ranges where:
  - BCC grows voids slower than FCC (expected at high T)
  - BCC grows voids faster than FCC (possible at low T due to different Lode angle sensitivity)
  - Both deviate from isotropic Rice-Tracey

### Step 6: Void Shape Evolution

- Under non-equibiaxial loading, the void does not remain circular
- The aspect ratio change rate: ḃ/ȧ = g(T, crystal structure)
- BCC and FCC produce different void shapes (ellipticity) at the same T
- This affects void coalescence and final fracture

### Step 7: Lode Angle Dependence

- For 2D plane strain, the Lode parameter is fixed
- But by varying the out-of-plane stress σ₃₃, different Lode states can be achieved
- Compute the activation surface in (T, L) space for both FCC and BCC
- Compare with phenomenological models (Nahshon-Hutchinson shear modification)

### Step 8: Multiple Orientations

- Solve for [100] and [111] void axes (both FCC and BCC)
- Map the full orientation dependence of void growth rate
- Produce inverse pole figure maps of ȧ/a(T) for BCC and FCC
- Identify the most void-resistant and most void-susceptible orientations

### Step 9: Numerical Verification

- CPFEM unit cell simulations for FCC and BCC under non-equibiaxial loading
- Verify:
  - Asymmetric sector boundaries vs analytical
  - Void growth rates vs analytical formula
  - Void shape evolution vs analytical prediction
- Use multiple stress triaxiality values: T = 1/3, 1, 2, 3

### Step 10: Write the Manuscript

**Outline:**

1. **Introduction** (1.5 pages)
   - Void growth rates in ductile fracture
   - Rice-Tracey (1969) isotropic formula and its limitations
   - Crystal anisotropy effects: computational results (Christodoulou et al. 2021) but no analytical formula
   - This paper: first analytical void growth rate for single crystals

2. **Problem Formulation** (1.5 pages)
   - Non-equibiaxial loading setup
   - Stress triaxiality and Lode angle definitions
   - Yield polygon review (from Paper 1 for BCC, Kysar 2005 for FCC)

3. **Asymmetric Sector Solutions** (3 pages)
   - FCC sector structure under general biaxial loading
   - BCC sector structure under general biaxial loading
   - How sector boundaries depend on stress triaxiality

4. **Void Growth Rate** (3 pages)
   - Velocity field derivation
   - Volumetric growth rate ȧ_eff/a = f(T)
   - Closed-form formula for FCC and BCC
   - Comparison with isotropic Rice-Tracey

5. **Void Shape Evolution** (2 pages)
   - Aspect ratio change rate
   - BCC vs FCC void shape anisotropy
   - Implications for coalescence

6. **Orientation Dependence** (2 pages)
   - [110], [100], [111] void axes
   - Inverse pole figure maps
   - Most/least resistant orientations

7. **Numerical Verification** (2 pages)
   - CPFEM model
   - Growth rate comparison at multiple T values
   - Void shape comparison

8. **Discussion and Implications** (1.5 pages)
   - Crystal-aware ductile fracture criteria
   - Bridging single-crystal to polycrystal (Taylor averaging)
   - Implications for radiation damage, AM defect tolerance

9. **Conclusions** (0.5 page)

**Total estimated length:** ~18 pages + figures

---

## Prerequisites

- Paper 1 must be completed (BCC yield polygon and sector solution)
- FCC solution must be re-derived in the same SymPy framework for consistency
- CPFEM solver capable of non-equibiaxial loading on single crystals

---

## Key Deliverable

The central result of this paper is a **closed-form void growth rate formula**:

$$\frac{\dot{a}}{a} = F\left(T, \text{crystal structure}, \text{orientation}\right)$$

This replaces the isotropic Rice-Tracey formula $\dot{a}/a \propto \exp(3T/2)$ with a crystal-specific version. It would be immediately usable in crystal plasticity damage models (Keralavarma-Benzerga, Mbiakop-Danas type).
