# Paper 1: Cylindrical Void in a Rigid-Ideally Plastic BCC Single Crystal

**Target journal:** JMPS or IJP
**Working title:** "Cylindrical void in a rigid-ideally plastic single crystal IV: Body-centered cubic crystal"

---

## Status: What Is Done

- [x] BCC {110}<111> slip systems transformed to plane strain frame ([110] void axis)
- [x] 3 distinct yield constraints identified (systems 1,2 have zero in-plane Schmid factor)
- [x] Exact hexagonal yield polygon: vertices (±√6/2, 0), (±√6/4, ±√3) τ_CRSS
- [x] 6 angular sectors with exact boundaries: arctan(2√2)/2 ≈ 35.26°, (π−arctan(2√2))/2 ≈ 54.74°, 90°
- [x] Exact void surface stress σ_ij(θ) in each sector (SymPy verified)
- [x] Activation pressure: p* = 3√6/4 τ_CRSS ≈ 1.837 (50% higher than FCC)
- [x] GitHub repo: https://github.com/stvsun/bcc-void-analytical

---

## What Remains

### Step 1: Complete the Interior Stress Field

The current solution gives stress at the void surface (r = a). The full solution requires the stress field for all r > a.

- Construct the slip-line characteristic network emanating from the void surface
- In each constant-stress sector: σ_ij is uniform (independent of r and θ) — already determined by the vertex stress
- At fan lines (sector boundaries): the characteristics are radial lines; σ_m jumps by the arc length along the yield face between adjacent vertices
- Compute the exact Δσ_m across each fan line using the yield polygon arc lengths
- Write the complete piecewise stress field σ_ij(r, θ) for r ≥ a
- Verify equilibrium ∇·σ = 0 symbolically in each region

### Step 2: Velocity Field (Deformation Kinematics)

- For each sector, determine the active slip system(s) and slip rate
- Construct the velocity field v(r, θ) consistent with:
  - Incompressibility: div(v) = 0
  - Flow rule: d_ij = Σ_α γ̇^α P^α_ij on active systems
  - Boundary condition: v_r(r=a) = ȧ (void growth rate)
- The velocity field differs from FCC due to different active systems at each angle — this is the kinematic signature of BCC
- Compute the void growth rate ȧ/a as a function of far-field pressure p

### Step 3: Numerical Verification (CPFEM)

- Implement a 2D plane-strain crystal plasticity FEM solver for BCC {110}<111>
- Single cylindrical void in an infinite medium (approximate with large outer boundary R/a ≥ 20)
- Rigid-ideally plastic constitutive law (rate-independent, or viscoplastic with m → 0)
- Equibiaxial far-field loading
- Compare:
  - Stress field σ_ij(r, θ) at the void surface against analytical
  - Sector boundary angles against analytical predictions
  - Activation pressure against p* = 3√6/4 τ_CRSS
  - Lattice rotation patterns (GND density)
- Alternatively: use the existing atlas-PINN framework with a BCC constitutive model on a cylindrical void geometry

### Step 4: FCC Comparison (Quantitative)

- Reproduce Kysar (2005) FCC results using the same SymPy framework for direct comparison
- Side-by-side plots: BCC vs FCC yield polygon, sector boundaries, stress fields, velocity fields
- Compute the ratio of activation pressures: p*_BCC / p*_FCC = (3√6/4) / (√6/2) = 3/2 — exactly 1.5
- Physical interpretation: the 50% higher activation pressure explains BCC radiation swelling resistance at the single-crystal level

### Step 5: Additional Orientations (Optional, strengthens paper)

- Solve for [100] void axis (4 effective in-plane systems → octagonal yield surface)
- Solve for [111] void axis (different symmetry)
- Show how activation pressure depends on void axis orientation
- Map out the orientation-dependent void growth resistance for BCC

### Step 6: Write the Manuscript

**Outline:**

1. **Introduction** (1.5 pages)
   - Void growth in ductile fracture
   - Kysar series: FCC (2005), experiments (2006), HCP (2007) — the BCC gap
   - BCC vs FCC radiation swelling resistance motivation
   - This paper: Part IV of the Kysar series for BCC

2. **Crystal Geometry and Effective Slip Systems** (2 pages)
   - BCC {110}<111> slip systems
   - Coordinate transformation to [110] void axis frame
   - Derivation of 3 distinct yield constraints
   - Zero in-plane Schmid factor for (110) plane systems — anti-plane only

3. **Yield Polygon Construction** (1.5 pages)
   - Hexagonal yield polygon vertices and faces
   - Comparison with FCC: same inscribed circle, rotated and elongated
   - Exact vertex coordinates in terms of τ_CRSS

4. **Stress Field Solution** (3 pages)
   - Rice (1973) framework: generalized Hencky equations
   - Sector structure: constant-stress regions and fan lines
   - Exact sector boundaries: arctan(2√2)/2, (π−arctan(2√2))/2, π/2
   - Complete stress field σ_ij(r, θ) in each sector
   - Activation pressure: p* = 3√6/4 τ_CRSS

5. **Velocity Field and Void Growth Rate** (2 pages)
   - Slip rates in each sector
   - Velocity field v(r, θ)
   - Void growth rate under equibiaxial stress

6. **Numerical Verification** (2 pages)
   - CPFEM model description
   - Stress field comparison (pointwise and contour plots)
   - Sector boundary verification
   - Activation pressure verification

7. **BCC vs FCC: Physical Interpretation** (2 pages)
   - Side-by-side yield polygon comparison
   - Activation pressure ratio = 3/2 (exact)
   - Different deformation kinematics → different void shapes
   - Implications for radiation void swelling resistance
   - Implications for ductile fracture models

8. **Conclusions** (0.5 page)

**Total estimated length:** ~15 pages + figures

### Step 7: Submit

- Format for JMPS (Elsevier) or IJP
- Suggested reviewers: J.W. Kysar (Columbia), S.M. Keralavarma (IIT Madras), K. Danas (Ecole Polytechnique), R. Brenner (Sorbonne)

---

## Timeline Estimate

| Step | Effort |
|------|--------|
| 1. Interior stress field | 1–2 days (SymPy derivation) |
| 2. Velocity field | 1–2 days |
| 3. Numerical verification | 3–5 days (CPFEM implementation) |
| 4. FCC comparison | 1 day |
| 5. Additional orientations | 2–3 days (optional) |
| 6. Write manuscript | 5–7 days |
| 7. Submit | 1 day |
