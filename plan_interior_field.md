# Plan: Exact Interior Stress Field via Airy Stress Functions

## Goal

Replace the approximate Section 5 (σ_rθ ≈ 0 leading-order) with the
exact Kysar-style solution: closed-form Airy stress functions in each
stress sector, giving σ_ij(r, θ) everywhere with no approximations.

## Method (following Kysar 2005, Sections 3.4–3.13)

### The key idea

In each stress sector, one slip system is active. The yield condition
constrains the stress to lie on one face of the yield polygon. This
reduces the Airy stress equation from a general PDE to a specific
hyperbolic equation whose characteristics are the α- and β-lines of
Rice's theory. In local coordinates (ξ, η) aligned with these
characteristics, the Airy stress function has a **closed-form** solution.

### Coordinate systems

For BCC with void axis || [110], the 3 effective slip systems define
3 pairs of characteristic directions. In Kysar's notation:

- **System (3,4)** (face Y = ±√3): characteristics at φ_A = 0° (horizontal)
  and φ_A + 90° = 90° (vertical)
- **System (5,12)** (face V3→V4): characteristics at φ_B and φ_B + 90°
- **System (6,11)** (face V5→V6): characteristics at φ_C and φ_C + 90°

The characteristic angles φ_B and φ_C are determined by the yield face
geometry. For a face with Schmid coefficients (a, b), the
characteristics make angle φ with the x₁-axis where:
  tan(2φ) = 2b/a  (for α-lines)
  → for sys (5,12): a = √6/3, b = √3/6 → tan(2φ) = (2·√3/6)/(√6/3) = 1/√2
    → 2φ = arctan(1/√2) ≈ 35.26° → φ ≈ 17.63°
  Wait, this doesn't match. Let me re-derive.

Actually, the characteristic directions come from the yield surface
normal, not the face coefficients directly. For a yield face with
normal direction n̂ = (a, 2b)/|(a, 2b)| in the (X, 2Y) Mohr plane,
the characteristics in physical space are at angles:
  φ_α = (1/2) · (angle of n̂ in Mohr plane)
  φ_β = φ_α + π/2

For the BCC faces:
- Face V3→V4 (sys 5,12): n̂ direction in Mohr plane at angle
  arctan(2b/a) = arctan((2·√3/6)/(√6/3)) = arctan(√3/3 / √6/3) = arctan(1/√2)
  So 2φ_α = arctan(1/√2), φ_α = arctan(1/√2)/2

Hmm, I need to be more careful. Let me look at what Kysar actually uses.

In Kysar (2005):
- FCC system (i): characteristics at φ₁ ≈ 54.74° = arctan(√2)
  → α-lines at φ₁, β-lines at φ₁ - 90° = -35.26°
- FCC system (ii): characteristics at φ₂ = 0°
  → α-lines horizontal, β-lines vertical

For BCC, the analogous characteristic angles need to be derived from
the yield face orientations. The relationship between the yield face
and the characteristic direction is:

The characteristics of the governing PDE (Airy equation restricted to
one yield face) are lines along the slip directions of the active
system. For the effective system with Schmid tensor at angle φ_eff:
  α-lines: at angle φ_eff from x₁
  β-lines: at angle φ_eff + π/2 from x₁ (perpendicular)

From derive_bcc_slip_systems.py, the effective systems have angles:
- System (3,4): φ = -45° (or equivalently, characteristics at 45° and -45°)
  Wait, from the output: P_11 = 0, P_12 = -0.7071 → pure shear
  The slip direction for pure shear is at 45° to the axes
  α-lines at -45°, β-lines at +45°

Actually, I need to determine the characteristic directions from the
Schmid tensor, which gives the slip line orientations. The Schmid tensor
P_ij = (s_i n_j + s_j n_i)/2 for the effective pair gives the strain
rate direction. The characteristics are at the angles of the effective
slip direction s and slip normal n in the 2D plane.

For the effective system (3,4) on (1-10):
  Effective s = [1/√3, 0] in primed coords (from computation)
  Effective n = [0, -1] in primed coords
  → Slip direction along e₁': α-lines are horizontal
  → Slip normal along -e₂': β-lines are vertical

For system (5,12):
  The effective pair produces strain at angle φ ≈ ... need to compute

This is getting complicated. Let me just implement it step by step in
the code, following Kysar's derivation exactly but with BCC angles.

## Steps

### Step 1: Identify characteristic angles for each BCC stress sector
- For each of the 3 yield faces (sys 3,4 / sys 5,12 / sys 6,11),
  determine the α-line and β-line directions in the (e₁', e₂') plane
- These directions are the slip direction and slip normal of the
  effective slip system

### Step 2: Set up local coordinate systems
- For each stress sector, define (ξ, η) or (x₁', x₂') aligned with
  the characteristics of the active system
- Express r, θ in terms of these local coordinates

### Step 3: Derive the Airy stress function Ψ in each sector
- The governing equation is hyperbolic (Eq. 7 in Kysar)
- In local coordinates aligned with characteristics, Ψ satisfies
  a specific equation that integrates to a closed form
- The stress components in the rotated frame are:
  σ'₁₁ = ∂²Ψ/∂η², σ'₂₂ = ∂²Ψ/∂ξ², σ'₁₂ = -∂²Ψ/∂ξ∂η

### Step 4: Apply boundary conditions
- Void surface (r = a): σ_rr = σ_rθ = 0
- Symmetry along x₁-axis: σ_rθ = 0 at θ = 0
- Matching at sector boundaries: stress continuity

### Step 5: Convert to polar coordinates
- Transform σ'₁₁, σ'₂₂, σ'₁₂ → σ_rr, σ_θθ, σ_rθ

### Step 6: Derive extended stress sectors (IV-VIII) and curved boundary
- Analogous to Kysar's Section 3.7 onward
- The curved sector boundary between sectors IV and V is determined
  by the characteristic construction (Eqs. 40-52)

### Step 7: Verify with SymPy
- Check that Ψ satisfies the governing PDE
- Check that boundary conditions are satisfied
- Compare with the void surface stress (must match)

### Step 8: Generate Table 3 and Figure 10 with full vertices
- Compute all sector boundary curves and intersection points
- Produce the complete stress sector map

## Deliverables
- Replace Section 5 of manuscript with exact solution
- New figures: full stress field contours (like Kysar Figs. 16-19)
- Updated Table (sector vertices with extended sectors)
- SymPy verification code
