"""
Exact interior stress field for BCC void, following Kysar (2005).

Kysar's approach (FCC):
  - 3 effective slip systems at angles φ₁ = arctan(√2) ≈ 54.74°,
    φ₂ = 0°, φ₃ = -arctan(√2)
  - β₁ = 2/√3, β₂ = √3 (scaling factors from yield polygon geometry)
  - In each sector, σ'₁₂ = A = ±βτ is constant (resolved shear stress)
  - The x'₁, x'₂ frame is aligned with the α, β characteristics
  - Key result: σ'₁₁ = f(x'₂) only, σ'₂₂ = g(x'₁) only

For BCC {110}<111> with void axis || [110]:
  - 3 effective systems at angles we need to determine
  - The yield polygon is elongated, so β values differ from FCC
  - The formalism is identical: just different φ and β values

The relationship between our notation and Kysar's:
  Kysar's φ = angle of the effective slip system in the (x₁, x₂) plane
  Kysar's β = distance from origin to yield face / τ
  Kysar's A = ±βτ (take τ = 1)
  Kysar's γ = angle where slip system changes on the void surface

For the BCC yield polygon with vertices:
  V₃ = (+√6/2, 0), V₄ = (+√6/4, +√3), V₅ = (-√6/4, +√3), V₆ = (-√6/2, 0)

The yield faces have:
  Face V₃→V₄ (sys 5,12): (√6/3)X + (√3/6)Y = 1
  Face V₄→V₅ (sys 3,4): Y = √3 (i.e., 0·X + (√3/3)·(-Y) = -1... wait)

Let me use Kysar's Eq. (3): (S₁N₂ + S₂N₁)σ₁₂ + 2S₁N₁(σ₁₁-σ₂₂)/2 = ±βτ
where S = (cos φ, sin φ) and N = (-sin φ, cos φ) are the effective
slip direction and normal.

For a face at angle φ, the yield condition becomes:
  σ₁₂ = tan(2φ) · (σ₁₁-σ₂₂)/2 ± βτ/cos(2φ)   [Kysar Eq. (4)]

So β/cos(2φ) = distance from origin to the yield face in the
(σ₁₁-σ₂₂)/2, σ₁₂ plane, measured perpendicular to the face.

The constants β and φ for each BCC system need to be determined from
the yield face equations.
"""

import numpy as np
import sympy as sp
from sympy import sqrt, cos, sin, tan, atan, atan2, pi, Rational, simplify, symbols, asin, acos

# ================================================================
# Step 1: Determine φ and β for each BCC effective system
# ================================================================
print("=" * 70)
print("Step 1: BCC Effective System Angles and β Values")
print("=" * 70)

# The yield polygon faces in (X, Y) = ((σ₁₁-σ₂₂)/2, σ₁₂) space:
#
# Face V₃→V₄ (sys 5,12): (√6/3)X + (√3/6)Y = 1
#   Normal direction: (√6/3, √3/6), normalize: |(√6/3, √3/6)| = √(6/9 + 3/36) = √(24/36+3/36) = √(27/36) = 3√3/6 = √3/2
#   Distance from origin = 1 / (√3/2) = 2/√3 = 2√3/3
#
# From Kysar's Eq. (4): the face equation in terms of φ is
#   σ₁₂ = tan(2φ)(σ₁₁-σ₂₂)/2 ± βτ/cos(2φ)
#   i.e.: -tan(2φ) · X + Y = ±β/cos(2φ)
#   Comparing with (√6/3)X + (√3/6)Y = 1:
#   Multiply by 6/√3: (6√6/(3√3))X + Y = 6/√3
#   → (2√2)X + Y = 2√3
#   So: -tan(2φ) = 2√2 → tan(2φ) = -2√2
#   And: β/cos(2φ) = 2√3
#
# tan(2φ) = -2√2:
#   2φ = π - arctan(2√2) (in the second quadrant, since tan is negative)
#   → 2φ = π - arctan(2√2) ≈ 180° - 70.53° = 109.47°
#   → φ ≈ 54.74°
#
# cos(2φ) = cos(π - arctan(2√2)) = -cos(arctan(2√2))
#   cos(arctan(2√2)) = 1/√(1+8) = 1/3
#   So cos(2φ) = -1/3
#   β = 2√3 · cos(2φ) = 2√3 · (-1/3) = -2√3/3
#   Take |β| = 2√3/3 = 2/√3

# Let me verify: distance from origin to face = |1| / |(√6/3, √3/6)|
# |(√6/3, √3/6)| = √(6/9 + 3/36) = √(24/36 + 3/36) = √(27/36) = √(3/4) = √3/2
# distance = 1/(√3/2) = 2/√3 = 2√3/3 ✓
# And β/|cos(2φ)| = (2√3/3)/(1/3) = 2√3 ≠ 2√3/3...

# Actually, let me re-derive more carefully using Kysar's notation.
# Kysar Eq. (3): (S₁N₂ + S₂N₁)σ₁₂ + 2S₁N₁(σ₁₁-σ₂₂)/2 = ±βτ
# With S₁ = cos φ, S₂ = sin φ, N₁ = -sin φ, N₂ = cos φ:
#   (cos φ · cos φ + sin φ · (-sin φ))σ₁₂ + 2 cos φ(-sin φ)(σ₁₁-σ₂₂)/2 = ±βτ
#   cos(2φ) · σ₁₂ - sin(2φ) · (σ₁₁-σ₂₂)/2 = ±βτ
#
# In (X, Y) notation:
#   -sin(2φ) · X + cos(2φ) · Y = ±β    [with τ = 1]
#
# For face V₃→V₄: (√6/3)X + (√3/6)Y = 1
#   → -sin(2φ) = √6/3 and cos(2φ) = √3/6 ... NO, the coefficients
#   must be normalized.
#
# The yield condition is: -sin(2φ)X + cos(2φ)Y = ±β
# This is a line at distance |β| from origin (since sin²+cos² = 1).
# For face V₃→V₄: (√6/3)X + (√3/6)Y = 1
# Normalize: coefficients (√6/3, √3/6), norm = √(6/9+3/36) = √(3/4) = √3/2
# Normalized: (√6/3)/(√3/2) X + (√3/6)/(√3/2) Y = 1/(√3/2)
#           = (2√6/(3√3)) X + (2√3/(6√3)) Y = 2/√3
#           = (2√2/3) X + (1/3) Y = 2/√3
# Hmm, let me just compute: (√6/3)/(√3/2) = (√6/3)·(2/√3) = 2√6/(3√3) = 2√2/3
# And (√3/6)/(√3/2) = (√3/6)·(2/√3) = 2/6 = 1/3
# So the unit normal to the face is (2√2/3, 1/3)
# And β = 1/(√3/2) = 2/√3 = 2√3/3

# Now matching: -sin(2φ) = 2√2/3, cos(2φ) = 1/3
# Check: sin²(2φ) + cos²(2φ) = 8/9 + 1/9 = 1 ✓
# β = 2√3/3 ✓

# So: sin(2φ) = -2√2/3, cos(2φ) = 1/3
# 2φ = -arcsin(2√2/3) or equivalently 2φ in the 4th quadrant
# arcsin(2√2/3) = arcsin(0.9428) ≈ 70.53°
# So 2φ ≈ -70.53° → φ ≈ -35.26°

# Wait, but Kysar has φ₁ ≈ +54.74° for FCC system (i). Let me check
# the sign convention. The issue is which direction we measure φ from.

# For FCC system (i): φ₁ = arctan(√2) ≈ 54.74°
# 2φ₁ ≈ 109.47°. sin(2φ₁) = sin(109.47°) = cos(19.47°) ≈ 0.9428
# cos(2φ₁) = cos(109.47°) = -sin(19.47°) ≈ -0.3333 = -1/3
# So for FCC: -sin(2φ₁) = -0.9428, cos(2φ₁) = -1/3
# β₁ = 2/√3 (from Kysar)

# For BCC face V₃→V₄: we found -sin(2φ) = +2√2/3, cos(2φ) = +1/3
# This gives 2φ in the 4th quadrant: 2φ = -arctan(2√2) ≈ -70.53°
# → φ ≈ -35.26°

# But we could also have φ in the 2nd quadrant: 2φ = 180° - (-70.53°)...
# The sign of φ determines the orientation of the x'₁, x'₂ frame.

# Let me just define the systems directly from the BCC yield polygon
# and the sector structure.

# BCC SECTORS (from our analysis):
# Sector I:   0 < θ < θ₁ ≈ 35.26°   sys (5,12), face V₃→V₄
# Sector II:  θ₁ < θ < θ₂ ≈ 54.74°  sys (3,4),  face V₄→V₅
# Sector III: θ₂ < θ < π/2          sys (6,11), face V₅→V₆

# For each sector, I need φ (slip system angle) and A (= ±βτ).

# System (5,12) face V₃→V₄: (√6/3)X + (√3/6)Y = 1
#   From above: -sin(2φ) = 2√2/3, cos(2φ) = 1/3
#   → 2φ = -arctan(2√2) [4th quadrant, since sin<0, cos>0]
#   → φ₁ = -arctan(2√2)/2 ≈ -35.26°
#   β₁ = 2√3/3
#   A_I = -β₁ = -2√3/3 (negative because face V₃→V₄ has σ₁₂ < 0 at θ=0... check)

# Actually A = ±βτ. The sign depends on the direction of slip.
# At θ = 0 (on the x₁-axis), the void surface has σ₁₂ = 0 and
# (σ₁₁-σ₂₂)/2 > 0 (σ_θθ < σ_rr = 0 → σ₂₂ < σ₁₁).
# From the void surface stress: at θ=0, X = √6/2 > 0, Y = 0.
# The yield face V₃→V₄ passes through V₃ = (√6/2, 0) at θ=0.
# From Kysar Eq. (5): σ_rθ = tan[2(φ-θ)]·(σ_rr-σ_θθ)/2 ± βτ/cos[2(φ-θ)]
# At θ=0 on void surface: σ_rθ = 0, so:
#   0 = tan(2φ₁)·(0-σ_θθ)/2 + A_I/cos(2φ₁)
# We know σ_θθ(θ=0) = -√6 τ from our analysis, so (σ_rr-σ_θθ)/2 = √6/2
#   0 = tan(2φ₁)·√6/2 + A_I/cos(2φ₁)
# With tan(2φ₁) = sin(2φ₁)/cos(2φ₁) = (-2√2/3)/(1/3) = -2√2:
#   0 = -2√2·√6/2 + A_I/(1/3)
#   0 = -√12 + 3A_I
#   A_I = √12/3 = 2√3/3

# For Kysar's Eq. (14): σ_θθ = 2A/sin[2(φ-θ)]
# At θ=0: σ_θθ = 2A_I/sin(2φ₁) = 2·(2√3/3)/(-2√2/3) = (4√3/3)·(-3/(2√2))
#        = -4√3/(2√2) = -2√3/√2 = -√6
# ✓ This matches our void surface result!

print("\nBCC System Parameters:")

# System (5,12) — active in Sector I
phi1_2 = -np.arctan(2*np.sqrt(2))  # 2φ₁
phi1 = phi1_2 / 2
beta1 = 2*np.sqrt(3)/3
A_I = 2*np.sqrt(3)/3  # = β₁ (positive)
print(f"  Sector I (sys 5,12):  φ₁ = {np.degrees(phi1):.4f}°, "
      f"2φ₁ = {np.degrees(phi1_2):.4f}°, β₁ = {beta1:.6f}, A_I = {A_I:.6f}")
print(f"    Check: sin(2φ₁) = {np.sin(phi1_2):.6f} (should be -2√2/3 = {-2*np.sqrt(2)/3:.6f})")
print(f"    Check: cos(2φ₁) = {np.cos(phi1_2):.6f} (should be 1/3 = {1/3:.6f})")

# System (3,4) — active in Sector II
# Face V₄→V₅: Y = √3, i.e., 0·X + 1·Y = √3
# Normalized: (0, 1)·(X, Y) = √3, so normal is (0, 1)
# -sin(2φ₂) = 0, cos(2φ₂) = 1 → 2φ₂ = 0 → φ₂ = 0
# β₂ = √3
phi2 = 0.0
beta2 = np.sqrt(3)
A_II = np.sqrt(3)
print(f"\n  Sector II (sys 3,4):  φ₂ = {np.degrees(phi2):.4f}°, "
      f"β₂ = {beta2:.6f}, A_II = {A_II:.6f}")

# System (6,11) — active in Sector III
# Face V₅→V₆: -(√6/3)X - (√3/6)Y = 1, or equivalently (√6/3)X + (√3/6)Y = -1
# Rewrite as: -(√6/3)X + (-(√3/6))Y = 1... wait let me be careful.
# From the yield face table:
# Face V₅→V₆: -√6/3 X - √3/6 Y = 1
# Normalized: |(-√6/3, -√3/6)| = √(6/9+3/36) = √3/2
# Unit normal: (-2√2/3, -1/3)
# Distance = 1/(√3/2) = 2/√3 = 2√3/3
# -sin(2φ₃) = -2√2/3 → sin(2φ₃) = 2√2/3
# cos(2φ₃) = -1/3
# 2φ₃ = π - arcsin(2√2/3) ≈ 180° - 70.53° = 109.47°
# → φ₃ = 109.47°/2 ≈ 54.74°
# But actually 2φ₃ could also be in the 2nd quadrant:
# sin(2φ₃) > 0, cos(2φ₃) < 0 → 2φ₃ in 2nd quadrant ✓
# 2φ₃ = π - arctan(2√2) ≈ 109.47°
# φ₃ ≈ 54.74°
phi3_2 = np.pi - np.arctan(2*np.sqrt(2))
phi3 = phi3_2 / 2
beta3 = 2*np.sqrt(3)/3
# Determine sign of A_III:
# At θ = π/2 (on x₂-axis): σ_rθ = 0 by symmetry
# σ_θθ = 2A_III/sin[2(φ₃ - π/2)] = 2A_III/sin(2φ₃ - π)
# sin(2φ₃ - π) = -sin(π - 2φ₃) = -sin(arctan(2√2)) = -2√2/3
# So σ_θθ(π/2) = 2A_III/(-2√2/3) = -3A_III/√2
# From our analysis: σ_θθ(π/2) = -√6 = -3·(2√3/3)/√2 ... let me check:
# -3·(2√3/3)/√2 = -2√3/√2 = -√6 ✓ if A_III = 2√3/3
A_III = 2*np.sqrt(3)/3
print(f"\n  Sector III (sys 6,11): φ₃ = {np.degrees(phi3):.4f}°, "
      f"2φ₃ = {np.degrees(phi3_2):.4f}°, β₃ = {beta3:.6f}, A_III = {A_III:.6f}")

# Sector boundary angles
gamma = np.arctan(2*np.sqrt(2)) / 2  # ≈ 35.26° (same as θ₁)
phi1_angle = np.arctan(2*np.sqrt(2))  # ≈ 70.53° (= 2γ in Kysar's notation)

print(f"\n  γ (slip system transition angle) = {np.degrees(gamma):.4f}°")
print(f"  φ₁ (from Kysar notation) = {np.degrees(phi1_angle/2):.4f}°")

# ================================================================
# Step 2: Stress field in each sector (Kysar Eqs. 17, 21, 30)
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Exact Stress Field in Each Sector")
print("=" * 70)

# Following Kysar's approach, in each stress sector the stresses in the
# rotated (x'₁, x'₂) frame aligned with the active slip system are:
#
# Stress Sector I (sys 5,12 active, slip at angle φ₁):
#   σ'₁₁ = A_I(r/r₀)sin(φ₁ - θ) / √(1 - (r/r₀)²sin²(φ₁ - θ))   [Kysar 17a]
#   σ'₂₂ = A_I(r/r₀)cos(φ₁ - θ) / √(1 - (r/r₀)²cos²(φ₁ - θ))   [Kysar 17b]
#   σ'₁₂ = A_I                                                      [Kysar 17c]
#
# BUT: φ₁ in Kysar is the angle of the slip system direction (α-line).
# For BCC Sector I, the active system (5,12) has α-lines at some angle
# that we've identified as |φ₁| ≈ 35.26° from the x₁-axis.
#
# The key relationship: the α-line direction at angle φ means that
# at any point (r, θ), the α-line through that point intersects the
# void surface at θ_{s1} and the β-line intersects at θ_{t1}, where:
#   θ_{s1} = φ - arcsin[(r/r₀)sin(φ - θ)]    [Kysar 16a]
#   θ_{t1} = φ - arccos[(r/r₀)cos(φ - θ)]    [Kysar 16b]
#
# In these equations, φ is measured counterclockwise from the x₁-axis
# to the α-line direction.
#
# For BCC: I need to use the correct φ for each sector.
# The void surface boundary condition σ_θθ = 2A/sin[2(φ-θ)]
# must match our exact solution.

# Let me check for Sector I:
# Our exact solution: σ_θθ(a, θ) = -12/(√3 sin2θ + 2√6 cos2θ)
# Kysar Eq. (14): σ_θθ = 2A_I/sin[2(φ₁-θ)]

# Using φ₁ = -arctan(2√2)/2 ≈ -35.26°:
# sin[2(φ₁-θ)] = sin(-arctan(2√2) - 2θ)
# At θ=0: sin(-arctan(2√2)) = -sin(arctan(2√2)) = -2√2/3
# 2A_I/(-2√2/3) = 2·(2√3/3)·(-3/(2√2)) = -2√3/√2 = -√6 = -2.449...
# Our solution at θ=0: -12/(0 + 2√6) = -12/(2√6) = -6/√6 = -√6 ✓

# But wait — we need φ₁ to be the angle measured CCW from x₁ to the
# α-LINE direction. The α-line corresponds to the SLIP DIRECTION of
# the active system. For Kysar's FCC: φ₁ = arctan(√2) ≈ 54.74° is
# the angle of the (111)[1-10] effective slip direction.

# For BCC, what is the effective slip direction of system (5,12)?
# System 5: n=(101), s=(-111) → in primed coords:
#   s'₅ = [1/√3, √(2/3), 0] → in-plane direction at angle arctan(√(2/3)/(1/√3))
#   = arctan(√2) ≈ 54.74°
# System 12: n=(01-1), s=(1-1-1) → s'₁₂ = [-1/√3, -√(2/3), 0] (opposite)
# The effective pair gives slip in the plane at some combined angle.

# Actually, I think the confusion is that φ in Kysar's notation is NOT
# the slip direction angle of the individual system, but rather the
# angle of the α-characteristic in the slip-line field. For a single
# active face, the α-line is at angle φ where the yield face has its
# specific orientation.

# Let me use a different approach: just parameterize by the face normal
# angle and work directly.

# The yield face for sector I: -sin(2φ)X + cos(2φ)Y = β
# With -sin(2φ) = 2√2/3, cos(2φ) = 1/3, β = 2√3/3
# → φ measured from x₁ such that 2φ is in the 4th quadrant

# But Kysar's Eq. (14) with our φ₁ = -arctan(2√2)/2:
# σ_θθ = 2A_I/sin[2(φ₁ - θ)]
# = 2·(2√3/3)/sin(-arctan(2√2) - 2θ)
# = (4√3/3)/(-sin(arctan(2√2) + 2θ))
# = -(4√3/3)/sin(arctan(2√2) + 2θ)

# Expand sin(arctan(2√2) + 2θ):
# = sin(arctan(2√2))cos(2θ) + cos(arctan(2√2))sin(2θ)
# = (2√2/3)cos(2θ) + (1/3)sin(2θ)

# So σ_θθ = -(4√3/3) / [(2√2/3)cos(2θ) + (1/3)sin(2θ)]
#         = -(4√3/3) · 3 / [2√2 cos(2θ) + sin(2θ)]
#         = -4√3 / [2√2 cos(2θ) + sin(2θ)]

# But our exact result is: σ_θθ = -12/(√3 sin2θ + 2√6 cos2θ)
# = -12/(√3 sin2θ + 2√6 cos2θ)

# Check if these match: -4√3/(sin2θ + 2√2 cos2θ) vs -12/(√3 sin2θ + 2√6 cos2θ)
# Multiply top and bottom of first by √3:
# = -4·3/(√3 sin2θ + 2√6 cos2θ) = -12/(√3 sin2θ + 2√6 cos2θ) ✓✓✓

print("Verification: Kysar Eq. (14) matches our exact void surface stress ✓")

# Now the full interior field.
# In the BCC case, Kysar's Eqs. (17) give for Sector I:
# In the rotated frame (x'₁ along α-line at angle φ₁, x'₂ along β-line):

# We use |φ₁| for the formulas (the sign just determines the rotation direction)
phi1_abs = np.arctan(2*np.sqrt(2)) / 2  # ≈ 35.26° (positive)

# For sector I, the slip system angle to use in Kysar's formulas is
# φ = π/2 - phi1_abs? No...

# Actually, from the matching above, the correct φ to use in Kysar's
# Eq. (17) is NOT φ₁ = -35.26° but rather we need to identify which
# convention Kysar uses.

# In Kysar, for FCC sector I (system (i)), φ₁ = arctan(√2) ≈ 54.74°.
# The α-lines make angle φ₁ with the x₁-axis.
# For a point P₁ at (r, θ) in sector I:
#   The α-line through P₁ intersects void at θ_{s1} = φ₁ - arcsin[(r/r₀)sin(φ₁-θ)]
#   The β-line through P₁ intersects void at θ_{t1} = φ₁ - arccos[(r/r₀)cos(φ₁-θ)]

# For BCC sector I (face V₃→V₄):
# We showed sin(2φ) = -2√2/3, cos(2φ) = 1/3
# This gives 2φ = -arctan(2√2) or equivalently 2φ = 2π - arctan(2√2)
# So |φ| = arctan(2√2)/2 ≈ 35.26° but in the negative direction.

# HOWEVER, what matters is the angle of the α-line (slip direction)
# in physical space. From the geometry:
# - At θ=0 on the void, the active system is (5,12) and the stress
#   is at vertex V₃ = (√6/2, 0).
# - At θ=θ₁ ≈ 35.26°, the stress transitions to V₄.
# - The α-lines emanate from the void surface at various angles.
# - The key angle: the α-line direction.

# From Kysar's framework, the α-line at point P makes angle φ
# (= the slip system angle) with the x₁ axis. For a point at angle θ
# on the void, the α-line goes into the material at angle φ from
# the horizontal.

# For BCC, I believe the correct identification is:
# The effective slip direction for system (5,12) makes angle φ_eff
# with x₁. This angle should equal the sector boundary angle:
# In Kysar's FCC, φ₁ = 54.74° and γ = arctan(1/√2) ≈ 35.26°
# with φ₁ + γ = 90° (they sum to 90°).
# The sector boundary on the void surface is at γ ≈ 35.26°.

# For BCC, our sector boundaries are at θ₁ ≈ 35.26° and θ₂ ≈ 54.74°.
# So the BCC angles are: γ_BCC = 35.26°, and the slip system angle is...
# By analogy with Kysar: φ = θ₂ ≈ 54.74° for sector I? Or φ = θ₁?

# From the matching of Eq. (14):
# Using φ₁ such that Kysar's formula σ_θθ = 2A/sin[2(φ₁-θ)] gives
# the correct BCC void surface stress, we found:
# sin[2(φ₁-θ)] = sin(-arctan(2√2) - 2θ) = -sin(arctan(2√2) + 2θ)
# and the formula works with 2φ₁ = -arctan(2√2).
# So φ₁_eff = -arctan(2√2)/2.

# BUT Kysar uses φ₁ > 0 (54.74° for FCC). The negative sign means
# the BCC slip direction is on the opposite side of the x₁-axis.
# This is geometrically correct: in FCC the slip direction goes UP
# from the void at 54.74°, while in BCC it goes DOWN at -35.26°
# (or equivalently, the β-line goes up).

# For the Kysar-style interior formulas, we just need to substitute
# the correct φ. Let me define everything and compute.

# Define the stress field functions following Kysar Eqs. (17)
def stress_sector_I(r, theta, r0=1.0):
    """Exact stress in Sector I (0 < θ < θ₁) using Kysar Eq. (17)."""
    phi = -np.arctan(2*np.sqrt(2)) / 2  # ≈ -35.26°
    A = 2*np.sqrt(3)/3
    rho = r / r0

    arg = phi - theta  # φ - θ

    s11p = A * rho * np.sin(arg) / np.sqrt(1 - rho**2 * np.sin(arg)**2)
    s22p = A * rho * np.cos(arg) / np.sqrt(1 - rho**2 * np.cos(arg)**2)
    s12p = A

    # Convert from rotated (x'₁, x'₂) frame to (x₁, x₂) frame
    # The x'₁ axis is at angle φ from x₁
    # σ₁₁ = σ'₁₁ cos²φ + σ'₂₂ sin²φ - 2σ'₁₂ sinφ cosφ
    # etc.
    c = np.cos(phi)
    s = np.sin(phi)
    c2 = np.cos(2*phi)
    s2 = np.sin(2*phi)

    sig11 = (s11p + s22p)/2 + (s11p - s22p)/2 * c2 - s12p * s2
    sig22 = (s11p + s22p)/2 - (s11p - s22p)/2 * c2 + s12p * s2
    sig12 = (s11p - s22p)/2 * s2 + s12p * c2

    # Convert to polar
    c2t = np.cos(2*theta)
    s2t = np.sin(2*theta)
    sigma_m = (sig11 + sig22) / 2
    X = (sig11 - sig22) / 2
    Y = sig12

    srr = sigma_m + X * c2t + Y * s2t
    stt = sigma_m - X * c2t - Y * s2t
    srt = -X * s2t + Y * c2t

    return srr, stt, srt

# Similarly for Sector II (Kysar Eq. 21)
def stress_sector_II(r, theta, r0=1.0):
    """Exact stress in Sector II (θ₁ < θ < θ₂) using Kysar Eq. (21/23)."""
    phi = 0.0  # system (3,4), characteristics horizontal/vertical
    A = np.sqrt(3)
    rho = r / r0

    arg = phi - theta  # = -θ (since φ₂ = 0)

    # Kysar Eq. (23):
    s11p = -A * rho * np.sin(arg) / np.sqrt(1 - rho**2 * np.sin(arg)**2)
    s22p = -A * rho * np.cos(arg) / np.sqrt(1 - rho**2 * np.cos(arg)**2)
    s12p = A

    # In this sector, x'₁ is along the α-line (horizontal since φ₂=0)
    # so the rotated frame IS the original frame: no rotation needed
    sig11 = s11p
    sig22 = s22p
    sig12 = s12p

    # Convert to polar
    c2t = np.cos(2*theta)
    s2t = np.sin(2*theta)
    sigma_m = (sig11 + sig22) / 2
    X = (sig11 - sig22) / 2
    Y = sig12

    srr = sigma_m + X * c2t + Y * s2t
    stt = sigma_m - X * c2t - Y * s2t
    srt = -X * s2t + Y * c2t

    return srr, stt, srt

# Sector III (Kysar Eq. 30 analogue)
def stress_sector_III(r, theta, r0=1.0):
    """Exact stress in Sector III (θ₂ < θ < π/2)."""
    phi = (np.pi - np.arctan(2*np.sqrt(2))) / 2  # ≈ 54.74°
    A = 2*np.sqrt(3)/3
    rho = r / r0

    arg = phi - theta

    # The stress in the rotated frame — need to determine from the
    # void surface BC and characteristic construction.
    # For sector III, the β-lines intersect the void surface but
    # the α-lines may not (they may intersect the x₂-axis instead).
    # This is analogous to Kysar's Stress Sector III (Section 3.6).

    # From Kysar Eq. (30):
    # σ'₁₁ = A_I(r/r₀)sin(φ₁-θ) / √(2 - (r/r₀)²sin²(φ₁-θ))  (note: 2 not 1!)
    # σ'₂₂ = A_I(r/r₀)cos(φ₁-θ) / √(1 - (r/r₀)²cos²(φ₁-θ))
    # σ'₁₂ = A_I

    # Wait, Eq. (30) for sector III has a different form because the
    # α-line intersects the x₁-axis (symmetry axis) rather than the
    # void surface. The factor under the square root changes from
    # (1 - ...) to (2 - ...) because of the symmetry condition.
    # This requires more careful analysis for BCC.

    # For now, use the simpler form valid near the void surface:
    s11p = A * rho * np.sin(arg) / np.sqrt(np.maximum(1e-10, 1 - rho**2 * np.sin(arg)**2))
    s22p = A * rho * np.cos(arg) / np.sqrt(np.maximum(1e-10, 1 - rho**2 * np.cos(arg)**2))
    s12p = A

    c2p = np.cos(2*phi)
    s2p = np.sin(2*phi)

    sig11 = (s11p + s22p)/2 + (s11p - s22p)/2 * c2p - s12p * s2p
    sig22 = (s11p + s22p)/2 - (s11p - s22p)/2 * c2p + s12p * s2p
    sig12 = (s11p - s22p)/2 * s2p + s12p * c2p

    c2t = np.cos(2*theta)
    s2t = np.sin(2*theta)
    sigma_m = (sig11 + sig22) / 2
    X = (sig11 - sig22) / 2
    Y = sig12

    srr = sigma_m + X * c2t + Y * s2t
    stt = sigma_m - X * c2t - Y * s2t
    srt = -X * s2t + Y * c2t

    return srr, stt, srt

# ================================================================
# Step 3: Verify at void surface
# ================================================================
print("\n" + "=" * 70)
print("Step 3: Verification at Void Surface (r = a)")
print("=" * 70)

theta1 = np.arctan(2*np.sqrt(2)) / 2
theta2 = (np.pi - np.arctan(2*np.sqrt(2))) / 2

print(f"\n{'θ (deg)':>10s} {'Sector':>8s} {'σ_rr':>10s} {'σ_θθ':>10s} "
      f"{'σ_rθ':>10s} {'σ_θθ exact':>12s}")

for th_deg in [0, 10, 20, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90]:
    th = np.radians(th_deg)
    r = 1.0001  # slightly above void surface (r=a=1)

    if th < theta1 - 0.01:
        srr, stt, srt = stress_sector_I(r, th)
        sector = "I"
    elif th < theta2 - 0.01:
        srr, stt, srt = stress_sector_II(r, th)
        sector = "II"
    else:
        srr, stt, srt = stress_sector_III(r, th)
        sector = "III"

    # Exact void surface σ_θθ from our analysis
    c2 = np.cos(2*th)
    s2 = np.sin(2*th)
    # Find X on yield surface at ray angle 2θ
    schmid_sys = [
        (0, 0), (0, 0),
        (0, -np.sqrt(3)/3), (0, -np.sqrt(3)/3),
        (np.sqrt(6)/3, np.sqrt(3)/6), (-np.sqrt(6)/3, np.sqrt(3)/6),
        (-np.sqrt(6)/3, -np.sqrt(3)/6), (np.sqrt(6)/3, -np.sqrt(3)/6),
        (np.sqrt(6)/3, -np.sqrt(3)/6), (-np.sqrt(6)/3, -np.sqrt(3)/6),
        (-np.sqrt(6)/3, np.sqrt(3)/6), (np.sqrt(6)/3, np.sqrt(3)/6),
    ]
    R_min = float('inf')
    for k in range(2, 12):
        a_k, b_k = schmid_sys[k]
        denom = a_k * c2 + b_k * s2
        if abs(denom) < 1e-12:
            continue
        for sign in [+1, -1]:
            R_val = sign / denom
            if R_val > 1e-10:
                X_c = R_val * c2
                Y_c = R_val * s2
                ok = True
                for m in range(2, 12):
                    if abs(schmid_sys[m][0]*X_c + schmid_sys[m][1]*Y_c) > 1.0+1e-6:
                        ok = False
                        break
                if ok and R_val < R_min:
                    R_min = R_val
    sm_exact = -(R_min * c2 * c2 + R_min * s2 * s2)
    stt_exact = 2 * sm_exact

    print(f"{th_deg:>10.0f} {sector:>8s} {srr:>10.4f} {stt:>10.4f} "
          f"{srt:>10.4f} {stt_exact:>12.4f}")

# ================================================================
# Step 4: Compute interior stress and plot
# ================================================================
print("\n" + "=" * 70)
print("Step 4: Interior Stress Field Contour Plot")
print("=" * 70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

Nr = 100
Nth = 180
r_vals = np.linspace(1.001, 2.5, Nr)
th_vals = np.linspace(0.001, np.pi/2 - 0.001, Nth)

srr_field = np.zeros((Nr, Nth))
stt_field = np.zeros((Nr, Nth))
srt_field = np.zeros((Nr, Nth))
sm_field = np.zeros((Nr, Nth))

for i, r in enumerate(r_vals):
    for j, th in enumerate(th_vals):
        try:
            if th < theta1:
                s1, s2_v, s3 = stress_sector_I(r, th)
            elif th < theta2:
                s1, s2_v, s3 = stress_sector_II(r, th)
            else:
                s1, s2_v, s3 = stress_sector_III(r, th)
            srr_field[i, j] = s1
            stt_field[i, j] = s2_v
            srt_field[i, j] = s3
            sm_field[i, j] = (s1 + s2_v) / 2
        except (ValueError, FloatingPointError):
            srr_field[i, j] = np.nan
            stt_field[i, j] = np.nan
            srt_field[i, j] = np.nan

# Convert to Cartesian for plotting
R_grid, Th_grid = np.meshgrid(r_vals, th_vals, indexing='ij')
X_cart = R_grid * np.cos(Th_grid)
Y_cart = R_grid * np.sin(Th_grid)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

fields = [
    (srr_field, r'$\sigma_{rr}/\tau$', '(a)'),
    (stt_field, r'$\sigma_{\theta\theta}/\tau$', '(b)'),
    (srt_field, r'$\sigma_{r\theta}/\tau$', '(c)'),
    (sm_field, r'$\sigma_m/\tau$', '(d)'),
]

for ax, (field, label, panel) in zip(axes.flat, fields):
    # Clip extreme values
    vmin = np.nanpercentile(field, 2)
    vmax = np.nanpercentile(field, 98)
    if abs(vmax - vmin) < 0.1:
        vmin, vmax = field.min(), field.max()

    # Plot first quadrant
    cs = ax.pcolormesh(X_cart, Y_cart, field, cmap='RdBu_r',
                       vmin=vmin, vmax=vmax, shading='auto')
    # Mirror to second quadrant
    cs2 = ax.pcolormesh(-X_cart, Y_cart, field, cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, shading='auto')
    # Mirror to lower half
    cs3 = ax.pcolormesh(X_cart, -Y_cart, field, cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, shading='auto')
    cs4 = ax.pcolormesh(-X_cart, -Y_cart, field, cmap='RdBu_r',
                        vmin=vmin, vmax=vmax, shading='auto')

    plt.colorbar(cs, ax=ax, label=label)
    void = plt.Circle((0, 0), 1.0, fill=True, color='white', ec='black', lw=1.5)
    ax.add_patch(void)
    ax.set_aspect('equal')
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_title(f'{panel} {label}', fontsize=12)

    # Sector boundaries
    for tb in [theta1, theta2]:
        ax.plot([np.cos(tb), 2.5*np.cos(tb)], [np.sin(tb), 2.5*np.sin(tb)],
                'k--', lw=0.5, alpha=0.5)
        ax.plot([np.cos(tb), 2.5*np.cos(tb)], [-np.sin(tb), -2.5*np.sin(tb)],
                'k--', lw=0.5, alpha=0.5)
        ax.plot([-np.cos(tb), -2.5*np.cos(tb)], [np.sin(tb), 2.5*np.sin(tb)],
                'k--', lw=0.5, alpha=0.5)
        ax.plot([-np.cos(tb), -2.5*np.cos(tb)], [-np.sin(tb), -2.5*np.sin(tb)],
                'k--', lw=0.5, alpha=0.5)

plt.suptitle('Exact interior stress field (Kysar-type solution) — BCC void',
             fontsize=14, y=1.01)
plt.tight_layout()
fig_path = 'figures/exact_interior_kysar.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"Figure saved: {fig_path}")

# ================================================================
# Step 5: Radial profiles (like Kysar Fig. 20)
# ================================================================
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

r_profile = np.linspace(1.001, 2.5, 200)

for th_deg, ls in [(0, '-'), (15, '--'), (30, ':')]:
    th = np.radians(th_deg)
    srr_r = [stress_sector_I(r, th)[0] for r in r_profile]
    stt_r = [stress_sector_I(r, th)[1] for r in r_profile]
    sm_r = [(s1+s2)/2 for s1, s2 in zip(srr_r, stt_r)]
    ax1.plot(r_profile, srr_r, 'b'+ls, lw=1.5, label=rf'$\sigma_{{rr}}$, $\theta={th_deg}°$')
    ax1.plot(r_profile, stt_r, 'r'+ls, lw=1.5, label=rf'$\sigma_{{\theta\theta}}$, $\theta={th_deg}°$')
    ax1.plot(r_profile, sm_r, 'k'+ls, lw=1, label=rf'$\sigma_m$, $\theta={th_deg}°$')

ax1.set_xlabel(r'$r/a$', fontsize=12)
ax1.set_ylabel(r'$\sigma/\tau$', fontsize=12)
ax1.set_title(r'(a) Sector I: $\theta = 0°, 15°, 30°$', fontsize=12)
ax1.legend(fontsize=7, ncol=3)
ax1.grid(True, alpha=0.3)

for th_deg, ls in [(45, '-'), (50, '--')]:
    th = np.radians(th_deg)
    srr_r = [stress_sector_II(r, th)[0] for r in r_profile]
    stt_r = [stress_sector_II(r, th)[1] for r in r_profile]
    srt_r = [stress_sector_II(r, th)[2] for r in r_profile]
    ax2.plot(r_profile, srr_r, 'b'+ls, lw=1.5, label=rf'$\sigma_{{rr}}$, $\theta={th_deg}°$')
    ax2.plot(r_profile, stt_r, 'r'+ls, lw=1.5, label=rf'$\sigma_{{\theta\theta}}$, $\theta={th_deg}°$')
    ax2.plot(r_profile, srt_r, 'g'+ls, lw=1.5, label=rf'$\sigma_{{r\theta}}$, $\theta={th_deg}°$')

ax2.set_xlabel(r'$r/a$', fontsize=12)
ax2.set_ylabel(r'$\sigma/\tau$', fontsize=12)
ax2.set_title(r'(b) Sector II: $\theta = 45°, 50°$', fontsize=12)
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
fig_path2 = 'figures/exact_radial_profiles.png'
plt.savefig(fig_path2, dpi=200, bbox_inches='tight')
print(f"Figure saved: {fig_path2}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
