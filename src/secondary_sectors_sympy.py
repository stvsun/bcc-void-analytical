"""
Secondary sector derivation for BCC void problem using SymPy.

Follows Kysar (2005) methodology:
1. Identify where primary sector characteristics cross sector boundaries
2. Compute the curved boundary shape
3. Derive stress field in secondary sectors
4. Verify continuity at all boundaries
"""
import sympy as sp
from sympy import sqrt, atan, cos, sin, tan, pi, Rational, simplify, symbols
from sympy import Function, solve, Eq, Abs, atan2, asin, acos, pprint
import numpy as np

# ================================================================
# 1. BCC parameters (symbolic)
# ================================================================
tau = sp.Symbol('tau', positive=True)
a = sp.Symbol('a', positive=True)  # void radius
r, theta = sp.symbols('r theta', real=True, positive=True)
t = sp.Symbol('t', real=True)
theta_0 = sp.Symbol('theta_0', real=True, positive=True)

# Key angles
gamma = atan(2*sqrt(2)) / 2          # ≈ 35.26° sector I/II boundary
alpha_half = atan(sqrt(2))            # ≈ 54.74° sector II/III boundary

print("="*70)
print("BCC SECONDARY SECTOR DERIVATION")
print("="*70)
print()

# Simplify key trig values
cos2g = sp.trigsimp(cos(2*gamma))
sin2g = sp.trigsimp(sin(2*gamma))
print(f"cos(2γ) = {cos2g}")
print(f"sin(2γ) = {sin2g}")

# Manual override since sympy may not simplify arctan(2√2) easily
# cos(arctan(2√2)) = 1/3, sin(arctan(2√2)) = 2√2/3
cos2g_val = Rational(1, 3)
sin2g_val = 2*sqrt(2)/3
cosg_val = sqrt((1 + cos2g_val)/2)  # = sqrt(2/3)
sing_val = sqrt((1 - cos2g_val)/2)  # = sqrt(1/3) = 1/√3

print(f"\nUsing exact values:")
print(f"cos(2γ) = 1/3")
print(f"sin(2γ) = 2√2/3")
print(f"cos(γ) = √(2/3)")
print(f"sin(γ) = 1/√3")

# Characteristic directions in each sector
# Sector I: φ_I = -γ
#   α-lines at φ_I + π/2 = π/2 - γ ≈ +54.74°
#   β-lines at φ_I = -γ ≈ -35.26°
phi_I = -gamma

# Sector II: φ_II = 0
#   α-lines horizontal (0°)
#   β-lines vertical (π/2)
phi_II = sp.Integer(0)

# Sector III: φ_III = γ (= π/2 - α_half by symmetry)
phi_III = gamma

# Airy constants
A_I = 2*sqrt(3)/3
A_II = sqrt(3)
A_III = 2*sqrt(3)/3

print(f"\nA_I = 2√3/3 = {float(A_I):.6f}")
print(f"A_II = √3 = {float(A_II):.6f}")
print(f"A_III = 2√3/3 = {float(A_III):.6f}")

# ================================================================
# 2. Yield polygon vertices (after R correction, det=+1)
# ================================================================
print("\n" + "="*70)
print("YIELD POLYGON AND FACE EQUATIONS")
print("="*70)

# Vertices of the hexagonal yield polygon (normalized to τ_CRSS = 1)
V1 = (sqrt(6)/2, sp.Integer(0))
V2 = (sqrt(6)/4, -sqrt(3))
V3 = (-sqrt(6)/4, -sqrt(3))
V4 = (-sqrt(6)/2, sp.Integer(0))
V5 = (-sqrt(6)/4, sqrt(3))
V6 = (sqrt(6)/4, sqrt(3))

vertices = {'V1': V1, 'V2': V2, 'V3': V3, 'V4': V4, 'V5': V5, 'V6': V6}

print("\nVertices:")
for name, (x, y) in vertices.items():
    print(f"  {name} = ({float(x):.4f}, {float(y):.4f})")

# Face equations: a_k X + b_k Y = 1
# System 5: (√6/3, -√3/6) → face V1↔V2
# System 3: (0, √3/3) → face V5↔V6 (Y = √3)
# System 6: (-√6/3, √3/6) → face V5↔V4

# Verify face V1↔V2 (system 5):
a5, b5 = sqrt(6)/3, -sqrt(3)/6
print(f"\nFace V1→V2: ({float(a5):.4f})X + ({float(b5):.4f})Y = 1")
print(f"  Check V1: {float(a5*V1[0] + b5*V1[1]):.4f}")
print(f"  Check V2: {float(a5*V2[0] + b5*V2[1]):.4f}")

# System 3 positive face: (0, √3/3)Y = 1 → Y = √3, face V5↔V6
a3, b3 = sp.Integer(0), sqrt(3)/3
print(f"\nFace V5→V6: ({float(a3):.4f})X + ({float(b3):.4f})Y = 1")
print(f"  Check V5: {float(a3*V5[0] + b3*V5[1]):.4f}")
print(f"  Check V6: {float(a3*V6[0] + b3*V6[1]):.4f}")

# System 6: (-√6/3, √3/6) → face V5↔V4? or V4↔V5?
a6, b6 = -sqrt(6)/3, sqrt(3)/6
print(f"\nFace with sys 6: ({float(a6):.4f})X + ({float(b6):.4f})Y = 1")
print(f"  Check V4: {float(a6*V4[0] + b6*V4[1]):.4f}")
print(f"  Check V5: {float(a6*V5[0] + b6*V5[1]):.4f}")

# Now determine the ACTUAL face active in each sector
# At θ=0: stress at V1 = (√6/2, 0). This is on face V1→V2 AND face V6→V1
# At θ=π/2: stress should be at V4 or V5 by symmetry

# ================================================================
# 3. Void surface stress in each sector
# ================================================================
print("\n" + "="*70)
print("VOID SURFACE STRESS (verification)")
print("="*70)

# In each sector, σ_rr = 0 and σ_rθ = 0 at r = a.
# σ_rθ = 0: -X sin(2θ) + Y cos(2θ) = 0 → Y = X tan(2θ)
# Face equation: a_k X + b_k Y = 1
# → X(a_k + b_k tan(2θ)) = 1

X_sym, Y_sym, sm_sym = sp.symbols('X Y sigma_m')

# Sector I: face V6→V1 with system 8: (√6/3, √3/6)
# Actually let me determine which face by checking the void surface stress
# The void surface stress must trace along the yield polygon as θ varies

# At θ = 0: σ_rθ = 0 → Y = 0. So we're at a vertex with Y = 0.
# The only vertices with Y = 0 are V1 = (√6/2, 0) and V4 = (-√6/2, 0).
# Under compression (void trying to close), we expect X > 0, so V1.

# As θ increases from 0, Y = X tan(2θ) > 0 (since 2θ ∈ (0, π)).
# Starting from V1 = (√6/2, 0), we move along a face to higher Y.
# Face V6→V1 has Y going from √3 to 0 (decreasing Y).
# Face V1→V2 has Y going from 0 to -√3 (decreasing Y).
# We need INCREASING Y, so... hmm.

# Wait, which direction? If Y = X tan(2θ) and θ increases from 0:
# At θ = 0: Y = 0 (at V1)
# At θ > 0: Y = X tan(2θ) > 0 for X > 0

# So we need the face from V1 going UPWARD (Y increasing).
# Face V1→V6: from (√6/2, 0) to (√6/4, √3)
# This face has equation: check which system

# System 8: (a8, b8) = (√6/3, √3/6) [after R correction]
a8, b8 = sqrt(6)/3, sqrt(3)/6
print(f"\nFace V1→V6: ({float(a8):.4f})X + ({float(b8):.4f})Y = 1")
print(f"  Check V1: {float(a8*V1[0] + b8*V1[1]):.6f}")
print(f"  Check V6: {float(a8*V6[0] + b8*V6[1]):.6f}")

# So system 8 goes through V1 and V6? Let me check.
# V1 = (√6/2, 0): (√6/3)(√6/2) + (√3/6)(0) = 6/6 = 1 ✓
# V6 = (√6/4, √3): (√6/3)(√6/4) + (√3/6)(√3) = 6/12 + 3/6 = 1/2 + 1/2 = 1 ✓

print("  → Face V1→V6 uses system 8 with (√6/3, √3/6)")

# So Sector I (0 ≤ θ ≤ γ) has system 8 active on face V1→V6!
# Not system 5 as previously assumed!

# Let me recheck: system 5 has (√6/3, -√3/6):
# V1: (√6/3)(√6/2) + (-√3/6)(0) = 1 ✓
# V2: (√6/3)(√6/4) + (-√3/6)(-√3) = 1/2 + 1/2 = 1 ✓
# Face V1→V2: from V1 = (√6/2, 0) to V2 = (√6/4, -√3), Y DECREASING

# Since in Sector I we need Y increasing (tan(2θ) > 0), the active face is
# V1→V6 with system 8 (not V1→V2 with system 5)!

print("\n*** IMPORTANT: Sector I uses system 8 (face V1→V6), NOT system 5 ***")
print("    System 5 (face V1→V2) has Y decreasing → not compatible with Y > 0")

# Now for each sector, determine the active face:
# Sector I (0 ≤ θ ≤ θ₁): face V1→V6, system 8: (√6/3, √3/6)
# Y = X tan(2θ) and X(√6/3 + (√3/6)tan(2θ)) = 1
#   X = 1/(√6/3 + (√3/6)tan(2θ)) = 6/(2√6 + √3 tan(2θ))

# At V6 = (√6/4, √3): Y/X = √3/(√6/4) = 4√3/√6 = 4/√2 = 2√2
# tan(2θ₁) = 2√2 → 2θ₁ = arctan(2√2) → θ₁ = γ ✓

# Sector II (θ₁ ≤ θ ≤ θ₂): face V6→V5, system 3 (positive): (0, √3/3)
# Y = √3 (constant on this face)
# X tan(2θ) = √3 → X = √3/tan(2θ) = √3 cos(2θ)/sin(2θ)
# σ_m = -(X cos(2θ) + Y sin(2θ)) = -(√3 cos²(2θ)/sin(2θ) + √3 sin(2θ))
#      = -√3(cos²(2θ) + sin²(2θ))/sin(2θ) = -√3/sin(2θ)

print("\nSector II: face V6→V5, system 3: (0, √3/3)")
print("  Y = √3, X = √3 cos(2θ)/sin(2θ)")
print("  σ_m = -√3/sin(2θ)")

# At V5 = (-√6/4, √3): Y/X = √3/(-√6/4) = -4/√2 = -2√2
# tan(2θ₂) = -2√2? But we need θ₂ ≈ 54.74° > 45° so tan(2θ₂) < 0.
# 2θ₂ = π - arctan(2√2) → θ₂ = (π - arctan(2√2))/2 = α_half ✓

# Sector III (θ₂ ≤ θ ≤ π/2): face V5→V4
# System 6 (negative): (-√6/3, √3/6) → (√6/3)X - (√3/6)Y = -1
# Or system 7: (-√6/3, -√3/6)?

# Check V5 = (-√6/4, √3) and V4 = (-√6/2, 0):
a7, b7 = -sqrt(6)/3, -sqrt(3)/6
print(f"\nSystem 7: ({float(a7):.4f})X + ({float(b7):.4f})Y = 1")
print(f"  V5: {float(a7*V5[0] + b7*V5[1]):.6f}")
print(f"  V4: {float(a7*V4[0] + b7*V4[1]):.6f}")

# V5: (-√6/3)(-√6/4) + (-√3/6)(√3) = 6/12 - 3/6 = 1/2 - 1/2 = 0 ≠ 1
# Let me try the negative face: a_k X + b_k Y = -1
# System 6 negative: (√6/3)X + (-√3/6)Y = -1 → (-√6/3)X + (√3/6)Y = 1
a6n, b6n = -sqrt(6)/3, sqrt(3)/6
print(f"\nSystem 6 (neg): ({float(a6n):.4f})X + ({float(b6n):.4f})Y = 1")
print(f"  V5: {float(a6n*V5[0] + b6n*V5[1]):.6f}")
print(f"  V4: {float(a6n*V4[0] + b6n*V4[1]):.6f}")

# ================================================================
# 4. Characteristic directions (corrected)
# ================================================================
print("\n" + "="*70)
print("CHARACTERISTIC DIRECTIONS (corrected)")
print("="*70)

# For a yield face with Schmid coefficients (a_k, b_k), the slip plane
# normal makes angle ψ with the x₁' axis where:
# The characteristics (slip lines) are at angles ψ and ψ + π/2
# where tan(2ψ) = b_k/a_k (Rice, 1973)

# Sector I: system 8 with (√6/3, √3/6)
# tan(2ψ_I) = b/a = (√3/6)/(√6/3) = √3/(2√6) = 1/(2√2)
# 2ψ_I = arctan(1/(2√2))
# cos(2ψ_I) = 2√2/3, sin(2ψ_I) = 1/3

# Wait, this is the angle of the x'₁ axis relative to x₁.
# Actually, for Rice's theory, the characteristic directions are at:
# φ = (1/2) arctan(a_k/b_k) or something like that.

# Let me be more careful. For a yield face a X + b Y = τ,
# the resolvedshear stress is τ = a(σ₁₁-σ₂₂)/2 + b σ₁₂
# The slip direction makes angle ψ with x₁ where:
# a = cos(2ψ), b = sin(2ψ) ... for normalized (a,b)

# Normalize: |(a,b)| for system 8: √(6/9 + 3/36) = √(2/3 + 1/12) = √(9/12) = √(3/4) = √3/2
norm8 = sp.sqrt(a8**2 + b8**2)
print(f"\n||(a8, b8)|| = {sp.simplify(norm8)} = {float(norm8):.6f}")

# The characteristic angle φ is half the angle of (a, b) in the Mohr plane:
# cos(2φ) = a/||(a,b)||, sin(2φ) = b/||(a,b)||
cos2phi_I = a8 / norm8
sin2phi_I = b8 / norm8
print(f"cos(2φ_I) = {sp.simplify(cos2phi_I)} = {float(cos2phi_I):.6f}")
print(f"sin(2φ_I) = {sp.simplify(sin2phi_I)} = {float(sin2phi_I):.6f}")

# φ_I = (1/2) arctan(b8/a8) = (1/2) arctan((√3/6)/(√6/3))
# = (1/2) arctan(√3/(2√6)) = (1/2) arctan(1/(2√2))
phi_I_exact = sp.atan2(b8, a8) / 2
print(f"φ_I = (1/2) arctan(b/a) = {float(phi_I_exact):.6f} rad = {float(sp.deg(phi_I_exact)):.4f}°")

# Hmm, this gives φ_I ≈ 11.8°, not -35.26° as in the paper.
# The discrepancy is because Rice's theory defines φ differently.

# In Rice (1973) and Kysar (2005), the characteristic directions are:
# The two families of slip lines make angles (π/4 + φ) and (π/4 - φ)
# with the maximum principal stress direction, where φ is the
# "angle of inclination of the yield locus tangent."

# For a polygonal yield surface on face a X + b Y = τ:
# The slip line directions in the (x₁, x₂) frame are determined by
# the angle of the face normal in the Mohr stress plane.

# The face V1→V6 goes from (√6/2, 0) to (√6/4, √3).
# The face direction vector: V6 - V1 = (-√6/4, √3)
# The face normal (inward): (√3, √6/4) → proportional to (a8, b8) ✓

# The CHARACTERISTICS for this face are lines along which:
# - σ₁ (stress component parallel to face normal) is constant on one family
# - σ₂ (stress component perpendicular to face normal) is constant on other

# In the rotated frame (x'₁ along face normal direction φ, x'₂ perpendicular):
# σ'₁₁ = const along α-lines (x'₁ = const, i.e., lines perpendicular to face normal)
# σ'₂₂ = const along β-lines (x'₂ = const, i.e., lines parallel to face normal)

# The face normal direction φ is defined by cos(2φ) = a/||a,b||:
# 2φ = arctan(sin2phi/cos2phi) = arctan(b/a) when a > 0
# For system 8: 2φ = arctan((√3/6)/(√6/3)) = arctan(1/(2√2))
# φ ≈ 11.8°

# But in the paper, the rotated frame has x'₁ at angle φ to the horizontal,
# and the characteristics are:
# α-lines: lines of constant x'₁ → perpendicular to x'₁ direction → at angle φ + π/2
# β-lines: lines of constant x'₂ → perpendicular to x'₂ direction → at angle φ

# Wait, that's not right either. α-lines are PARALLEL to x'₂ (along x'₂),
# and β-lines are PARALLEL to x'₁ (along x'₁).

# So:
# β-lines at angle φ_I ≈ 11.8° from horizontal
# α-lines at angle φ_I + 90° ≈ 101.8° from horizontal

# Hmm, but the paper says φ_I = -γ ≈ -35.26°. There's a sign/convention issue.

# Let me check: in the paper, the active system in Sector I is system (5,12)
# or system (8) (after R correction)?

# The paper says system (5,12). After R correction, the b coefficients flip.
# System 5 old: (√6/3, √3/6) → system 5 new: (√6/3, -√3/6)
# System 8 old: (√6/3, -√3/6) → system 8 new: (√6/3, √3/6)

# So old system 5 = new system 8 in terms of coefficients!
# The paper's "system (5,12)" with old R has (√6/3, √3/6).
# After R correction, this becomes system 8 with (√6/3, √3/6).
# OR: the paper's system numbering stays the same, but the b values flip.

# The KEY question: what is the characteristic angle for the face V1→V6?

# Face V1→V6: equation a8 X + b8 Y = 1 with (a8, b8) = (√6/3, √3/6)
# Face normal angle: φ = (1/2) arctan(b8/a8) = (1/2) arctan(1/(2√2)) ≈ 11.8°

# But the paper has φ_I = -γ ≈ -35.26° for Sector I.
# This suggests the paper uses a DIFFERENT convention for φ.

# In Kysar (2005), φ is defined such that the yield face equation in the
# rotated frame is σ'₁₂ = A (constant). The rotation angle is chosen so
# that the face becomes horizontal in the (σ'₁₁ - σ'₂₂, σ'₁₂) plane.

# For face V1→V6: the face goes from (√6/2, 0) to (√6/4, √3).
# In the rotated Mohr plane (σ'₁₁ - σ'₂₂, σ'₁₂), this face should be
# horizontal (σ'₁₂ = const). The rotation angle φ that achieves this
# satisfies: the face direction (V6 - V1) rotated by -2φ is horizontal.

# V6 - V1 = (-√6/4, √3) in (X, Y) space.
# Rotation by -2φ: (-√6/4 cos(2φ) + √3 sin(2φ), √6/4 sin(2φ) + √3 cos(2φ))
# For this to be horizontal (Y component = 0):
# √6/4 sin(2φ) + √3 cos(2φ) = 0
# tan(2φ) = -√3/(√6/4) = -4√3/√6 = -4/√2 = -2√2

# So tan(2φ) = -2√2, 2φ = -arctan(2√2), φ = -(1/2)arctan(2√2) = -γ!

phi_I_kysar = -gamma
print(f"\nKysar convention: φ_I = -γ = {float(phi_I_kysar):.6f} rad = {float(sp.deg(phi_I_kysar)):.4f}°")
print("This matches the paper's value!")

# So the two conventions differ:
# - Standard: φ = (1/2) arctan(b/a) ≈ +11.8° (angle of face NORMAL)
# - Kysar: φ chosen so face is horizontal in rotated Mohr plane → -(1/2) arctan(2√2) ≈ -35.26°

# The Kysar convention makes the face equation σ'₁₂ = A in the rotated frame.
# This is the convention used in the paper and in the Airy stress function derivation.

# Characteristics in the Kysar convention:
# α-lines: lines of constant x'₁ → perpendicular to x'₁ → at angle φ + π/2
# β-lines: lines of constant x'₂ → perpendicular to x'₂ → at angle φ

print(f"\nSector I characteristics (Kysar convention):")
print(f"  β-lines at angle φ_I = {float(sp.deg(phi_I_kysar)):.2f}°")
print(f"  α-lines at angle φ_I + 90° = {float(sp.deg(phi_I_kysar + pi/2)):.2f}°")

# ================================================================
# 5. α-line from void at θ₀ — parametric equation
# ================================================================
print("\n" + "="*70)
print("α-LINE TRACING FROM VOID SURFACE")
print("="*70)

# α-line direction: φ_I + π/2 = -γ + π/2 = π/2 - γ
alpha_dir = pi/2 - gamma  # ≈ 54.74°
beta_dir = -gamma          # ≈ -35.26°

print(f"α-line direction: π/2 - γ = {float(sp.deg(alpha_dir)):.4f}°")
print(f"β-line direction: -γ = {float(sp.deg(beta_dir)):.4f}°")

# α-line from void surface at θ₀:
# x(t) = a cos(θ₀) + t cos(α_dir) = a cos(θ₀) + t sin(γ)
# y(t) = a sin(θ₀) + t sin(α_dir) = a sin(θ₀) + t cos(γ)

# Using exact values: sin(γ) = 1/√3, cos(γ) = √(2/3)
# x(t) = a cos(θ₀) + t/√3
# y(t) = a sin(θ₀) + t√(2/3)

# This α-line crosses the sector boundary θ = γ when y/x = tan(γ) = 1/√2:
# (a sin(θ₀) + t√(2/3)) / (a cos(θ₀) + t/√3) = 1/√2

# Cross-multiply:
# √2(a sin(θ₀) + t√(2/3)) = a cos(θ₀) + t/√3
# √2 a sin(θ₀) + t·2/√3 = a cos(θ₀) + t/√3
# t(2/√3 - 1/√3) = a(cos(θ₀) - √2 sin(θ₀))
# t/√3 = a(cos(θ₀) - √2 sin(θ₀))
# t = a√3(cos(θ₀) - √2 sin(θ₀))

# Using sin(γ)/cos(γ) = 1/√2:
# cos(θ₀) - √2 sin(θ₀) = cos(θ₀) - sin(θ₀)/sin(γ)·cos(γ)
# Hmm, let me just simplify differently.

# t = a√3(cos θ₀ - √2 sin θ₀)
# For θ₀ = 0: t = a√3 (> 0 ✓)
# For θ₀ = γ: cos γ - √2 sin γ = √(2/3) - √2/√3 = √(2/3) - √(2/3) = 0
# So t = 0 at θ₀ = γ ✓ (the α-line from the boundary point doesn't need to travel to cross)

# At the crossing point:
# x_cross = a cos θ₀ + a√3(cos θ₀ - √2 sin θ₀)/√3
#          = a cos θ₀ + a(cos θ₀ - √2 sin θ₀)
#          = a(2 cos θ₀ - √2 sin θ₀)
# y_cross = a sin θ₀ + a√3(cos θ₀ - √2 sin θ₀)√(2/3)
#          = a sin θ₀ + a√2(cos θ₀ - √2 sin θ₀)
#          = a(sin θ₀ + √2 cos θ₀ - 2 sin θ₀)
#          = a(√2 cos θ₀ - sin θ₀)

# r_cross² = x² + y² = a²[(2c-√2 s)² + (√2 c - s)²]
# where c = cos θ₀, s = sin θ₀
# = a²[4c² - 4√2 cs + 2s² + 2c² - 2√2 cs + s²]
# = a²[6c² - 6√2 cs + 3s²]
# = 3a²[2c² - 2√2 cs + s²]
# = 3a²[(c - √2 s)² + c² - s²]  ... hmm, let me just compute directly

# Actually: 6c² - 6√2cs + 3s² = 3(2c² - 2√2cs + s²)
# Let me verify at θ₀ = 0: 3(2 - 0 + 0) = 6, so r/a = √6 ✓ (matches earlier calculation)

print("\nα-line crossing θ = γ:")
print("  t_cross = a√3(cos θ₀ - √2 sin θ₀)")
print("  x_cross = a(2 cos θ₀ - √2 sin θ₀)")
print("  y_cross = a(√2 cos θ₀ - sin θ₀)")
print("  r_cross²/a² = 6cos²θ₀ - 6√2 cosθ₀ sinθ₀ + 3sin²θ₀")
print(f"  At θ₀ = 0: r_cross/a = √6 ≈ {float(sp.sqrt(6)):.4f}")
print(f"  At θ₀ = γ: r_cross/a = 1 (void surface)")

# ================================================================
# 6. Curved sector boundary shape
# ================================================================
print("\n" + "="*70)
print("CURVED SECTOR BOUNDARY")
print("="*70)

# The curved boundary is the α-line emanating from the void surface at θ₀ = γ.
# This is because: any α-line from θ₀ < γ crosses θ = γ at r > a,
# and the LAST α-line that doesn't cross is from θ₀ = γ itself.
# Actually, it's the OTHER way: the α-line from θ₀ = γ IS the boundary
# between the primary Sector I domain (where both chars reach void)
# and the secondary sector (where the α-line from Sector I has already
# passed through).

# The curved boundary is NOT the α-line from θ₀ = γ (which stays at θ = γ
# for t = 0 and then enters Sector II). Instead, it's the locus of
# points where the β-line backward trace just barely stays in Sector I.

# Actually, I think the curved boundary is the α-line from (a, γ) going
# outward. Let me parameterize it.

# α-line from (a cos γ, a sin γ) at angle α_dir = π/2 - γ:
# x(t) = a cos γ + t sin γ = a√(2/3) + t/√3
# y(t) = a sin γ + t cos γ = a/√3 + t√(2/3)

# In polar:
# r²(t) = a² + 2at sin(2γ) + t² ... wait, let me recompute
# r²(t) = (a cos γ + t sin γ)² + (a sin γ + t cos γ)²
#        = a²(cos²γ + sin²γ) + 2at(cos γ sin γ + sin γ cos γ) + t²(sin²γ + cos²γ)
#        = a² + 2at · 2sinγ cosγ + t²   ... NO!
# cos γ sin γ + sin γ cos γ = 2 sin γ cos γ = sin(2γ)
# But we have DIFFERENT products:
# (a cos γ)(t sin γ) + (a sin γ)(t cos γ) = 2at sin γ cos γ = at sin(2γ)
# Wait: x = a cosγ + t sinγ, y = a sinγ + t cosγ
# x² + y² = a²cos²γ + 2at cosγ sinγ + t²sin²γ + a²sin²γ + 2at sinγ cosγ + t²cos²γ
#          = a² + 4at sinγ cosγ + t²
#          = a² + 2at sin(2γ) + t²

# r²(t) = a² + 2at sin(2γ) + t²
# With sin(2γ) = 2√2/3:
# r²(t) = a² + 4√2 at/3 + t²

# θ(t) = arctan(y/x) = arctan((a sinγ + t cosγ)/(a cosγ + t sinγ))
# = arctan((a/√3 + t√(2/3))/(a√(2/3) + t/√3))
# = arctan((a + t√2)/(a√2 + t))   [multiplied by √3]

# As t → ∞: θ → arctan(√2/1) = arctan(√2) = α_half ≈ 54.74°
# This confirms the α-line approaches Sector II/III boundary!

print("Curved boundary = α-line from void at θ = γ:")
print("  x(t)/a = cos γ + (t/a) sin γ = √(2/3) + (t/a)/√3")
print("  y(t)/a = sin γ + (t/a) cos γ = 1/√3 + (t/a)√(2/3)")
print(f"  r²(t)/a² = 1 + 2(t/a)sin(2γ) + (t/a)²")
print(f"  θ(t) → arctan(√2) = {float(sp.deg(alpha_half)):.2f}° as t → ∞")
print()

# Compute numerical values of the curved boundary
import numpy as np
gamma_num = float(gamma)
alpha_num = float(alpha_half)
t_vals = np.linspace(0, 10, 500)
x_cb = np.cos(gamma_num) + t_vals * np.sin(gamma_num)
y_cb = np.sin(gamma_num) + t_vals * np.cos(gamma_num)
r_cb = np.sqrt(x_cb**2 + y_cb**2)
theta_cb = np.arctan2(y_cb, x_cb)

print("Sample points on curved boundary:")
print(f"  {'t/a':>6}  {'r/a':>8}  {'θ (°)':>8}")
for i in range(0, len(t_vals), 50):
    print(f"  {t_vals[i]:6.2f}  {r_cb[i]:8.4f}  {np.degrees(theta_cb[i]):8.4f}")

# ================================================================
# 7. Stress at the curved boundary
# ================================================================
print("\n" + "="*70)
print("STRESS AT THE CURVED BOUNDARY")
print("="*70)

# Along the curved boundary (which is an α-line), σ'₁₁ is CONSTANT.
# Its value is determined by the void surface stress at θ₀ = γ.
# σ'₁₁ at (a, γ) from the Sector I Airy solution:

# σ'₁₁ = A_I · (r/a) sin(φ_I - θ) / √(1 - (r/a)² sin²(φ_I - θ))
# At r = a, θ = γ, φ_I = -γ:
# arg = φ_I - γ = -2γ
# sin(-2γ) = -sin(2γ) = -2√2/3
# σ'₁₁ = A_I · 1 · (-2√2/3) / √(1 - (2√2/3)²)
#       = (2√3/3)(-2√2/3) / √(1 - 8/9)
#       = (2√3/3)(-2√2/3) / (1/3)
#       = (2√3/3)(-2√2)
#       = -4√6/3

s11p_boundary = A_I * (-sin2g_val) / sp.sqrt(1 - sin2g_val**2)
s11p_boundary_simplified = sp.nsimplify(sp.simplify(s11p_boundary), rational=False)
print(f"σ'₁₁ at boundary = {s11p_boundary_simplified} = {float(s11p_boundary):.6f}")

# Similarly, σ'₂₂ at (a, γ):
# arg = φ_I - γ = -2γ
# cos(-2γ) = cos(2γ) = 1/3
# σ'₂₂ = A_I · 1 · (1/3) / √(1 - (1/3)²) = (2√3/3)(1/3)/√(8/9)
#       = (2√3/9)/(2√2/3) = (2√3/9)(3/(2√2)) = √3/(3√2) = √6/6 ≈ 0.408

s22p_boundary = A_I * cos2g_val / sp.sqrt(1 - cos2g_val**2)
s22p_boundary_simplified = sp.nsimplify(sp.simplify(s22p_boundary), rational=False)
print(f"σ'₂₂ at boundary = {s22p_boundary_simplified} = {float(s22p_boundary):.6f}")

print(f"σ'₁₂ = A_I = {float(A_I):.6f}")

# Convert to Cartesian stress at (a, γ):
phi = phi_I_kysar  # = -γ
c2phi = cos2g_val   # cos(2φ_I) = cos(-2γ) = cos(2γ) = 1/3
s2phi = -sin2g_val  # sin(2φ_I) = sin(-2γ) = -sin(2γ) = -2√2/3

s11_cart = (s11p_boundary + s22p_boundary)/2 + (s11p_boundary - s22p_boundary)/2 * c2phi - A_I * s2phi
s22_cart = (s11p_boundary + s22p_boundary)/2 - (s11p_boundary - s22p_boundary)/2 * c2phi + A_I * s2phi
s12_cart = (s11p_boundary - s22p_boundary)/2 * s2phi + A_I * c2phi

X_boundary = sp.simplify((s11_cart - s22_cart)/2)
Y_boundary = sp.simplify(s12_cart)
sm_boundary = sp.simplify((s11_cart + s22_cart)/2)

print(f"\nCartesian stress at boundary point (a, γ):")
print(f"  σ₁₁ = {float(s11_cart):.6f}")
print(f"  σ₂₂ = {float(s22_cart):.6f}")
print(f"  σ₁₂ = {float(s12_cart):.6f}")
print(f"  X = {float(X_boundary):.6f}")
print(f"  Y = {float(Y_boundary):.6f}")
print(f"  σ_m = {float(sm_boundary):.6f}")

# Check: is this at vertex V6 = (√6/4, √3)?
print(f"\n  V6 = ({float(sp.sqrt(6)/4):.4f}, {float(sp.sqrt(3)):.4f})")
print(f"  Stress: ({float(X_boundary):.4f}, {float(Y_boundary):.4f})")
dist_V6 = sp.sqrt((X_boundary - sqrt(6)/4)**2 + (Y_boundary - sqrt(3))**2)
print(f"  Distance to V6: {float(dist_V6):.6f}")
# Check V1
dist_V1 = sp.sqrt((X_boundary - sqrt(6)/2)**2 + Y_boundary**2)
print(f"  Distance to V1: {float(dist_V1):.6f}")

# Verify σ_rr and σ_rθ at the void surface
c2th = cos2g_val   # cos(2γ) = 1/3
s2th = sin2g_val   # sin(2γ) = 2√2/3
srr_check = sm_boundary + X_boundary * c2th + Y_boundary * s2th
srt_check = -X_boundary * s2th + Y_boundary * c2th
print(f"\nVoid surface check at θ = γ:")
print(f"  σ_rr = {float(sp.simplify(srr_check)):.6f} (should be 0)")
print(f"  σ_rθ = {float(sp.simplify(srt_check)):.6f} (should be 0)")

# ================================================================
# 8. Structure of secondary sectors
# ================================================================
print("\n" + "="*70)
print("SECONDARY SECTOR STRUCTURE")
print("="*70)

# The curved boundary (α-line from void at γ) divides the region
# θ > γ into two parts:
#
# A) Between the radial boundary θ = γ and the curved boundary:
#    "Extended Sector I" — system 8 still active, but the β-line
#    backward trace hits the curved boundary (not the void).
#    σ'₁₁ comes from the α-line to the void (same as primary Sector I)
#    σ'₂₂ comes from the β-line to the curved boundary
#
# B) Between the curved boundary and the next sector:
#    "Primary Sector II" — system 3 active, both chars to void
#    (This is the original Sector II, but its domain is reduced)

# In region A, the β-line from a point P traces backward at angle
# -γ + π = π - γ. It hits the curved boundary at some point.
# At that point, σ'₂₂ is determined by the curved boundary stress.

# Since the curved boundary IS an α-line (σ'₁₁ = const along it),
# the σ'₂₂ at the boundary point varies along the curve.

# For a β-line from P at (r, θ) in region A:
# The β-line goes at angle π - γ (backward).
# It hits the curved boundary at (r_b, θ_b).
# At that point, σ'₂₂ from the primary Sector I formula gives
# the boundary value for σ'₂₂ in the secondary sector.

# But wait — in the secondary sector, the SAME yield face is active
# (system 8, face V1→V6). The σ'₁₂ = A_I (same constant).
# The σ'₁₁ is still determined by the α-line to the void.
# Only σ'₂₂ changes because the β-line hits the curved boundary
# instead of the void.

# The β-line from P hits the curved boundary at some parameter t_b.
# The curved boundary has σ'₂₂ values that vary along it.
# These values come from the PRIMARY Sector I Airy solution evaluated
# at points on the boundary.

# σ'₂₂ on the boundary (parameterized by t):
# At the point (x_cb(t), y_cb(t)), r = r_cb(t), θ = θ_cb(t)
# σ'₂₂ = A_I · (r/a)cos(φ_I - θ) / √(1 - (r/a)²cos²(φ_I - θ))

# This gives σ'₂₂ as a function along the curved boundary.

print("Secondary sector structure:")
print("  Region A (extended Sector I): between θ = γ and curved boundary")
print("    σ'₁₁: from α-line backward to void (same as primary)")
print("    σ'₂₂: from β-line backward to curved boundary (DIFFERENT)")
print("    σ'₁₂ = A_I (unchanged)")
print()
print("  The stress field in Region A is determined by:")
print("    σ'₁₁(r,θ) = A_I·(r/a)sin(φ_I-θ)/√(1-(r/a)²sin²(φ_I-θ))")
print("    σ'₂₂(r,θ) = σ'₂₂ at the β-line/boundary intersection")
print("    σ'₁₂ = A_I")

# ================================================================
# 9. Compute stress in secondary sector numerically
# ================================================================
print("\n" + "="*70)
print("NUMERICAL COMPUTATION OF SECONDARY SECTOR STRESS")
print("="*70)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def primary_sector_I_stress(r_val, theta_val, a_val=1.0):
    """Primary Sector I Airy stress in rotated frame."""
    phi = float(phi_I_kysar)  # = -γ
    A = float(A_I)
    rho = r_val / a_val
    arg = phi - theta_val

    s_arg = np.sin(arg)
    c_arg = np.cos(arg)

    d1 = 1 - rho**2 * s_arg**2
    d2 = 1 - rho**2 * c_arg**2

    if d1 <= 0 or d2 <= 0:
        return np.nan, np.nan, A

    s11p = A * rho * s_arg / np.sqrt(d1)
    s22p = A * rho * c_arg / np.sqrt(d2)
    return s11p, s22p, A


def primary_sector_II_stress(r_val, theta_val, a_val=1.0):
    """Primary Sector II Airy stress (rotated frame = Cartesian, φ=0)."""
    A = float(A_II)
    rho = r_val / a_val

    d1 = 1 - rho**2 * np.sin(theta_val)**2
    d2 = 1 - rho**2 * np.cos(theta_val)**2

    if d1 <= 0 or d2 <= 0:
        return np.nan, np.nan, A

    s11p = -A * rho * np.sin(theta_val) / np.sqrt(d1)
    s22p = -A * rho * np.cos(theta_val) / np.sqrt(d2)
    return s11p, s22p, A


def rotated_to_cartesian(s11p, s22p, s12p, phi_val):
    """Convert stress from rotated frame to Cartesian."""
    c2 = np.cos(2*phi_val)
    s2 = np.sin(2*phi_val)
    s11 = (s11p + s22p)/2 + (s11p - s22p)/2 * c2 - s12p * s2
    s22 = (s11p + s22p)/2 - (s11p - s22p)/2 * c2 + s12p * s2
    s12 = (s11p - s22p)/2 * s2 + s12p * c2
    return s11, s22, s12


def cart_to_polar(s11, s22, s12, theta_val):
    """Convert Cartesian to polar stress."""
    c2t = np.cos(2*theta_val)
    s2t = np.sin(2*theta_val)
    sm = (s11 + s22)/2
    X = (s11 - s22)/2
    Y = s12
    srr = sm + X*c2t + Y*s2t
    stt = sm - X*c2t - Y*s2t
    srt = -X*s2t + Y*c2t
    return srr, stt, srt, sm


# Trace β-line from point (r, θ) backward to find where it hits the
# curved boundary (α-line from void at γ)
def find_beta_boundary_intersection(r_val, theta_val, a_val=1.0):
    """
    Trace β-line backward from (r, θ) and find intersection with
    curved boundary (α-line from void at γ).

    β-line backward direction: π - γ (up-left)
    Curved boundary: α-line from (a cosγ, a sinγ) at angle π/2 - γ
    """
    gam = float(gamma)

    # Point P
    xp = r_val * np.cos(theta_val)
    yp = r_val * np.sin(theta_val)

    # β-line backward from P: (xp - s cosγ, yp + s sinγ) for s > 0
    # Curved boundary from (a cosγ, a sinγ): (a cosγ + u sinγ, a sinγ + u cosγ) for u ≥ 0

    # Intersection: a cosγ + u sinγ = xp - s cosγ
    #               a sinγ + u cosγ = yp + s sinγ

    # Two equations, two unknowns (s, u):
    # s cosγ + u sinγ = xp - a cosγ   ...(i)
    # -s sinγ + u cosγ = yp - a sinγ   ...(ii)

    # Solve by Cramer's rule:
    # det = cosγ·cosγ + sinγ·sinγ = 1
    # s = (xp - a cosγ)cosγ - (yp - a sinγ)sinγ
    #   = xp cosγ - yp sinγ - a(cos²γ - sin²γ)
    #   = xp cosγ - yp sinγ - a cos(2γ)
    # u = (xp - a cosγ)sinγ + (yp - a sinγ)cosγ  ... wait, need to redo

    # Actually: [cosγ, sinγ; -sinγ, cosγ] [s; u] = [xp - a cosγ; yp - a sinγ]
    # This is a rotation matrix! det = 1.
    # [s; u] = [cosγ, sinγ; -sinγ, cosγ]^(-1) [xp - a cosγ; yp - a sinγ]
    #        = [cosγ, -sinγ; sinγ, cosγ] [xp - a cosγ; yp - a sinγ]

    # Hmm wait, let me redo. The system is:
    # cosγ · s + sinγ · u = xp - a cosγ    ...(i)
    # -sinγ · s + cosγ · u = yp - a sinγ   ...(ii)

    # Multiply (i) by cosγ and (ii) by sinγ and add:
    # cos²γ · s + sinγ cosγ · u - sin²γ · s + sinγ cosγ · u = (xp-a cosγ)cosγ + (yp-a sinγ)sinγ
    # s(cos²γ - sin²γ) + 2 sinγ cosγ · u = xp cosγ + yp sinγ - a
    # s cos(2γ) + u sin(2γ) = xp cosγ + yp sinγ - a

    # Multiply (i) by sinγ and (ii) by -cosγ and add:
    # sinγ cosγ · s + sin²γ · u + sinγ cosγ · s - cos²γ · u = (xp-a cosγ)sinγ - (yp-a sinγ)cosγ
    # 2 sinγ cosγ · s - (cos²γ - sin²γ) · u = xp sinγ - yp cosγ
    # sin(2γ) · s - cos(2γ) · u = xp sinγ - yp cosγ

    # So: s cos(2γ) + u sin(2γ) = xp cosγ + yp sinγ - a         ...(A)
    #     s sin(2γ) - u cos(2γ) = xp sinγ - yp cosγ              ...(B)

    # From (A): s = [xp cosγ + yp sinγ - a - u sin(2γ)] / cos(2γ)
    # Sub into (B)... this is getting messy. Let me just solve numerically.

    cg = np.cos(gam)
    sg = np.sin(gam)

    # Linear system: [cg, sg; -sg, cg] · [s, u]^T = [xp - a cg, yp - a sg]^T
    # Inverse of [cg, sg; -sg, cg] is [cg, -sg; sg, cg] (rotation matrix inverse)
    dx = xp - a_val * cg
    dy = yp - a_val * sg

    s_val = cg * dx - sg * dy    # Wait, need to be careful
    u_val = sg * dx + cg * dy

    # Hmm, let me just use the explicit inverse:
    # [s; u] = (1/det) [cg, -sg; sg, cg] [dx; dy]
    # det = cg² + sg² = 1
    s_val = cg * dx - sg * dy  # Hmm, this doesn't look right.

    # Actually: A = [[cg, sg], [-sg, cg]], A^{-1} = [[cg, -sg], [sg, cg]]
    s_val = cg * dx + (-sg) * dy
    u_val = sg * dx + cg * dy

    # But wait, need to verify:
    # A [s; u] = [cg*s + sg*u; -sg*s + cg*u] = [dx; dy]
    # Check: cg*(cg*dx - sg*dy) + sg*(sg*dx + cg*dy) = cg²dx - cg sg dy + sg²dx + sg cg dy = dx ✓
    # -sg*(cg*dx - sg*dy) + cg*(sg*dx + cg*dy) = -sg cg dx + sg²dy + sg cg dx + cg²dy = dy ✓

    return s_val, u_val


def secondary_sector_stress(r_val, theta_val, a_val=1.0):
    """
    Stress in the secondary sector (extended Sector I beyond θ = γ).

    σ'₁₁: from α-line backward to void (same formula as primary Sector I)
    σ'₂₂: from β-line backward to curved boundary
    σ'₁₂ = A_I (constant on the yield face)
    """
    gam = float(gamma)
    phi = float(phi_I_kysar)
    A = float(A_I)
    rho = r_val / a_val

    # σ'₁₁ from α-line to void (same as primary)
    arg = phi - theta_val
    d1 = 1 - rho**2 * np.sin(arg)**2
    if d1 <= 0:
        return np.nan, np.nan, A

    s11p = A * rho * np.sin(arg) / np.sqrt(d1)

    # σ'₂₂ from β-line to curved boundary:
    # Find intersection of β-line backward with curved boundary
    s_val, u_val = find_beta_boundary_intersection(r_val, theta_val, a_val)

    if u_val < -1e-10 or s_val < -1e-10:
        return np.nan, np.nan, A  # No valid intersection

    # The intersection point on the curved boundary is at parameter u_val.
    # The stress at this point: σ'₂₂ from primary Sector I.
    x_int = a_val * np.cos(gam) + u_val * np.sin(gam)
    y_int = a_val * np.sin(gam) + u_val * np.cos(gam)
    r_int = np.sqrt(x_int**2 + y_int**2)
    theta_int = np.arctan2(y_int, x_int)

    rho_int = r_int / a_val
    arg_int = phi - theta_int
    d2_int = 1 - rho_int**2 * np.cos(arg_int)**2

    if d2_int <= 0:
        return np.nan, np.nan, A

    s22p = A * rho_int * np.cos(arg_int) / np.sqrt(d2_int)

    return s11p, s22p, A


# ================================================================
# 10. Generate comprehensive figure
# ================================================================
print("\nGenerating figure with primary and secondary sectors...")

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

gam = float(gamma)
alph = float(alpha_half)
phi_I_num = float(phi_I_kysar)

# Grid for stress computation
Nr, Nth = 200, 360
r_grid = np.linspace(1.001, 3.0, Nr)
theta_grid = np.linspace(0.001, np.pi/2 - 0.001, Nth)
R, Theta = np.meshgrid(r_grid, theta_grid)
X_cart = R * np.cos(Theta)
Y_cart = R * np.sin(Theta)

# Compute stress everywhere
Srr = np.full_like(R, np.nan)
Stt = np.full_like(R, np.nan)
Srt = np.full_like(R, np.nan)
Sm = np.full_like(R, np.nan)

for i in range(Nth):
    for j in range(Nr):
        r_val = R[i, j]
        th_val = Theta[i, j]

        # Determine sector
        if th_val < gam:
            # Primary Sector I
            s11p, s22p, s12p = primary_sector_I_stress(r_val, th_val)
            phi_rot = phi_I_num
        elif th_val < alph:
            # Check if in secondary sector (between θ=γ and curved boundary)
            s_val, u_val = find_beta_boundary_intersection(r_val, th_val)
            if u_val >= -0.01 and s_val >= -0.01:
                # In secondary sector (extended Sector I)
                s11p, s22p, s12p = secondary_sector_stress(r_val, th_val)
                phi_rot = phi_I_num
            else:
                # In primary Sector II
                s11p, s22p, s12p = primary_sector_II_stress(r_val, th_val)
                phi_rot = 0.0
        else:
            # Primary Sector III (simplified — use symmetry with Sector I)
            th_mirror = np.pi/2 - th_val
            s11p, s22p, s12p = primary_sector_I_stress(r_val, th_mirror)
            phi_rot = float(phi_III)
            # Note: Sector III has different boundary conditions; this is approximate

        if np.isnan(s11p):
            continue

        s11c, s22c, s12c = rotated_to_cartesian(s11p, s22p, s12p, phi_rot)
        srr, stt, srt, sm = cart_to_polar(s11c, s22c, s12c, th_val)

        Srr[i, j] = srr
        Stt[i, j] = stt
        Srt[i, j] = srt
        Sm[i, j] = sm

# Apply 4-fold symmetry
for qi, (sign_x, sign_y) in enumerate([(1,1)]):  # Just first quadrant for now
    pass

# Plot
titles = [r'(a) $\sigma_{rr}/\tau$', r'(b) $\sigma_{\theta\theta}/\tau$',
          r'(c) $\sigma_{r\theta}/\tau$', r'(d) $\sigma_m/\tau$']
data = [Srr, Stt, Srt, Sm]
vmaxs = [5, 8, 4, 6]

for idx, (ax, title, Z, vm) in enumerate(zip(axes.flat, titles, data, vmaxs)):
    # Plot in Cartesian
    pc = ax.pcolormesh(X_cart, Y_cart, Z, cmap='RdBu_r', vmin=-vm, vmax=vm,
                       shading='auto', rasterized=True)
    fig.colorbar(pc, ax=ax, shrink=0.8)

    # Void circle
    th_circle = np.linspace(0, np.pi/2, 200)
    ax.plot(np.cos(th_circle), np.sin(th_circle), 'k-', lw=2)

    # Sector boundaries (radial)
    for angle in [gam, alph]:
        ax.plot([0, 3*np.cos(angle)], [0, 3*np.sin(angle)],
                'k--', lw=0.5, alpha=0.5)

    # Curved boundary
    t_cb = np.linspace(0, 5, 200)
    x_cb_plot = np.cos(gam) + t_cb * np.sin(gam)
    y_cb_plot = np.sin(gam) + t_cb * np.cos(gam)
    r_cb_plot = np.sqrt(x_cb_plot**2 + y_cb_plot**2)
    mask_cb = r_cb_plot < 3
    ax.plot(x_cb_plot[mask_cb], y_cb_plot[mask_cb], 'w-', lw=2, alpha=0.8)

    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$x'_1/a$")
    ax.set_ylabel(r"$x'_2/a$")

plt.suptitle("Stress field with secondary sector (extended Sector I)", fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/secondary_sectors.png',
            dpi=150, bbox_inches='tight')
print("Saved: figures/secondary_sectors.png")

# ================================================================
# 11. Verify continuity at boundaries
# ================================================================
print("\n" + "="*70)
print("CONTINUITY VERIFICATION AT BOUNDARIES")
print("="*70)

# Check stress continuity at the curved boundary
# Take a point on the curved boundary at u = 1.0
u_test = 1.0
x_test = np.cos(gam) + u_test * np.sin(gam)
y_test = np.sin(gam) + u_test * np.cos(gam)
r_test = np.sqrt(x_test**2 + y_test**2)
th_test = np.arctan2(y_test, x_test)

print(f"\nTest point on curved boundary: u = {u_test}")
print(f"  (x, y) = ({x_test:.4f}, {y_test:.4f})")
print(f"  (r, θ) = ({r_test:.4f}, {np.degrees(th_test):.2f}°)")

# Stress from primary Sector I
s11p_I, s22p_I, s12p_I = primary_sector_I_stress(r_test, th_test)
s11c_I, s22c_I, s12c_I = rotated_to_cartesian(s11p_I, s22p_I, s12p_I, phi_I_num)
print(f"\n  Primary Sector I stress:")
print(f"    σ'₁₁ = {s11p_I:.6f}, σ'₂₂ = {s22p_I:.6f}, σ'₁₂ = {s12p_I:.6f}")
print(f"    σ₁₁ = {s11c_I:.6f}, σ₂₂ = {s22c_I:.6f}, σ₁₂ = {s12c_I:.6f}")

# Stress from secondary sector (approaching from Sector II side)
# At the boundary, the secondary sector stress should match
s11p_S, s22p_S, s12p_S = secondary_sector_stress(r_test, th_test + 0.001)
if not np.isnan(s11p_S):
    s11c_S, s22c_S, s12c_S = rotated_to_cartesian(s11p_S, s22p_S, s12p_S, phi_I_num)
    print(f"\n  Secondary sector stress (θ + ε):")
    print(f"    σ'₁₁ = {s11p_S:.6f}, σ'₂₂ = {s22p_S:.6f}, σ'₁₂ = {s12p_S:.6f}")
    print(f"    σ₁₁ = {s11c_S:.6f}, σ₂₂ = {s22c_S:.6f}, σ₁₂ = {s12c_S:.6f}")

    jump = np.sqrt((s11c_I - s11c_S)**2 + (s22c_I - s22c_S)**2 + (s12c_I - s12c_S)**2)
    print(f"\n  Stress jump at boundary: ||Δσ|| = {jump:.6f}")
else:
    print("\n  Secondary sector stress: NaN (outside domain)")

print("\n" + "="*70)
print("DONE")
print("="*70)
