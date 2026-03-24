"""
Derive the exact Sector III σ'₁₁ formula using SymPy.

The α-line from P at (r, θ) hits the x₂-axis (symmetry boundary),
then reflects. The σ'₁₁ value is determined by the image construction:
the reflected α-line hits the void at the mirror point, and the
axis vertex stress provides the offset.

Following Kysar (2005) Section 3.6 exactly.
"""
import sympy as sp
from sympy import sqrt, cos, sin, tan, atan, pi, Rational, simplify, symbols
from sympy import solve, Eq, trigsimp, nsimplify
import numpy as np

# ================================================================
# Symbolic setup
# ================================================================
r, a, theta, tau = symbols('r a theta tau', positive=True)
rho = r / a  # normalized radius

gamma = atan(2*sqrt(2)) / 2
phi_III = gamma
A_III = 2*sqrt(3) / 3

print("="*70)
print("SECTOR III σ'₁₁ DERIVATION")
print("="*70)

# ================================================================
# Step 1: α-line geometry
# ================================================================
print("\n--- Step 1: α-line from P to x₂-axis ---")

# α-line direction in Sector III: φ_III + π/2
# φ_III = γ, so α-line at γ + π/2
# Direction vector: (cos(γ+π/2), sin(γ+π/2)) = (-sinγ, cosγ)

# From P = (r cosθ, r sinθ), the α-line going toward x₂-axis:
# x(t) = r cosθ - t sinγ
# y(t) = r sinθ + t cosγ
# Hits x₁ = 0 when: r cosθ - t₁ sinγ = 0
# t₁ = r cosθ / sinγ

t1 = r * cos(theta) / sin(gamma)
print(f"t₁ = r cosθ / sinγ")

# Point on x₂-axis:
# y₁ = r sinθ + t₁ cosγ = r sinθ + r cosθ cosγ/sinγ
# = r(sinθ sinγ + cosθ cosγ)/sinγ = r cos(θ-γ)/sinγ
y1 = r * cos(theta - gamma) / sin(gamma)
print(f"y₁ = r cos(θ-γ) / sinγ")

# ================================================================
# Step 2: Reflected α-line (image construction)
# ================================================================
print("\n--- Step 2: Image construction ---")

# At the x₂-axis, the stress is at vertex V₄ = (-√6/2, 0)
# By symmetry about the x₂-axis, the reflected α-line goes in
# direction (+sinγ, cosγ) from (0, y₁).
#
# The reflected α-line hits the void at (x₂, y₂):
# x₂ = s sinγ, y₂ = y₁ + s cosγ
# x₂² + y₂² = a²
# s² sin²γ + (y₁ + s cosγ)² = a²
# s² + 2s y₁ cosγ + y₁² - a² = 0
#
# s = -y₁ cosγ ± √(y₁² cos²γ - y₁² + a²)
#   = -y₁ cosγ ± √(a² - y₁² sin²γ)

# Substitute y₁ = r cos(θ-γ)/sinγ:
# y₁² sin²γ = r² cos²(θ-γ)
# Discriminant: a² - r² cos²(θ-γ)

disc = a**2 - r**2 * cos(theta - gamma)**2
print(f"Discriminant: a² - r² cos²(θ-γ)")
print(f"Valid when r/a ≤ 1/|cos(θ-γ)| = sec(θ-γ)")

# The physical root (s > 0, going upward from axis):
# s = -y₁ cosγ + √(disc)  if y₁ cosγ < √(disc)
# or s = -y₁ cosγ - √(disc) (both negative → no valid root going up)
# For points near the void, s should be positive.

# At the void hit point:
# x₂ = s sinγ, y₂ = y₁ + s cosγ
# θ₂ = atan2(y₂, x₂) = atan2(y₁ + s cosγ, s sinγ)

# ================================================================
# Step 3: σ'₂₂ at the reflected void point
# ================================================================
print("\n--- Step 3: σ'₂₂ at reflected void point ---")

# The β-line from the reflected void point carries σ'₂₂.
# At the void surface, σ'₂₂ has the same formula as for Sector I
# (since Sector III's φ_III = -φ_I by the symmetry of the problem).
#
# Actually, the reflected void point is in the "Sector III mirror image"
# where the same yield face is active. The σ'₂₂ at this point is
# determined by the void surface stress at angle (π - θ₂) (mirrored).
#
# At the void surface at angle θ_void:
# σ'₂₂ = -A_III · cos(φ_III - θ_void) / √(1 - cos²(φ_III - θ_void))
#       = -A_III · cos(φ_III - θ_void) / |sin(φ_III - θ_void)|

# But actually, σ'₁₁ is what's constant along the α-line, not σ'₂₂.
# Let me reconsider.

# ================================================================
# Step 3 (revised): σ'₁₁ from α-line tracing
# ================================================================
print("\n--- Step 3 (revised): σ'₁₁ via α-line/axis/image ---")

# σ'₁₁ is constant along each α-line (lines of constant x'₁).
# The α-line from P hits the x₂-axis, where the stress is at vertex V₄.
# On the x₂-axis, we know the RELATIONSHIP between σ'₁₁ and σ'₂₂:
#   σ'₁₁ - σ'₂₂ = -2τ · √6/6 = -√6/3 τ
#   (from vertex V₄ = (-√6/2, 0) in the rotated frame)
#
# At the axis point (0, y₁), we also know σ'₂₂ because the β-line
# from (0, y₁) goes to the void surface.
#
# The β-line from (0, y₁) goes at angle φ_III = γ (the β-direction):
# x(t) = 0 + t cosγ, y(t) = y₁ + t sinγ  ... wait, β-line direction
# is φ_III = +γ, so (cosγ, sinγ)? No...

# In the Kysar convention:
# β-lines are parallel to x'₁, at angle φ_III from horizontal
# α-lines are perpendicular to x'₁, at angle φ_III + π/2

# For Sector III: φ_III = +γ
# β-line direction: angle +γ from horizontal → (cosγ, sinγ)
# α-line direction: angle γ+π/2 from horizontal → (-sinγ, cosγ)

# β-line from (0, y₁) going at angle γ:
# x(t) = t cosγ, y(t) = y₁ + t sinγ
# Hits void: t² cos²γ + (y₁ + t sinγ)² = a²
# t² + 2t y₁ sinγ + y₁² - a² = 0
# t = -y₁ sinγ ± √(y₁² sin²γ - y₁² + a²) = -y₁ sinγ ± √(a² - y₁²cos²γ)

disc_beta = a**2 - y1**2 * cos(gamma)**2

# Substitute y₁ = r cos(θ-γ)/sinγ:
# y₁² cos²γ = r² cos²(θ-γ) cos²γ/sin²γ
disc_beta_sub = simplify(disc_beta.subs(y1, r*cos(theta-gamma)/sin(gamma)))
print(f"β-line discriminant: {disc_beta_sub}")

# Actually, the β-line from (0, y₁) going TOWARD the void (negative t,
# since the void is to the right of the axis only for x > 0).
# Wait, x = t cosγ > 0 for t > 0. So going right (toward the void)
# means t > 0... but the void center is at origin, and the axis point
# is at (0, y₁) with y₁ > a. The void surface at x = 0 has y = ±a.
# If y₁ > a, the β-line goes DOWN (t < 0) to reach the void at y = a.

# For t < 0: x < 0, which is the wrong side of the axis!
# So the β-line from (0, y₁) going at angle γ goes to the RIGHT and UP,
# away from the void.
# Going at angle γ + π goes LEFT and DOWN, also away from void.

# The issue: the β-line from the axis point doesn't directly reach the
# void. Instead, it reaches the void via the IMAGE construction.

# By symmetry about x₂-axis: the void at (a cosθ_v, a sinθ_v) has
# an image at (-a cosθ_v, a sinθ_v). The β-line from (0, y₁) going
# LEFT (at angle γ + π) reaches the image void, and by symmetry this
# gives the same σ'₂₂ as at the real void point.

# So: the EFFECTIVE β-line distance from (0, y₁) to the void (image)
# is: t going at angle γ + π: x = -t cosγ, y = y₁ - t sinγ
# Hits image void: t² - 2t y₁ sinγ + y₁² = a²
# t = y₁ sinγ ± √(y₁² sin²γ - y₁² + a²) = y₁ sinγ ± √(a² - y₁²cos²γ)

# The effective void point is at angle:
# θ_eff = π - arctan(y₂/x₂) where (x₂, y₂) is on the image void

# OK this is getting complicated. Let me just do the KEY computation:
# at the axis point (0, y₁), σ'₂₂ is determined by the β-line that
# goes from the void at some angle θ* to the axis.
# Since σ'₂₂ is constant on β-lines, and the β-line from the axis
# comes from the void at θ* where the β-line through (0, y₁) hits the void:
# The void point: β-line at angle γ from (a cosθ*, a sinθ*) reaches (0, y₁).
# a cosθ* + t cosγ = 0 → t = -a cosθ*/cosγ
# a sinθ* + t sinγ = y₁ → a sinθ* - a cosθ* tanγ = y₁
# a(sinθ* - cosθ* tanγ) = y₁ = r cos(θ-γ)/sinγ

# sinθ* - cosθ* tanγ = (r/a) cos(θ-γ)/sinγ

# On the void (r* = a, angle θ*): σ'₂₂ from the Sector III formula.
# σ'₂₂(a, θ*) = -A_III cos(φ_III - θ*)/|sin(φ_III - θ*)|

# But θ* is determined by the β-line equation above.
# Let α* = γ - θ* (relative angle):
# sin(γ - α*) - cos(γ - α*) tanγ = rho cos(θ-γ)/sinγ
# This is a transcendental equation for α* (or θ*).

# Let me try a DIFFERENT approach: use the r*/a ratio from Kysar Eq. (28).

print("\n--- Step 4: Using Kysar's r* construction ---")

# Kysar Eq. (28): r*/a = (r/a)|sin(φ-θ)|/sinφ
# For Sector III: φ = γ
rstar_over_a = rho * sp.Abs(sin(gamma - theta)) / sin(gamma)
print(f"r*/a = (r/a)|sin(γ-θ)|/sinγ")

# At the axis point (0, r*/sinγ·something), the β-line that connects
# to the void at angle π/2 (on the x₂ axis) gives:
# σ'₂₂(axis) = σ'₂₂ from Sector III at (r*, π/2)
# = -A_III · r*/a · cos(γ - π/2) / √(1 - (r*/a)² cos²(γ - π/2))
# = -A_III · r*/a · sinγ / √(1 - (r*/a)² sin²γ)

# But wait, the β-line from the axis goes to the void at some angle,
# not necessarily π/2. Let me re-derive.

# Actually, in Kysar's construction, the image void point is obtained
# by reflecting the α-line origin about the symmetry axis. The
# effective radial distance r* accounts for the total path length
# of the α-line from P to the axis and from the axis to the image void.

# The key result (Kysar Eq. 30a adapted):
# σ'₁₁(P) = σ'₁₁(image) = σ'₂₂(axis) + (√6/3)τ ... need to be careful
# where σ'₂₂(axis) comes from the void via a β-line of length r*.

# Let's compute σ'₂₂(axis) numerically for several θ values and
# see if it matches a formula involving r*.

print("\n--- Numerical verification ---")

gamma_n = float(gamma)
A_n = float(A_III)

for theta_n in [56, 65, 75, 85, 89]:
    theta_r = np.radians(theta_n)
    rho_n = 1.0  # at void surface

    # r*/a
    rstar = rho_n * abs(np.sin(gamma_n - theta_r)) / np.sin(gamma_n)

    # σ'₂₂ at the axis point, from a β-line that traverses distance r*
    # In the rotated frame, the β-line sees an effective void at distance r*
    # σ'₂₂ = -A · rstar · cos(γ - π/2) / √(1 - rstar² cos²(γ - π/2))
    # cos(γ - π/2) = sinγ
    s22p_axis = -A_n * rstar * np.sin(gamma_n) / np.sqrt(max(1e-10, 1 - rstar**2 * np.sin(gamma_n)**2))

    # σ'₁₁ = σ'₂₂ - √6/3  (from vertex V₄ condition)
    s11p_from_axis = s22p_axis - np.sqrt(6)/3

    # Exact σ'₁₁ from void surface
    c2 = np.cos(2*theta_r); s2 = np.sin(2*theta_r)
    if abs(c2) > 1e-10:
        X = 6/(-2*np.sqrt(6) + np.sqrt(3)*np.tan(2*theta_r))
        Y = X * np.tan(2*theta_r)
    else:
        X = -np.sqrt(6)/2; Y = 0
    sm = -(X*c2 + Y*s2)
    sig11 = sm+X; sig22 = sm-X; sig12 = Y
    c2p = np.cos(2*gamma_n); s2p = np.sin(2*gamma_n)
    s11p_exact = (sig11+sig22)/2 + (sig11-sig22)/2*c2p + sig12*s2p

    print(f"  θ={theta_n}°  r*={rstar:.5f}  s22p(axis)={s22p_axis:.5f}  "
          f"s11p(axis)={s11p_from_axis:.5f}  s11p(exact)={s11p_exact:.5f}  "
          f"err={abs(s11p_from_axis - s11p_exact):.6f}")

# Try with the CORRECT axis relationship:
# At axis, vertex V₄ in rotated frame (φ = γ):
# V₄ = (-√6/2, 0) in (X, Y)
# σ'₁₁ - σ'₂₂ = 2X cos(2γ) + 2Y sin(2γ) = 2(-√6/2)(1/3) = -√6/3
# So σ'₁₁ = σ'₂₂ - √6/3  ← This is what we used

# Try the OTHER interpretation: σ'₂₂ at axis from reflected void point
# The reflected void point is at angle (π - θ_void_mirror)
# The β-line from axis to this point has specific geometry

# Actually, let me try σ'₂₂ = A · rstar / √(1 - rstar²) (without specific angle)
print("\n  Alternative: σ'₂₂(axis) = -A · rstar / √(1 - rstar²):")
for theta_n in [56, 65, 75, 85, 89]:
    theta_r = np.radians(theta_n)
    rstar = abs(np.sin(gamma_n - theta_r)) / np.sin(gamma_n)
    if rstar >= 1: continue
    s22p_alt = -A_n * rstar / np.sqrt(1 - rstar**2)
    s11p_alt = s22p_alt - np.sqrt(6)/3

    c2 = np.cos(2*theta_r); s2 = np.sin(2*theta_r)
    X = 6/(-2*np.sqrt(6) + np.sqrt(3)*np.tan(2*theta_r)) if abs(c2)>1e-10 else -np.sqrt(6)/2
    Y = X*np.tan(2*theta_r) if abs(c2)>1e-10 else 0
    sm = -(X*c2+Y*s2)
    sig11=sm+X;sig22=sm-X;sig12=Y
    c2p=np.cos(2*gamma_n);s2p=np.sin(2*gamma_n)
    s11p_exact = (sig11+sig22)/2+(sig11-sig22)/2*c2p+sig12*s2p

    print(f"  θ={theta_n}°  r*={rstar:.5f}  s11p(alt)={s11p_alt:.5f}  s11p(exact)={s11p_exact:.5f}  err={abs(s11p_alt-s11p_exact):.6f}")

# Try: σ'₂₂(axis) from the Sector I formula at the reflected angle
print("\n  Using Sector I σ'₂₂ at reflected angle:")
for theta_n in [56, 65, 75, 85, 89]:
    theta_r = np.radians(theta_n)
    theta_reflected = np.pi - theta_r  # Mirror about x₂-axis
    # At the reflected void point, use Sector I formula (which covers 0 to γ)
    # But θ_reflected = π - θ is in the range [91°, 124°], not in Sector I!
    # The mirror of Sector III about x₂ maps to Sector IV (π/2 to π/2+γ)
    # which has the same structure as Sector I by the π-periodicity.

    # Actually, the correct approach: the image void point angle θ_v
    # satisfies sinθ_v - cosθ_v tanγ = cos(θ-γ)/sinγ (from step 3)
    # Let me solve this numerically.
    from scipy.optimize import brentq

    target = np.cos(theta_r - gamma_n) / np.sin(gamma_n)
    f = lambda tv: np.sin(tv) - np.cos(tv)*np.tan(gamma_n) - target

    # θ_v should be in the range (π/2, π)
    try:
        theta_v = brentq(f, np.pi/2 + 0.01, np.pi - 0.01)
    except:
        theta_v = np.nan

    if not np.isnan(theta_v):
        # σ'₂₂ at the image void point (angle θ_v) using Sector III formula
        arg_v = gamma_n - theta_v
        d2_v = 1 - np.cos(arg_v)**2
        s22p_v = -A_n * np.cos(arg_v) / np.sqrt(d2_v) if d2_v > 0 else np.nan

        s11p_v = s22p_v - np.sqrt(6)/3

        c2=np.cos(2*theta_r);s2=np.sin(2*theta_r)
        X = 6/(-2*np.sqrt(6)+np.sqrt(3)*np.tan(2*theta_r)) if abs(c2)>1e-10 else -np.sqrt(6)/2
        Y = X*np.tan(2*theta_r) if abs(c2)>1e-10 else 0
        sm=-(X*c2+Y*s2);sig11=sm+X;sig22=sm-X;sig12=Y
        c2p=np.cos(2*gamma_n);s2p=np.sin(2*gamma_n)
        s11p_exact=(sig11+sig22)/2+(sig11-sig22)/2*c2p+sig12*s2p

        print(f"  θ={theta_n}°  θ_v={np.degrees(theta_v):.2f}°  s22p(v)={s22p_v:.5f}  "
              f"s11p={s11p_v:.5f}  exact={s11p_exact:.5f}  err={abs(s11p_v-s11p_exact):.6f}")
    else:
        print(f"  θ={theta_n}°  no solution for θ_v")
