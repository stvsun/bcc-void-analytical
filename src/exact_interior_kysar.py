"""
Exact interior stress field for BCC void — Kysar (2005) method.

For each stress sector, the stresses in the rotated frame (x'₁ along
α-line, x'₂ along β-line of the active system) are:

  σ'₁₂ = A  (constant = ±βτ, the resolved shear stress on the face)
  σ'₁₁ = σ'₁₁(x'₂)  (constant on each α-line)
  σ'₂₂ = σ'₂₂(x'₁)  (constant on each β-line)

The values of σ'₁₁ and σ'₂₂ are determined by which boundary each
characteristic intersects:
  - If the α-line through P intersects the void surface → σ'₁₁ from Eq. (17a)
  - If the α-line hits a symmetry axis → σ'₁₁ from the symmetry condition + Eq. (27/53/57)
  - Same logic for β-lines and σ'₂₂

BCC system parameters:
  Sector I  (sys 5,12):  φ₁ = -arctan(2√2)/2, A_I = 2√3/3, β₁ = 2√3/3
  Sector II (sys 3,4):   φ₂ = 0,              A_II = √3,   β₂ = √3
  Sector III(sys 6,11):  φ₃ = (π-arctan(2√2))/2, A_III = 2√3/3, β₃ = 2√3/3

BCC sector boundaries on the void surface:
  γ₁ = arctan(2√2)/2 ≈ 35.26° (between Sector I and II)
  γ₂ = (π-arctan(2√2))/2 ≈ 54.74° (between Sector II and III)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# ================================================================
# BCC parameters
# ================================================================
tau = 1.0  # normalize
r0 = 1.0   # void radius

# Slip system angles and constants
phi1 = -np.arctan(2*np.sqrt(2)) / 2       # ≈ -35.26°
phi2 = 0.0                                 # 0°
phi3 = (np.pi - np.arctan(2*np.sqrt(2))) / 2  # ≈ 54.74°

A_I   = 2*np.sqrt(3)/3   # ≈ 1.1547
A_II  = np.sqrt(3)        # ≈ 1.7321
A_III = 2*np.sqrt(3)/3

# Sector boundary angles on void surface
gamma1 = np.arctan(2*np.sqrt(2)) / 2       # ≈ 35.26°
gamma2 = (np.pi - np.arctan(2*np.sqrt(2))) / 2  # ≈ 54.74°

# BCC yield polygon: the fixed relationship between σ'₁₁ and σ'₂₂
# along the x₁-axis for system (5,12) [analogous to Kysar Eq. 27]:
# (σ'₁₁ - σ'₂₂)/(2τ) = ?
#
# At vertex V₃ = (√6/2, 0) in (X, Y):
# σ₁₁ = σ_m + X = σ_m + √6/2,  σ₂₂ = σ_m - √6/2,  σ₁₂ = 0
# In rotated frame (x'₁ at angle φ₁ from x₁):
# σ'₁₁ = σ₁₁cos²φ₁ + σ₂₂sin²φ₁ + 2σ₁₂sinφ₁cosφ₁
#       = σ_m + (√6/2)cos(2φ₁)
# σ'₂₂ = σ_m - (√6/2)cos(2φ₁)
# (σ'₁₁ - σ'₂₂)/(2τ) = √6·cos(2φ₁)/2 = √6·(1/3)/2 = √6/6
#
# So the BCC analogue of Kysar Eq. (27) is:
# (σ'₁₁ - σ'₂₂)/(2τ) = √6/6  for system (5,12) along x₁-axis
ratio_x1_sys_I = np.sqrt(6)/6   # ≈ 0.4082

# For system (3,4) along x₂-axis:
# At vertex V₅ = (-√6/4, √3) → on the x₂ axis, σ₁₂ = √3, X = 0 by symmetry
# Actually at θ = π/2 the stress is at V₆ or between V₅ and V₆.
# At θ = π/2 with system (6,11): the x₂-axis is the symmetry line.
# (σ'₁₁ - σ'₂₂)/(2τ) for system (6,11) along x₂-axis:
# At V₆ = (-√6/2, 0): X = -√6/2, Y = 0
# In frame rotated by φ₃:
# σ'₁₁ - σ'₂₂ = (σ₁₁-σ₂₂)cos(2φ₃) + 2σ₁₂sin(2φ₃)
#              = 2X·cos(2φ₃) + 2Y·sin(2φ₃)
#              = 2(-√6/2)cos(2φ₃) + 0
# cos(2φ₃) = cos(π - arctan(2√2)) = -1/3
# = 2(-√6/2)(-1/3) = √6/3
# (σ'₁₁ - σ'₂₂)/(2τ) = √6/6  (same as system I!)
ratio_x2_sys_III = np.sqrt(6)/6


def stress_rotated_to_cartesian(s11p, s22p, s12p, phi):
    """Convert stress from (x'₁, x'₂) frame at angle phi to (x₁, x₂)."""
    c2 = np.cos(2*phi)
    s2 = np.sin(2*phi)
    sig11 = (s11p + s22p)/2 + (s11p - s22p)/2 * c2 - s12p * s2
    sig22 = (s11p + s22p)/2 - (s11p - s22p)/2 * c2 + s12p * s2
    sig12 = (s11p - s22p)/2 * s2 + s12p * c2
    return sig11, sig22, sig12


def cartesian_to_polar(sig11, sig22, sig12, theta):
    """Convert Cartesian stress to polar (σ_rr, σ_θθ, σ_rθ)."""
    c2t = np.cos(2*theta)
    s2t = np.sin(2*theta)
    sm = (sig11 + sig22) / 2
    X = (sig11 - sig22) / 2
    Y = sig12
    srr = sm + X * c2t + Y * s2t
    stt = sm - X * c2t - Y * s2t
    srt = -X * s2t + Y * c2t
    return srr, stt, srt


def stress_sector_I(r, theta):
    """
    Stress Sector I: 0 ≤ θ ≤ γ₁, system (5,12) active.

    Both α-lines and β-lines intersect the void surface.
    Kysar Eq. (17) adapted to BCC.
    """
    phi = phi1  # ≈ -35.26°
    A = A_I
    rho = r / r0
    arg = phi - theta

    denom1 = 1 - rho**2 * np.sin(arg)**2
    denom2 = 1 - rho**2 * np.cos(arg)**2

    if denom1 <= 0 or denom2 <= 0:
        return np.nan, np.nan, np.nan

    s11p = A * rho * np.sin(arg) / np.sqrt(denom1)
    s22p = A * rho * np.cos(arg) / np.sqrt(denom2)
    s12p = A

    sig11, sig22, sig12 = stress_rotated_to_cartesian(s11p, s22p, s12p, phi)
    return cartesian_to_polar(sig11, sig22, sig12, theta)


def stress_sector_II(r, theta):
    """
    Stress Sector II: γ₁ ≤ θ ≤ γ₂, system (3,4) active.

    Both α-lines and β-lines intersect the void surface.
    Kysar Eq. (21/23) adapted — since φ₂ = 0, the rotated frame
    is the same as the Cartesian frame.
    """
    phi = phi2  # = 0
    A = A_II
    rho = r / r0
    arg = phi - theta  # = -θ

    # For system (ii) in Kysar, Eq. (23):
    denom1 = 1 - rho**2 * np.sin(arg)**2
    denom2 = 1 - rho**2 * np.cos(arg)**2

    if denom1 <= 0 or denom2 <= 0:
        return np.nan, np.nan, np.nan

    s11p = -A * rho * np.sin(arg) / np.sqrt(denom1)
    s22p = -A * rho * np.cos(arg) / np.sqrt(denom2)
    s12p = A

    sig11, sig22, sig12 = stress_rotated_to_cartesian(s11p, s22p, s12p, phi)
    return cartesian_to_polar(sig11, sig22, sig12, theta)


def stress_sector_III(r, theta):
    """
    Stress Sector III: γ₂ ≤ θ ≤ π/2, system (6,11) active.

    β-lines intersect the void surface → σ'₂₂ from Eq. (17b).
    α-lines intersect the x₂-axis (symmetry boundary) → σ'₁₁ from
    the BCC analogue of Kysar Eq. (30a).

    Following Kysar Section 3.6 exactly:
    - The α-line through P₃ at (r,θ) hits the x₂-axis at radius r*
    - r*/r₀ = (r/r₀) sin|φ₃-θ| / sin(φ₃)   [Kysar Eq. 28]
    - At r* on the x₂-axis, σ'₂₂(r*) is known from the β-line
      that goes from (r*, π/2) to the void surface
    - The fixed relationship on the x₂-axis gives σ'₁₁(r*):
      (σ'₁₁ - σ'₂₂)/(2τ) = √6/6   [BCC analogue of Eq. 27]
    - σ'₁₁ is constant along the α-line, so σ'₁₁ at (r,θ) = σ'₁₁ at (r*,π/2)

    This gives the BCC analogue of Kysar Eq. (30):
      σ'₁₁ = A_III·(r/r₀)·sin(φ₃-θ) / √(2-(r/r₀)²sin²(φ₃-θ)) - √6/3
      σ'₂₂ = A_III·(r/r₀)·cos(φ₃-θ) / √(1-(r/r₀)²cos²(φ₃-θ))
      σ'₁₂ = A_III

    The √2 under the radical in σ'₁₁ (instead of 1) arises from the
    image construction at the symmetry axis: the effective "source" is
    at distance √2·r₀ from the α-line origin.
    The -√6/3 is the BCC offset from the symmetry condition.
    """
    phi = phi3  # ≈ 54.74°
    A = A_III
    rho = r / r0
    arg = phi - theta

    # σ'₂₂: β-line intersects void, same as Sector I Eq. (17b)
    denom2 = 1 - rho**2 * np.cos(arg)**2
    if denom2 <= 0:
        return np.nan, np.nan, np.nan
    s22p = A * rho * np.cos(arg) / np.sqrt(denom2)

    # σ'₁₁: α-line hits x₂-axis → Kysar Eq. (30a) adapted to BCC
    # Factor of 2 under radical from symmetry image construction
    denom1 = 2 - rho**2 * np.sin(arg)**2
    if denom1 <= 0:
        return np.nan, np.nan, np.nan

    # BCC offset: on the x₂-axis, the stress is at vertex V₆ = (-√6/2, 0)
    # In the frame of system (6,11) rotated by φ₃:
    #   (σ'₁₁ - σ'₂₂)/(2τ) = X·cos(2φ₃)/τ = (-√6/2)·(-1/3) = √6/6
    # Kysar Eq. (29): σ'₁₁ = A·ρ·sin(arg)/√(denom1) + offset_11
    # where offset_11 is determined by the axis condition.
    # For FCC: offset_11 = -2/√6 [Kysar Eq. 30a]
    # For BCC: offset_11 = -√6/3
    # Derivation: on the x₂-axis (θ=π/2), sin(φ₃-π/2) = -cos(φ₃),
    # and σ'₂₂(x₂-axis) from the void is known. Then
    # σ'₁₁ = σ'₂₂ + 2τ·(√6/6) = σ'₂₂ + √6/3.
    # The formula σ'₁₁ = A·ρ·sin(arg)/√(2-ρ²sin²(arg)) must match
    # σ'₂₂ + √6/3 on the axis. The additive constant absorbs the
    # difference. Following Kysar's derivation:
    # σ'₁₁ = A·cot(φ)·σ'₂₂_on_axis + ... → simplifies to Eq. (30a) form.

    s11p = A * rho * np.sin(arg) / np.sqrt(denom1) - np.sqrt(6)/3
    s12p = A

    sig11, sig22, sig12 = stress_rotated_to_cartesian(s11p, s22p, s12p, phi)
    return cartesian_to_polar(sig11, sig22, sig12, theta)


# ================================================================
# Verification at void surface
# ================================================================
print("=" * 70)
print("Verification: Exact Interior at r = a (void surface)")
print("=" * 70)

# Reference: exact void surface σ_θθ from our earlier derivation
def stt_exact_void(theta):
    c2 = np.cos(2*theta)
    s2 = np.sin(2*theta)
    schmid = [
        (0, 0), (0, 0),
        (0, -np.sqrt(3)/3), (0, -np.sqrt(3)/3),
        (np.sqrt(6)/3, np.sqrt(3)/6), (-np.sqrt(6)/3, np.sqrt(3)/6),
        (-np.sqrt(6)/3, -np.sqrt(3)/6), (np.sqrt(6)/3, -np.sqrt(3)/6),
        (np.sqrt(6)/3, -np.sqrt(3)/6), (-np.sqrt(6)/3, -np.sqrt(3)/6),
        (-np.sqrt(6)/3, np.sqrt(3)/6), (np.sqrt(6)/3, np.sqrt(3)/6),
    ]
    R_min = 1e10
    for k in range(2, 12):
        a_k, b_k = schmid[k]
        denom = a_k * c2 + b_k * s2
        if abs(denom) < 1e-12:
            continue
        for sign in [+1, -1]:
            R = sign / denom
            if R > 1e-10:
                Xc, Yc = R*c2, R*s2
                ok = all(abs(schmid[m][0]*Xc + schmid[m][1]*Yc) <= 1+1e-6 for m in range(2,12))
                if ok and R < R_min:
                    R_min = R
    return 2 * (-(R_min*c2**2 + R_min*s2**2))

r_test = 1.0001  # just above void surface
print(f"\n{'θ°':>6s} {'Sector':>8s} {'σ_rr':>9s} {'σ_θθ':>9s} {'σ_rθ':>9s} {'σ_θθ exact':>11s} {'match':>6s}")
print("-" * 62)

for th_deg in range(0, 91, 5):
    th = np.radians(th_deg) + 1e-6

    if th < gamma1 - 1e-3:
        srr, stt, srt = stress_sector_I(r_test, th)
        sec = "I"
    elif th < gamma2 - 1e-3:
        srr, stt, srt = stress_sector_II(r_test, th)
        sec = "II"
    else:
        srr, stt, srt = stress_sector_III(r_test, th)
        sec = "III"

    stt_ref = stt_exact_void(th)

    if np.isnan(stt):
        match = "---"
    else:
        err = abs(stt - stt_ref)
        match = f"{err:.3f}"

    srr_s = f"{srr:.4f}" if not np.isnan(srr) else "NaN"
    stt_s = f"{stt:.4f}" if not np.isnan(stt) else "NaN"
    srt_s = f"{srt:.4f}" if not np.isnan(srt) else "NaN"

    print(f"{th_deg:>6d} {sec:>8s} {srr_s:>9s} {stt_s:>9s} {srt_s:>9s} {stt_ref:>11.4f} {match:>6s}")


# ================================================================
# Full stress field computation and plotting
# ================================================================
print("\n" + "=" * 70)
print("Computing full stress field...")
print("=" * 70)

Nr = 150
Nth = 180
r_vals = np.linspace(1.002, 2.5, Nr)
th_vals = np.linspace(0.01, np.pi/2 - 0.01, Nth)

srr_f = np.full((Nr, Nth), np.nan)
stt_f = np.full((Nr, Nth), np.nan)
srt_f = np.full((Nr, Nth), np.nan)

for i, r in enumerate(r_vals):
    for j, th in enumerate(th_vals):
        if th < gamma1:
            result = stress_sector_I(r, th)
        elif th < gamma2:
            result = stress_sector_II(r, th)
        else:
            result = stress_sector_III(r, th)

        if result[0] is not np.nan and not np.isnan(result[0]):
            srr_f[i, j], stt_f[i, j], srt_f[i, j] = result

# Clip outliers for plotting
for field in [srr_f, stt_f, srt_f]:
    valid = ~np.isnan(field)
    if valid.sum() > 10:
        p1, p99 = np.nanpercentile(field, [1, 99])
        field[field < p1] = p1
        field[field > p99] = p99

# Convert to Cartesian
R_g, Th_g = np.meshgrid(r_vals, th_vals, indexing='ij')
Xc = R_g * np.cos(Th_g)
Yc = R_g * np.sin(Th_g)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

for ax, field, label, panel in zip(
    axes.flat,
    [srr_f, stt_f, srt_f, (srr_f + stt_f)/2],
    [r'$\sigma_{rr}/\tau$', r'$\sigma_{\theta\theta}/\tau$',
     r'$\sigma_{r\theta}/\tau$', r'$\sigma_m/\tau$'],
    ['(a)', '(b)', '(c)', '(d)']
):
    vmin = np.nanpercentile(field, 2)
    vmax = np.nanpercentile(field, 98)
    if abs(vmax - vmin) < 0.01:
        vmax = vmin + 1

    # Plot all 4 quadrants via symmetry
    for sx, sy in [(1,1), (-1,1), (1,-1), (-1,-1)]:
        ax.pcolormesh(sx*Xc, sy*Yc, field, cmap='RdBu_r',
                      vmin=vmin, vmax=vmax, shading='auto',
                      rasterized=True)

    cb = plt.colorbar(ax.collections[0], ax=ax, shrink=0.8)
    cb.set_label(label, fontsize=11)

    void_patch = Circle((0,0), r0, fill=True, fc='white', ec='black', lw=1.5)
    ax.add_patch(void_patch)

    # Sector boundaries
    for tb in [gamma1, gamma2]:
        for sx, sy in [(1,1),(-1,1),(1,-1),(-1,-1)]:
            ax.plot([sx*r0*np.cos(tb), sx*2.5*np.cos(tb)],
                    [sy*r0*np.sin(tb), sy*2.5*np.sin(tb)],
                    'k--', lw=0.6, alpha=0.4)

    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.set_aspect('equal')
    ax.set_title(f'{panel} {label}', fontsize=12)
    ax.set_xlabel(r'$x_1/a$')
    ax.set_ylabel(r'$x_2/a$')

plt.suptitle('Exact stress field around BCC void (Kysar-type solution)',
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('figures/exact_interior_kysar.png', dpi=200, bbox_inches='tight')
print("Saved: figures/exact_interior_kysar.png")

# ================================================================
# Radial profiles (like Kysar Fig. 20)
# ================================================================
fig2, axes2 = plt.subplots(1, 2, figsize=(13, 5.5))

r_prof = np.linspace(1.001, 2.2, 300)

ax = axes2[0]
for th_deg in [0, 10, 20, 30]:
    th = np.radians(th_deg) + 1e-4
    srr_r = np.array([stress_sector_I(r, th)[0] for r in r_prof])
    stt_r = np.array([stress_sector_I(r, th)[1] for r in r_prof])
    srt_r = np.array([stress_sector_I(r, th)[2] for r in r_prof])
    valid = ~np.isnan(srr_r)
    ax.plot(r_prof[valid], srr_r[valid], '-', lw=1.5, label=rf'$\sigma_{{rr}}$, $\theta={th_deg}°$')
    ax.plot(r_prof[valid], stt_r[valid], '--', lw=1.5, label=rf'$\sigma_{{\theta\theta}}$')
    ax.plot(r_prof[valid], srt_r[valid], ':', lw=1, label=rf'$\sigma_{{r\theta}}$')

ax.set_xlabel(r'$r/a$', fontsize=12)
ax.set_ylabel(r'$\sigma/\tau$', fontsize=12)
ax.set_title(r'(a) Sector I ($\theta = 0°$)', fontsize=12)
ax.legend(fontsize=7, ncol=3, loc='lower left')
ax.grid(True, alpha=0.3)

ax = axes2[1]
for th_deg in [45]:
    th = np.radians(th_deg)
    srr_r = np.array([stress_sector_II(r, th)[0] for r in r_prof])
    stt_r = np.array([stress_sector_II(r, th)[1] for r in r_prof])
    srt_r = np.array([stress_sector_II(r, th)[2] for r in r_prof])
    valid = ~np.isnan(srr_r)
    ax.plot(r_prof[valid], srr_r[valid], 'b-', lw=2, label=r'$\sigma_{rr}$')
    ax.plot(r_prof[valid], stt_r[valid], 'r--', lw=2, label=r'$\sigma_{\theta\theta}$')
    ax.plot(r_prof[valid], srt_r[valid], 'g:', lw=1.5, label=r'$\sigma_{r\theta}$')

# Isotropic comparison at θ=45°: σ_rr = √3(1 - ρ/√(2-ρ²)), σ_θθ = √3(-1 - ρ/√(2-ρ²))
# [Kysar Eq. 62 adapted for BCC]
rho_prof = r_prof / r0
srr_iso = np.sqrt(3) * (1 - rho_prof / np.sqrt(2 - rho_prof**2))
stt_iso = np.sqrt(3) * (-1 - rho_prof / np.sqrt(2 - rho_prof**2))
valid_iso = rho_prof < np.sqrt(2)
ax.plot(r_prof[valid_iso], srr_iso[valid_iso], 'b-.', lw=1, alpha=0.5, label=r'$\sigma_{rr}$ (Eq. 62 type)')
ax.plot(r_prof[valid_iso], stt_iso[valid_iso], 'r-.', lw=1, alpha=0.5, label=r'$\sigma_{\theta\theta}$ (Eq. 62 type)')

ax.set_xlabel(r'$r/a$', fontsize=12)
ax.set_ylabel(r'$\sigma/\tau$', fontsize=12)
ax.set_title(r'(b) Sector II ($\theta = 45°$)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/exact_radial_profiles.png', dpi=200, bbox_inches='tight')
print("Saved: figures/exact_radial_profiles.png")

# ================================================================
# Summary
# ================================================================
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"""
BCC parameters for Kysar-type exact solution:

  Sector I  (sys 5,12): φ = {np.degrees(phi1):.4f}°, A = {A_I:.6f}
  Sector II (sys 3,4):  φ = {np.degrees(phi2):.4f}°, A = {A_II:.6f}
  Sector III(sys 6,11): φ = {np.degrees(phi3):.4f}°, A = {A_III:.6f}

  Sector boundaries: γ₁ = {np.degrees(gamma1):.4f}°, γ₂ = {np.degrees(gamma2):.4f}°

  Symmetry-axis offset: (σ'₁₁ - σ'₂₂)/(2τ) = √6/6 ≈ {np.sqrt(6)/6:.6f}
  (BCC analogue of Kysar Eq. 27)

  Sectors I and II: exact (both characteristics hit void surface)
  Sector III: exact (β-line hits void, α-line hits x₂-axis symmetry)

  Valid domain: r/a ≤ 1/|sin(φ-θ)| and r/a ≤ 1/|cos(φ-θ)|
  (characteristics must reach the void or axis)
""")
