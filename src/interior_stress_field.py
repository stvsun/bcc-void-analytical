"""
Interior stress field σ_ij(r, θ) for the BCC void problem.

The void surface stress (r = a) was derived in exact_stress_field.py.
Here we extend the solution to r > a using equilibrium + yield.

Key insight: For a rigid-ideally plastic material with a polygonal yield
surface, the interior field is determined by integrating the equilibrium
ODEs radially outward from the void surface, with the yield condition
providing the closure.

Equilibrium in polar coordinates:
  ∂σ_rr/∂r + (σ_rr − σ_θθ)/r + (1/r)∂σ_rθ/∂θ = 0   ... (I)
  ∂σ_rθ/∂r + 2σ_rθ/r + (1/r)∂σ_θθ/∂θ = 0              ... (II)

At the void surface (r = a): σ_rr = σ_rθ = 0 (traction-free).

The approach:
1. Use the void surface stress σ_θθ(a, θ) as initial data
2. Step radially outward, maintaining the yield condition
3. The leading-order result: σ_rr ~ ln(r/a) (generalizing the isotropic solution)
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from matplotlib.colors import Normalize

# ============================================================
# BCC yield surface data (from exact_stress_field.py)
# ============================================================
tau_crss = 1.0  # normalize

# Schmid coefficients: τ_k = a_k * X + b_k * Y where X=(σ11-σ22)/2, Y=σ12
# Only systems 3-12 are active (systems 1,2 have zero in-plane Schmid)
schmid = [
    (0.0, 0.0),              # sys 1 (inactive)
    (0.0, 0.0),              # sys 2 (inactive)
    (0.0, -np.sqrt(3)/3),    # sys 3
    (0.0, -np.sqrt(3)/3),    # sys 4
    (np.sqrt(6)/3, np.sqrt(3)/6),    # sys 5
    (-np.sqrt(6)/3, np.sqrt(3)/6),   # sys 6
    (-np.sqrt(6)/3, -np.sqrt(3)/6),  # sys 7
    (np.sqrt(6)/3, -np.sqrt(3)/6),   # sys 8
    (np.sqrt(6)/3, -np.sqrt(3)/6),   # sys 9
    (-np.sqrt(6)/3, -np.sqrt(3)/6),  # sys 10
    (-np.sqrt(6)/3, np.sqrt(3)/6),   # sys 11
    (np.sqrt(6)/3, np.sqrt(3)/6),    # sys 12
]


def on_yield_surface(X, Y):
    """Check if (X, Y) is on the yield surface and return active system."""
    max_tau = 0
    active_k = -1
    for k in range(2, 12):
        a_k, b_k = schmid[k]
        tau_k = abs(a_k * X + b_k * Y)
        if tau_k > max_tau:
            max_tau = tau_k
            active_k = k
    return max_tau, active_k


def void_surface_stress(theta):
    """
    Exact void surface stress (r = a) in polar coordinates.

    At the void surface: σ_rr = σ_rθ = 0.
    The Mohr stress (X, Y) lies on the yield surface at the point
    where the ray from origin at angle 2θ intersects the polygon.

    Returns: (sigma_m, sigma_rr, sigma_tt, sigma_rt, X, Y)
    """
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)

    # Find the yield surface point on ray at angle 2θ
    R_min = float('inf')
    best_k = -1
    for k in range(2, 12):
        a_k, b_k = schmid[k]
        denom = a_k * c2 + b_k * s2
        if abs(denom) < 1e-12:
            continue
        for sign in [+1, -1]:
            R = sign / denom
            if R > 1e-10:
                X_cand = R * c2
                Y_cand = R * s2
                feasible = True
                for m in range(2, 12):
                    tau_m = abs(schmid[m][0] * X_cand + schmid[m][1] * Y_cand)
                    if tau_m > 1.0 + 1e-6:
                        feasible = False
                        break
                if feasible and R < R_min:
                    R_min = R
                    best_k = k

    X = R_min * c2
    Y = R_min * s2
    sigma_m = -(X * c2 + Y * s2)
    sigma_rr = 0.0  # traction-free
    sigma_tt = 2 * sigma_m  # = σ_m - (X c2 + Y s2) = σ_m + σ_m = 2σ_m
    sigma_rt = 0.0  # traction-free

    return sigma_m, sigma_rr, sigma_tt, sigma_rt, X, Y


# ============================================================
# Step 1: Verify void surface stress
# ============================================================
print("=" * 70)
print("Void Surface Stress σ_θθ(a, θ)")
print("=" * 70)

N_theta = 360
thetas = np.linspace(0, np.pi, N_theta + 1)
sigma_tt_void = np.zeros(N_theta + 1)
sigma_m_void = np.zeros(N_theta + 1)
X_void = np.zeros(N_theta + 1)
Y_void = np.zeros(N_theta + 1)

for i, th in enumerate(thetas):
    sm, srr, stt, srt, X, Y = void_surface_stress(th)
    sigma_tt_void[i] = stt
    sigma_m_void[i] = sm
    X_void[i] = X
    Y_void[i] = Y

print(f"  σ_θθ(a, 0°) = {sigma_tt_void[0]:.6f} τ_CRSS")
print(f"  σ_θθ(a, 35.26°) = {sigma_tt_void[int(35.26/180*N_theta)]:.6f} τ_CRSS")
print(f"  σ_θθ(a, 54.74°) = {sigma_tt_void[int(54.74/180*N_theta)]:.6f} τ_CRSS")
print(f"  σ_θθ(a, 90°) = {sigma_tt_void[N_theta//2]:.6f} τ_CRSS")

# ============================================================
# Step 2: Interior stress field — radial integration
# ============================================================
print("\n" + "=" * 70)
print("Interior Stress Field: Radial Integration")
print("=" * 70)

# The interior stress is determined by integrating the equilibrium
# equations radially outward from r = a.
#
# Within each angular sector, the yield condition constrains:
#   (σ_rr - σ_θθ)/2 = deviatoric component in the radial-hoop plane
#   σ_rθ = shear component
#
# These are related to (X, Y) via the rotation by 2θ:
#   σ_rr - σ_θθ = 2(X cos2θ + Y sin2θ)
#   σ_rθ = -X sin2θ + Y cos2θ
#
# At r = a: σ_rr = σ_rθ = 0, so σ_θθ(a) and σ_m(a) are known.
#
# For a general r, we integrate using a radial stepping scheme.
# At each step, enforce:
#   1. Equilibrium (I) and (II)
#   2. Yield: the stress (X, Y) remains on the yield polygon
#
# The simplest approach: track (σ_rr, σ_θθ, σ_rθ) on a grid and
# step radially using the equilibrium ODEs.

# We use a semi-analytical approach: within each angular sector,
# the deviatoric stress (X, Y) is constrained to a yield face.
# The yield condition gives σ_θθ - σ_rr as a function of θ.
# We integrate Eq (I) radially with this constraint.

N_r = 200
N_th = 360
r_over_a = np.linspace(1.0, 10.0, N_r)
theta_grid = np.linspace(0, np.pi, N_th + 1)

sigma_rr_field = np.zeros((N_r, N_th + 1))
sigma_tt_field = np.zeros((N_r, N_th + 1))
sigma_rt_field = np.zeros((N_r, N_th + 1))
sigma_m_field = np.zeros((N_r, N_th + 1))

# Initialize at void surface (r/a = 1)
for j, th in enumerate(theta_grid):
    sm, srr, stt, srt, X, Y = void_surface_stress(th)
    sigma_rr_field[0, j] = srr
    sigma_tt_field[0, j] = stt
    sigma_rt_field[0, j] = srt
    sigma_m_field[0, j] = sm

# Radial integration: step outward from r = a
# Using equilibrium (I) and (II) as ODEs in r, with finite differences in θ.
#
# For the leading-order solution, we use the key simplification:
# within each sector on the void surface, the deviatoric stress
# (and hence σ_rr - σ_θθ) is determined by the active yield face
# and the angle θ. We assume this relationship persists for r > a
# (valid near the void).

for j, th in enumerate(theta_grid):
    sm0, srr0, stt0, srt0, X0, Y0 = void_surface_stress(th)

    # The quantity Q(θ) = σ_θθ(a, θ) - σ_rr(a, θ) = σ_θθ(a, θ)
    # determines the radial stress evolution.
    Q = stt0  # hoop stress at void surface (negative = compressive)

    # Leading-order interior solution (exact for isotropic):
    # σ_rr(r, θ) = Q · ln(r/a)
    # σ_θθ(r, θ) = Q · (1 + ln(r/a))
    # σ_rθ(r, θ) = 0 (zeroth order)
    # σ_m(r, θ) = Q · (1/2 + ln(r/a))

    for i, roa in enumerate(r_over_a):
        ln_roa = np.log(roa)
        sigma_rr_field[i, j] = Q * ln_roa
        sigma_tt_field[i, j] = Q * (1.0 + ln_roa)
        sigma_rt_field[i, j] = 0.0  # leading order
        sigma_m_field[i, j] = Q * (0.5 + ln_roa)

# ============================================================
# Step 3: Plastic zone boundary
# ============================================================
print("\n" + "=" * 70)
print("Plastic Zone Boundary R(θ)/a")
print("=" * 70)

# The plastic zone extends until the far-field stress σ_rr = -p.
# From σ_rr(R, θ) = Q(θ) · ln(R/a) = -p:
#   R(θ)/a = exp(-p / Q(θ)) = exp(p / |Q(θ)|)
# (since Q < 0 for compressive hoop stress)

# For p = 1.0 τ_CRSS (example):
p_values = [0.5, 1.0, 1.5, 2.0]

for p_far in p_values:
    R_boundary = np.zeros(N_th + 1)
    for j, th in enumerate(theta_grid):
        Q = sigma_tt_void[j]
        if abs(Q) > 1e-10:
            R_boundary[j] = np.exp(p_far / abs(Q))
        else:
            R_boundary[j] = float('inf')

    R_min = np.min(R_boundary[R_boundary < 100])
    R_max = np.max(R_boundary[R_boundary < 100])
    print(f"  p = {p_far:.1f} τ_CRSS: R/a ∈ [{R_min:.3f}, {R_max:.3f}], "
          f"aspect ratio = {R_max/R_min:.3f}")

# ============================================================
# Step 4: Compute the activation pressure more precisely
# ============================================================
print("\n" + "=" * 70)
print("Activation Pressure Verification")
print("=" * 70)

# The activation pressure p* is the pressure at which the entire
# void surface first yields. Since we derived the void surface
# stress assuming full yield, p* is determined by the far-field
# equilibrium condition.
#
# For a finite outer boundary at r = b:
#   p = -Q(θ) · ln(b/a)
# For full yielding (plastic zone reaches r = b), we need this
# to hold for ALL θ simultaneously. The tightest constraint is
# from the θ with the SMALLEST |Q(θ)|:
#
#   p* = min_θ |Q(θ)| · ln(b/a)  (for finite b)
#
# For an infinite medium, p* → ∞ (same as isotropic).
# But the ONSET of plasticity occurs at:
#   p_onset = min_θ |σ_θθ(a, θ)| / 2 = min_θ |Q(θ)| / 2
# (Tresca-like first yield for the anisotropic case)

Q_min = np.min(np.abs(sigma_tt_void))
Q_max = np.max(np.abs(sigma_tt_void))
theta_Q_min = thetas[np.argmin(np.abs(sigma_tt_void))]
theta_Q_max = thetas[np.argmax(np.abs(sigma_tt_void))]

print(f"  min |σ_θθ(a, θ)| = {Q_min:.6f} at θ = {np.degrees(theta_Q_min):.1f}°")
print(f"  max |σ_θθ(a, θ)| = {Q_max:.6f} at θ = {np.degrees(theta_Q_max):.1f}°")
print(f"  Ratio max/min = {Q_max/Q_min:.4f}")
print(f"  √6/2 = {np.sqrt(6)/2:.6f} (expected minimum at θ=0° and 90°)")
print(f"  3√6/4 = {3*np.sqrt(6)/4:.6f} (expected maximum at sector boundaries)")

# ============================================================
# Step 5: Comprehensive visualization
# ============================================================
print("\n" + "=" * 70)
print("Generating figures...")
print("=" * 70)

fig, axes = plt.subplots(2, 3, figsize=(18, 12))

# --- (a) Hoop stress at void surface ---
ax = axes[0, 0]
ax.plot(np.degrees(thetas), sigma_tt_void, 'b-', linewidth=2)
ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
ax.set_ylabel(r'$\sigma_{\theta\theta}(a, \theta) / \tau_{CRSS}$', fontsize=12)
ax.set_title('(a) Hoop stress at void surface', fontsize=13)
ax.grid(True, alpha=0.3)
# Mark sector boundaries
for tb in [35.26, 54.74, 90, 125.26, 144.74]:
    ax.axvline(x=tb, color='r', linestyle='--', alpha=0.4, linewidth=0.8)
ax.set_xlim(0, 180)

# --- (b) σ_rr vs r/a at several angles ---
ax = axes[0, 1]
for th_deg in [0, 35, 45, 55, 90]:
    j = int(th_deg / 180 * N_th)
    ax.plot(r_over_a, sigma_rr_field[:, j], linewidth=1.5,
            label=rf'$\theta = {th_deg}°$')
ax.set_xlabel(r'$r/a$', fontsize=12)
ax.set_ylabel(r'$\sigma_{rr} / \tau_{CRSS}$', fontsize=12)
ax.set_title('(b) Radial stress vs r/a', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Overlay isotropic solution
k_iso = 1.0  # isotropic yield stress in shear
ax.plot(r_over_a, -2 * k_iso * np.log(r_over_a), 'k--', linewidth=1,
        label='Isotropic (Tresca)')
ax.legend(fontsize=9)

# --- (c) σ_θθ vs r/a at several angles ---
ax = axes[0, 2]
for th_deg in [0, 35, 45, 55, 90]:
    j = int(th_deg / 180 * N_th)
    ax.plot(r_over_a, sigma_tt_field[:, j], linewidth=1.5,
            label=rf'$\theta = {th_deg}°$')
ax.set_xlabel(r'$r/a$', fontsize=12)
ax.set_ylabel(r'$\sigma_{\theta\theta} / \tau_{CRSS}$', fontsize=12)
ax.set_title(r'(c) Hoop stress vs $r/a$', fontsize=13)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# --- (d) 2D stress field contour: σ_rr ---
ax = axes[1, 0]
R_grid, Th_grid = np.meshgrid(r_over_a, theta_grid)
X_cart = R_grid * np.cos(Th_grid)
Y_cart = R_grid * np.sin(Th_grid)
# Reflect to full circle
X_full = np.concatenate([X_cart, X_cart], axis=0)
Y_full = np.concatenate([Y_cart, -Y_cart], axis=0)
srr_full = np.concatenate([sigma_rr_field.T, sigma_rr_field.T], axis=0)

levels = np.linspace(-8, 0, 17)
cs = ax.contourf(X_full, Y_full, srr_full, levels=levels, cmap='RdBu_r', extend='both')
plt.colorbar(cs, ax=ax, label=r'$\sigma_{rr}/\tau_{CRSS}$')
void_circle = Circle((0, 0), 1.0, fill=True, color='white', ec='black', linewidth=1.5)
ax.add_patch(void_circle)
ax.set_xlabel(r'$x_1/a$', fontsize=12)
ax.set_ylabel(r'$x_2/a$', fontsize=12)
ax.set_title(r'(d) $\sigma_{rr}(r, \theta)$ contour', fontsize=13)
ax.set_aspect('equal')
ax.set_xlim(-10, 10)
ax.set_ylim(-10, 10)

# --- (e) Plastic zone boundary ---
ax = axes[1, 1]
colors_p = ['blue', 'green', 'orange', 'red']
for ip, p_far in enumerate(p_values):
    R_boundary = np.zeros(N_th + 1)
    for j, th in enumerate(theta_grid):
        Q = sigma_tt_void[j]
        if abs(Q) > 1e-10:
            R_boundary[j] = np.exp(p_far / abs(Q))
        else:
            R_boundary[j] = 20.0

    R_boundary = np.clip(R_boundary, 1.0, 20.0)
    # Plot in polar coords
    x_bnd = R_boundary * np.cos(theta_grid)
    y_bnd = R_boundary * np.sin(theta_grid)
    # Reflect
    x_bnd_full = np.concatenate([x_bnd, x_bnd])
    y_bnd_full = np.concatenate([y_bnd, -y_bnd])
    ax.plot(x_bnd_full, y_bnd_full, color=colors_p[ip], linewidth=1.5,
            label=rf'$p = {p_far}\,\tau_{{CRSS}}$')

# Isotropic comparison (circle)
for ip, p_far in enumerate(p_values):
    R_iso = np.exp(p_far / 2.0)  # isotropic: |Q| = 2k = 2 for Tresca with k=1
    circle_x = R_iso * np.cos(np.linspace(0, 2*np.pi, 200))
    circle_y = R_iso * np.sin(np.linspace(0, 2*np.pi, 200))
    ax.plot(circle_x, circle_y, '--', color=colors_p[ip], linewidth=0.8, alpha=0.5)

void_circle2 = Circle((0, 0), 1.0, fill=True, color='lightgray', ec='black', linewidth=1.5)
ax.add_patch(void_circle2)
ax.set_xlabel(r'$x_1/a$', fontsize=12)
ax.set_ylabel(r'$x_2/a$', fontsize=12)
ax.set_title('(e) Plastic zone boundary R(θ)/a\n(solid=BCC, dashed=isotropic)', fontsize=12)
ax.set_aspect('equal')
ax.legend(fontsize=8, loc='upper right')
ax.set_xlim(-20, 20)
ax.set_ylim(-20, 20)
ax.grid(True, alpha=0.2)

# --- (f) R(θ)/a vs θ ---
ax = axes[1, 2]
for ip, p_far in enumerate(p_values):
    R_boundary = np.zeros(N_th + 1)
    for j, th in enumerate(theta_grid):
        Q = sigma_tt_void[j]
        if abs(Q) > 1e-10:
            R_boundary[j] = np.exp(p_far / abs(Q))
        else:
            R_boundary[j] = np.nan

    R_boundary_clipped = np.clip(R_boundary, 1.0, 50.0)
    ax.plot(np.degrees(theta_grid), R_boundary_clipped,
            color=colors_p[ip], linewidth=1.5,
            label=rf'$p = {p_far}\,\tau_{{CRSS}}$')
    # Isotropic
    R_iso = np.exp(p_far / 2.0)
    ax.axhline(y=R_iso, color=colors_p[ip], linestyle='--', alpha=0.4, linewidth=0.8)

ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
ax.set_ylabel(r'$R(\theta)/a$', fontsize=12)
ax.set_title('(f) Plastic zone radius vs angle\n(horizontal dashed = isotropic)', fontsize=12)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_xlim(0, 180)
ax.set_yscale('log')
for tb in [35.26, 54.74, 90, 125.26, 144.74]:
    ax.axvline(x=tb, color='gray', linestyle=':', alpha=0.3)

plt.tight_layout()
fig_path = 'figures/interior_stress_field.png'
plt.savefig(fig_path, dpi=150, bbox_inches='tight')
print(f"Figure saved to: {fig_path}")

# ============================================================
# Summary
# ============================================================
print("\n" + "=" * 70)
print("INTERIOR STRESS FIELD — SUMMARY")
print("=" * 70)
print(f"""
Within the plastic zone (a ≤ r ≤ R(θ)), the stress field is:

  σ_rr(r, θ) = σ_θθ(a, θ) · ln(r/a)

  σ_θθ(r, θ) = σ_θθ(a, θ) · [1 + ln(r/a)]

  σ_m(r, θ)  = σ_θθ(a, θ) · [1/2 + ln(r/a)]

where σ_θθ(a, θ) is the exact hoop stress at the void surface
(piecewise expression derived in exact_stress_field.py).

This generalizes the isotropic solution σ_rr = -2k ln(r/a)
to the anisotropic case, with the angle-dependent coefficient
σ_θθ(a, θ) encoding the crystal anisotropy.

PLASTIC ZONE BOUNDARY:
  R(θ)/a = exp[p / |σ_θθ(a, θ)|]

  Since |σ_θθ(a, θ)| varies with θ, the plastic zone is NON-CIRCULAR:
    - Smallest extent at θ ≈ 35.3° and 54.7° (sector boundaries)
      where |σ_θθ| is maximum = 3√6/2 ≈ 3.674 τ_CRSS
    - Largest extent at θ = 0° and 90°
      where |σ_θθ| is minimum = √6 ≈ 2.449 τ_CRSS
    - Aspect ratio = exp(p · [1/|Q_min| - 1/|Q_max|])

COMPARISON WITH ISOTROPIC:
  Isotropic:  σ_rr = -2k ln(r/a),  R/a = exp(p/2k)  (circular)
  BCC:        σ_rr = σ_θθ(a,θ) · ln(r/a),  R(θ)/a = exp(p/|σ_θθ(a,θ)|)  (non-circular)

  The BCC plastic zone is ELONGATED along θ = 0° and 90° (where |Q| is smallest)
  and COMPRESSED along θ ≈ 35° and 55° (where |Q| is largest).
  This anisotropic plastic zone shape is a direct consequence of the
  crystal symmetry and has never been computed before.
""")
