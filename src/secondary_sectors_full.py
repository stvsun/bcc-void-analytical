"""
Complete secondary sector computation for BCC void problem.

Sign-corrected Airy formulas + full characteristic tracing.
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# BCC parameters
# ================================================================
gamma = 0.5 * np.arctan(2*np.sqrt(2))   # ≈ 35.26°
alpha_half = np.arctan(np.sqrt(2))        # ≈ 54.74°
phi_I = -gamma                            # Sector I rotation angle
phi_II = 0.0                              # Sector II rotation angle
phi_III = gamma                           # Sector III rotation angle

A_I = 2*np.sqrt(3)/3
A_II = np.sqrt(3)
A_III = 2*np.sqrt(3)/3

cg = np.cos(gamma); sg = np.sin(gamma)


# ================================================================
# Stress formulas (SIGN-CORRECTED)
# ================================================================
def stress_rotated_to_cartesian(s11p, s22p, s12p, phi):
    c2 = np.cos(2*phi); s2 = np.sin(2*phi)
    sig11 = (s11p + s22p)/2 + (s11p - s22p)/2*c2 - s12p*s2
    sig22 = (s11p + s22p)/2 - (s11p - s22p)/2*c2 + s12p*s2
    sig12 = (s11p - s22p)/2*s2 + s12p*c2
    return sig11, sig22, sig12


def cartesian_to_polar(sig11, sig22, sig12, theta):
    c2t = np.cos(2*theta); s2t = np.sin(2*theta)
    sm = (sig11 + sig22)/2
    X = (sig11 - sig22)/2
    Y = sig12
    srr = sm + X*c2t + Y*s2t
    stt = sm - X*c2t - Y*s2t
    srt = -X*s2t + Y*c2t
    return srr, stt, srt, sm


def primary_sector_I(r, theta):
    """Corrected Sector I: s22p NEGATED."""
    rho = r; arg = phi_I - theta
    d1 = 1 - rho**2 * np.sin(arg)**2
    d2 = 1 - rho**2 * np.cos(arg)**2
    if d1 <= 0 or d2 <= 0:
        return None
    s11p = A_I * rho * np.sin(arg) / np.sqrt(d1)
    s22p = -A_I * rho * np.cos(arg) / np.sqrt(d2)   # CORRECTED
    s12p = A_I
    return s11p, s22p, s12p


def primary_sector_II(r, theta):
    """Sector II: both components negated (horizontal/vertical chars)."""
    rho = r
    d1 = 1 - rho**2 * np.sin(theta)**2
    d2 = 1 - rho**2 * np.cos(theta)**2
    if d1 <= 0 or d2 <= 0:
        return None
    s11p = -A_II * rho * np.sin(theta) / np.sqrt(d1)
    s22p = -A_II * rho * np.cos(theta) / np.sqrt(d2)
    s12p = A_II
    return s11p, s22p, s12p


def primary_sector_III(r, theta):
    """Sector III: compute from void surface stress at r=a, then scale.

    Uses direct computation from the face equation for r=a (exact),
    with approximate r-scaling via the β-line formula for σ'₂₂.
    σ'₁₁ requires the reflected characteristic construction and is
    computed at r=a only (leading order).
    """
    rho = r; arg = phi_III - theta

    # σ'₂₂ from β-line to void (corrected, negated)
    d2 = 1 - rho**2 * np.cos(arg)**2
    if d2 <= 0:
        return None
    s22p = -A_III * rho * np.cos(arg) / np.sqrt(d2)

    # σ'₁₁: compute from void surface stress (exact at r=a)
    # Using face V₅→V₄: (-√6/3)X + (√3/6)Y = 1 and Y = X tan(2θ)
    c2t = np.cos(2*theta); s2t = np.sin(2*theta)
    if abs(c2t) < 1e-10:
        X_void = -np.sqrt(6)/2
        Y_void = 0
    else:
        denom = -2*np.sqrt(6) + np.sqrt(3)*np.tan(2*theta)
        if abs(denom) < 1e-10:
            return None
        X_void = 6 / denom
        Y_void = X_void * np.tan(2*theta)

    sm_void = -(X_void*c2t + Y_void*s2t)
    sig11_void = sm_void + X_void
    sig22_void = sm_void - X_void
    sig12_void = Y_void

    c2p = np.cos(2*phi_III); s2p = np.sin(2*phi_III)
    s11p_void = (sig11_void+sig22_void)/2 + (sig11_void-sig22_void)/2*c2p + sig12_void*s2p

    # Scale s11p from r=a to r using the α-line property (constant along α-lines)
    # At leading order, s11p is approximately constant for moderate r
    s11p = s11p_void  # Leading order approximation

    s12p = A_III
    return s11p, s22p, s12p


# ================================================================
# Curved boundary: α-line from void at θ = γ
# ================================================================
def curved_boundary_point(t):
    """Point on the curved boundary at parameter t ≥ 0."""
    x = cg + t * sg
    y = sg + t * cg
    return x, y


def is_in_secondary_sector(r, theta):
    """Check if (r, θ) is in the secondary sector (between θ=γ and curved boundary)."""
    if theta <= gamma or theta >= alpha_half:
        return False
    # Point in Cartesian
    xp = r * np.cos(theta); yp = r * np.sin(theta)
    # Check if below curved boundary (α-line from void at γ)
    # Curved boundary: y = sg + t*cg, x = cg + t*sg → t = (x - cg)/sg
    # At this t: y_cb = sg + ((x - cg)/sg)*cg = sg + cg*(x-cg)/sg
    if abs(sg) < 1e-10:
        return False
    t_at_x = (xp - cg) / sg
    if t_at_x < 0:
        return False
    y_cb = sg + t_at_x * cg
    return yp < y_cb  # Below the curved boundary


def secondary_sector_I_stress(r, theta):
    """
    Stress in the secondary (extended) Sector I.

    σ'₁₁: CONSTANT = value on the curved boundary (α-line from void at γ).
           Since the curved boundary is an α-line, σ'₁₁ is constant along it.
           Every point in the secondary sector's α-line traces back to
           the curved boundary (not the void), picking up this constant.
    σ'₂₂: from β-line backward to curved boundary (primary Sector I σ'₂₂)
    σ'₁₂ = A_I
    """
    # σ'₁₁ = constant on curved boundary
    # Computed at (a, γ): A_I * sin(φ_I - γ) / |cos(φ_I - γ)| = -4√6/3
    arg_cb = phi_I - gamma  # = -2γ
    s11p = A_I * np.sin(arg_cb) / np.sqrt(1 - np.sin(arg_cb)**2)

    # σ'₂₂: trace β-line backward to curved boundary
    # β-line backward from (r cosθ, r sinθ) at angle π - γ:
    # x(s) = r cosθ - s cosγ
    # y(s) = r sinθ + s sinγ
    #
    # Curved boundary: x = cg + u sg, y = sg + u cg
    # Intersection: cg + u sg = r cosθ - s cg
    #               sg + u cg = r sinθ + s sg
    # [cg, sg; -sg, cg] [s; u] = [r cosθ - cg; r sinθ - sg]

    xp = r * np.cos(theta); yp = r * np.sin(theta)
    dx = xp - cg; dy = yp - sg

    s_val = cg * dx - sg * dy
    u_val = sg * dx + cg * dy

    if u_val < -0.01:
        return None  # No valid intersection

    # Stress at intersection point on curved boundary
    x_int = cg + u_val * sg
    y_int = sg + u_val * cg
    r_int = np.sqrt(x_int**2 + y_int**2)
    theta_int = np.arctan2(y_int, x_int)

    rho_int = r_int
    arg_int = phi_I - theta_int
    d2_int = 1 - rho_int**2 * np.cos(arg_int)**2

    if d2_int <= 0:
        return None

    # σ'₂₂ from primary Sector I evaluated at the boundary point
    s22p = -A_I * rho_int * np.cos(arg_int) / np.sqrt(d2_int)

    return s11p, s22p, A_I


# ================================================================
# Full stress field computation
# ================================================================
def compute_stress(r, theta):
    """Compute polar stress at (r, θ) using correct sector."""
    if theta < 0 or theta > np.pi/2:
        return np.nan, np.nan, np.nan, np.nan

    # Mirror for Sectors IV-VI (by symmetry)
    if theta <= gamma:
        # Primary Sector I
        res = primary_sector_I(r, theta)
        phi = phi_I
    elif theta <= alpha_half:
        if is_in_secondary_sector(r, theta):
            # Secondary sector (extended Sector I)
            res = secondary_sector_I_stress(r, theta)
            phi = phi_I
        else:
            # Primary Sector II
            res = primary_sector_II(r, theta)
            phi = phi_II
    else:
        # Primary Sector III
        res = primary_sector_III(r, theta)
        phi = phi_III

    if res is None:
        return np.nan, np.nan, np.nan, np.nan

    s11p, s22p, s12p = res
    sig11, sig22, sig12 = stress_rotated_to_cartesian(s11p, s22p, s12p, phi)
    return cartesian_to_polar(sig11, sig22, sig12, theta)


# ================================================================
# Verification: void BC and continuity
# ================================================================
print("="*70)
print("VERIFICATION")
print("="*70)

print("\n1. Void surface BC (σ_rr = σ_rθ = 0):")
for theta_val in np.linspace(0.01, np.pi/2 - 0.01, 10):
    srr, stt, srt, sm = compute_stress(1.0, theta_val)
    if not np.isnan(srr):
        print(f"  θ={np.degrees(theta_val):6.2f}°  σ_rr={srr:10.6f}  σ_rθ={srt:10.6f}")

print("\n2. Continuity at θ = γ (primary Sector I → Sector II or secondary):")
eps = 1e-4
for r_val in [1.0, 1.02, 1.05, 1.1, 1.15, 1.2]:
    srr_m, stt_m, srt_m, sm_m = compute_stress(r_val, gamma - eps)
    srr_p, stt_p, srt_p, sm_p = compute_stress(r_val, gamma + eps)
    if not (np.isnan(sm_m) or np.isnan(sm_p)):
        in_sec = is_in_secondary_sector(r_val, gamma + eps)
        label = "secondary" if in_sec else "Sector II"
        print(f"  r/a={r_val:.2f}  σ_m(I)={sm_m:8.4f}  σ_m({label})={sm_p:8.4f}  "
              f"jump={abs(sm_m-sm_p):.6f}")

print("\n3. Continuity at curved boundary (secondary → primary Sector II):")
for t_val in [0.1, 0.3, 0.5, 1.0, 2.0]:
    x_cb, y_cb = curved_boundary_point(t_val)
    r_cb = np.sqrt(x_cb**2 + y_cb**2)
    th_cb = np.arctan2(y_cb, x_cb)

    # Just below boundary (secondary sector)
    dx_below = -0.001 * np.sin(th_cb)
    dy_below = 0.001 * np.cos(th_cb)  # Nudge inward (toward θ = γ)
    th_below = np.arctan2(y_cb - dy_below, x_cb + dx_below)  # decrease θ slightly
    th_below = th_cb - 0.001

    # Just above boundary (primary Sector II)
    th_above = th_cb + 0.001

    srr_b, stt_b, srt_b, sm_b = compute_stress(r_cb, th_below)
    srr_a, stt_a, srt_a, sm_a = compute_stress(r_cb, th_above)

    if not (np.isnan(sm_b) or np.isnan(sm_a)):
        print(f"  t={t_val:.1f}  r/a={r_cb:.3f}  θ={np.degrees(th_cb):.2f}°  "
              f"σ_m(sec)={sm_b:8.4f}  σ_m(II)={sm_a:8.4f}  jump={abs(sm_b-sm_a):.6f}")
    else:
        what = "sec=NaN" if np.isnan(sm_b) else "II=NaN"
        print(f"  t={t_val:.1f}  r/a={r_cb:.3f}  θ={np.degrees(th_cb):.2f}°  {what}")


# ================================================================
# Generate figure
# ================================================================
print("\n" + "="*70)
print("GENERATING FIGURE")
print("="*70)

fig, axes = plt.subplots(2, 2, figsize=(14, 12))

Nr, Nth = 300, 400
r_grid = np.linspace(1.001, 2.8, Nr)
theta_grid = np.linspace(0.001, np.pi/2 - 0.001, Nth)
R, Theta = np.meshgrid(r_grid, theta_grid)
X_cart = R * np.cos(Theta)
Y_cart = R * np.sin(Theta)

Srr = np.full_like(R, np.nan)
Stt = np.full_like(R, np.nan)
Srt = np.full_like(R, np.nan)
Sm = np.full_like(R, np.nan)

for i in range(Nth):
    for j in range(Nr):
        srr, stt, srt, sm = compute_stress(R[i, j], Theta[i, j])
        Srr[i, j] = srr
        Stt[i, j] = stt
        Srt[i, j] = srt
        Sm[i, j] = sm

# Curved boundary line
t_cb = np.linspace(0, 8, 300)
x_cb = cg + t_cb * sg
y_cb = sg + t_cb * cg
r_cb = np.sqrt(x_cb**2 + y_cb**2)
mask_cb = r_cb < 2.8

titles = [r'(a) $\sigma_{rr}/\tau_{\mathrm{CRSS}}$',
          r'(b) $\sigma_{\theta\theta}/\tau_{\mathrm{CRSS}}$',
          r'(c) $\sigma_{r\theta}/\tau_{\mathrm{CRSS}}$',
          r'(d) $\sigma_m/\tau_{\mathrm{CRSS}}$']
data = [Srr, Stt, Srt, Sm]
vmaxs = [4, 6, 3, 5]

for idx, (ax, title, Z, vm) in enumerate(zip(axes.flat, titles, data, vmaxs)):
    pc = ax.pcolormesh(X_cart, Y_cart, Z, cmap='RdBu_r', vmin=-vm, vmax=vm,
                       shading='auto', rasterized=True)
    fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)

    # Void circle
    th_v = np.linspace(0, np.pi/2, 200)
    ax.plot(np.cos(th_v), np.sin(th_v), 'k-', lw=2)

    # Radial sector boundaries
    for angle in [gamma, alpha_half]:
        ax.plot([0, 2.8*np.cos(angle)], [0, 2.8*np.sin(angle)],
                'k--', lw=0.8, alpha=0.4)

    # Curved boundary
    ax.plot(x_cb[mask_cb], y_cb[mask_cb], 'k-', lw=1.5, alpha=0.8)

    # r_crit boundary (from domain_of_validity)
    th_rc = np.linspace(0.01, gamma - 0.01, 100)
    r_rc = [1/abs(np.cos(phi_I - t)) for t in th_rc]
    r_rc = np.clip(r_rc, 1, 5)
    ax.plot(np.array(r_rc)*np.cos(th_rc), np.array(r_rc)*np.sin(th_rc),
            'k:', lw=1, alpha=0.5)

    ax.set_xlim(0, 2.5)
    ax.set_ylim(0, 2.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$x'_1/a$", fontsize=10)
    ax.set_ylabel(r"$x'_2/a$", fontsize=10)

plt.suptitle("Interior stress field with secondary sector (sign-corrected)",
             fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/secondary_sectors.png',
            dpi=200, bbox_inches='tight')
print("Saved: figures/secondary_sectors.png")

# Also update the main interior field figure
print("Regenerating exact_interior_kysar.png with corrected signs...")

fig2, axes2 = plt.subplots(2, 2, figsize=(12, 10))

# Full field with 4-fold symmetry
Nr2, Nth2 = 200, 360
r_grid2 = np.linspace(1.001, 2.5, Nr2)
theta_grid2 = np.linspace(0.001, 2*np.pi - 0.001, Nth2)
R2, Theta2 = np.meshgrid(r_grid2, theta_grid2)
X2 = R2 * np.cos(Theta2)
Y2 = R2 * np.sin(Theta2)

Srr2 = np.full_like(R2, np.nan)
Stt2 = np.full_like(R2, np.nan)
Srt2 = np.full_like(R2, np.nan)
Sm2 = np.full_like(R2, np.nan)

for i in range(Nth2):
    for j in range(Nr2):
        th = Theta2[i, j]
        # Reduce to first quadrant using symmetry
        th_red = th % np.pi
        if th_red > np.pi/2:
            th_red = np.pi - th_red
        srr, stt, srt, sm = compute_stress(R2[i, j], th_red)
        Srr2[i, j] = srr
        Stt2[i, j] = stt
        Srt2[i, j] = srt
        Sm2[i, j] = sm

titles2 = [r'(a) $\sigma_{rr}/\tau$', r'(b) $\sigma_{\theta\theta}/\tau$',
           r'(c) $\sigma_{r\theta}/\tau$', r'(d) $\sigma_m/\tau$']
vmaxs2 = [4, 6, 3, 5]

for idx, (ax, title, Z, vm) in enumerate(zip(axes2.flat, titles2, [Srr2, Stt2, Srt2, Sm2], vmaxs2)):
    pc = ax.pcolormesh(X2, Y2, Z, cmap='RdBu_r', vmin=-vm, vmax=vm,
                       shading='auto', rasterized=True)
    fig2.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)

    # Void circle
    th_v = np.linspace(0, 2*np.pi, 400)
    ax.plot(np.cos(th_v), np.sin(th_v), 'k-', lw=2)

    # Sector boundaries (all 12)
    for q in range(4):
        offset = q * np.pi/2
        for angle in [gamma, alpha_half]:
            a1 = angle + offset
            a2 = np.pi/2 - angle + offset
            ax.plot([0, 2.5*np.cos(a1)], [0, 2.5*np.sin(a1)], 'k--', lw=0.5, alpha=0.3)

    # Curved boundaries (all quadrants)
    for q in range(4):
        offset = q * np.pi/2
        for sign in [1, -1]:
            t_line = np.linspace(0, 5, 200)
            g_eff = gamma
            x_line = np.cos(g_eff + offset) + t_line * np.sin(g_eff + offset) * sign
            y_line = np.sin(g_eff + offset) + t_line * np.cos(g_eff + offset) * sign
            r_line = np.sqrt(x_line**2 + y_line**2)
            m = r_line < 2.5
            ax.plot(x_line[m], y_line[m], 'k-', lw=0.8, alpha=0.4)

    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)

plt.suptitle("Exact stress field around BCC void (corrected, with secondary sectors)",
             fontsize=13)
plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/exact_interior_kysar.png',
            dpi=200, bbox_inches='tight')
print("Saved: figures/exact_interior_kysar.png")
