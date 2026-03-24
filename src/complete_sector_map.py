"""
Complete sector map for BCC void: primary + secondary sectors.

Structure (first quadrant, by symmetry about θ = π/4):

Near void (r < r_crit):
  Sector I:   0 < θ < γ           (system 8, face V1→V6)
  Sector II:  γ < θ < α_half      (system 3, face V6→V5)
  Sector III: α_half < θ < π/2    (system 6, face V5→V4)

Far from void (r > r_crit):
  Secondary Ia: between θ = γ and curved boundary C₁ (α-line from void at γ)
     → system 8 still active, σ'₁₁ = const from C₁
  Secondary IIIa: between curved boundary C₂ (β-line from void at α_half)
     and θ = α_half → system 6 still active, σ'₂₂ = const from C₂
  Reduced Sector II: between C₁ and C₂
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

gamma = 0.5 * np.arctan(2*np.sqrt(2))
alpha_half = np.arctan(np.sqrt(2))
phi_I = -gamma; phi_II = 0.0; phi_III = gamma
A_I = 2*np.sqrt(3)/3; A_II = np.sqrt(3); A_III = 2*np.sqrt(3)/3
cg = np.cos(gamma); sg = np.sin(gamma)


def stress_to_polar(s11p, s22p, s12p, phi, theta):
    c2 = np.cos(2*phi); s2 = np.sin(2*phi)
    sig11 = (s11p+s22p)/2 + (s11p-s22p)/2*c2 - s12p*s2
    sig22 = (s11p+s22p)/2 - (s11p-s22p)/2*c2 + s12p*s2
    sig12 = (s11p-s22p)/2*s2 + s12p*c2
    c2t = np.cos(2*theta); s2t = np.sin(2*theta)
    sm = (sig11+sig22)/2; X = (sig11-sig22)/2; Y = sig12
    return sm+X*c2t+Y*s2t, sm-X*c2t-Y*s2t, -X*s2t+Y*c2t, sm


# ================================================================
# Curved boundaries
# ================================================================

# C₁: α-line from void at θ = γ (Sector I/II boundary)
# Direction: φ_I + π/2 = π/2 - γ, so (sinγ, cosγ)
# x(t) = cosγ + t sinγ, y(t) = sinγ + t cosγ
def C1_point(t):
    return cg + t*sg, sg + t*cg

# C₂: β-line from void at θ = α_half (Sector II/III boundary)
# β-line in Sector II is vertical (φ_II = 0, β at angle 90°)
# From (cos α_half, sin α_half) going upward:
# x = cos(α_half) = const, y = sin(α_half) + t
ca = np.cos(alpha_half); sa = np.sin(alpha_half)
def C2_point(t):
    return ca, sa + t


def is_in_secondary_Ia(x, y):
    """Between θ = γ (radial) and C₁ (α-line from void at γ)."""
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)
    if th <= gamma or th >= alpha_half or r <= 1.0:
        return False
    # Below C₁? C₁: y_cb(x) = sg + ((x - cg)/sg)*cg if x > cg
    if x < cg:
        return False
    t_at_x = (x - cg) / sg
    if t_at_x < 0:
        return False
    y_cb = sg + t_at_x * cg
    return y < y_cb


def is_in_secondary_IIIa(x, y):
    """Between C₂ (vertical at x = cos α_half) and θ = α_half (radial)."""
    r = np.sqrt(x**2 + y**2)
    th = np.arctan2(y, x)
    if th <= gamma or th >= alpha_half or r <= 1.0:
        return False
    # Right of C₂? C₂ is the vertical line x = cos(α_half)
    # The secondary IIIa sector is between C₂ and θ = α_half,
    # i.e., x < cos(α_half) (left of the vertical line) and θ < α_half
    # Wait, C₂ is from (cos α, sin α) going UP. Points between C₂
    # and the radial line θ = α_half have x > cos α (to the right of C₂)
    # but θ < α_half. Hmm, actually C₂ is a vertical line, and points
    # between it and θ = α_half are those with:
    # x < cos(α_half) / cos(θ) ... this is confusing. Let me think geometrically.
    #
    # The radial line θ = α_half: y = x tan(α_half) = x√2
    # C₂: x = cos(α_half) = 1/√3 (vertical line)
    # Region between: to the LEFT of C₂ and ABOVE the x-axis
    # i.e., x < 1/√3 and θ < α_half
    # BUT this is in Sector II (γ < θ < α_half), so we also need θ > γ.
    #
    # More precisely: the secondary III sector exists where the β-line
    # (vertical, from Sector II) has crossed θ = α_half before reaching the void.
    # A vertical β-line at x-position hits the void at y = √(1-x²) when x < 1.
    # It crosses θ = α_half at y = x√2.
    # The β-line crosses α_half BEFORE reaching the void when x√2 < √(1-x²):
    # 2x² < 1 - x², 3x² < 1, x < 1/√3 = cos(α_half)
    # So the secondary IIIa region has x < cos(α_half).
    return x < ca and th > gamma


# ================================================================
# Stress computation in each region
# ================================================================

def compute_stress_full(r_val, theta_val):
    """Full stress including secondary sectors."""
    x = r_val * np.cos(theta_val)
    y = r_val * np.sin(theta_val)

    if r_val <= 1.0 or theta_val <= 0 or theta_val >= np.pi/2:
        return np.nan, np.nan, np.nan, np.nan

    if theta_val < gamma:
        # Primary Sector I
        arg = phi_I - theta_val; rho = r_val
        d1 = 1 - rho**2*np.sin(arg)**2; d2 = 1 - rho**2*np.cos(arg)**2
        if d1 <= 0 or d2 <= 0: return np.nan, np.nan, np.nan, np.nan
        s11p = A_I*rho*np.sin(arg)/np.sqrt(d1)
        s22p = -A_I*rho*np.cos(arg)/np.sqrt(d2)
        return stress_to_polar(s11p, s22p, A_I, phi_I, theta_val)

    elif theta_val < alpha_half:
        if is_in_secondary_Ia(x, y):
            # Secondary Ia: extended Sector I
            arg_cb = phi_I - gamma
            s11p = A_I * np.sin(arg_cb) / np.sqrt(1 - np.sin(arg_cb)**2)
            # β-line backward to C₁
            dx = x - cg; dy = y - sg
            s_val = cg*dx - sg*dy; u_val = sg*dx + cg*dy
            if u_val < -0.01: return np.nan, np.nan, np.nan, np.nan
            x_int = cg + u_val*sg; y_int = sg + u_val*cg
            r_int = np.sqrt(x_int**2 + y_int**2)
            th_int = np.arctan2(y_int, x_int)
            arg_int = phi_I - th_int
            d2 = 1 - r_int**2*np.cos(arg_int)**2
            if d2 <= 0: return np.nan, np.nan, np.nan, np.nan
            s22p = -A_I*r_int*np.cos(arg_int)/np.sqrt(d2)
            return stress_to_polar(s11p, s22p, A_I, phi_I, theta_val)

        elif is_in_secondary_IIIa(x, y):
            # Secondary IIIa: extended Sector III
            # σ'₂₂ = const from C₂ (β-line from void at α_half)
            arg_cb = phi_III - alpha_half
            s22p = -A_III * np.cos(arg_cb) / np.sqrt(1 - np.cos(arg_cb)**2)
            # σ'₁₁: from α-line backward to void (Sector III formula)
            # Actually, in the secondary IIIa, the active system is Sector III's,
            # so the α-line goes at angle φ_III + π/2 = γ + π/2
            # Tracing backward from (r, θ) at angle γ - π/2 hits the void.
            arg = phi_III - theta_val; rho = r_val
            d1 = 1 - rho**2*np.sin(arg)**2
            if d1 <= 0: return np.nan, np.nan, np.nan, np.nan
            s11p = A_III*rho*np.sin(arg)/np.sqrt(d1)
            return stress_to_polar(s11p, s22p, A_III, phi_III, theta_val)

        else:
            # Primary (reduced) Sector II
            rho = r_val
            d1 = 1 - rho**2*np.sin(theta_val)**2
            d2 = 1 - rho**2*np.cos(theta_val)**2
            if d1 <= 0 or d2 <= 0: return np.nan, np.nan, np.nan, np.nan
            s11p = -A_II*rho*np.sin(theta_val)/np.sqrt(d1)
            s22p = -A_II*rho*np.cos(theta_val)/np.sqrt(d2)
            return stress_to_polar(s11p, s22p, A_II, phi_II, theta_val)

    else:
        # Primary Sector III
        arg = phi_III - theta_val; rho = r_val
        d1 = 1 - rho**2*np.sin(arg)**2; d2 = 1 - rho**2*np.cos(arg)**2
        if d1 <= 0 or d2 <= 0: return np.nan, np.nan, np.nan, np.nan
        s11p = A_III*rho*np.sin(arg)/np.sqrt(d1)
        s22p = -A_III*rho*np.cos(arg)/np.sqrt(d2)
        return stress_to_polar(s11p, s22p, A_III, phi_III, theta_val)


# ================================================================
# Verification
# ================================================================
print("="*70)
print("VERIFICATION")
print("="*70)

print("\n1. Void surface BC:")
for th in np.linspace(0.01, np.pi/2 - 0.01, 15):
    srr, stt, srt, sm = compute_stress_full(1.0001, th)
    if not np.isnan(srr):
        ok = "✓" if abs(srr) < 0.01 and abs(srt) < 0.01 else "✗"
        print(f"  θ={np.degrees(th):6.2f}°  σ_rr={srr:9.5f}  σ_rθ={srt:9.5f}  {ok}")

print("\n2. Sector identification at r = 1.5a:")
for th_deg in range(1, 90, 5):
    th = np.radians(th_deg)
    x = 1.5*np.cos(th); y = 1.5*np.sin(th)
    sec = "I" if th < gamma else ("III" if th > alpha_half else "II")
    if gamma < th < alpha_half:
        if is_in_secondary_Ia(x, y):
            sec = "Ia(sec)"
        elif is_in_secondary_IIIa(x, y):
            sec = "IIIa(sec)"
        else:
            sec = "II"
    srr, stt, srt, sm = compute_stress_full(1.5, th)
    status = f"σ_m={sm:7.3f}" if not np.isnan(sm) else "NaN"
    print(f"  θ={th_deg:3d}°  sector={sec:10s}  {status}")

# ================================================================
# Figure: complete sector map
# ================================================================
print("\n" + "="*70)
print("GENERATING COMPLETE SECTOR MAP")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

# Panel (a): Sector map with boundaries
ax = axes[0]
th_void = np.linspace(0, np.pi/2, 200)
ax.plot(np.cos(th_void), np.sin(th_void), 'k-', lw=2.5)

# Radial sector boundaries
for angle, label in [(gamma, r'$\theta_1$'), (alpha_half, r'$\theta_2$')]:
    ax.plot([0, 3*np.cos(angle)], [0, 3*np.sin(angle)], 'k--', lw=1, alpha=0.5)

# C₁: α-line from void at γ
t_c1 = np.linspace(0, 5, 200)
x_c1, y_c1 = zip(*[C1_point(t) for t in t_c1])
x_c1, y_c1 = np.array(x_c1), np.array(y_c1)
r_c1 = np.sqrt(x_c1**2 + y_c1**2)
m = r_c1 < 2.8
ax.plot(x_c1[m], y_c1[m], 'b-', lw=2, label=r'$C_1$ ($\alpha$-line from $\gamma$)')

# C₂: β-line from void at α_half (vertical)
t_c2 = np.linspace(0, 3, 200)
x_c2 = np.full_like(t_c2, ca)
y_c2 = np.array([sa + t for t in t_c2])
r_c2 = np.sqrt(x_c2**2 + y_c2**2)
m2 = r_c2 < 2.8
ax.plot(x_c2[m2], y_c2[m2], 'r-', lw=2, label=r'$C_2$ ($\beta$-line from $\alpha$)')

# Shade primary sectors
th_I = np.linspace(0, gamma, 50); r_max = 2.5
ax.fill_between(r_max*np.cos(th_I), 0, r_max*np.sin(th_I), alpha=0.08, color='blue')
th_III = np.linspace(alpha_half, np.pi/2, 50)
ax.fill_between(r_max*np.cos(th_III), 0, r_max*np.sin(th_III), alpha=0.08, color='green')

# Labels
ax.text(1.8*np.cos(np.radians(17)), 1.8*np.sin(np.radians(17)), 'I', fontsize=16,
        fontweight='bold', color='blue', ha='center')
ax.text(0.6, 1.1, 'Ia\n(sec)', fontsize=10, color='blue', ha='center',
        bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='blue'))
ax.text(1.5*np.cos(np.radians(45)), 1.5*np.sin(np.radians(45)), 'II', fontsize=16,
        fontweight='bold', color='red', ha='center')
ax.text(0.3, 1.5, 'IIIa\n(sec)', fontsize=10, color='green', ha='center',
        bbox=dict(facecolor='lightyellow', alpha=0.8, edgecolor='green'))
ax.text(1.5*np.cos(np.radians(73)), 1.5*np.sin(np.radians(73)), 'III', fontsize=16,
        fontweight='bold', color='green', ha='center')

ax.set_xlim(-0.1, 2.5); ax.set_ylim(-0.1, 2.5)
ax.set_aspect('equal')
ax.set_xlabel(r"$x'_1/a$", fontsize=12)
ax.set_ylabel(r"$x'_2/a$", fontsize=12)
ax.set_title("(a) Complete sector map", fontsize=12)
ax.legend(fontsize=8, loc='upper left')

# Panel (b): σ_m contour
ax = axes[1]
Nr, Nth = 250, 350
r_grid = np.linspace(1.001, 2.5, Nr)
theta_grid = np.linspace(0.001, np.pi/2 - 0.001, Nth)
R, Theta = np.meshgrid(r_grid, theta_grid)
Sm = np.full_like(R, np.nan)
for i in range(Nth):
    for j in range(Nr):
        _, _, _, sm = compute_stress_full(R[i,j], Theta[i,j])
        Sm[i,j] = sm

X_c = R*np.cos(Theta); Y_c = R*np.sin(Theta)
pc = ax.pcolormesh(X_c, Y_c, Sm, cmap='RdBu_r', vmin=-6, vmax=0,
                   shading='auto', rasterized=True)
fig.colorbar(pc, ax=ax, shrink=0.8)

ax.plot(np.cos(th_void), np.sin(th_void), 'k-', lw=2)
for angle in [gamma, alpha_half]:
    ax.plot([0, 2.5*np.cos(angle)], [0, 2.5*np.sin(angle)], 'k--', lw=0.5, alpha=0.5)
ax.plot(x_c1[m], y_c1[m], 'w-', lw=1.5, alpha=0.8)
ax.plot(x_c2[m2], y_c2[m2], 'w-', lw=1.5, alpha=0.8)

ax.set_xlim(0, 2.2); ax.set_ylim(0, 2.2)
ax.set_aspect('equal')
ax.set_xlabel(r"$x'_1/a$", fontsize=12)
ax.set_title(r"(b) $\sigma_m / \tau_{\mathrm{CRSS}}$", fontsize=12)

# Panel (c): σ_θθ contour
ax = axes[2]
Stt = np.full_like(R, np.nan)
for i in range(Nth):
    for j in range(Nr):
        _, stt, _, _ = compute_stress_full(R[i,j], Theta[i,j])
        Stt[i,j] = stt

pc = ax.pcolormesh(X_c, Y_c, Stt, cmap='RdBu_r', vmin=-8, vmax=0,
                   shading='auto', rasterized=True)
fig.colorbar(pc, ax=ax, shrink=0.8)

ax.plot(np.cos(th_void), np.sin(th_void), 'k-', lw=2)
for angle in [gamma, alpha_half]:
    ax.plot([0, 2.5*np.cos(angle)], [0, 2.5*np.sin(angle)], 'k--', lw=0.5, alpha=0.5)
ax.plot(x_c1[m], y_c1[m], 'w-', lw=1.5, alpha=0.8)
ax.plot(x_c2[m2], y_c2[m2], 'w-', lw=1.5, alpha=0.8)

ax.set_xlim(0, 2.2); ax.set_ylim(0, 2.2)
ax.set_aspect('equal')
ax.set_xlabel(r"$x'_1/a$", fontsize=12)
ax.set_title(r"(c) $\sigma_{\theta\theta} / \tau_{\mathrm{CRSS}}$", fontsize=12)

plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/complete_sector_map.png',
            dpi=200, bbox_inches='tight')
print("Saved: figures/complete_sector_map.png")
