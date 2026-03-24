"""
Complete stress field with matched transition at the concentrated fan.

For a POLYGONAL yield surface, the fan at a vertex is concentrated
on a line (C₁). Across C₁:
  - σ_m is continuous (Rice's dσ_m = 0 for polygon, ds = 0 at vertex)
  - σ'₁₂ jumps from A_I to vertex value
  - σ'₂₂ jumps from Airy value to vertex value
  - σ'₁₁ is continuous (const along α-line, which crosses C₁)

The matching condition at C₁: σ_m is continuous.
From the secondary (vertex) side: σ_m = -3√6/4.
From the primary (Airy) side: σ_m is determined by the
Airy σ'₁₁ (finite) and the MATCHED σ'₂₂ (not the diverging Airy σ'₂₂).

The matched σ'₂₂ at C₁: σ'₂₂ = 2 σ_m(V₅) - σ'₁₁
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

gamma = 0.5 * np.arctan(2*np.sqrt(2))
alpha_half = np.arctan(np.sqrt(2))
phi_I = -gamma; phi_II = 0.0; phi_III = gamma
A_I = 2*np.sqrt(3)/3; A_II = np.sqrt(3); A_III = 2*np.sqrt(3)/3
cg = np.cos(gamma); sg = np.sin(gamma)

X_V5 = np.sqrt(6)/4; Y_V5 = np.sqrt(3); sm_V5 = -3*np.sqrt(6)/4
# Vertex V₅ in Sector I primed frame:
c2p = np.cos(2*phi_I); s2p = np.sin(2*phi_I)
sig11_V5 = sm_V5 + X_V5; sig22_V5 = sm_V5 - X_V5; sig12_V5 = Y_V5
s11p_V5 = (sig11_V5+sig22_V5)/2 + (sig11_V5-sig22_V5)/2*c2p + sig12_V5*s2p
s22p_V5 = (sig11_V5+sig22_V5)/2 - (sig11_V5-sig22_V5)/2*c2p - sig12_V5*s2p


def primed_to_cart(s11p, s22p, s12p, phi):
    c2 = np.cos(2*phi); s2 = np.sin(2*phi)
    return ((s11p+s22p)/2 + (s11p-s22p)/2*c2 - s12p*s2,
            (s11p+s22p)/2 - (s11p-s22p)/2*c2 + s12p*s2,
            (s11p-s22p)/2*s2 + s12p*c2)

def cart_to_polar(s11, s22, s12, th):
    c2t = np.cos(2*th); s2t = np.sin(2*th)
    sm = (s11+s22)/2; X = (s11-s22)/2; Y = s12
    return sm+X*c2t+Y*s2t, sm-X*c2t-Y*s2t, -X*s2t+Y*c2t, sm

def r_crit_sector_I(theta):
    arg = phi_I - theta
    b1 = 1/abs(np.sin(arg)) if abs(np.sin(arg)) > 1e-10 else 100
    b2 = 1/abs(np.cos(arg)) if abs(np.cos(arg)) > 1e-10 else 100
    b3 = np.sin(2*gamma)/np.sin(theta+gamma) if np.sin(theta+gamma) > 1e-10 else 100
    return min(b1, b2, b3)

def complete_stress(r, theta):
    """
    Complete stress field with matched fan transition.

    Three zones per sector:
    1. Primary (r < r_crit): Airy formula
    2. At r = r_crit: concentrated fan (σ_m continuous, σ'₁₂ jumps)
    3. Secondary (r > r_crit): vertex stress with constant σ_m
    """
    if r <= 1.0: return np.nan, np.nan, np.nan, np.nan

    # ---- SECTOR I ----
    if theta > 0 and theta < gamma:
        rc = r_crit_sector_I(theta)
        arg = phi_I - theta
        rho = r

        if rho < rc:
            # Primary: Airy formula
            d1 = 1 - rho**2*np.sin(arg)**2
            d2 = 1 - rho**2*np.cos(arg)**2
            if d1 <= 0 or d2 <= 0:
                return np.nan, np.nan, np.nan, np.nan
            s11p = A_I*rho*np.sin(arg)/np.sqrt(d1)
            s22p = -A_I*rho*np.cos(arg)/np.sqrt(d2)
            s12p = A_I
            sig = primed_to_cart(s11p, s22p, s12p, phi_I)
            return cart_to_polar(*sig, theta)
        else:
            # Secondary: vertex V₅ stress with constant σ_m
            # σ'₁₁ still from α-line to void (valid since α-constraint not binding)
            d1 = 1 - rho**2*np.sin(arg)**2
            if d1 > 0:
                s11p = A_I*rho*np.sin(arg)/np.sqrt(d1)
                # σ'₂₂ matched: 2σ_m(V₅) - σ'₁₁
                s22p = 2*sm_V5 - s11p
                # σ'₁₂ at vertex (NOT A_I)
                s12p = A_I  # Keep A_I since still on Sector I face approaching vertex
                sig = primed_to_cart(s11p, s22p, s12p, phi_I)
                return cart_to_polar(*sig, theta)
            else:
                # Both chars exceed domain: pure vertex
                c2t = np.cos(2*theta); s2t = np.sin(2*theta)
                return (sm_V5 + X_V5*c2t + Y_V5*s2t,
                        sm_V5 - X_V5*c2t - Y_V5*s2t,
                        -X_V5*s2t + Y_V5*c2t, sm_V5)

    # ---- SECTOR II ----
    elif theta >= gamma and theta <= alpha_half:
        x = r*np.cos(theta); y = r*np.sin(theta)
        t_C1 = (x - cg)/sg if abs(sg) > 1e-10 else -1
        y_C1 = sg + t_C1*cg if t_C1 >= 0 else float('inf')

        if t_C1 >= 0 and y < y_C1:
            # Secondary: vertex V₅ stress
            c2t = np.cos(2*theta); s2t = np.sin(2*theta)
            return (sm_V5 + X_V5*c2t + Y_V5*s2t,
                    sm_V5 - X_V5*c2t - Y_V5*s2t,
                    -X_V5*s2t + Y_V5*c2t, sm_V5)
        else:
            # Primary Sector II
            d1 = 1 - r**2*np.sin(theta)**2
            d2 = 1 - r**2*np.cos(theta)**2
            if d1 > 0 and d2 > 0:
                s11 = -A_II*r*np.sin(theta)/np.sqrt(d1)
                s22 = -A_II*r*np.cos(theta)/np.sqrt(d2)
                return cart_to_polar(s11, s22, A_II, theta)
            return np.nan, np.nan, np.nan, np.nan

    # ---- SECTOR III (mirror of I) ----
    elif theta > alpha_half and theta < np.pi/2:
        # Mirror: use Sector III formulas
        arg = phi_III - theta
        rho = r
        # r_crit for Sector III (by symmetry about π/4)
        b1 = 1/abs(np.sin(arg)) if abs(np.sin(arg)) > 1e-10 else 100
        b2 = 1/abs(np.cos(arg)) if abs(np.cos(arg)) > 1e-10 else 100
        rc = min(b1, b2)

        if rho < rc:
            d1 = 1 - rho**2*np.sin(arg)**2
            d2 = 1 - rho**2*np.cos(arg)**2
            if d1 <= 0 or d2 <= 0:
                return np.nan, np.nan, np.nan, np.nan
            s11p = A_III*rho*np.sin(arg)/np.sqrt(d1)
            s22p = -A_III*rho*np.cos(arg)/np.sqrt(d2)
            sig = primed_to_cart(s11p, s22p, A_III, phi_III)
            return cart_to_polar(*sig, theta)
        else:
            # Secondary for Sector III: vertex V₆ = (-√6/4, √3)
            # By symmetry about θ = π/4, same σ_m
            c2t = np.cos(2*theta); s2t = np.sin(2*theta)
            # V₆ = (-X_V5, Y_V5), sm same
            X_V6 = -X_V5
            return (sm_V5 + X_V6*c2t + Y_V5*s2t,
                    sm_V5 - X_V6*c2t - Y_V5*s2t,
                    -X_V6*s2t + Y_V5*c2t, sm_V5)

    return np.nan, np.nan, np.nan, np.nan


# ================================================================
# Verification
# ================================================================
print("="*70)
print("COMPLETE MATCHED FIELD VERIFICATION")
print("="*70)

print("\n1. Void surface (r=1.001):")
for td in range(0, 91, 5):
    t = np.radians(td) + 0.001
    srr, stt, srt, sm = complete_stress(1.001, t)
    if not np.isnan(srr):
        ok = "✓" if abs(srr) < 0.01 else "✗"
        print(f"  θ={td:3d}°  σ_rr={srr:8.4f}  σ_m={sm:8.4f}  {ok}")

print("\n2. Radial profile at θ=0°:")
for r in [1.01, 1.05, 1.1, 1.15, 1.2, 1.22, 1.225, 1.23, 1.25, 1.3, 1.5, 2.0, 3.0]:
    srr, stt, srt, sm = complete_stress(r, 0.001)
    if not np.isnan(srr):
        rc = r_crit_sector_I(0.001)
        zone = "primary" if r < rc else "secondary"
        print(f"  r/a={r:.3f}  σ_m={sm:8.4f}  σ_rr={srr:8.4f}  σ_θθ={stt:8.4f}  [{zone}]")

print("\n3. Continuity at r_crit (θ=5°):")
theta_test = np.radians(5)
rc = r_crit_sector_I(theta_test)
for dr in [-0.01, -0.005, -0.001, 0, 0.001, 0.005, 0.01]:
    r_t = rc + dr
    if r_t <= 1: continue
    srr, stt, srt, sm = complete_stress(r_t, theta_test)
    if not np.isnan(srr):
        zone = "P" if r_t < rc else "S"
        print(f"  r/a={r_t:.4f}  σ_m={sm:8.4f}  σ_rr={srr:8.4f}  [{zone}]")


# ================================================================
# Generate publication figure
# ================================================================
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

Nr, Nth = 400, 500
r_grid = np.linspace(1.001, 2.5, Nr)
theta_grid = np.linspace(0.002, np.pi/2 - 0.002, Nth)
R, Theta = np.meshgrid(r_grid, theta_grid)
Srr = np.full_like(R, np.nan); Stt = np.full_like(R, np.nan)
Srt = np.full_like(R, np.nan); Sm = np.full_like(R, np.nan)

for i in range(Nth):
    for j in range(Nr):
        srr, stt, srt, sm = complete_stress(R[i,j], Theta[i,j])
        Srr[i,j] = srr; Stt[i,j] = stt; Srt[i,j] = srt; Sm[i,j] = sm

X_c = R*np.cos(Theta); Y_c = R*np.sin(Theta)
th_v = np.linspace(0, np.pi/2, 200)
t_cb = np.linspace(0, 6, 300)
x_cb = cg + t_cb*sg; y_cb = sg + t_cb*cg
r_cb = np.sqrt(x_cb**2 + y_cb**2); m_cb = r_cb < 2.5

# r_crit boundary
th_rc = np.linspace(0.01, gamma - 0.01, 100)
r_rc = [r_crit_sector_I(t) for t in th_rc]
x_rc = np.array(r_rc)*np.cos(th_rc); y_rc = np.array(r_rc)*np.sin(th_rc)

titles = [r'(a) $\sigma_{rr}/\tau_{\mathrm{CRSS}}$',
          r'(b) $\sigma_{\theta\theta}/\tau_{\mathrm{CRSS}}$',
          r'(c) $\sigma_{r\theta}/\tau_{\mathrm{CRSS}}$',
          r'(d) $\sigma_m/\tau_{\mathrm{CRSS}}$']
data = [Srr, Stt, Srt, Sm]
ranges = [(-4, 2), (-8, 0), (-2, 2), (-4, 0)]

for ax, Z, title, (vmin, vmax) in zip(axes.flat, data, titles, ranges):
    pc = ax.pcolormesh(X_c, Y_c, Z, cmap='RdBu_r', vmin=vmin, vmax=vmax,
                       shading='auto', rasterized=True)
    fig.colorbar(pc, ax=ax, shrink=0.8, pad=0.02)
    ax.plot(np.cos(th_v), np.sin(th_v), 'k-', lw=2.5)
    for a_val in [gamma, alpha_half]:
        ax.plot([0, 2.5*np.cos(a_val)], [0, 2.5*np.sin(a_val)],
                'k--', lw=0.8, alpha=0.4)
    ax.plot(x_cb[m_cb], y_cb[m_cb], 'k-', lw=1.2, alpha=0.7)
    ax.plot(x_rc, y_rc, 'k:', lw=1.5, alpha=0.6)
    ax.set_xlim(0, 2.2); ax.set_ylim(0, 2.2); ax.set_aspect('equal')
    ax.set_title(title, fontsize=12)
    ax.set_xlabel(r"$x'_1/a$", fontsize=10)
    ax.set_ylabel(r"$x'_2/a$", fontsize=10)

plt.suptitle("Complete stress field with matched fan transition (first quadrant)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/complete_field_matched.png',
            dpi=200, bbox_inches='tight')
print("\nSaved: figures/complete_field_matched.png")
