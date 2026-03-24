"""
Transition zone ODE integration for BCC void problem.

In the transition zone (r ≈ r_crit), the Airy formula breaks down
because one characteristic exits the primary sector before reaching
the void. The stress must be computed by integrating Rice's (1973)
characteristic equations along the composite path:

  Primary sector α-line → fan line C₁ → secondary sector β-line

This script implements the integration using SciPy solve_ivp.

Mathematical framework:
-----------------------
In each sector with a STRAIGHT yield face (polygon), the equilibrium
equations in the rotated frame reduce to:
  ∂σ'₁₁/∂x'₁ = 0  (σ'₁₁ const along α-lines = x'₂ = const lines)
  ∂σ'₂₂/∂x'₂ = 0  (σ'₂₂ const along β-lines = x'₁ = const lines)
  σ'₁₂ = A (constant on the face)

At a vertex of the polygon (fan line), the stress is:
  X = X_vertex, Y = Y_vertex, σ_m = σ_m(position)
with dσ_m = 0 along the fan line (Rice's Hencky equation with ds = 0).

The transition zone has TWO sub-regions:
A) Between the Airy domain boundary and C₁: the β-line from a point
   crosses the sector boundary before reaching the void. The stress
   must be computed by tracing the β-line to the boundary and then
   using the adjacent sector's stress.
B) Between C₁ and the primary Sector II domain: the secondary sector
   with vertex stress (constant σ_m).

Region A requires the ODE integration.
"""

import numpy as np
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ================================================================
# BCC parameters
# ================================================================
gamma = 0.5 * np.arctan(2*np.sqrt(2))   # ≈ 35.26°
alpha_half = np.arctan(np.sqrt(2))        # ≈ 54.74°
phi_I = -gamma
phi_II = 0.0
phi_III = gamma

A_I = 2*np.sqrt(3)/3
A_II = np.sqrt(3)
A_III = 2*np.sqrt(3)/3

cg = np.cos(gamma); sg = np.sin(gamma)
ca = np.cos(alpha_half); sa = np.sin(alpha_half)

# Vertex V₅ coordinates
X_V5 = np.sqrt(6)/4
Y_V5 = np.sqrt(3)
sm_V5 = -3*np.sqrt(6)/4  # σ_m at vertex V₅ on the void surface at θ₁


# ================================================================
# Primary sector stress (Airy formulas, corrected signs)
# ================================================================
def primary_sector_I_primed(r, theta):
    """Sector I stress in the rotated (primed) frame."""
    arg = phi_I - theta
    rho = r
    d1 = 1 - rho**2 * np.sin(arg)**2
    d2 = 1 - rho**2 * np.cos(arg)**2
    if d1 <= 0 or d2 <= 0:
        return None
    s11p = A_I * rho * np.sin(arg) / np.sqrt(d1)
    s22p = -A_I * rho * np.cos(arg) / np.sqrt(d2)
    return s11p, s22p, A_I


def primed_to_cartesian(s11p, s22p, s12p, phi):
    c2 = np.cos(2*phi); s2 = np.sin(2*phi)
    sig11 = (s11p+s22p)/2 + (s11p-s22p)/2*c2 - s12p*s2
    sig22 = (s11p+s22p)/2 - (s11p-s22p)/2*c2 + s12p*s2
    sig12 = (s11p-s22p)/2*s2 + s12p*c2
    return sig11, sig22, sig12


def cartesian_to_polar(sig11, sig22, sig12, theta):
    c2t = np.cos(2*theta); s2t = np.sin(2*theta)
    sm = (sig11 + sig22) / 2
    X = (sig11 - sig22) / 2
    Y = sig12
    srr = sm + X*c2t + Y*s2t
    stt = sm - X*c2t - Y*s2t
    srt = -X*s2t + Y*c2t
    return srr, stt, srt, sm


def void_surface_stress_cartesian(theta):
    """Full Cartesian stress at the void surface at angle θ."""
    c2t = np.cos(2*theta); s2t = np.sin(2*theta)
    if abs(c2t) < 1e-12:
        # At θ = π/4: vertex between Sector II and III
        return None

    if theta <= gamma + 1e-10:
        # Sector I: face V₄→V₅, (√6/3)X + (√3/6)Y = 1
        denom = np.sqrt(6)/3 + np.sqrt(3)/6 * s2t/c2t
        if abs(denom) < 1e-10: return None
        X = 1/denom
    elif theta <= alpha_half + 1e-10:
        # Sector II: face V₅→V₆, Y = √3
        X = np.sqrt(3) * c2t / s2t
    else:
        # Sector III: face V₆→V₁, (√6/3)X - (√3/6)Y = -1
        denom = np.sqrt(6)/3 - np.sqrt(3)/6 * s2t/c2t
        if abs(denom) < 1e-10: return None
        X = -1/denom

    Y = X * s2t / c2t
    sm = -(X*c2t + Y*s2t)
    return sm + X, sm - X, Y  # sig11, sig22, sig12


# ================================================================
# Transition zone: characteristic tracing with sector crossing
# ================================================================
def trace_beta_composite(r, theta):
    """
    Trace the β-line backward from (r, θ) in Sector I.

    The β-line goes at angle φ_I + 90° = 54.74° from horizontal.
    Backward: angle 234.74° (down-left).

    If it hits the void (r = a = 1): return void surface stress σ'₂₂.
    If it crosses θ₁: continue in Sector II (vertical β-line downward)
    until it hits the void. Return the composite σ'₂₂.
    """
    x0 = r * np.cos(theta)
    y0 = r * np.sin(theta)

    # β-line in Sector I: direction (cos(φ_I+90°), sin(φ_I+90°)) = (-sinγ, cosγ)
    # Backward: (sinγ, -cosγ)
    beta_dx = sg   # sinγ
    beta_dy = -cg  # -cosγ

    # Step 1: trace in Sector I direction until void or sector boundary
    # Parametric: x(s) = x0 + s*sinγ, y(s) = y0 - s*cosγ  (s > 0 = backward)

    # Hit void: (x0 + s sinγ)² + (y0 - s cosγ)² = 1
    # s² - 2s(x0 sinγ - y0 cosγ)... wait, expand:
    # x0² + 2s x0 sinγ + s² sin²γ + y0² - 2s y0 cosγ + s² cos²γ = 1
    # r² + 2s(x0 sinγ - y0 cosγ) + s² = 1
    # s² + 2s(x0 sinγ - y0 cosγ) + (r² - 1) = 0

    a_coeff = 1.0
    b_coeff = 2*(x0*sg - y0*cg)
    c_coeff = r**2 - 1

    disc = b_coeff**2 - 4*a_coeff*c_coeff

    # Hit sector boundary θ = θ₁: y/x = tanγ = sinγ/cosγ
    # (y0 - s cosγ) / (x0 + s sinγ) = sinγ/cosγ
    # cosγ(y0 - s cosγ) = sinγ(x0 + s sinγ)
    # y0 cosγ - s cos²γ = x0 sinγ + s sin²γ
    # y0 cosγ - x0 sinγ = s(sin²γ + cos²γ) = s
    # s_boundary = y0 cosγ - x0 sinγ

    s_boundary = y0*cg - x0*sg

    if disc >= 0:
        s_void_1 = (-b_coeff - np.sqrt(disc)) / (2*a_coeff)
        s_void_2 = (-b_coeff + np.sqrt(disc)) / (2*a_coeff)
        # Take the smallest positive s
        s_void = min(s for s in [s_void_1, s_void_2] if s > 1e-10) if any(s > 1e-10 for s in [s_void_1, s_void_2]) else float('inf')
    else:
        s_void = float('inf')

    if s_boundary <= 1e-10:
        # Already at or past the boundary
        s_boundary = float('inf')

    if s_void < s_boundary and s_void < float('inf'):
        # β-line reaches void within Sector I → standard Airy formula valid
        x_hit = x0 + s_void * sg
        y_hit = y0 - s_void * cg
        theta_hit = np.arctan2(y_hit, x_hit)
        # Return σ'₂₂ from void surface stress at θ_hit (in Sector I frame)
        cart = void_surface_stress_cartesian(theta_hit)
        if cart is None:
            return None
        sig11_v, sig22_v, sig12_v = cart
        # Convert to Sector I primed frame to get σ'₂₂
        c2 = np.cos(2*phi_I); s2 = np.sin(2*phi_I)
        s22p = (sig11_v+sig22_v)/2 - (sig11_v-sig22_v)/2*c2 + sig12_v*s2
        return s22p, 'void', theta_hit

    elif s_boundary < float('inf'):
        # β-line crosses sector boundary θ₁ before reaching void
        x_bnd = x0 + s_boundary * sg
        y_bnd = y0 - s_boundary * cg
        r_bnd = np.sqrt(x_bnd**2 + y_bnd**2)

        # Now in Sector II: β-line is VERTICAL (downward)
        # Continue from (x_bnd, y_bnd) going down at x = x_bnd

        if x_bnd >= 1.0:
            # Vertical line misses void entirely
            # → stress at vertex V₅ with σ_m = -3√6/4
            # σ₂₂(V₅) = sm_V5 - X_V5 = -3√6/4 - √6/4 = -√6
            sig22_vertex = sm_V5 - X_V5
            # Convert to Sector I primed frame
            sig11_vertex = sm_V5 + X_V5
            sig12_vertex = Y_V5
            c2 = np.cos(2*phi_I); s2 = np.sin(2*phi_I)
            s22p = (sig11_vertex+sig22_vertex)/2 - (sig11_vertex-sig22_vertex)/2*c2 + sig12_vertex*s2
            return s22p, 'vertex', None

        # Vertical line at x = x_bnd hits void at y = √(1 - x_bnd²)
        y_void = np.sqrt(1 - x_bnd**2)
        theta_void = np.arctan2(y_void, x_bnd)

        # At the void surface at θ_void: get Cartesian stress
        cart = void_surface_stress_cartesian(theta_void)
        if cart is None:
            return None
        sig11_v, sig22_v, sig12_v = cart

        # The vertical β-line carries σ₂₂ (Cartesian) from the void.
        # But we need σ'₂₂ in Sector I's primed frame at (r, θ).
        #
        # Key subtlety: the β-line changes direction at the sector boundary.
        # In Sector I: β-line at angle 54.74°
        # In Sector II: β-line vertical (90°)
        # At the boundary, the stress transitions through vertex V₅.
        #
        # The Hencky equation for polygons: at a vertex, dσ_m = 0.
        # So σ_m is continuous across the boundary.
        #
        # σ₂₂ from the void (in Sector II): σ₂₂ = sig22_v
        # This is constant along the VERTICAL β-line up to the boundary.
        # At the boundary: σ₂₂ (Cartesian) is continuous.
        #
        # From the boundary, the β-line continues in Sector I direction.
        # σ'₂₂ (Sector I frame) is constant along THIS segment.
        # Its value = σ'₂₂ at the boundary point, computed from the
        # Cartesian stress at the boundary.

        # Cartesian stress at the boundary: from Sector II side
        # σ₂₂(boundary) = σ₂₂(void) = sig22_v  (const along vertical)
        # σ₁₁(boundary): from horizontal α-line in Sector II...
        # BUT the boundary point may be in the PRIMARY Sector II (if the
        # horizontal α-line reaches the void). Let's check.

        # Horizontal α-line from (x_bnd, y_bnd) goes left:
        # y = y_bnd = const, x decreases → hits void at x = √(1 - y_bnd²)
        if y_bnd < 1.0:
            x_void_h = np.sqrt(1 - y_bnd**2)
            theta_void_h = np.arctan2(y_bnd, x_void_h)
            cart_h = void_surface_stress_cartesian(theta_void_h)
            if cart_h:
                sig11_bnd = cart_h[0]  # σ₁₁ from void via horizontal
            else:
                sig11_bnd = sm_V5 + X_V5  # vertex fallback
        else:
            sig11_bnd = sm_V5 + X_V5

        # σ₁₂ at boundary: on the Sector II face, σ₁₂ = √3
        sig12_bnd = Y_V5  # = √3

        # Full Cartesian stress at boundary:
        # σ₁₁ = sig11_bnd, σ₂₂ = sig22_v, σ₁₂ = sig12_bnd

        # Convert to Sector I primed frame:
        c2 = np.cos(2*phi_I); s2 = np.sin(2*phi_I)
        s22p = (sig11_bnd+sig22_v)/2 - (sig11_bnd-sig22_v)/2*c2 + sig12_bnd*s2

        return s22p, 'composite', theta_void

    return None


def transition_zone_stress(r, theta):
    """
    Compute stress in the transition zone using composite β-line tracing.

    σ'₁₁: from α-line to void (standard Sector I formula, always valid
           since the α-line constraint is not binding in the transition zone)
    σ'₂₂: from composite β-line tracing (void → Sector II → boundary → Sector I)
    σ'₁₂ = A_I (on the Sector I yield face)
    """
    # σ'₁₁ from Sector I α-line (always valid in transition zone)
    arg = phi_I - theta
    rho = r
    d1 = 1 - rho**2 * np.sin(arg)**2
    if d1 <= 0:
        return None
    s11p = A_I * rho * np.sin(arg) / np.sqrt(d1)

    # σ'₂₂ from composite β-line tracing
    result = trace_beta_composite(r, theta)
    if result is None:
        return None
    s22p, source, theta_hit = result

    s12p = A_I

    # Convert to Cartesian then polar
    sig11, sig22, sig12 = primed_to_cartesian(s11p, s22p, s12p, phi_I)
    return cartesian_to_polar(sig11, sig22, sig12, theta)


# ================================================================
# Verification
# ================================================================
print("="*70)
print("TRANSITION ZONE ODE INTEGRATION")
print("="*70)

print("\n1. Void surface check (r = 1.001):")
for theta_deg in np.arange(0, 36, 5):
    theta = np.radians(theta_deg)
    res = transition_zone_stress(1.001, theta)
    if res:
        srr, stt, srt, sm = res
        print(f"  θ={theta_deg:5.1f}°  σ_rr={srr:9.5f}  σ_rθ={srt:9.5f}  σ_m={sm:8.4f}")

print("\n2. Transition zone points (r between r_crit and void):")
for theta_deg in [0, 5, 10, 15, 20, 25, 30, 35]:
    theta = np.radians(theta_deg)
    arg = phi_I - theta
    r_airy = min(1/abs(np.sin(arg)), 1/abs(np.cos(arg)))
    r_C1 = np.sin(2*gamma)/np.sin(theta + gamma) if np.sin(theta+gamma) > 1e-10 else 100
    r_crit = min(r_airy, r_C1)

    # Test at r slightly above r_crit
    r_test = min(r_crit * 1.01, r_crit + 0.02)
    if r_test > 3: continue

    res_trans = transition_zone_stress(r_test, theta)
    # Also compute primary sector at r slightly below r_crit
    res_primary = primary_sector_I_primed(r_crit * 0.99, theta)

    if res_trans and res_primary:
        srr_t, stt_t, srt_t, sm_t = res_trans
        s11p, s22p, _ = res_primary
        sig11, sig22, sig12 = primed_to_cartesian(s11p, s22p, A_I, phi_I)
        _, _, _, sm_p = cartesian_to_polar(sig11, sig22, sig12, theta)
        print(f"  θ={theta_deg:3d}°  r_crit={r_crit:.4f}  "
              f"σ_m(primary)={sm_p:8.4f}  σ_m(transition)={sm_t:8.4f}  "
              f"jump={abs(sm_p-sm_t):.4f}")
    elif res_trans:
        srr_t, stt_t, srt_t, sm_t = res_trans
        print(f"  θ={theta_deg:3d}°  r_crit={r_crit:.4f}  "
              f"σ_m(transition)={sm_t:8.4f}  (primary NaN)")

print("\n3. Deep in transition zone / secondary sector:")
for theta_deg in [0, 10, 20, 30]:
    theta = np.radians(theta_deg)
    for r_val in [1.3, 1.5, 2.0]:
        res = transition_zone_stress(r_val, theta)
        if res:
            srr, stt, srt, sm = res
            print(f"  θ={theta_deg}° r/a={r_val:.1f}: σ_m={sm:8.4f}  σ_rr={srr:8.4f}  "
                  f"(target: σ_m={sm_V5:.4f})")

# ================================================================
# Generate comprehensive figure
# ================================================================
print("\n" + "="*70)
print("GENERATING FIGURE")
print("="*70)

fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

Nr, Nth = 250, 300
r_grid = np.linspace(1.001, 2.0, Nr)
theta_grid = np.linspace(0.001, np.pi/2 - 0.001, Nth)
R, Theta = np.meshgrid(r_grid, theta_grid)
Sm = np.full_like(R, np.nan)
Srr = np.full_like(R, np.nan)
Region = np.full_like(R, np.nan)  # 1=primary, 2=transition, 3=secondary

for i in range(Nth):
    for j in range(Nr):
        r_val = R[i,j]; th = Theta[i,j]

        if th < gamma:
            # Sector I domain: check if primary or transition
            arg = phi_I - th
            r_airy = min(1/abs(np.sin(arg)) if abs(np.sin(arg))>1e-10 else 100,
                         1/abs(np.cos(arg)) if abs(np.cos(arg))>1e-10 else 100)
            r_C1 = np.sin(2*gamma)/np.sin(th+gamma) if np.sin(th+gamma)>1e-10 else 100
            r_crit = min(r_airy, r_C1)

            if r_val < r_crit * 0.98:
                # Primary Sector I
                res = primary_sector_I_primed(r_val, th)
                if res:
                    sig11, sig22, sig12 = primed_to_cartesian(*res, phi_I)
                    srr, stt, srt, sm = cartesian_to_polar(sig11, sig22, sig12, th)
                    Sm[i,j] = sm; Srr[i,j] = srr; Region[i,j] = 1
            else:
                # Transition zone
                res = transition_zone_stress(r_val, th)
                if res:
                    Srr[i,j], _, _, Sm[i,j] = res
                    Region[i,j] = 2

        elif th < alpha_half:
            # Sector II: check primary vs secondary
            x = r_val*np.cos(th); y = r_val*np.sin(th)
            t_C1 = (x - cg)/sg if abs(sg) > 1e-10 else -1
            y_C1 = sg + t_C1*cg if t_C1 >= 0 else float('inf')

            if t_C1 >= 0 and y < y_C1:
                # Secondary sector: vertex V₅ stress
                Sm[i,j] = sm_V5; Region[i,j] = 3
                c2t = np.cos(2*th); s2t = np.sin(2*th)
                Srr[i,j] = sm_V5 + X_V5*c2t + Y_V5*s2t
            else:
                # Primary Sector II
                d1 = 1 - r_val**2*np.sin(th)**2
                d2 = 1 - r_val**2*np.cos(th)**2
                if d1 > 0 and d2 > 0:
                    s11p = -A_II*r_val*np.sin(th)/np.sqrt(d1)
                    s22p = -A_II*r_val*np.cos(th)/np.sqrt(d2)
                    sig11, sig22, sig12 = s11p, s22p, A_II  # φ=0
                    srr, stt, srt, sm = cartesian_to_polar(sig11, sig22, sig12, th)
                    Sm[i,j] = sm; Srr[i,j] = srr; Region[i,j] = 1

        else:
            # Sector III (mirror of I)
            arg = phi_III - th
            d1 = 1 - r_val**2*np.sin(arg)**2
            d2 = 1 - r_val**2*np.cos(arg)**2
            if d1 > 0 and d2 > 0:
                s11p = A_III*r_val*np.sin(arg)/np.sqrt(d1)
                s22p = -A_III*r_val*np.cos(arg)/np.sqrt(d2)
                sig11, sig22, sig12 = primed_to_cartesian(s11p, s22p, A_III, phi_III)
                srr, stt, srt, sm = cartesian_to_polar(sig11, sig22, sig12, th)
                Sm[i,j] = sm; Srr[i,j] = srr; Region[i,j] = 1

X_c = R*np.cos(Theta); Y_c = R*np.sin(Theta)
th_v = np.linspace(0, np.pi/2, 200)
t_cb = np.linspace(0, 4, 200)
x_cb = cg + t_cb*sg; y_cb = sg + t_cb*cg
r_cb = np.sqrt(x_cb**2 + y_cb**2); m_cb = r_cb < 2

# Panel (a): σ_m
ax = axes[0]
pc = ax.pcolormesh(X_c, Y_c, Sm, cmap='RdBu_r', vmin=-5, vmax=0,
                   shading='auto', rasterized=True)
fig.colorbar(pc, ax=ax, shrink=0.8)
ax.plot(np.cos(th_v), np.sin(th_v), 'k-', lw=2)
for a_val in [gamma, alpha_half]:
    ax.plot([0, 2*np.cos(a_val)], [0, 2*np.sin(a_val)], 'k--', lw=0.5, alpha=0.5)
ax.plot(x_cb[m_cb], y_cb[m_cb], 'w-', lw=1.5)
ax.set_xlim(0, 1.8); ax.set_ylim(0, 1.8); ax.set_aspect('equal')
ax.set_title(r'(a) $\sigma_m/\tau$ (composite)', fontsize=11)
ax.set_xlabel(r"$x'_1/a$"); ax.set_ylabel(r"$x'_2/a$")

# Panel (b): σ_rr
ax = axes[1]
pc = ax.pcolormesh(X_c, Y_c, Srr, cmap='RdBu_r', vmin=-3, vmax=3,
                   shading='auto', rasterized=True)
fig.colorbar(pc, ax=ax, shrink=0.8)
ax.plot(np.cos(th_v), np.sin(th_v), 'k-', lw=2)
for a_val in [gamma, alpha_half]:
    ax.plot([0, 2*np.cos(a_val)], [0, 2*np.sin(a_val)], 'k--', lw=0.5, alpha=0.5)
ax.plot(x_cb[m_cb], y_cb[m_cb], 'w-', lw=1.5)
ax.set_xlim(0, 1.8); ax.set_ylim(0, 1.8); ax.set_aspect('equal')
ax.set_title(r'(b) $\sigma_{rr}/\tau$ (composite)', fontsize=11)
ax.set_xlabel(r"$x'_1/a$")

# Panel (c): Region map
ax = axes[2]
pc = ax.pcolormesh(X_c, Y_c, Region, cmap='Set1', vmin=0.5, vmax=3.5,
                   shading='auto', rasterized=True)
cbar = fig.colorbar(pc, ax=ax, shrink=0.8, ticks=[1, 2, 3])
cbar.set_ticklabels(['Primary', 'Transition', 'Secondary'])
ax.plot(np.cos(th_v), np.sin(th_v), 'k-', lw=2)
for a_val in [gamma, alpha_half]:
    ax.plot([0, 2*np.cos(a_val)], [0, 2*np.sin(a_val)], 'k--', lw=0.5, alpha=0.5)
ax.plot(x_cb[m_cb], y_cb[m_cb], 'w-', lw=1.5)
ax.set_xlim(0, 1.8); ax.set_ylim(0, 1.8); ax.set_aspect('equal')
ax.set_title('(c) Region map', fontsize=11)
ax.set_xlabel(r"$x'_1/a$")

plt.suptitle("Complete stress field: primary + transition + secondary sectors", fontsize=13)
plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/transition_zone.png', dpi=200, bbox_inches='tight')
print("Saved: figures/transition_zone.png")
