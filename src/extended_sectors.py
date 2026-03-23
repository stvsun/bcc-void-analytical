"""
Extended stress sectors IV-VIII for BCC void (Kysar Secs. 3.7-3.13).

After computing the stress in sectors I-III (which directly border
the void), we extend to sectors IV-VIII using the characteristic
construction. The key new element is the curved boundary between
sectors IV and V.

BCC parameters:
  System I  (sys 5,12): φ₁ = arctan(2√2)/2, A_I = 2√3/3, β₁ = 2√3/3
  System II (sys 3,4):  φ₂ = 0,              A_II = √3,   β₂ = √3
  γ = arctan(2√2)/2 ≈ 35.26° (sector boundary angle)

Kysar's key equations adapted to BCC:
  Eq. (34): l_I  = (3/2)√(3/2) √(sin[2(γ-θ_{p1})] / sin[2(φ₁-θ_{p1})])
  Eq. (37): l_II = (3/2)√(3/2) √(sin[2(θ_{p2}-γ)] / sin[2(θ_{p2}-φ₂)])
  Eq. (40): governing equation for sector boundary
  Eq. (47): tan θ_{p2} = f(θ_{p1}) — BCC-specific relationship
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle

# BCC parameters
s2 = np.sqrt(2)
s3 = np.sqrt(3)
s6 = np.sqrt(6)

phi1 = np.arctan(2*s2) / 2      # ≈ 35.26° (system I direction)
phi2 = 0.0                       # system II direction
gamma_bcc = phi1                  # ≈ 35.26° (sector boundary)

A_I  = 2*s3/3    # ≈ 1.1547
A_II = s3         # ≈ 1.7321
B_I  = A_I        # β₁ = 2√3/3
B_II = A_II       # β₂ = √3

r0 = 1.0  # void radius

print("=" * 70)
print("Extended Sectors: BCC Parameters")
print("=" * 70)
print(f"  φ₁ = {np.degrees(phi1):.4f}°")
print(f"  φ₂ = {np.degrees(phi2):.4f}°")
print(f"  γ  = {np.degrees(gamma_bcc):.4f}°")
print(f"  A_I = {A_I:.6f}, A_II = {A_II:.6f}")

# ================================================================
# Step 1: Curved sector boundary (Kysar Eqs. 40-52 adapted to BCC)
# ================================================================
print("\n" + "=" * 70)
print("Step 1: Curved Sector Boundary (between sectors IV and V)")
print("=" * 70)

# For FCC, Kysar derived: tan θ_{p2} = 1/(tan θ_{p1} + 1/√2)  [Eq. 47]
# This comes from the condition A_II - B_II sin(2γ) = 0 [Eq. 45]
# which simplifies the general Eq. (44).
#
# For BCC, we need to check if the same simplification holds:
# A_II - B_II sin(2γ) = √3 - √3 · sin(2·35.26°)
# sin(2γ) = sin(arctan(2√2)) = 2√2/3
# A_II - B_II sin(2γ) = √3 - √3 · (2√2/3) = √3(1 - 2√2/3) = √3(3-2√2)/3
# = √3(3-2√2)/3 ≈ √3·(3-2.828)/3 = √3·0.172/3 ≈ 0.0990
# This is NOT zero! So the FCC simplification does NOT apply to BCC.
# We must use the general Eq. (44) for BCC.

check = A_II - B_II * np.sin(2*gamma_bcc)
print(f"\n  A_II - B_II sin(2γ) = {check:.6f}")
print(f"  (For FCC this is 0; for BCC it is NOT 0)")
print(f"  Must use general Eq. (44) for BCC sector boundary.")

# General Eq. (44):
# tan θ_{p2} = (-Q ± √(Q² - (A_II² - B_II² sin²(2γ)))) / (A_II + B_II sin(2γ))
#
# where Q = (A_I - B_I sin[2(γ-θ_{p1})]) / sin[2(φ₁-θ_{p1})] + B_II cos(2γ)
#
# This gives θ_{p2} as a function of θ_{p1}, defining the sector boundary
# parametrically.

def theta_p2_from_p1(theta_p1):
    """Compute θ_{p2} from θ_{p1} using BCC Eq. (44)."""
    # Q from Eq. (42):
    arg1 = gamma_bcc - theta_p1
    arg2 = phi1 - theta_p1
    if abs(np.sin(2*arg2)) < 1e-10:
        return np.nan

    Q = (A_I - B_I * np.sin(2*arg1)) / np.sin(2*arg2) + B_II * np.cos(2*gamma_bcc)

    # Discriminant
    denom = A_II + B_II * np.sin(2*gamma_bcc)
    disc = Q**2 - (A_II**2 - B_II**2 * np.sin(2*gamma_bcc)**2)

    if disc < 0:
        return np.nan

    # Take the branch that gives θ_{p2} > γ (in Sector V region)
    tp2_plus  = np.arctan((-Q + np.sqrt(disc)) / denom)
    tp2_minus = np.arctan((-Q - np.sqrt(disc)) / denom)

    # θ_{p2} should be between γ and φ₁ (the Sector II range on the void)
    for tp2 in [tp2_plus, tp2_minus]:
        if tp2 > gamma_bcc - 0.01 and tp2 < np.pi/2:
            return tp2

    return tp2_plus if tp2_plus > 0 else tp2_minus

# Compute the sector boundary curve parametrically
print("\n  Sector boundary: θ_{p2}(θ_{p1}) for 0 ≤ θ_{p1} ≤ γ:")
n_pts = 50
theta_p1_vals = np.linspace(0, gamma_bcc - 0.01, n_pts)
theta_p2_vals = np.array([theta_p2_from_p1(tp1) for tp1 in theta_p1_vals])

for i in range(0, n_pts, 10):
    tp1 = theta_p1_vals[i]
    tp2 = theta_p2_vals[i]
    if not np.isnan(tp2):
        print(f"    θ_p1 = {np.degrees(tp1):6.2f}° → θ_p2 = {np.degrees(tp2):6.2f}°")

# Physical coordinates of the sector boundary (Kysar Eqs. 48-52)
# The α-line of system I through P_I at (r₀, θ_{p1}):
#   x₁ᴵ/r₀ = [tan φ₁](x₂ᴵ/r₀) + [sin θ_{p1} - tan φ₁ cos θ_{p1}]
# The α-line of system II through P_II at (r₀, θ_{p2}):
#   x₂ᴵᴵ/r₀ = sin θ_{p2}

def sector_boundary_point(theta_p1):
    """Compute (x₁, x₂) of sector boundary at parameter θ_{p1}."""
    theta_p2 = theta_p2_from_p1(theta_p1)
    if np.isnan(theta_p2):
        return np.nan, np.nan

    # Kysar Eq. (49): x₂ᴵᴵ/r₀ = sin θ_{p2}
    # But need to use Eq. (50) for BCC:
    # sin θ_{p2} = 1/√(1 + (tan θ_{p1} + 1/√2)²) for FCC
    # For BCC, use the general form from Eq. (47) → (49)

    # Actually, the boundary point is at the intersection of:
    # α-line of system I from (r₀, θ_{p1}): straight line at angle φ₁
    # α-line of system II from (r₀, θ_{p2}): horizontal line (φ₂ = 0)
    #
    # System I α-line through void surface at θ_{p1}:
    #   Point: (r₀ cos θ_{p1}, r₀ sin θ_{p1})
    #   Direction: (cos φ₁, sin φ₁) (going outward)
    #   Parametric: x₁ = r₀ cos θ_{p1} + t cos φ₁
    #               x₂ = r₀ sin θ_{p1} + t sin φ₁
    #
    # System II α-line through void surface at θ_{p2}:
    #   Point: (r₀ cos θ_{p2}, r₀ sin θ_{p2})
    #   Direction: (cos φ₂, sin φ₂) = (1, 0) (horizontal)
    #   Parametric: x₁ = r₀ cos θ_{p2} + s
    #               x₂ = r₀ sin θ_{p2}
    #
    # Intersection: x₂ must match:
    # r₀ sin θ_{p1} + t sin φ₁ = r₀ sin θ_{p2}
    # t = r₀ (sin θ_{p2} - sin θ_{p1}) / sin φ₁
    if abs(np.sin(phi1)) < 1e-10:
        return np.nan, np.nan

    t = r0 * (np.sin(theta_p2) - np.sin(theta_p1)) / np.sin(phi1)
    x1 = r0 * np.cos(theta_p1) + t * np.cos(phi1)
    x2 = r0 * np.sin(theta_p2)

    return x1, x2

# Compute boundary curve
x1_sb = []
x2_sb = []
for tp1 in theta_p1_vals:
    x1, x2 = sector_boundary_point(tp1)
    if not np.isnan(x1) and x1 > 0 and x2 > 0:
        x1_sb.append(x1)
        x2_sb.append(x2)

x1_sb = np.array(x1_sb)
x2_sb = np.array(x2_sb)
r_sb = np.sqrt(x1_sb**2 + x2_sb**2)
theta_sb = np.arctan2(x2_sb, x1_sb)

print(f"\n  Sector boundary curve: {len(x1_sb)} points")
if len(x1_sb) > 0:
    print(f"    r/a range: [{r_sb.min():.4f}, {r_sb.max():.4f}]")
    print(f"    θ range: [{np.degrees(theta_sb.min()):.2f}°, {np.degrees(theta_sb.max()):.2f}°]")

# ================================================================
# Step 2: Stress sector vertex coordinates (BCC Table 3)
# ================================================================
print("\n" + "=" * 70)
print("Step 2: Stress Sector Vertices (BCC Table 3)")
print("=" * 70)

# Vertices on the void surface
a = (r0, 0)
b = (r0*np.cos(gamma_bcc), r0*np.sin(gamma_bcc))
c = (r0*np.cos(gamma_bcc + (np.pi/2 - 2*gamma_bcc)),  # θ₂ ≈ 54.74°
     r0*np.sin(gamma_bcc + (np.pi/2 - 2*gamma_bcc)))
d = (0, r0)

# Extended vertices
# e: intersection of sector boundary with the radial line at θ = 0
#    (or the x₁-axis extended from Sector I)
# Actually, vertices in Kysar Fig. 10:
# e = point on x₁-axis where Sector III meets Sector VI
# For BCC, the structure is different. Let me compute the key vertices.

# Point i: where the curved sector boundary meets the sector line at θ = γ₁
# This is the endpoint of the curved boundary at θ_{p1} = 0
x1_i, x2_i = sector_boundary_point(0.01)
# Point b: on the void at θ = γ₁
# The curved boundary goes from b (on void) to i (in the interior)

# Special radial lines and their intersections define additional vertices.
# For now, compute vertices that we can:

vertices_bcc = {
    'a': a,
    'b': b,
    'c': c,
    'd': d,
}
if len(x1_sb) > 0:
    vertices_bcc['i'] = (x1_sb[0], x2_sb[0])  # first point of boundary

print(f"\n  {'Vertex':>8s} {'x₁/a':>10s} {'x₂/a':>10s} {'r/a':>10s} {'θ°':>8s}")
print("  " + "-" * 42)
for name, (x, y) in sorted(vertices_bcc.items()):
    r_v = np.sqrt(x**2 + y**2)
    th_v = np.degrees(np.arctan2(y, x))
    print(f"  {name:>8s} {x:>10.5f} {y:>10.5f} {r_v:>10.5f} {th_v:>8.2f}")

# ================================================================
# Step 3: Full sector map figure
# ================================================================
print("\n" + "=" * 70)
print("Step 3: Full Sector Map with Curved Boundary")
print("=" * 70)

fig, ax = plt.subplots(1, 1, figsize=(8, 8))

# Void
void = Circle((0, 0), r0, fill=True, fc='#f5f5f5', ec='black', lw=2, zorder=5)
ax.add_patch(void)

# Sector colors
from matplotlib.patches import Polygon as MPoly
sector_info = [
    (0, gamma_bcc, '#FFE0B2', 'I'),
    (gamma_bcc, np.pi/2 - gamma_bcc, '#BBDEFB', 'II'),
    (np.pi/2 - gamma_bcc, np.pi/2, '#C8E6C9', 'III'),
]

r_max = 2.5

# Draw primary sectors I-III as wedges
for th_lo, th_hi, color, label in sector_info:
    n = 50
    th = np.linspace(th_lo, th_hi, n)
    inner = np.column_stack([r0*np.cos(th), r0*np.sin(th)])
    outer = np.column_stack([r_max*np.cos(th[::-1]), r_max*np.sin(th[::-1])])
    pts = np.vstack([inner, outer])
    poly = MPoly(pts, alpha=0.2, fc=color, ec='none', zorder=1)
    ax.add_patch(poly)

    # Mirror below x₁-axis
    pts_mirror = pts.copy()
    pts_mirror[:, 1] *= -1
    poly_m = MPoly(pts_mirror, alpha=0.2, fc=color, ec='none', zorder=1)
    ax.add_patch(poly_m)

    # Label
    th_mid = (th_lo + th_hi) / 2
    ax.text(1.6*np.cos(th_mid), 1.6*np.sin(th_mid), label,
            fontsize=13, fontweight='bold', ha='center', va='center')

# Sector boundaries (radial lines)
for tb in [0, gamma_bcc, np.pi/2 - gamma_bcc, np.pi/2]:
    ax.plot([r0*np.cos(tb), r_max*np.cos(tb)],
            [r0*np.sin(tb), r_max*np.sin(tb)], 'k-', lw=1.2, zorder=3)
    ax.plot([r0*np.cos(tb), r_max*np.cos(tb)],
            [-r0*np.sin(tb), -r_max*np.sin(tb)], 'k-', lw=1.2, zorder=3)

# Curved sector boundary
if len(x1_sb) > 1:
    ax.plot(x1_sb, x2_sb, 'r-', lw=2, zorder=4, label='Curved boundary (IV-V)')
    ax.plot(x1_sb, -x2_sb, 'r-', lw=2, zorder=4)
    # Mirror to other quadrants
    ax.plot(-x2_sb, x1_sb, 'r-', lw=2, zorder=4)
    ax.plot(-x2_sb, -x1_sb, 'r-', lw=2, zorder=4)

    # Label extended sectors
    if len(x1_sb) > 5:
        mid = len(x1_sb)//2
        # Sector IV: between radial line at γ and the curved boundary (above)
        ax.text(x1_sb[mid]+0.2, x2_sb[mid]+0.2, 'IV',
                fontsize=12, fontweight='bold', color='darkred')
        # Sector V: between curved boundary and radial line at γ₂
        ax.text(x1_sb[mid]-0.3, x2_sb[mid]+0.3, 'V',
                fontsize=12, fontweight='bold', color='darkblue')

# Vertex labels
for name, (x, y) in vertices_bcc.items():
    ax.plot(x, y, 'ko', markersize=5, zorder=6)
    ax.text(x-0.12, y+0.1, name, fontsize=11, fontweight='bold', zorder=7)

# 45° symmetry line
ax.plot([0, r_max*np.cos(np.pi/4)], [0, r_max*np.sin(np.pi/4)],
        'k--', lw=0.8, alpha=0.4)

# Axes
ax.set_xlim(-0.5, r_max+0.3)
ax.set_ylim(-r_max-0.1, r_max+0.3)
ax.set_aspect('equal')
ax.set_xlabel(r'$x_1/a$', fontsize=13)
ax.set_ylabel(r'$x_2/a$', fontsize=13)
ax.set_title('BCC void: stress sectors with curved boundary\n'
             '(BCC analogue of Kysar Fig. 10)', fontsize=13)
ax.legend(fontsize=10, loc='upper right')
ax.grid(True, alpha=0.15)

plt.tight_layout()
fig_path = 'figures/extended_sectors_map.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {fig_path}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
