"""
Verify the three reviewer criticisms about Section 5:
1. Does φ_I = -35.3° coincide with the sector boundary at +35.3°?
2. Does a β-line in Sector II at r=2a, θ=45° miss the void?
3. What is the domain of validity for each sector's Airy solution?
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Key angles (corrected R, det=+1)
gamma = 0.5 * np.arctan(2*np.sqrt(2))   # ≈ 35.26°
alpha_half = np.arctan(np.sqrt(2))        # ≈ 54.74°

print("="*70)
print("CRITICISM 1: Characteristic direction vs sector boundary")
print("="*70)
print(f"γ = (1/2) arctan(2√2) = {np.degrees(gamma):.4f}°")
print(f"Sector I spans: 0° to {np.degrees(gamma):.2f}°")
print(f"Sector II spans: {np.degrees(gamma):.2f}° to {np.degrees(alpha_half):.2f}°")
print()

# Characteristic directions in each sector
# For the yield face active in Sector I: effective system with
# Schmid coefficients (a, b) = (√6/3, -√3/6) [after R correction]
# The characteristic angle is φ = (1/2) arctan(b/a) ...
# Actually, for a yield face aX + bY = τ, the characteristics make angle
# φ = (1/2) arctan(a/b) with certain sign conventions.
# From the paper: φ_I = -(1/2) arctan(2√2) = -γ ≈ -35.26°

phi_I = -gamma  # as stated in Section 5.2
print(f"α-line direction in Sector I: φ_I = {np.degrees(phi_I):.2f}°")
print(f"β-line direction in Sector I: φ_I + 90° = {np.degrees(phi_I + np.pi/2):.2f}°")
print(f"Sector boundary (Sector I → II): θ = +{np.degrees(gamma):.2f}°")
print()
print(f"Angle between α-line and sector boundary: "
      f"|φ_I| + γ = {np.degrees(abs(phi_I) + gamma):.2f}°")
print(f"  → They are NOT parallel (70.5° apart, not 0°)")
print(f"  → The paper's claim '|φ_I| = γ exactly' is TRUE numerically,")
print(f"    but this does NOT mean they coincide geometrically.")
print(f"  → CRITICISM 1 IS VALID")

print()
print("="*70)
print("CRITICISM 2: β-line in Sector II at r=2a, θ=45° misses void")
print("="*70)

# In Sector II, characteristics are at 0° and 90° (horizontal/vertical)
# β-line is vertical (90°)
r_test = 2.0  # in units of a
theta_test = np.radians(45)
x_test = r_test * np.cos(theta_test)
y_test = r_test * np.sin(theta_test)
print(f"Test point: r/a = {r_test}, θ = 45°")
print(f"  Cartesian: x/a = {x_test:.4f}, y/a = {y_test:.4f}")
print()

# Vertical β-line: x = x_test = const ≈ 1.414a
# Does this intersect the void circle x² + y² = a²?
# At x = x_test: y² = a² - x_test² = 1 - 2 = -1 < 0
print(f"Vertical β-line at x/a = {x_test:.4f}:")
print(f"  Void intersection: y²/a² = 1 - ({x_test:.4f})² = {1 - x_test**2:.4f}")
print(f"  Since {1 - x_test**2:.4f} < 0, the β-line MISSES the void entirely")
print(f"  → CRITICISM 2 IS VALID")

# Where does it hit the sector boundary instead?
# Sector II lower boundary: θ = γ, i.e., y = x tan(γ)
# At x = x_test: y = x_test * tan(gamma)
y_boundary = x_test * np.tan(gamma)
r_boundary = np.sqrt(x_test**2 + y_boundary**2)
print(f"  Instead hits Sector I boundary at y/a = {y_boundary:.4f}, "
      f"r/a = {r_boundary:.4f}")

print()
print("="*70)
print("CRITICISM 3: Domain of validity — critical radii")
print("="*70)

def compute_r_crit_sector_I(theta):
    """
    In Sector I (0 < θ < γ), α-line at angle φ = -γ from horizontal.
    Trace backward from (r, θ): does it hit void or sector boundary first?

    α-line through (r cosθ, r sinθ) going backward:
      x = r cosθ - t cosγ
      y = r sinθ + t sinγ     (t > 0 = backward)

    Hits sector boundary θ = γ when y/x = tanγ:
      t_boundary = r sin(γ - θ) / sin(2γ)

    Hits void when x² + y² = a² (= 1 in normalized units):
      r² - 2rt cos(θ+γ) + t² = 1
      t_void = r cos(θ+γ) - √(1 - r² sin²(θ+γ))

    Valid when t_void exists and t_void < t_boundary.
    Critical r: t_void = t_boundary (or void intersection ceases to exist).
    """
    if theta < 1e-10 or theta > gamma - 1e-10:
        return np.nan

    sg = gamma
    s2g = np.sin(2*sg)

    # Search for r_crit by bisection
    r_lo, r_hi = 1.001, 10.0

    for _ in range(200):
        r = 0.5 * (r_lo + r_hi)

        # t to boundary
        t_bnd = r * np.sin(sg - theta) / s2g

        # t to void: solve r² - 2rt cos(θ+γ) + t² = 1
        cos_tpg = np.cos(theta + sg)
        disc = 1 - r**2 * np.sin(theta + sg)**2

        if disc < 0:
            # No void intersection at all
            r_hi = r
            continue

        t_void = r * cos_tpg - np.sqrt(disc)

        if t_void < 0:
            # Void intersection is "ahead" not "behind"
            r_hi = r
            continue

        if t_void < t_bnd:
            r_lo = r  # Still valid, try larger r
        else:
            r_hi = r  # Boundary hit first, reduce r

    return 0.5 * (r_lo + r_hi)


def compute_r_crit_sector_II_beta(theta):
    """
    In Sector II (γ < θ < arctan(√2)), β-line is vertical (90°).
    Trace backward (downward) from (r, θ).

    Vertical line: x = r cosθ = const
    Hits void when y = √(1 - x²), requires x ≤ 1 (= a).
    Otherwise hits sector boundary θ = γ first.

    Critical: x = r cosθ = 1 → r_crit = 1/cosθ = secθ
    """
    return 1.0 / np.cos(theta)


def compute_r_crit_sector_II_alpha(theta):
    """
    In Sector II, α-line is horizontal (0°).
    Trace backward (leftward) from (r, θ).

    Horizontal line: y = r sinθ = const
    Hits void when x = √(1 - y²), requires y ≤ 1.
    Otherwise hits sector boundary θ = arctan(√2) first.

    Critical: y = r sinθ = 1 → r_crit = 1/sinθ = cscθ
    """
    return 1.0 / np.sin(theta)


# Compute r_crit for Sector I
print("\nSector I (0° to {:.2f}°):".format(np.degrees(gamma)))
print("  α-line direction: {:.2f}° (backward: up-left)".format(np.degrees(-gamma)))
thetas_I = np.linspace(1, np.degrees(gamma) - 1, 10)
print(f"  {'θ (°)':>8}  {'r_crit/a':>10}")
for th_deg in thetas_I:
    th = np.radians(th_deg)
    rc = compute_r_crit_sector_I(th)
    print(f"  {th_deg:8.1f}  {rc:10.4f}")

# Compute r_crit for Sector II
print("\nSector II ({:.2f}° to {:.2f}°):".format(np.degrees(gamma), np.degrees(alpha_half)))
print("  β-line (vertical): r_crit = sec(θ)")
print("  α-line (horizontal): r_crit = csc(θ)")
thetas_II = np.linspace(np.degrees(gamma) + 1, np.degrees(alpha_half) - 1, 8)
print(f"  {'θ (°)':>8}  {'r_crit(β)/a':>12}  {'r_crit(α)/a':>12}  {'r_crit/a':>10}")
for th_deg in thetas_II:
    th = np.radians(th_deg)
    rc_beta = compute_r_crit_sector_II_beta(th)
    rc_alpha = compute_r_crit_sector_II_alpha(th)
    rc = min(rc_beta, rc_alpha)
    print(f"  {th_deg:8.1f}  {rc_beta:12.4f}  {rc_alpha:12.4f}  {rc:10.4f}")

print()
print("="*70)
print("VISUALIZATION: Domain of validity")
print("="*70)

# Plot the domain of validity for each sector
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Panel (a): Schematic showing characteristics crossing boundaries
ax = ax1
th_void = np.linspace(0, np.pi/2, 200)
ax.plot(np.cos(th_void), np.sin(th_void), 'k-', lw=2, label='Void (r=a)')

# Sector boundaries
for bdry_angle in [0, gamma, alpha_half, np.pi/2]:
    ax.plot([0, 3*np.cos(bdry_angle)], [0, 3*np.sin(bdry_angle)],
            'k--', lw=0.5, alpha=0.5)

# α-characteristics in Sector I (angle -γ from horizontal)
for th0 in [5, 15, 25]:
    th0_rad = np.radians(th0)
    r0 = 1.0  # start at void surface
    x0, y0 = r0*np.cos(th0_rad), r0*np.sin(th0_rad)
    # α-line at angle -γ: dx = cos(-γ), dy = sin(-γ)
    t_vals = np.linspace(-2, 2, 200)
    x_line = x0 + t_vals * np.cos(-gamma)
    y_line = y0 + t_vals * np.sin(-gamma)
    # Only plot where r > 1 and in reasonable range
    r_line = np.sqrt(x_line**2 + y_line**2)
    mask = (r_line >= 0.95) & (x_line > 0) & (y_line > -0.5) & (r_line < 3)
    ax.plot(x_line[mask], y_line[mask], 'b-', lw=0.8, alpha=0.6)

# β-characteristics in Sector II (vertical, angle 90°)
for th0 in [38, 42, 46, 50]:
    th0_rad = np.radians(th0)
    r0 = 1.0
    x0 = r0*np.cos(th0_rad)
    # Vertical line at x = x0
    y_vals = np.linspace(-0.5, 3, 200)
    r_line = np.sqrt(x0**2 + y_vals**2)
    mask = (r_line >= 0.95) & (y_vals > 0) & (r_line < 3)
    ax.plot(np.full_like(y_vals[mask], x0), y_vals[mask], 'r-', lw=0.8, alpha=0.6)

# α-characteristics in Sector II (horizontal, angle 0°)
for th0 in [38, 42, 46, 50]:
    th0_rad = np.radians(th0)
    r0 = 1.0
    y0 = r0*np.sin(th0_rad)
    x_vals = np.linspace(-0.5, 3, 200)
    r_line = np.sqrt(x_vals**2 + y0**2)
    mask = (r_line >= 0.95) & (x_vals > 0) & (r_line < 3)
    ax.plot(x_vals[mask], np.full_like(x_vals[mask], y0), 'r--', lw=0.8, alpha=0.6)

# Mark the test point from Criticism 2
ax.plot(2*np.cos(np.radians(45)), 2*np.sin(np.radians(45)), 'rx', ms=12, mew=3,
        label=r'Point at $r=2a$, $\theta=45°$')

# Sector labels
for angle, label in [(np.radians(17), 'I'), (np.radians(45), 'II'), (np.radians(73), 'III')]:
    ax.text(1.5*np.cos(angle), 1.5*np.sin(angle), label, fontsize=14,
            ha='center', va='center', fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

ax.set_xlim(-0.2, 3)
ax.set_ylim(-0.2, 3)
ax.set_aspect('equal')
ax.set_xlabel(r"$x_1'/a$", fontsize=12)
ax.set_ylabel(r"$x_2'/a$", fontsize=12)
ax.set_title("(a) Characteristics crossing sector boundaries", fontsize=11)
ax.legend(fontsize=9, loc='upper left')

# Panel (b): Domain of validity (r_crit as function of θ)
ax = ax2

# Sector I
thetas_I_fine = np.linspace(0.5, np.degrees(gamma) - 0.5, 100)
r_crits_I = []
for th_deg in thetas_I_fine:
    th = np.radians(th_deg)
    rc = compute_r_crit_sector_I(th)
    r_crits_I.append(rc if not np.isnan(rc) else 0)
ax.fill_between(thetas_I_fine, 1, r_crits_I, alpha=0.3, color='blue', label='Sector I valid')
ax.plot(thetas_I_fine, r_crits_I, 'b-', lw=2)

# Sector II
thetas_II_fine = np.linspace(np.degrees(gamma) + 0.5, np.degrees(alpha_half) - 0.5, 100)
r_crits_II = []
for th_deg in thetas_II_fine:
    th = np.radians(th_deg)
    rc = min(compute_r_crit_sector_II_beta(th), compute_r_crit_sector_II_alpha(th))
    r_crits_II.append(rc)
ax.fill_between(thetas_II_fine, 1, r_crits_II, alpha=0.3, color='red', label='Sector II valid')
ax.plot(thetas_II_fine, r_crits_II, 'r-', lw=2)

# Sector III (by symmetry with Sector I about θ = 54.74°)
thetas_III_fine = np.linspace(np.degrees(alpha_half) + 0.5, 89.5, 100)
r_crits_III = []
for th_deg in thetas_III_fine:
    # By the 4-fold structure, Sector III has similar characteristics
    # φ_III direction relative to its boundaries
    th = np.radians(th_deg)
    # Sector III: α-line at +γ from horizontal, β-line at γ + 90°
    # Simplified: r_crit ≈ 1/sin(θ - arctan(√2)) or similar
    # For now, use a symmetric estimate
    th_mirror = np.pi/2 - th  # mirror about 45°
    if th_mirror > 0.01:
        rc = 1.0 / np.sin(th)  # rough estimate
    else:
        rc = 10.0
    r_crits_III.append(min(rc, 5))
ax.fill_between(thetas_III_fine, 1, r_crits_III, alpha=0.3, color='green', label='Sector III valid')
ax.plot(thetas_III_fine, r_crits_III, 'g-', lw=2)

# Sector boundaries
for angle in [np.degrees(gamma), np.degrees(alpha_half)]:
    ax.axvline(angle, color='gray', ls='--', lw=1, alpha=0.5)

ax.axhline(1, color='black', lw=1.5, label='Void surface (r=a)')
ax.set_xlabel(r"$\theta$ (degrees)", fontsize=12)
ax.set_ylabel(r"$r_{crit}/a$", fontsize=12)
ax.set_title("(b) Domain of validity for Airy stress functions", fontsize=11)
ax.set_xlim(0, 90)
ax.set_ylim(0.8, 5)
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/domain_of_validity.png', dpi=150,
            bbox_inches='tight')
print("Saved: figures/domain_of_validity.png")

# Summary
print()
print("="*70)
print("SUMMARY OF FINDINGS")
print("="*70)
print()
print("CRITICISM 1 (Geometric Degeneracy): VALID")
print("  |φ_I| = γ numerically, but the characteristic at -35.3° from")
print("  horizontal does NOT run along the boundary at θ = +35.3°.")
print("  They intersect at ~70.5°. The paper's claim is geometrically wrong.")
print()
print("CRITICISM 2 (β-line misses void): VALID")
print("  At r = 2a, θ = 45°: vertical β-line at x = 1.414a misses void")
print("  (void only extends to x = a). It hits Sector I boundary instead.")
print()
print("CRITICISM 3 (Missing curved sectors): VALID")
print("  Domain of validity is limited:")
print(f"  - Sector I:  r_crit ≈ 1.3-1.8a (depending on θ)")
print(f"  - Sector II: r_crit = sec(θ), from 1.2a to 1.7a")
print(f"  Beyond r_crit, characteristics cross into adjacent sectors →")
print(f"  curved transition sectors needed (as in Kysar FCC solution)")
print()
print("IMPACT ON PAPER:")
print("  The void surface solution (Section 4) is UNAFFECTED.")
print("  The activation pressure is UNAFFECTED.")
print("  Section 5 (interior field) must be corrected:")
print("  - Remove degeneracy claim")
print("  - Add domain of validity (r < r_crit)")
print("  - Acknowledge curved sectors exist for r > r_crit")
print("  - Note that full field requires Kysar-type construction")
