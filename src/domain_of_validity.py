"""
Generate the domain-of-validity figure for the BCC void interior field.

Shows:
(a) Characteristic network with sector boundaries and validity domain
(b) r_crit(θ) for each sector
"""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Circle
from matplotlib.collections import PatchCollection

# Key angles
gamma = 0.5 * np.arctan(2*np.sqrt(2))   # ≈ 35.26°
alpha_half = np.arctan(np.sqrt(2))        # ≈ 54.74°
phi_I = -gamma
phi_II = 0.0
phi_III = 0.5 * (np.pi - np.arctan(2*np.sqrt(2)))


def r_crit_sector_I(theta):
    """r_crit/a for Sector I: min of two characteristic constraints."""
    arg = phi_I - theta
    r1 = 1.0 / abs(np.sin(arg)) if abs(np.sin(arg)) > 1e-10 else 100
    r2 = 1.0 / abs(np.cos(arg)) if abs(np.cos(arg)) > 1e-10 else 100
    return min(r1, r2)


def r_crit_sector_II(theta):
    """r_crit/a for Sector II: min(csc θ, sec θ)."""
    r1 = 1.0 / abs(np.sin(theta)) if abs(np.sin(theta)) > 1e-10 else 100
    r2 = 1.0 / abs(np.cos(theta)) if abs(np.cos(theta)) > 1e-10 else 100
    return min(r1, r2)


def r_crit_sector_III(theta):
    """r_crit/a for Sector III."""
    arg = phi_III - theta
    r1 = 1.0 / abs(np.sin(arg)) if abs(np.sin(arg)) > 1e-10 else 100
    r2 = 1.0 / abs(np.cos(arg)) if abs(np.cos(arg)) > 1e-10 else 100
    return min(r1, r2)


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# =========================================================
# Panel (a): Characteristic network around void
# =========================================================
ax = ax1

# Void circle
th = np.linspace(0, np.pi/2, 200)
ax.plot(np.cos(th), np.sin(th), 'k-', lw=2.5)
ax.fill_between(np.cos(th), 0, np.sin(th), color='lightgray', alpha=0.3)

# Sector boundaries (radial lines)
for angle, ls in [(0, '-'), (gamma, '--'), (alpha_half, '--'), (np.pi/2, '-')]:
    ax.plot([0, 2.8*np.cos(angle)], [0, 2.8*np.sin(angle)],
            'k', ls=ls, lw=0.8, alpha=0.5)

# Shade validity domain for each sector
# Sector I
thI = np.linspace(0.01, gamma - 0.01, 80)
rI = np.array([r_crit_sector_I(t) for t in thI])
rI = np.clip(rI, 1, 5)
xI_out = rI * np.cos(thI)
yI_out = rI * np.sin(thI)
xI_in = np.cos(thI)[::-1]
yI_in = np.sin(thI)[::-1]
ax.fill(np.concatenate([xI_out, xI_in]),
        np.concatenate([yI_out, yI_in]),
        color='royalblue', alpha=0.2, label='Sector I valid')
ax.plot(xI_out, yI_out, 'b-', lw=2)

# Sector II
thII = np.linspace(gamma + 0.01, alpha_half - 0.01, 80)
rII = np.array([r_crit_sector_II(t) for t in thII])
rII = np.clip(rII, 1, 5)
xII_out = rII * np.cos(thII)
yII_out = rII * np.sin(thII)
xII_in = np.cos(thII)[::-1]
yII_in = np.sin(thII)[::-1]
ax.fill(np.concatenate([xII_out, xII_in]),
        np.concatenate([yII_out, yII_in]),
        color='firebrick', alpha=0.2, label='Sector II valid')
ax.plot(xII_out, yII_out, 'r-', lw=2)

# Sector III
thIII = np.linspace(alpha_half + 0.01, np.pi/2 - 0.01, 80)
rIII = np.array([r_crit_sector_III(t) for t in thIII])
rIII = np.clip(rIII, 1, 5)
xIII_out = rIII * np.cos(thIII)
yIII_out = rIII * np.sin(thIII)
xIII_in = np.cos(thIII)[::-1]
yIII_in = np.sin(thIII)[::-1]
ax.fill(np.concatenate([xIII_out, xIII_in]),
        np.concatenate([yIII_out, yIII_in]),
        color='forestgreen', alpha=0.2, label='Sector III valid')
ax.plot(xIII_out, yIII_out, 'g-', lw=2)

# Draw some characteristic lines from void surface
# Sector I: α-lines at angle phi_I + 90° = +54.74°, β-lines at phi_I = -35.26°
alpha_dir_I = phi_I + np.pi/2  # ≈ +54.74°
beta_dir_I = phi_I             # ≈ -35.26°

for th0 in np.linspace(2, 33, 5):
    th0_rad = np.radians(th0)
    x0, y0 = np.cos(th0_rad), np.sin(th0_rad)
    # α-line (forward from void)
    t_vals = np.linspace(0, 1.5, 100)
    xl = x0 + t_vals * np.cos(alpha_dir_I)
    yl = y0 + t_vals * np.sin(alpha_dir_I)
    rl = np.sqrt(xl**2 + yl**2)
    angle_l = np.arctan2(yl, xl)
    mask = (rl > 0.95) & (xl > -0.1) & (yl > -0.1) & (angle_l < gamma + 0.1) & (angle_l > -0.05)
    if np.any(mask):
        ax.plot(xl[mask], yl[mask], 'b-', lw=0.5, alpha=0.4)
    # β-line (forward from void)
    xl = x0 + t_vals * np.cos(beta_dir_I)
    yl = y0 + t_vals * np.sin(beta_dir_I)
    rl = np.sqrt(xl**2 + yl**2)
    angle_l = np.arctan2(yl, xl)
    mask = (rl > 0.95) & (xl > -0.1) & (yl > -0.1) & (angle_l < gamma + 0.05) & (angle_l > -0.05)
    if np.any(mask):
        ax.plot(xl[mask], yl[mask], 'b--', lw=0.5, alpha=0.4)

# Sector II: α-lines horizontal (0°), β-lines vertical (90°)
for th0 in np.linspace(37, 53, 4):
    th0_rad = np.radians(th0)
    x0, y0 = np.cos(th0_rad), np.sin(th0_rad)
    # horizontal α-line
    t_vals = np.linspace(-0.5, 1.5, 100)
    xl = x0 + t_vals
    yl = np.full_like(t_vals, y0)
    rl = np.sqrt(xl**2 + yl**2)
    mask = (rl > 0.95) & (xl > 0) & (rl < 3)
    if np.any(mask):
        ax.plot(xl[mask], yl[mask], 'r-', lw=0.5, alpha=0.4)
    # vertical β-line
    yl2 = y0 + t_vals
    xl2 = np.full_like(t_vals, x0)
    rl2 = np.sqrt(xl2**2 + yl2**2)
    mask2 = (rl2 > 0.95) & (yl2 > 0) & (rl2 < 3)
    if np.any(mask2):
        ax.plot(xl2[mask2], yl2[mask2], 'r--', lw=0.5, alpha=0.4)

# Sector labels
for angle, label, color in [(np.radians(17), 'I', 'royalblue'),
                              (np.radians(45), 'II', 'firebrick'),
                              (np.radians(73), 'III', 'forestgreen')]:
    ax.text(0.55*np.cos(angle), 0.55*np.sin(angle), label, fontsize=13,
            ha='center', va='center', fontweight='bold', color=color)

# Boundary labels
ax.text(2.3*np.cos(gamma) + 0.05, 2.3*np.sin(gamma) + 0.05,
        r'$\theta_1 = \gamma$', fontsize=9, rotation=np.degrees(gamma),
        ha='left', va='bottom')
ax.text(2.0*np.cos(alpha_half) - 0.15, 2.0*np.sin(alpha_half) + 0.05,
        r'$\theta_2$', fontsize=9, ha='right', va='bottom')

ax.set_xlim(-0.15, 2.5)
ax.set_ylim(-0.15, 2.5)
ax.set_aspect('equal')
ax.set_xlabel(r"$x'_1/a$", fontsize=12)
ax.set_ylabel(r"$x'_2/a$", fontsize=12)
ax.set_title("(a) Validity domain (shaded) and characteristics", fontsize=11)
ax.legend(fontsize=8, loc='upper left', framealpha=0.9)

# =========================================================
# Panel (b): r_crit(θ) for each sector
# =========================================================
ax = ax2

# Sector I
thI = np.linspace(0.5, np.degrees(gamma) - 0.5, 200)
rcI = [r_crit_sector_I(np.radians(t)) for t in thI]
ax.fill_between(thI, 1, rcI, alpha=0.25, color='royalblue')
ax.plot(thI, rcI, 'b-', lw=2, label='Sector I')

# Sector II
thII = np.linspace(np.degrees(gamma) + 0.5, np.degrees(alpha_half) - 0.5, 200)
rcII = [r_crit_sector_II(np.radians(t)) for t in thII]
ax.fill_between(thII, 1, rcII, alpha=0.25, color='firebrick')
ax.plot(thII, rcII, 'r-', lw=2, label='Sector II')

# Sector III
thIII = np.linspace(np.degrees(alpha_half) + 0.5, 89.5, 200)
rcIII = [r_crit_sector_III(np.radians(t)) for t in thIII]
ax.fill_between(thIII, 1, rcIII, alpha=0.25, color='forestgreen')
ax.plot(thIII, rcIII, 'g-', lw=2, label='Sector III')

# Sector boundaries
for angle in [np.degrees(gamma), np.degrees(alpha_half)]:
    ax.axvline(angle, color='gray', ls='--', lw=1, alpha=0.5)
    ax.text(angle + 0.5, 2.4, f'{angle:.1f}°', fontsize=8, color='gray')

ax.axhline(1, color='black', lw=1.5, ls='-', alpha=0.5)
ax.text(45, 1.03, r'void surface ($r = a$)', fontsize=8, ha='center', va='bottom',
        color='black', alpha=0.6)

# Annotations
ax.annotate(r'$r_{\mathrm{crit}} = \sec\gamma \approx 1.22$',
            xy=(1, 1.0/np.cos(gamma)), fontsize=9,
            xytext=(8, 2.0), arrowprops=dict(arrowstyle='->', color='blue'),
            color='blue')
ax.annotate(r'$r_{\mathrm{crit}} = \sqrt{2} \approx 1.41$',
            xy=(45, np.sqrt(2)), fontsize=9,
            xytext=(50, 2.0), arrowprops=dict(arrowstyle='->', color='red'),
            color='red')

ax.set_xlabel(r"$\theta$ (degrees)", fontsize=12)
ax.set_ylabel(r"$r_{\mathrm{crit}}/a$", fontsize=12)
ax.set_title("(b) Critical radius for Airy solution validity", fontsize=11)
ax.set_xlim(0, 90)
ax.set_ylim(0.9, 2.6)
ax.legend(fontsize=9, loc='upper right')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('/tmp/bcc-void-analytical/figures/domain_of_validity.png',
            dpi=200, bbox_inches='tight')
print("Saved: figures/domain_of_validity.png")
