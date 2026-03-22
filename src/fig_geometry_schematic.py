"""
Geometry schematic for the BCC void problem.
Shows: cylindrical void, crystal axes, polar coordinates, loading.
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, Arc
import matplotlib.patches as mpatches

fig, ax = plt.subplots(1, 1, figsize=(7, 7))

# Void
void = Circle((0, 0), 1.0, fill=True, facecolor='#f0f0f0',
              edgecolor='black', linewidth=2, zorder=5)
ax.add_patch(void)
ax.text(0, 0, r'Void', ha='center', va='center', fontsize=11,
        style='italic', color='gray')
ax.text(0.55, 0.55, r'$a$', fontsize=13, ha='center', va='center')

# Radius line
ax.plot([0, 0.707], [0, 0.707], 'k-', linewidth=1)

# Outer region (annular domain hint)
outer = Circle((0, 0), 4.0, fill=False, edgecolor='gray',
               linewidth=0.8, linestyle='--', zorder=1)
ax.add_patch(outer)

# Crystal axes
arrow_len = 4.5
# e1' = [001]
ax.annotate('', xy=(arrow_len, 0), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='blue', lw=2))
ax.text(arrow_len + 0.15, 0.15,
        r"$\mathbf{e}'_1 = [001]$", fontsize=12, color='blue',
        ha='left', va='bottom')

# e2' = [-110]/√2
ax.annotate('', xy=(0, arrow_len), xytext=(0, 0),
            arrowprops=dict(arrowstyle='->', color='red', lw=2))
ax.text(0.15, arrow_len + 0.1,
        r"$\mathbf{e}'_2 = [\bar{1}10]/\sqrt{2}$", fontsize=12,
        color='red', ha='left', va='bottom')

# Polar coordinates
r_line = 3.0
theta_line = np.radians(35)
x_p = r_line * np.cos(theta_line)
y_p = r_line * np.sin(theta_line)

ax.plot([0, x_p], [0, y_p], 'k-', linewidth=1, zorder=2)
ax.plot(x_p, y_p, 'ko', markersize=4, zorder=6)
ax.text(x_p + 0.2, y_p + 0.15, r'$(r, \theta)$', fontsize=12)

# r label
ax.text(r_line/2 * np.cos(theta_line) + 0.15,
        r_line/2 * np.sin(theta_line) + 0.15,
        r'$r$', fontsize=13, ha='center', va='center')

# θ arc
theta_arc = Arc((0, 0), 1.5, 1.5, angle=0, theta1=0,
                theta2=np.degrees(theta_line), color='green', linewidth=1.5)
ax.add_patch(theta_arc)
ax.text(1.0, 0.3, r'$\theta$', fontsize=13, color='green')

# Far-field loading arrows (equibiaxial pressure)
for angle in np.linspace(0, 2*np.pi, 12, endpoint=False):
    r_start = 4.3
    r_end = 3.7
    x0 = r_start * np.cos(angle)
    y0 = r_start * np.sin(angle)
    x1 = r_end * np.cos(angle)
    y1 = r_end * np.sin(angle)
    ax.annotate('', xy=(x1, y1), xytext=(x0, y0),
                arrowprops=dict(arrowstyle='->', color='darkgreen',
                                lw=1.5, shrinkA=0, shrinkB=0))

ax.text(3.8, 3.8, r'$p$', fontsize=14, color='darkgreen',
        ha='center', va='center')
ax.text(-3.8, 3.8, r'$p$', fontsize=14, color='darkgreen',
        ha='center', va='center')

# σ_rr = 0 at void surface
ax.text(1.15, -0.35, r'$\sigma_{rr}=0$', fontsize=10,
        color='black', ha='left', style='italic')
ax.text(1.15, -0.65, r'$\sigma_{r\theta}=0$', fontsize=10,
        color='black', ha='left', style='italic')

# Void axis label
ax.text(0, -1.4, r'Void axis $\parallel [110]$', fontsize=11,
        ha='center', va='top', style='italic',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow',
                  edgecolor='gray', alpha=0.8))

# Sector boundaries (faint)
for theta_b in [np.radians(35.26), np.radians(54.74)]:
    r_outer = 4.0
    ax.plot([0, r_outer*np.cos(theta_b)], [0, r_outer*np.sin(theta_b)],
            'k:', linewidth=0.7, alpha=0.4)

ax.text(3.5*np.cos(np.radians(17)), 3.5*np.sin(np.radians(17)),
        r'I', fontsize=11, color='gray', ha='center', va='center',
        style='italic')
ax.text(3.5*np.cos(np.radians(45)), 3.5*np.sin(np.radians(45)),
        r'II', fontsize=11, color='gray', ha='center', va='center',
        style='italic')
ax.text(3.5*np.cos(np.radians(72)), 3.5*np.sin(np.radians(72)),
        r'III', fontsize=11, color='gray', ha='center', va='center',
        style='italic')

# Formatting
ax.set_xlim(-5, 5.5)
ax.set_ylim(-2.5, 5.5)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title('Problem geometry: plane strain, void axis $\\parallel [110]$',
             fontsize=13, pad=15)

plt.tight_layout()
fig_path = 'figures/geometry_schematic.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight', facecolor='white')
print(f"Saved: {fig_path}")
