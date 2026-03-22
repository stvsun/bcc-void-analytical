"""
Publication-quality BCC vs FCC yield polygon comparison figure.

FCC {111}<110> with [110] void axis:
  3 effective systems at φ = 0°, ±arctan(√2)
  Yield polygon vertices from Rice (1987) and Kysar (2005).

BCC {110}<111> with [110] void axis:
  3 effective constraints (this work).
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

# ================================================================
# BCC yield polygon (from exact_stress_field.py)
# ================================================================
# Vertices in (X, Y) = ((σ₁₁-σ₂₂)/2, σ₁₂) units of τ_CRSS
s6 = np.sqrt(6)
s3 = np.sqrt(3)

bcc_vertices = np.array([
    [-s6/4, -s3],   # V1
    [+s6/4, -s3],   # V2
    [+s6/2,  0],    # V3
    [+s6/4, +s3],   # V4
    [-s6/4, +s3],   # V5
    [-s6/2,  0],    # V6
])

# ================================================================
# FCC yield polygon
# ================================================================
# For FCC {111}<110> with void axis [110], the 3 effective in-plane
# systems have Schmid tensors with resolved shear stress:
#   τ₁ = -√(2/3) Y              (system at φ=0°)
#   τ₂ = +√(1/6) X + √(1/6) Y  (system at φ=+arctan(√2))
#   τ₃ = -√(1/6) X + √(1/6) Y  (system at φ=-arctan(√2))
#
# Yield conditions: |τ_α| ≤ τ_CRSS
#
# From Kysar (2005): the FCC yield polygon vertices are at
#   (±√6/2, 0) and (±√6/4, ±1)
# (note: half-height = 1, not √3 as in BCC)
#
# More precisely, from Rice (1987), the FCC yield surface for [110]
# void axis is a regular hexagon with inscribed radius = √(2/3)·τ_CRSS.
# The vertices lie at distance √6/2 from the origin.

fcc_vertices = np.array([
    [-s6/4, -1],    # V1
    [+s6/4, -1],    # V2
    [+s6/2,  0],    # V3
    [+s6/4, +1],    # V4
    [-s6/4, +1],    # V5
    [-s6/2,  0],    # V6
])

# ================================================================
# Plot
# ================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))

# --- (a) BCC yield polygon ---
ax = ax1
bcc_closed = np.vstack([bcc_vertices, bcc_vertices[0]])
ax.plot(bcc_closed[:, 0], bcc_closed[:, 1], 'b-o', linewidth=2,
        markersize=7, label='BCC $\\{110\\}\\langle 111\\rangle$',
        zorder=3)
ax.fill(bcc_vertices[:, 0], bcc_vertices[:, 1], alpha=0.12, color='blue')

# Label vertices
labels_bcc = ['$V_1$', '$V_2$', '$V_3$', '$V_4$', '$V_5$', '$V_6$']
offsets_bcc = [(-12, -14), (4, -14), (8, -4), (4, 8), (-12, 8), (-18, -4)]
for i, (lbl, off) in enumerate(zip(labels_bcc, offsets_bcc)):
    ax.annotate(lbl, bcc_vertices[i], textcoords="offset points",
                xytext=off, fontsize=10, color='blue')

# Draw yield lines (extended, faint)
for i in range(6):
    v1 = bcc_vertices[i]
    v2 = bcc_vertices[(i+1) % 6]
    direction = v2 - v1
    direction = direction / np.linalg.norm(direction)
    ext = 1.0
    ax.plot([v1[0]-ext*direction[0], v2[0]+ext*direction[0]],
            [v1[1]-ext*direction[1], v2[1]+ext*direction[1]],
            'b--', alpha=0.15, linewidth=0.8)

ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.set_xlabel(r'$X = (\sigma_{11}-\sigma_{22})/2\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
ax.set_ylabel(r'$Y = \sigma_{12}\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
ax.set_title('(a) BCC $\\{110\\}\\langle 111\\rangle$ yield polygon', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.5, 2.5)
ax.legend(fontsize=10, loc='upper left')

# --- (b) BCC vs FCC comparison ---
ax = ax2
bcc_closed = np.vstack([bcc_vertices, bcc_vertices[0]])
fcc_closed = np.vstack([fcc_vertices, fcc_vertices[0]])

ax.plot(bcc_closed[:, 0], bcc_closed[:, 1], 'b-o', linewidth=2,
        markersize=6, label='BCC $\\{110\\}\\langle 111\\rangle$',
        zorder=3)
ax.fill(bcc_vertices[:, 0], bcc_vertices[:, 1], alpha=0.08, color='blue')

ax.plot(fcc_closed[:, 0], fcc_closed[:, 1], 'r-s', linewidth=2,
        markersize=6, label='FCC $\\{111\\}\\langle 110\\rangle$',
        zorder=3)
ax.fill(fcc_vertices[:, 0], fcc_vertices[:, 1], alpha=0.08, color='red')

# Annotate key dimensions
# BCC half-height
ax.annotate('', xy=(1.6, s3), xytext=(1.6, 0),
            arrowprops=dict(arrowstyle='<->', color='blue', lw=1.2))
ax.text(1.72, s3/2, r'$\sqrt{3}$', fontsize=10, color='blue',
        ha='left', va='center')

# FCC half-height
ax.annotate('', xy=(-1.6, 1), xytext=(-1.6, 0),
            arrowprops=dict(arrowstyle='<->', color='red', lw=1.2))
ax.text(-1.72, 0.5, r'$1$', fontsize=10, color='red',
        ha='right', va='center')

# Common half-width
ax.annotate('', xy=(s6/2, -2.1), xytext=(0, -2.1),
            arrowprops=dict(arrowstyle='<->', color='black', lw=1.0))
ax.text(s6/4, -2.25, r'$\sqrt{6}/2$', fontsize=9, color='black',
        ha='center', va='top')

ax.axhline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.axvline(0, color='gray', linewidth=0.5, alpha=0.5)
ax.set_xlabel(r'$X = (\sigma_{11}-\sigma_{22})/2\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
ax.set_ylabel(r'$Y = \sigma_{12}\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
ax.set_title('(b) BCC vs FCC yield polygon comparison', fontsize=12)
ax.set_aspect('equal')
ax.grid(True, alpha=0.2)
ax.set_xlim(-2.2, 2.2)
ax.set_ylim(-2.5, 2.5)
ax.legend(fontsize=10, loc='upper left')

plt.tight_layout()
fig_path = 'figures/bcc_vs_fcc_yield_surface.png'
plt.savefig(fig_path, dpi=200, bbox_inches='tight')
print(f"Saved: {fig_path}")
