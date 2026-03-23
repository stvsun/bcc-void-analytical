"""
Derive effective in-plane slip systems for BCC with BOTH slip families:
  {110}<111>  (12 systems)
  {112}<111>  (12 systems)
  Total: 24 systems

This gives the complete BCC yield polygon under plane strain with
void axis || [110].

The {112}<111> systems are:
  Slip plane normals: {112} family -> 12 planes
  Slip directions: <111> family
  Each {112} plane has one <111> direction in it.
"""

import numpy as np
import sympy as sp
from sympy import sqrt, Matrix, simplify, Rational

# ============================================================
# Step 1: Define all 24 BCC slip systems
# ============================================================

# {110}<111>: 12 systems (same as before)
slip_110 = [
    ((1, 1, 0), (-1, 1, 1)),    # 1
    ((1, 1, 0), (1, -1, 1)),    # 2
    ((1, -1, 0), (1, 1, 1)),    # 3
    ((1, -1, 0), (-1, -1, 1)),  # 4
    ((1, 0, 1), (-1, 1, 1)),    # 5
    ((1, 0, 1), (1, 1, -1)),    # 6
    ((1, 0, -1), (1, 1, 1)),    # 7
    ((1, 0, -1), (-1, 1, -1)),  # 8
    ((0, 1, 1), (1, -1, 1)),    # 9
    ((0, 1, 1), (1, 1, -1)),    # 10
    ((0, 1, -1), (1, 1, 1)),    # 11
    ((0, 1, -1), (1, -1, -1)),  # 12
]

# {112}<111>: 12 systems
# Each {112} plane has exactly one <111> direction lying in it.
# The 12 systems are:
slip_112 = [
    ((1, 1, 2), (-1, 1, 1)),    # 13  -- n·s = -1+1+2 = 2 ≠ 0! Wrong.
]
# Let me be more careful. For {112}<111>:
# n = (hkl) with h²+k²+l² = 6, s = [uvw] with u²+v²+w² = 3
# Constraint: n·s = 0

# Systematic enumeration of {112} planes:
# (112), (1-12), (-112), (121), (1-21), (-121),
# (211), (2-11), (-211), (122)... wait, let me be systematic.
# {112} means all permutations of (1,1,2) with sign changes.
# That gives: (112), (1-12), (-112), (-1-12),
#             (121), (1-21), (-121), (-1-21),
#             (211), (2-11), (-211), (-2-11)
# But planes (hkl) and (-h,-k,-l) are the same, so 12 distinct planes.

# For each plane, find the <111> direction in it:
# n·s = 0 with s ∈ <111>

planes_112 = []
for signs_n in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1)]:
    for perm in [(0,1,2), (1,2,0), (2,0,1)]:  # permutations of (1,1,2)
        base = [1, 1, 2]
        n = [0, 0, 0]
        for i in range(3):
            n[perm[i]] = signs_n[i] * base[i]
        n = tuple(n)
        # Check we haven't already added the negative
        neg_n = tuple(-x for x in n)
        already = False
        for existing_n, _ in planes_112:
            if existing_n == n or existing_n == neg_n:
                already = True
                break
        if already:
            continue
        # Find <111> direction in this plane
        for s_signs in [(1,1,1), (1,1,-1), (1,-1,1), (1,-1,-1),
                        (-1,1,1), (-1,1,-1), (-1,-1,1), (-1,-1,-1)]:
            s = s_signs
            dot = n[0]*s[0] + n[1]*s[1] + n[2]*s[2]
            if dot == 0:
                planes_112.append((n, s))
                break

slip_112 = planes_112

print("=" * 70)
print(f"BCC {{112}}<111> Slip Systems: {len(slip_112)} found")
print("=" * 70)
for i, (n, s) in enumerate(slip_112):
    n_vec = np.array(n, dtype=float)
    s_vec = np.array(s, dtype=float)
    dot = int(np.dot(n_vec, s_vec))
    print(f"  System {i+13:2d}: n = {str(n):14s}  s = {str(s):14s}  n·s = {dot}")

# ============================================================
# Step 2: Coordinate transformation (same as before)
# ============================================================
R = np.array([
    [0, 0, 1],
    [1/np.sqrt(2), -1/np.sqrt(2), 0],
    [1/np.sqrt(2), 1/np.sqrt(2), 0],
])

# ============================================================
# Step 3: Compute Schmid coefficients for ALL 24 systems
# ============================================================
print("\n" + "=" * 70)
print("In-plane Schmid coefficients: τ = a·X + b·Y")
print("X = (σ₁₁-σ₂₂)/2, Y = σ₁₂")
print("=" * 70)

all_systems = slip_110 + slip_112
schmid_all = []

for idx, (n, s) in enumerate(all_systems):
    n_vec = np.array(n, dtype=float)
    s_vec = np.array(s, dtype=float)
    n_hat = n_vec / np.linalg.norm(n_vec)
    s_hat = s_vec / np.linalg.norm(s_vec)

    n_p = R @ n_hat
    s_p = R @ s_hat

    P_11 = s_p[0] * n_p[0]
    P_12 = (s_p[0]*n_p[1] + s_p[1]*n_p[0]) / 2

    a_k = 2 * P_11
    b_k = 2 * P_12

    family = "{110}" if idx < 12 else "{112}"
    schmid_all.append((a_k, b_k))
    if abs(a_k) > 1e-10 or abs(b_k) > 1e-10:
        print(f"  Sys {idx+1:2d} ({family}): a = {a_k:+.6f}, b = {b_k:+.6f}")
    else:
        print(f"  Sys {idx+1:2d} ({family}): a = {a_k:+.6f}, b = {b_k:+.6f}  (INACTIVE)")

# ============================================================
# Step 4: Construct yield polygon from all 24 systems
# ============================================================
print("\n" + "=" * 70)
print("Combined yield polygon (all 24 systems)")
print("=" * 70)

# Find all vertices: intersections of pairs of yield lines
# Yield line for system k: a_k X + b_k Y = ±1
all_vertices = []
n_sys = len(schmid_all)

for i in range(n_sys):
    for si in [+1, -1]:
        for j in range(i+1, n_sys):
            for sj in [+1, -1]:
                ai, bi = schmid_all[i]
                aj, bj = schmid_all[j]
                A = np.array([[ai, bi], [aj, bj]])
                det = np.linalg.det(A)
                if abs(det) < 1e-10:
                    continue
                sol = np.linalg.solve(A, np.array([si, sj]))

                # Check feasibility: all constraints satisfied
                ok = True
                for k in range(n_sys):
                    tau_k = abs(schmid_all[k][0]*sol[0] + schmid_all[k][1]*sol[1])
                    if tau_k > 1.0 + 1e-6:
                        ok = False
                        break
                if not ok:
                    continue

                # Check for duplicates
                is_dup = False
                for v in all_vertices:
                    if abs(v[0]-sol[0]) < 1e-6 and abs(v[1]-sol[1]) < 1e-6:
                        is_dup = True
                        break
                if not is_dup:
                    all_vertices.append(sol)

# Sort by angle
angles = [np.arctan2(v[1], v[0]) for v in all_vertices]
order = np.argsort(angles)
all_vertices = [all_vertices[i] for i in order]

print(f"\nVertices: {len(all_vertices)}")
for i, v in enumerate(all_vertices):
    angle = np.degrees(np.arctan2(v[1], v[0]))
    r = np.sqrt(v[0]**2 + v[1]**2)
    # Which systems are active at this vertex?
    active = []
    for k in range(n_sys):
        tau_k = abs(schmid_all[k][0]*v[0] + schmid_all[k][1]*v[1])
        if abs(tau_k - 1.0) < 1e-4:
            family = "{110}" if k < 12 else "{112}"
            active.append(f"{k+1}({family})")
    print(f"  V{i+1:2d}: ({v[0]:+.5f}, {v[1]:+.5f})  angle={angle:+7.2f}°  "
          f"r={r:.5f}  active: {', '.join(active[:4])}")

# ============================================================
# Step 5: Compare with {110}-only polygon
# ============================================================
print("\n" + "=" * 70)
print("Comparison: {110}-only vs {110}+{112} combined")
print("=" * 70)

# {110}-only vertices
s6 = np.sqrt(6)
s3 = np.sqrt(3)
v110_only = [
    (-s6/4, -s3), (s6/4, -s3), (s6/2, 0),
    (s6/4, s3), (-s6/4, s3), (-s6/2, 0),
]

print(f"  {{110}}-only: 6 vertices, max|X| = {s6/2:.4f}, max|Y| = {s3:.4f}")
if all_vertices:
    max_X = max(abs(v[0]) for v in all_vertices)
    max_Y = max(abs(v[1]) for v in all_vertices)
    print(f"  Combined:    {len(all_vertices)} vertices, max|X| = {max_X:.4f}, max|Y| = {max_Y:.4f}")

    # Did the {112} systems truncate any faces?
    # Check: is each {110}-only vertex still a vertex of the combined polygon?
    for i, v110 in enumerate(v110_only):
        still_vertex = False
        for vc in all_vertices:
            if abs(vc[0]-v110[0]) < 1e-4 and abs(vc[1]-v110[1]) < 1e-4:
                still_vertex = True
                break
        status = "RETAINED" if still_vertex else "TRUNCATED by {112}"
        print(f"    {110}-only V{i+1}: ({v110[0]:+.5f}, {v110[1]:+.5f}) -> {status}")

# ============================================================
# Step 6: Plot comparison
# ============================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # (a) Combined yield polygon
    ax = ax1
    vc = np.array(all_vertices)
    vc_closed = np.vstack([vc, vc[0]])
    ax.plot(vc_closed[:, 0], vc_closed[:, 1], 'b-o', linewidth=2,
            markersize=5, label=r'$\{110\}+\{112\}$ combined')
    ax.fill(vc[:, 0], vc[:, 1], alpha=0.1, color='blue')

    # Overlay {110}-only
    v110 = np.array(v110_only)
    v110_closed = np.vstack([v110, v110[0]])
    ax.plot(v110_closed[:, 0], v110_closed[:, 1], 'r--s', linewidth=1.5,
            markersize=5, alpha=0.7, label=r'$\{110\}$ only')

    ax.set_xlabel(r'$X = (\sigma_{11}-\sigma_{22})/2\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
    ax.set_ylabel(r'$Y = \sigma_{12}\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
    ax.set_title(r'(a) BCC yield polygon: $\{110\}+\{112\}$ combined', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.5, 2.5)

    # (b) All three: {110}-only, combined, FCC
    ax = ax2
    ax.plot(vc_closed[:, 0], vc_closed[:, 1], 'b-o', linewidth=2,
            markersize=4, label=r'BCC $\{110\}+\{112\}$')
    ax.fill(vc[:, 0], vc[:, 1], alpha=0.08, color='blue')

    ax.plot(v110_closed[:, 0], v110_closed[:, 1], 'b--', linewidth=1,
            alpha=0.4, label=r'BCC $\{110\}$ only')

    # FCC
    fcc = np.array([(-s6/4, -1), (s6/4, -1), (s6/2, 0),
                    (s6/4, 1), (-s6/4, 1), (-s6/2, 0)])
    fcc_closed = np.vstack([fcc, fcc[0]])
    ax.plot(fcc_closed[:, 0], fcc_closed[:, 1], 'r-s', linewidth=2,
            markersize=4, label=r'FCC $\{111\}\langle 110\rangle$')
    ax.fill(fcc[:, 0], fcc[:, 1], alpha=0.08, color='red')

    ax.set_xlabel(r'$X\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
    ax.set_ylabel(r'$Y\;\;[\tau_{\mathrm{CRSS}}]$', fontsize=11)
    ax.set_title(r'(b) BCC (both families) vs FCC', fontsize=12)
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.2)
    ax.legend(fontsize=9)
    ax.set_xlim(-2.2, 2.2)
    ax.set_ylim(-2.5, 2.5)

    plt.tight_layout()
    fig_path = 'figures/bcc_combined_yield_surface.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

except ImportError:
    print("Matplotlib not available.")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
