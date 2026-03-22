"""
Sector solution for the combined {110}+{112}<111> BCC yield polygon.

The combined polygon has 10 vertices (decagon). This script:
1. Traces the void surface stress around the decagonal yield polygon
2. Identifies sector boundaries
3. Computes exact void surface stress in each sector
4. Computes the new activation pressure
5. Compares with {110}-only result
"""

import numpy as np

# ================================================================
# Combined yield polygon vertices (from derive_bcc_combined_slip.py)
# ================================================================
# 10 vertices, sorted by Mohr angle:
# Using exact values where possible.
# From the output:
#   V1: (-1.06066, 0)     = (-3√2/4, 0)     angle = 180°
#   V2: (-0.53033, -1.5)  = (-3√2/8, -3/2)  angle = -109.47°
#   V3: (-0.28420, -√3)                      angle = -99.32°
#   V4: (+0.28420, -√3)                      angle = -80.68°
#   V5: (+0.53033, -1.5)  = (+3√2/8, -3/2)  angle = -70.53°
#   V6: (+1.06066, 0)     = (+3√2/4, 0)     angle = 0°
#   V7: (+0.53033, +1.5)                     angle = +70.53°
#   V8: (+0.28420, +√3)                      angle = +80.68°
#   V9: (-0.28420, +√3)                      angle = +99.32°
#   V10:(-0.53033, +1.5)                     angle = +109.47°
#
# Note: 3√2/4 = 3/(2√2) = 1.06066, 3√2/8 = 0.53033

s2 = np.sqrt(2)
s3 = np.sqrt(3)
s6 = np.sqrt(6)

vertices = np.array([
    [-3*s2/4,    0],       # V1
    [-3*s2/8, -1.5],       # V2
    [-s6/4+s6/4*0.5361, -s3],  # V3 -- need exact value
    [ s6/4-s6/4*0.5361, -s3],  # V4
    [ 3*s2/8, -1.5],       # V5
    [ 3*s2/4,    0],       # V6
    [ 3*s2/8,  1.5],       # V7
    [ s6/4-s6/4*0.5361,  s3],  # V8
    [-s6/4+s6/4*0.5361,  s3],  # V9
    [-3*s2/8,  1.5],       # V10
])

# Actually let me recompute V3 and V4 exactly.
# V3 is at the intersection of:
#   Y = -√3 (from {110} systems 3,4: b = -√3/3, so Y = -√3 at |τ|=1)
#   and a {112} constraint.
# From the output: V3 = (-0.28420, -1.73205)
# The {112} system active at V3: systems 17 and 21
# System 17: a = +0.471405, b = +0.500000 → τ = 0.471405 X + 0.5 Y
#   At V3: 0.471405*(-0.28420) + 0.5*(-1.73205) = -0.13389 - 0.86603 = -0.99992 ≈ -1 ✓
# System 21: a = -0.471405, b = -0.500000 → τ = -0.471405 X - 0.5 Y
#   At V3: -0.471405*(-0.28420) - 0.5*(-1.73205) = 0.13389 + 0.86603 = 0.99992 ≈ 1 ✓
# So V3 is at intersection of Y = -√3 and 0.471405 X + 0.5 Y = -1
# → 0.471405 X + 0.5*(-√3) = -1 → 0.471405 X = -1 + √3/2 = -1 + 0.86603 = -0.13397
# → X = -0.13397/0.471405 = -0.28416 ✓
# Exact: 0.471405 = √(2/9) = √2/3? Let me check: (√2/3)² = 2/9 = 0.2222, √0.2222 = 0.4714 ✓
# So a = √2/3, b = 1/2 for system 17.
# X = (-1 + √3/2) / (√2/3) = (√3/2 - 1) * 3/√2 = 3(√3-2)/(2√2)
# = 3(1.7321-2)/(2*1.4142) = 3*(-0.2679)/2.8284 = -0.8038/2.8284 = -0.2842 ✓

# Exact vertices:
# V3: X = 3(√3-2)/(2√2), Y = -√3
# V4: X = -3(√3-2)/(2√2) = 3(2-√3)/(2√2), Y = -√3
# V1: X = -3/(2√2) = -3√2/4, Y = 0
# V6: X = +3/(2√2) = +3√2/4, Y = 0
# V2: intersection of {112} system 19 (a=-√2·√(2/9)·3=-...) and system 17...
# Actually, V2 = (-3√2/8, -3/2). Let me verify:
# At V2: system 19: a=-0.942809, b=-0.333333 → -0.942809*(-0.53033) - 0.333333*(-1.5)
#   = 0.5 + 0.5 = 1.0 ✓
# And system 22: a=-0.942809, b=+0.333333 → -0.942809*(-0.53033) + 0.333333*(-1.5)
#   = 0.5 - 0.5 = 0.0 ✗  Not active.
# Let me check which systems are active at V2 more carefully.

# Let me just use the numerical vertices from the derivation code.
vertices = np.array([
    [-1.06066,  0.00000],  # V1
    [-0.53033, -1.50000],  # V2
    [-0.28420, -1.73205],  # V3
    [ 0.28420, -1.73205],  # V4
    [ 0.53033, -1.50000],  # V5
    [ 1.06066,  0.00000],  # V6
    [ 0.53033,  1.50000],  # V7
    [ 0.28420,  1.73205],  # V8
    [-0.28420,  1.73205],  # V9
    [-0.53033,  1.50000],  # V10
])

# ================================================================
# Schmid coefficients for all 24 systems
# ================================================================
# From the derivation output (only non-zero ones matter):
schmid_all = [
    (0, 0),                      # 1  {110} INACTIVE
    (0, 0),                      # 2  {110} INACTIVE
    (0, -0.577350),              # 3  {110}
    (0, -0.577350),              # 4  {110}
    (+0.816497, +0.288675),      # 5  {110}
    (-0.816497, +0.288675),      # 6  {110}
    (-0.816497, -0.288675),      # 7  {110}
    (+0.816497, -0.288675),      # 8  {110}
    (+0.816497, -0.288675),      # 9  {110}
    (-0.816497, -0.288675),      # 10 {110}
    (-0.816497, +0.288675),      # 11 {110}
    (+0.816497, +0.288675),      # 12 {110}
    (-0.942809,  0.000000),      # 13 {112}
    (-0.471405, -0.166667),      # 14 {112}
    (+0.471405, -0.166667),      # 15 {112}
    (-0.942809,  0.000000),      # 16 {112}
    (+0.471405, +0.500000),      # 17 {112}
    (+0.471405, -0.500000),      # 18 {112}
    (-0.942809, -0.333333),      # 19 {112}
    (-0.471405, +0.166667),      # 20 {112}
    (-0.471405, -0.500000),      # 21 {112}
    (-0.942809, +0.333333),      # 22 {112}
    (+0.471405, -0.500000),      # 23 {112}  same as 18
    (-0.471405, -0.166667),      # 24 {112}  same as 14
]

# ================================================================
# Trace void surface stress on the combined polygon
# ================================================================
print("=" * 70)
print("Void Surface Stress: Combined {110}+{112} Yield Polygon")
print("=" * 70)

N_theta = 360
thetas = np.linspace(0, np.pi, N_theta + 1)
void_X = np.zeros(N_theta + 1)
void_Y = np.zeros(N_theta + 1)
void_sm = np.zeros(N_theta + 1)
void_stt = np.zeros(N_theta + 1)
void_active = np.zeros(N_theta + 1, dtype=int)

for ti, theta in enumerate(thetas):
    c2 = np.cos(2*theta)
    s2_val = np.sin(2*theta)

    R_min = float('inf')
    best_k = -1
    for k in range(24):
        a_k, b_k = schmid_all[k]
        if abs(a_k) < 1e-10 and abs(b_k) < 1e-10:
            continue
        denom = a_k * c2 + b_k * s2_val
        if abs(denom) < 1e-12:
            continue
        for sign in [+1, -1]:
            R = sign / denom
            if R > 1e-10:
                X_c = R * c2
                Y_c = R * s2_val
                ok = True
                for m in range(24):
                    am, bm = schmid_all[m]
                    if abs(am*X_c + bm*Y_c) > 1.0 + 1e-6:
                        ok = False
                        break
                if ok and R < R_min:
                    R_min = R
                    best_k = k

    void_X[ti] = R_min * c2
    void_Y[ti] = R_min * s2_val
    void_sm[ti] = -(void_X[ti]*c2 + void_Y[ti]*s2_val)
    void_stt[ti] = 2 * void_sm[ti]
    void_active[ti] = best_k + 1

# Identify sector boundaries
print("\nSector boundaries:")
boundaries = []
for ti in range(1, len(thetas)):
    if void_active[ti] != void_active[ti-1]:
        theta_deg = np.degrees(thetas[ti])
        boundaries.append(theta_deg)
        fam_prev = "{110}" if void_active[ti-1] <= 12 else "{112}"
        fam_next = "{110}" if void_active[ti] <= 12 else "{112}"
        print(f"  θ ≈ {theta_deg:6.1f}°: sys {void_active[ti-1]}({fam_prev}) "
              f"→ sys {void_active[ti]}({fam_next})")

print(f"\nTotal sectors in [0°, 180°]: {len(boundaries) + 1}")

# Activation pressure
sm_max = np.max(np.abs(void_sm))
theta_max = np.degrees(thetas[np.argmax(np.abs(void_sm))])
print(f"\nActivation pressure:")
print(f"  p* = max|σ_m| = {sm_max:.6f} τ_CRSS  (at θ ≈ {theta_max:.1f}°)")
print(f"\nComparison:")
print(f"  {{110}}-only:     p* = 3√6/4 = {3*s6/4:.6f} τ_CRSS")
print(f"  {{110}}+{{112}}:    p* = {sm_max:.6f} τ_CRSS")
print(f"  FCC:             p* = √6/2  = {s6/2:.6f} τ_CRSS")
print(f"  Ratio BCC/FCC:   {sm_max / (s6/2):.4f}")

# Key angles
print(f"\nVoid surface stress at key angles:")
for th_deg in [0, 20, 35, 45, 55, 70, 90]:
    ti = int(th_deg / 180 * N_theta)
    fam = "{110}" if void_active[ti] <= 12 else "{112}"
    print(f"  θ = {th_deg:3d}°: σ_θθ = {void_stt[ti]:+.5f}, "
          f"active = sys {void_active[ti]} ({fam})")

# ================================================================
# Plot
# ================================================================
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # Also compute {110}-only for comparison
    schmid_110 = schmid_all[:12]
    void_stt_110 = np.zeros(N_theta + 1)
    for ti, theta in enumerate(thetas):
        c2 = np.cos(2*theta)
        s2_val = np.sin(2*theta)
        R_min = float('inf')
        for k in range(12):
            a_k, b_k = schmid_110[k]
            if abs(a_k) < 1e-10 and abs(b_k) < 1e-10:
                continue
            denom = a_k * c2 + b_k * s2_val
            if abs(denom) < 1e-12:
                continue
            for sign in [+1, -1]:
                R = sign / denom
                if R > 1e-10:
                    X_c = R * c2
                    Y_c = R * s2_val
                    ok = True
                    for m in range(12):
                        am, bm = schmid_110[m]
                        if abs(am*X_c + bm*Y_c) > 1.0 + 1e-6:
                            ok = False
                            break
                    if ok and R < R_min:
                        R_min = R
        sm_110 = -(R_min*c2*c2 + R_min*s2_val*s2_val)
        void_stt_110[ti] = 2 * sm_110

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

    # (a) Void surface σ_θθ comparison
    ax = axes[0]
    ax.plot(np.degrees(thetas), void_stt, 'b-', linewidth=2,
            label=r'$\{110\}+\{112\}$ combined')
    ax.plot(np.degrees(thetas), void_stt_110, 'r--', linewidth=1.5,
            label=r'$\{110\}$ only')
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$\sigma_{\theta\theta}(a, \theta) / \tau_{\mathrm{CRSS}}$', fontsize=12)
    ax.set_title(r'(a) Hoop stress at void surface', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)
    for b in boundaries:
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.3)

    # (b) Active system vs θ
    ax = axes[1]
    ax.plot(np.degrees(thetas), void_active, 'b-', linewidth=1.5)
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel('Active slip system', fontsize=12)
    ax.set_title('(b) Active system around void', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=12.5, color='red', linestyle='--', alpha=0.5,
               label=r'$\{110\}/\{112\}$ boundary')
    ax.legend(fontsize=9)
    for b in boundaries:
        ax.axvline(x=b, color='gray', linestyle=':', alpha=0.3)

    # (c) σ_m comparison
    ax = axes[2]
    sm_combined = void_sm
    sm_110only = void_stt_110 / 2
    ax.plot(np.degrees(thetas), np.abs(sm_combined), 'b-', linewidth=2,
            label=r'$|σ_m|$: $\{110\}+\{112\}$')
    ax.plot(np.degrees(thetas), np.abs(sm_110only), 'r--', linewidth=1.5,
            label=r'$|σ_m|$: $\{110\}$ only')
    ax.axhline(y=sm_max, color='blue', linestyle=':', alpha=0.5)
    ax.axhline(y=3*s6/4, color='red', linestyle=':', alpha=0.5)
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$|\sigma_m| / \tau_{\mathrm{CRSS}}$', fontsize=12)
    ax.set_title('(c) Mean stress magnitude (→ activation pressure)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)

    plt.tight_layout()
    fig_path = 'figures/combined_sector_solution.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"\nFigure saved: {fig_path}")

except ImportError:
    pass

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
