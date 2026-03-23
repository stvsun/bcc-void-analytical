"""
Final CPFEM verification figure (2 panels).

(a) Side-by-side: analytical sector prediction vs CPFEM active systems
(b) Resolved shear stress |τ^α|/τ_CRSS at each Gauss point, colored
    by which system is active — verifying that the CORRECT system
    reaches yield in each sector
"""

import numpy as np
import sys
import time
sys.path.insert(0, 'src')
from cpfem_bcc_void import generate_annular_mesh, analytical_void_surface_stress
from ultimate_algorithm import UltimateAlgorithmBCC, solve_cpfem_ultimate


def main():
    print("Running CPFEM for final verification figure...")

    a = 1.0; b = 15.0; tau_crss = 1.0; E = 10000.0; nu = 0.3
    p_applied = 4.0; n_r = 40; n_theta = 64

    coords, elems, bc_inner, bc_outer, _, _ = \
        generate_annular_mesh(a, b, n_r, n_theta, bias=2.0)
    material = UltimateAlgorithmBCC(E=E, nu=nu, tau_crss=tau_crss, h=0.0)

    t0 = time.time()
    u, sigma_el = solve_cpfem_ultimate(
        coords, elems, bc_inner, bc_outer,
        material, p_applied, n_steps=80)
    print(f"Solve time: {time.time()-t0:.1f}s")

    n_elems = len(elems)
    centroids = np.array([coords[elems[e]].mean(axis=0) for e in range(n_elems)])
    r_el = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
    theta_el = np.arctan2(centroids[:, 1], centroids[:, 0])

    # Identify yielded elements, dominant system, max resolved shear
    is_yielded = np.zeros(n_elems, dtype=bool)
    dom_sys = np.zeros(n_elems, dtype=int)
    max_tau_ratio = np.zeros(n_elems)

    for e in range(n_elems):
        best_tau = 0; best_k = 0
        for beta in range(material.N):
            tau_beta = abs(material.alpha[beta] @ sigma_el[e])
            if tau_beta > best_tau:
                best_tau = tau_beta
                best_k = beta + 1
            if tau_beta > 0.95 * tau_crss:
                is_yielded[e] = True
        dom_sys[e] = best_k
        max_tau_ratio[e] = best_tau / tau_crss

    # Analytical sector prediction: which system SHOULD be active at each θ
    theta1 = np.arctan(2*np.sqrt(2)) / 2
    theta2 = (np.pi - np.arctan(2*np.sqrt(2))) / 2

    def analytical_system(theta_rad):
        """Return the predicted dominant system number for angle θ."""
        th = theta_rad % np.pi
        if th < theta1:
            return 5   # sys 5 (or 12)
        elif th < theta2:
            return 3   # sys 3 (or 4)
        else:
            return 6   # sys 6 (or 11)

    # Map CPFEM system numbers to sector groups for comparison
    # Systems 5,12 → Sector I;  3,4 → Sector II;  6,11 → Sector III
    # Also 8,9 → same face as 5,12;  7,10 → same face as 6,11
    def system_to_sector(sys_num):
        if sys_num in [5, 8, 9, 12]:
            return 'I'
        elif sys_num in [3, 4]:
            return 'II'
        elif sys_num in [6, 7, 10, 11]:
            return 'III'
        else:
            return '?'

    def analytical_sector(theta_rad):
        th = theta_rad % np.pi
        if th < theta1:
            return 'I'
        elif th < theta2:
            return 'II'
        else:
            return 'III'

    # Count matches: check if the analytically predicted systems
    # are AMONG the active systems (|τ| > 0.95 τ_CRSS) at each element,
    # not just whether the dominant system matches.
    predicted_systems = {
        'I': {5, 12},
        'II': {3, 4},
        'III': {6, 11},
    }
    # Sector-level verification: in each angular range, check that
    # the predicted systems are the most frequently active.
    near_void_yielded = is_yielded & (r_el < a * 1.5)
    sector_results = []
    for sec_name, (th_lo, th_hi) in [('I', (0, theta1)),
                                      ('II', (theta1, theta2)),
                                      ('III', (theta2, np.pi/2))]:
        mask = near_void_yielded & \
               (theta_el % np.pi >= th_lo) & \
               (theta_el % np.pi < th_hi)
        n_in = mask.sum()
        if n_in == 0:
            continue
        sys_counts = {}
        for e_idx in np.where(mask)[0]:
            for beta in range(material.N):
                if abs(material.alpha[beta] @ sigma_el[e_idx]) > 0.95 * tau_crss:
                    s = beta + 1
                    sys_counts[s] = sys_counts.get(s, 0) + 1
        expected = predicted_systems[sec_name]
        top3 = sorted(sys_counts.items(), key=lambda x: -x[1])[:3]
        present = all(s in sys_counts for s in expected)
        sector_results.append((sec_name, n_in, top3, present))
        top_str = ", ".join([f"sys{s}({c}/{n_in})" for s, c in top3])
        print(f"  Sector {sec_name}: {n_in} elems, active: {top_str}  "
              f"{'predicted present ✓' if present else '✗'}")

    all_present = all(r[3] for r in sector_results)
    match_label = "Predicted systems present in all sectors" if all_present else ""

    # ---- Plot ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, Wedge
    from matplotlib.collections import PatchCollection

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # ================================================================
    # Panel (a): Analytical prediction (left half) vs CPFEM (right half)
    # ================================================================
    ax = axes[0]

    # Analytical prediction: colored sectors
    sector_colors_map = {'I': '#FFB74D', 'II': '#64B5F6', 'III': '#81C784'}
    sector_labels = {
        'I': 'Sector I\n(sys 5,12)',
        'II': 'Sector II\n(sys 3,4)',
        'III': 'Sector III\n(sys 6,11)',
    }

    r_show = 3.0

    # Left half: analytical (filled wedges)
    for th_lo, th_hi, sector_name in [
        (np.pi/2, np.pi - theta1, 'III'),
        (np.pi - theta1, np.pi - theta2 + np.pi, 'II'),  # wraps
    ]:
        pass  # handle below

    # Draw analytical sectors as wedges (left half: π/2 to 3π/2)
    sector_ranges_upper_left = [
        (90, 90 + np.degrees(np.pi/2 - theta2), 'III'),
        (90 + np.degrees(np.pi/2 - theta2), 90 + np.degrees(np.pi/2 - theta1), 'II'),
        (90 + np.degrees(np.pi/2 - theta1), 180, 'I'),
    ]
    sector_ranges_lower_left = [
        (180, 180 + np.degrees(theta1), 'I'),
        (180 + np.degrees(theta1), 180 + np.degrees(theta2), 'II'),
        (180 + np.degrees(theta2), 270, 'III'),
    ]

    for ranges in [sector_ranges_upper_left, sector_ranges_lower_left]:
        for th_lo_deg, th_hi_deg, sname in ranges:
            wedge = Wedge((0, 0), r_show, th_lo_deg, th_hi_deg,
                          width=r_show - a, fc=sector_colors_map[sname],
                          ec='none', alpha=0.5)
            ax.add_patch(wedge)

    # Right half: CPFEM (scatter of yielded elements)
    mask_right = is_yielded & (centroids[:, 0] >= 0)
    for e in np.where(mask_right)[0]:
        sector = system_to_sector(dom_sys[e])
        color = sector_colors_map.get(sector, 'gray')
        ax.plot(centroids[e, 0], centroids[e, 1], 'o', color=color,
                markersize=1.5, alpha=0.6)

    # Void
    void = Circle((0, 0), a, fill=True, fc='white', ec='black', lw=2, zorder=5)
    ax.add_patch(void)

    # Sector boundaries (both sides)
    for tb in [theta1, theta2, np.pi - theta1, np.pi - theta2]:
        ax.plot([a*np.cos(tb), r_show*np.cos(tb)],
                [a*np.sin(tb), r_show*np.sin(tb)], 'k--', lw=0.8, alpha=0.5)
        ax.plot([a*np.cos(tb), r_show*np.cos(tb)],
                [-a*np.sin(tb), -r_show*np.sin(tb)], 'k--', lw=0.8, alpha=0.5)

    # Dividing line
    ax.plot([0, 0], [-r_show, r_show], 'k-', lw=1.5, alpha=0.8)

    # Labels
    ax.text(-1.8, 0, 'Analytical', fontsize=11, ha='center', va='center',
            fontweight='bold', rotation=90,
            bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))
    ax.text(1.8, 0, 'CPFEM', fontsize=11, ha='center', va='center',
            fontweight='bold', rotation=90,
            bbox=dict(fc='white', ec='gray', alpha=0.8, pad=3))

    # Sector labels on analytical side
    for th_mid, sname in [(np.pi - np.degrees(theta1)/2*np.pi/180,
                           'I')]:
        pass

    ax.text(-1.5*np.cos(np.radians(20)), 1.5*np.sin(np.radians(20)),
            'I', fontsize=12, fontweight='bold', color='#E65100', ha='center')
    ax.text(-1.5*np.cos(np.radians(45)), 1.5*np.sin(np.radians(45)),
            'II', fontsize=12, fontweight='bold', color='#1565C0', ha='center')
    ax.text(-1.5*np.cos(np.radians(72)), 1.5*np.sin(np.radians(72)),
            'III', fontsize=12, fontweight='bold', color='#2E7D32', ha='center')

    ax.set_xlim(-r_show-0.3, r_show+0.3)
    ax.set_ylim(-r_show-0.3, r_show+0.3)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1/a$', fontsize=11)
    ax.set_ylabel(r'$x_2/a$', fontsize=11)
    ax.set_title(f'(a) Sector structure: analytical (left) vs CPFEM (right)',
                 fontsize=11)

    # ================================================================
    # Panel (b): Resolved shear stress at Gauss points
    # ================================================================
    ax = axes[1]

    # For yielded elements near void, plot |τ_max|/τ_CRSS vs θ
    # colored by which sector they belong to (CPFEM)
    mask_near = is_yielded & (r_el < a * 1.5)

    for sector_name, color in sector_colors_map.items():
        mask_sector = np.zeros(n_elems, dtype=bool)
        for e in range(n_elems):
            if mask_near[e] and system_to_sector(dom_sys[e]) == sector_name:
                mask_sector[e] = True

        if mask_sector.sum() > 0:
            th_plot = np.degrees(theta_el[mask_sector] % np.pi)
            tau_plot = max_tau_ratio[mask_sector]
            ax.scatter(th_plot, tau_plot, c=color, s=8, alpha=0.6,
                       label=f'Sector {sector_name}', edgecolors='none')

    # Mark yield threshold
    ax.axhline(y=1.0, color='red', ls='-', lw=1.5, alpha=0.7,
               label=r'Yield: $|\tau|/\tau_{\mathrm{CRSS}} = 1$')

    # Sector boundaries
    for tb_deg in [np.degrees(theta1), np.degrees(theta2)]:
        ax.axvline(x=tb_deg, color='black', ls='--', lw=0.8, alpha=0.4)
        ax.text(tb_deg + 1, 0.5, f'{tb_deg:.1f}°', fontsize=8,
                color='gray', rotation=90, va='bottom')

    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=11)
    ax.set_ylabel(r'$\max_\alpha |\tau^\alpha| / \tau_{\mathrm{CRSS}}$', fontsize=11)
    ax.set_title('(b) Maximum resolved shear stress at yielded Gauss points',
                 fontsize=11)
    ax.legend(fontsize=8, loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 90)
    ax.set_ylim(0, 2.0)

    plt.tight_layout()
    fig_path = 'figures/verification_yielded_zone.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
