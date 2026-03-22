"""
Verify CPFEM against analytical solution ONLY in the yielded zone.

The analytical rigid-plastic solution applies only inside the plastic zone.
The elastic-plastic FEM naturally produces a yielded zone near the void
and an elastic zone far away. We compare only where both solutions are valid.

We also verify:
1. Which slip systems are active at each angle (sector structure)
2. The hoop stress at the void surface
3. The elastic-plastic boundary location R(θ)
"""

import numpy as np
import sys
import time

sys.path.insert(0, 'src')
from cpfem_bcc_void import generate_annular_mesh, analytical_void_surface_stress
from ultimate_algorithm import UltimateAlgorithmBCC, solve_cpfem_ultimate


def main():
    print("=" * 70)
    print("Verification in Yielded Zone Only")
    print("=" * 70)

    a = 1.0
    b = 15.0
    tau_crss = 1.0
    E = 10000.0  # moderate E — results are E-independent anyway
    nu = 0.3
    p_applied = 4.0  # high pressure to yield more of the domain

    n_r = 40
    n_theta = 64

    coords, elems, bc_inner, bc_outer, _, _ = \
        generate_annular_mesh(a, b, n_r, n_theta, bias=2.0)
    print(f"Mesh: {len(coords)} nodes, {len(elems)} elements")
    print(f"E = {E}, p = {p_applied}")

    material = UltimateAlgorithmBCC(E=E, nu=nu, tau_crss=tau_crss, h=0.0)

    t0 = time.time()
    u, sigma_el = solve_cpfem_ultimate(
        coords, elems, bc_inner, bc_outer,
        material, p_applied, n_steps=80
    )
    print(f"Solve time: {time.time()-t0:.1f}s")

    # ---- Classify elements ----
    n_elems = len(elems)
    centroids = np.array([coords[elems[e]].mean(axis=0) for e in range(n_elems)])
    r_el = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
    theta_el = np.arctan2(centroids[:, 1], centroids[:, 0])

    # Convert to polar stress
    sigma_rr = np.zeros(n_elems)
    sigma_tt = np.zeros(n_elems)
    sigma_rt = np.zeros(n_elems)

    for e in range(n_elems):
        s11, s22, s33, s12 = sigma_el[e]
        th = theta_el[e]
        c2, s2 = np.cos(2*th), np.sin(2*th)
        sigma_rr[e] = (s11+s22)/2 + (s11-s22)/2*c2 + s12*s2
        sigma_tt[e] = (s11+s22)/2 - (s11-s22)/2*c2 - s12*s2
        sigma_rt[e] = -(s11-s22)/2*s2 + s12*c2

    # Identify yielded elements
    is_yielded = np.zeros(n_elems, dtype=bool)
    active_systems = [[] for _ in range(n_elems)]
    for e in range(n_elems):
        for beta in range(material.N):
            tau_beta = abs(material.alpha[beta] @ sigma_el[e])
            if tau_beta > 0.95 * tau_crss:
                is_yielded[e] = True
                active_systems[e].append(beta + 1)  # 1-indexed

    n_yielded = is_yielded.sum()
    print(f"\nYielded elements: {n_yielded}/{n_elems} ({100*n_yielded/n_elems:.1f}%)")

    # ---- Find elastic-plastic boundary ----
    print("\n" + "=" * 70)
    print("Elastic-Plastic Boundary R(θ)")
    print("=" * 70)

    # For each angular bin, find the outermost yielded element
    n_bins = 36
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    R_boundary_fem = np.zeros(n_bins)
    theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2

    for k in range(n_bins):
        mask = (theta_el % np.pi >= theta_bins[k]) & \
               (theta_el % np.pi < theta_bins[k+1]) & is_yielded
        if mask.sum() > 0:
            R_boundary_fem[k] = r_el[mask].max()
        else:
            R_boundary_fem[k] = a

    for k in range(0, n_bins, n_bins//6):
        print(f"  θ = {np.degrees(theta_mid[k]):5.1f}°: R/a = {R_boundary_fem[k]:.3f}")

    # Analytical prediction: R(θ)/a = exp(p / |σ_θθ(a,θ)|)
    R_boundary_ana = np.zeros(n_bins)
    for k in range(n_bins):
        _, stt_ana, _, _ = analytical_void_surface_stress(theta_mid[k], tau_crss)
        if abs(stt_ana) > 1e-10:
            R_boundary_ana[k] = a * np.exp(p_applied / abs(stt_ana))
        else:
            R_boundary_ana[k] = b

    # ---- Compare void surface stress (yielded elements only) ----
    print("\n" + "=" * 70)
    print("Void Surface Hoop Stress Comparison (yielded elements only)")
    print("=" * 70)

    # Elements near void that are yielded
    near_void = is_yielded & (r_el < a * 1.5)
    n_near = near_void.sum()
    print(f"  Yielded elements near void (r < 1.5a): {n_near}")

    if n_near > 0:
        theta_near = theta_el[near_void] % np.pi
        stt_near = sigma_tt[near_void]

        # Analytical
        stt_ana_near = np.array([
            analytical_void_surface_stress(th, tau_crss)[1]
            for th in theta_near
        ])

        error = np.abs(stt_near - stt_ana_near)
        print(f"  Mean |σ_θθ error| = {error.mean():.4f} τ_CRSS")
        print(f"  Max  |σ_θθ error| = {error.max():.4f} τ_CRSS")
        print(f"  Mean relative error = {(error / np.abs(stt_ana_near + 1e-10)).mean():.2%}")

    # ---- Verify active slip systems (sector structure) ----
    print("\n" + "=" * 70)
    print("Active Slip System Verification (sector structure)")
    print("=" * 70)
    print("Analytical sectors:")
    print("  Sector I:   0° < θ < 35.26°  → systems 5, 12")
    print("  Sector II:  35.26° < θ < 54.74° → systems 3, 4")
    print("  Sector III: 54.74° < θ < 90°  → systems 6, 11")
    print()

    # Check which systems are active in each angular range (near void)
    sector_ranges = [
        (0, 35.26, "I", {5, 12}),
        (35.26, 54.74, "II", {3, 4}),
        (54.74, 90, "III", {6, 11}),
    ]

    for th_lo, th_hi, name, expected_sys in sector_ranges:
        mask = near_void & \
               (np.degrees(theta_el % np.pi) >= th_lo) & \
               (np.degrees(theta_el % np.pi) < th_hi)
        n_in = mask.sum()
        if n_in == 0:
            print(f"  Sector {name}: no yielded elements")
            continue

        # Count active systems
        sys_counts = {}
        for e_idx in np.where(mask)[0]:
            for s in active_systems[e_idx]:
                sys_counts[s] = sys_counts.get(s, 0) + 1

        top_sys = sorted(sys_counts.items(), key=lambda x: -x[1])[:4]
        top_str = ", ".join([f"sys{s}({c}/{n_in})" for s, c in top_sys])
        match = all(s in sys_counts for s in expected_sys)
        print(f"  Sector {name} ({th_lo}°-{th_hi}°): {n_in} elems, "
              f"active: {top_str}  {'✓' if match else '✗'}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 3, figsize=(18, 11))

        # (a) Void surface σ_θθ: CPFEM vs analytical (yielded only)
        ax = axes[0, 0]
        th_ana = np.linspace(0.01, np.pi-0.01, 200)
        stt_ana = np.array([analytical_void_surface_stress(t, tau_crss)[1] for t in th_ana])
        ax.plot(np.degrees(th_ana), stt_ana, 'r-', linewidth=2.5, label='Analytical')
        if n_near > 0:
            order = np.argsort(theta_near)
            ax.plot(np.degrees(theta_near[order]), stt_near[order],
                    'bo', markersize=4, alpha=0.6, label='CPFEM (yielded)')
        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\sigma_{\theta\theta}(a, \theta) / \tau_{CRSS}$', fontsize=12)
        ax.set_title('(a) Void surface hoop stress (yielded zone only)', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

        # (b) Elastic-plastic boundary R(θ)
        ax = axes[0, 1]
        ax.plot(np.degrees(theta_mid), R_boundary_fem, 'bo-', markersize=5,
                linewidth=1.5, label='CPFEM')
        R_ana_clipped = np.clip(R_boundary_ana, a, b)
        ax.plot(np.degrees(theta_mid), R_ana_clipped, 'r-', linewidth=2,
                label='Analytical')
        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$R(\theta)/a$', fontsize=12)
        ax.set_title('(b) Elastic-plastic boundary', fontsize=11)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

        # (c) Active slip systems map
        ax = axes[0, 2]
        # Color by dominant active system
        dom_sys = np.zeros(n_elems)
        for e in range(n_elems):
            if active_systems[e]:
                dom_sys[e] = active_systems[e][0]
        mask_y = is_yielded
        scatter = ax.scatter(centroids[mask_y, 0], centroids[mask_y, 1],
                           c=dom_sys[mask_y], cmap='tab10', s=3, vmin=1, vmax=12)
        plt.colorbar(scatter, ax=ax, label='Dominant active system')
        circle = plt.Circle((0, 0), a, fill=True, color='white', ec='black', lw=1.5)
        ax.add_patch(circle)
        ax.set_xlabel(r'$x_1/a$', fontsize=12)
        ax.set_ylabel(r'$x_2/a$', fontsize=12)
        ax.set_title('(c) Active slip systems (yielded zone)', fontsize=11)
        ax.set_aspect('equal')
        lim = R_boundary_fem.max() * 1.2
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # (d) σ_rr contour (full domain)
        ax = axes[1, 0]
        triang = plt.matplotlib.tri.Triangulation(centroids[:, 0], centroids[:, 1])
        levels = np.linspace(np.percentile(sigma_rr, 1), np.percentile(sigma_rr, 99), 20)
        if len(levels) > 2:
            cs = ax.tricontourf(triang, sigma_rr, levels=levels, cmap='RdBu_r', extend='both')
            plt.colorbar(cs, ax=ax, label=r'$\sigma_{rr}/\tau_{CRSS}$')
        circle = plt.Circle((0, 0), a, fill=True, color='white', ec='black', lw=1.5)
        ax.add_patch(circle)
        ax.set_title(r'(d) $\sigma_{rr}$ contour', fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # (e) σ_θθ contour (full domain)
        ax = axes[1, 1]
        levels = np.linspace(np.percentile(sigma_tt, 1), np.percentile(sigma_tt, 99), 20)
        if len(levels) > 2:
            cs = ax.tricontourf(triang, sigma_tt, levels=levels, cmap='RdBu_r', extend='both')
            plt.colorbar(cs, ax=ax, label=r'$\sigma_{\theta\theta}/\tau_{CRSS}$')
        circle = plt.Circle((0, 0), a, fill=True, color='white', ec='black', lw=1.5)
        ax.add_patch(circle)
        ax.set_title(r'(e) $\sigma_{\theta\theta}$ contour', fontsize=11)
        ax.set_aspect('equal')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

        # (f) σ_rr and σ_θθ along radial lines (yielded zone)
        ax = axes[1, 2]
        for th_target in [0, 45, 90]:
            th_rad = np.radians(th_target)
            mask = is_yielded & (np.abs(theta_el % np.pi - th_rad) < np.radians(5))
            if mask.sum() < 3:
                continue
            r_sel = r_el[mask]
            stt_sel = sigma_tt[mask]
            order = np.argsort(r_sel)
            ax.plot(r_sel[order], stt_sel[order], 'o', markersize=3,
                    label=rf'$\sigma_{{\theta\theta}}$ CPFEM $\theta={th_target}°$')

            # Analytical in yielded zone
            _, stt_void, _, _ = analytical_void_surface_stress(th_rad, tau_crss)
            r_ana = np.linspace(a, r_sel.max(), 50)
            stt_ana_r = stt_void * (1 + np.log(r_ana / a))
            ax.plot(r_ana, stt_ana_r, '-', linewidth=1.5)

        ax.set_xlabel(r'$r/a$', fontsize=12)
        ax.set_ylabel(r'$\sigma_{\theta\theta} / \tau_{CRSS}$', fontsize=12)
        ax.set_title(r'(f) $\sigma_{\theta\theta}$ vs $r/a$ (dots=CPFEM, lines=analytical)', fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fig_path = 'figures/verification_yielded_zone.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")

    except ImportError:
        print("Matplotlib not available.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
