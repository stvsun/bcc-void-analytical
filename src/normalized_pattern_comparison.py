"""
Normalized angular pattern comparison.

The rigid-plastic analytical solution predicts the ANGULAR VARIATION
of σ_θθ(a, θ), not the absolute magnitude (which depends on loading
and the elastic-plastic transition). The shape of the curve:

    σ_θθ(a, θ) / σ_θθ(a, 0°)

is a purely geometric quantity determined by the yield polygon and
sector structure. This normalized pattern is independent of E and p,
and should match between CPFEM and analytical.

Also compare: the angular location of stress extrema (sector boundaries).
"""

import numpy as np
import sys
import time

sys.path.insert(0, 'src')
from cpfem_bcc_void import generate_annular_mesh, analytical_void_surface_stress
from ultimate_algorithm import UltimateAlgorithmBCC, solve_cpfem_ultimate


def main():
    print("=" * 70)
    print("Normalized Angular Pattern Comparison")
    print("=" * 70)

    a = 1.0
    b = 10.0
    tau_crss = 1.0
    nu = 0.3

    # Run with several E and p to show pattern is universal
    configs = [
        (1000.0,   2.0,  40, 64, 60, "E=1e3, p=2.0"),
        (10000.0,  2.5,  40, 64, 70, "E=1e4, p=2.5"),
        (10000.0,  4.0,  40, 64, 80, "E=1e4, p=4.0"),
        (100000.0, 2.5,  40, 64, 70, "E=1e5, p=2.5"),
    ]

    all_results = []

    for E, p_applied, n_r, n_theta, n_steps, label in configs:
        print(f"\n--- {label} ---")
        coords, elems, bc_inner, bc_outer, _, _ = \
            generate_annular_mesh(a, b, n_r, n_theta, bias=2.5)

        material = UltimateAlgorithmBCC(E=E, nu=nu, tau_crss=tau_crss, h=0.0)

        t0 = time.time()
        u, sigma_el = solve_cpfem_ultimate(
            coords, elems, bc_inner, bc_outer,
            material, p_applied, n_steps=n_steps
        )
        elapsed = time.time() - t0

        # Extract void surface stress
        n_elems = len(elems)
        centroids = np.array([coords[elems[e]].mean(axis=0) for e in range(n_elems)])
        r_el = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
        theta_el = np.arctan2(centroids[:, 1], centroids[:, 0])

        # Innermost ring
        r_sorted = np.sort(np.unique(np.round(r_el, 6)))
        r_cut = (r_sorted[0] + r_sorted[1]) / 2 if len(r_sorted) > 1 else r_sorted[0]*1.1
        mask = r_el <= r_cut

        # Check if yielded
        is_yielded = np.zeros(n_elems, dtype=bool)
        for e in range(n_elems):
            for beta in range(material.N):
                if abs(material.alpha[beta] @ sigma_el[e]) > 0.95 * tau_crss:
                    is_yielded[e] = True
                    break

        mask_yielded = mask & is_yielded

        theta_sel = theta_el[mask_yielded] % np.pi
        sigma_tt = np.zeros(mask_yielded.sum())
        idx = 0
        for e in np.where(mask_yielded)[0]:
            s11, s22, s33, s12 = sigma_el[e]
            th = theta_el[e]
            c2, s2 = np.cos(2*th), np.sin(2*th)
            sigma_tt[idx] = (s11+s22)/2 - (s11-s22)/2*c2 - s12*s2
            idx += 1

        # Normalize by value at θ ≈ 0
        near_0 = np.abs(theta_sel) < np.radians(10)
        if near_0.sum() > 0:
            stt_at_0 = sigma_tt[near_0].mean()
        else:
            stt_at_0 = sigma_tt.min()

        sigma_tt_norm = sigma_tt / stt_at_0 if abs(stt_at_0) > 1e-10 else sigma_tt

        n_y = is_yielded.sum()
        print(f"  Yielded: {n_y}/{n_elems} ({100*n_y/n_elems:.1f}%), time={elapsed:.1f}s")
        print(f"  σ_θθ(0°) = {stt_at_0:.4f}, range: [{sigma_tt.min():.4f}, {sigma_tt.max():.4f}]")

        all_results.append({
            'label': label,
            'theta': theta_sel,
            'stt_norm': sigma_tt_norm,
            'stt_raw': sigma_tt,
            'stt_at_0': stt_at_0,
        })

    # Analytical normalized pattern
    th_ana = np.linspace(0.02, np.pi-0.02, 300)
    stt_ana = np.array([analytical_void_surface_stress(t, tau_crss)[1] for t in th_ana])
    stt_ana_at_0 = analytical_void_surface_stress(0.01, tau_crss)[1]
    stt_ana_norm = stt_ana / stt_ana_at_0

    # Compute pattern error (normalized)
    print(f"\n{'='*70}")
    print("NORMALIZED PATTERN COMPARISON")
    print(f"{'='*70}")
    print(f"\nAnalytical: σ_θθ(a, 0°) = {stt_ana_at_0:.4f}")
    print(f"Analytical pattern range: [{stt_ana_norm.min():.4f}, {stt_ana_norm.max():.4f}]")

    for res in all_results:
        if len(res['theta']) < 5:
            continue
        # Interpolate analytical at FEM angles
        from scipy.interpolate import interp1d
        f_ana = interp1d(th_ana, stt_ana_norm, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
        ana_at_fem = f_ana(res['theta'])
        pattern_err = np.abs(res['stt_norm'] - ana_at_fem)
        print(f"\n  {res['label']}:")
        print(f"    σ_θθ(0°) = {res['stt_at_0']:.4f}")
        print(f"    Mean normalized pattern error = {pattern_err.mean():.4f}")
        print(f"    Max  normalized pattern error = {pattern_err.max():.4f}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # (a) Raw σ_θθ (all configs)
        ax = axes[0]
        ax.plot(np.degrees(th_ana), stt_ana, 'k-', linewidth=2.5,
                label='Analytical (rigid-plastic)')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        for i, res in enumerate(all_results):
            order = np.argsort(res['theta'])
            ax.plot(np.degrees(res['theta'][order]), res['stt_raw'][order],
                    'o', markersize=3, color=colors[i], alpha=0.7,
                    label=res['label'])
        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\sigma_{\theta\theta}(a, \theta) / \tau_{CRSS}$', fontsize=12)
        ax.set_title('(a) Raw hoop stress (magnitude differs)', fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

        # (b) Normalized pattern
        ax = axes[1]
        ax.plot(np.degrees(th_ana), stt_ana_norm, 'k-', linewidth=2.5,
                label='Analytical')
        for i, res in enumerate(all_results):
            order = np.argsort(res['theta'])
            ax.plot(np.degrees(res['theta'][order]), res['stt_norm'][order],
                    'o', markersize=3+i, color=colors[i], alpha=0.7,
                    label=res['label'])
        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\sigma_{\theta\theta}(\theta) / \sigma_{\theta\theta}(0°)$',
                      fontsize=12)
        ax.set_title(r'(b) Normalized pattern $\sigma_{\theta\theta}(\theta)/\sigma_{\theta\theta}(0°)$',
                     fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        # Mark sector boundaries
        for tb in [35.26, 54.74, 90, 125.26, 144.74]:
            ax.axvline(x=tb, color='gray', linestyle=':', alpha=0.4)

        # (c) Pattern error
        ax = axes[2]
        from scipy.interpolate import interp1d
        f_ana = interp1d(th_ana, stt_ana_norm, kind='linear',
                         bounds_error=False, fill_value='extrapolate')
        for i, res in enumerate(all_results):
            if len(res['theta']) < 5:
                continue
            ana_at_fem = f_ana(res['theta'])
            pattern_err = np.abs(res['stt_norm'] - ana_at_fem)
            order = np.argsort(res['theta'])
            ax.plot(np.degrees(res['theta'][order]), pattern_err[order],
                    'o-', markersize=3, color=colors[i], alpha=0.7,
                    label=res['label'])
        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
        ax.set_ylabel('Normalized pattern error', fontsize=12)
        ax.set_title('(c) Error in normalized pattern', fontsize=12)
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)
        for tb in [35.26, 54.74, 90, 125.26, 144.74]:
            ax.axvline(x=tb, color='gray', linestyle=':', alpha=0.4)

        plt.tight_layout()
        fig_path = 'figures/normalized_pattern_comparison.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")

    except ImportError:
        print("Matplotlib not available.")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()
