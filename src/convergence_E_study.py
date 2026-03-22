"""
Convergence study: increase E/τ_CRSS toward the rigid-plastic limit.

For each E value, solve the BCC void problem and compare the void surface
hoop stress σ_θθ(a, θ) against the analytical rigid-plastic solution.

As E → ∞, the elastic-plastic FEM solution should approach the
analytical rigid-ideally plastic solution.
"""

import numpy as np
import sys
import time

sys.path.insert(0, 'src')
from cpfem_bcc_void import generate_annular_mesh, analytical_void_surface_stress
from ultimate_algorithm import UltimateAlgorithmBCC, solve_cpfem_ultimate


def extract_void_surface_stress(coords, elems, sigma_el, a=1.0, r_tol=0.3):
    """Extract σ_θθ at elements near the void surface."""
    n_elems = len(elems)
    centroids = np.array([coords[elems[e]].mean(axis=0) for e in range(n_elems)])
    r_el = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
    theta_el = np.arctan2(centroids[:, 1], centroids[:, 0])

    # Select elements near void surface
    mask = (r_el < a * (1 + r_tol)) & (r_el > a * 0.99)

    theta_sel = theta_el[mask] % np.pi  # fold to [0, π]
    sigma_sel = sigma_el[mask]

    # Convert to polar stress
    sigma_tt = np.zeros(mask.sum())
    for i, e_idx in enumerate(np.where(mask)[0]):
        s11, s22, s33, s12 = sigma_el[e_idx]
        th = theta_el[e_idx]
        c2, s2 = np.cos(2*th), np.sin(2*th)
        sigma_tt[i] = (s11 + s22)/2 - (s11 - s22)/2 * c2 - s12 * s2

    return theta_sel, sigma_tt


def run_convergence_study():
    print("=" * 70)
    print("Convergence Study: E/τ_CRSS → ∞ (rigid-plastic limit)")
    print("=" * 70)

    # Fixed parameters
    a = 1.0
    b = 8.0
    tau_crss = 1.0
    nu = 0.3

    # Mesh (moderate resolution for speed)
    n_r = 25
    n_theta = 48
    coords, elems, bc_inner, bc_outer, _, _ = \
        generate_annular_mesh(a, b, n_r, n_theta, bias=2.0)
    print(f"Mesh: {len(coords)} nodes, {len(elems)} elements\n")

    # E values to test (start high, approach rigid-plastic limit)
    E_values = [1e7, 5e7, 1e8, 5e8, 1e9]

    # For each E, we set p proportional to E so that the elastic strain
    # at the void is comparable. But for the rigid-plastic comparison,
    # we want the void to be fully yielded.
    # First yield at void surface: p_yield ≈ 2/√3 · τ_CRSS ≈ 1.155
    # (inscribed radius of BCC yield polygon)
    # Use p = 1.5 * p_yield to ensure significant plasticity
    p_applied = 2.0  # well above first yield

    # Analytical solution
    th_ana = np.linspace(0.01, np.pi - 0.01, 200)
    stt_ana = np.array([analytical_void_surface_stress(t, tau_crss)[1] for t in th_ana])

    results = {}

    for E in E_values:
        print(f"\n{'='*50}")
        print(f"E = {E}, E/τ_CRSS = {E/tau_crss:.0f}")
        print(f"{'='*50}")

        material = UltimateAlgorithmBCC(E=E, nu=nu, tau_crss=tau_crss, h=0.0)

        # More steps for higher E (stiffer → smaller elastic increments needed)
        n_steps = max(30, int(E / 50))
        n_steps = min(n_steps, 200)

        t0 = time.time()
        u, sigma_el = solve_cpfem_ultimate(
            coords, elems, bc_inner, bc_outer,
            material, p_applied, n_steps=n_steps
        )
        elapsed = time.time() - t0
        print(f"  Time: {elapsed:.1f}s")

        # Extract void surface stress
        theta_fem, stt_fem = extract_void_surface_stress(
            coords, elems, sigma_el, a, r_tol=0.4)

        # Compute error vs analytical at matched angles
        errors = []
        for i, th in enumerate(theta_fem):
            _, stt_exact, _, _ = analytical_void_surface_stress(th, tau_crss)
            errors.append(abs(stt_fem[i] - stt_exact))

        mean_err = np.mean(errors) if errors else float('nan')
        max_err = np.max(errors) if errors else float('nan')

        # Count yielded elements
        n_yielded = 0
        for e in range(len(elems)):
            for beta in range(material.N):
                if abs(material.alpha[beta] @ sigma_el[e]) > 0.99 * tau_crss:
                    n_yielded += 1
                    break

        results[E] = {
            'theta': theta_fem,
            'stt': stt_fem,
            'mean_err': mean_err,
            'max_err': max_err,
            'n_yielded': n_yielded,
            'n_elems': len(elems),
            'time': elapsed,
        }

        print(f"  Yielded: {n_yielded}/{len(elems)} ({100*n_yielded/len(elems):.1f}%)")
        print(f"  |σ_θθ error| at void surface: mean={mean_err:.4f}, max={max_err:.4f}")

    # ---- Summary table ----
    print(f"\n{'='*70}")
    print("CONVERGENCE SUMMARY")
    print(f"{'='*70}")
    print(f"{'E/τ':>10s} {'Yielded%':>10s} {'Mean err':>12s} {'Max err':>12s} {'Time(s)':>10s}")
    print("-" * 60)
    for E in E_values:
        r = results[E]
        pct = 100 * r['n_yielded'] / r['n_elems']
        print(f"{E:>10.0f} {pct:>9.1f}% {r['mean_err']:>12.4f} {r['max_err']:>12.4f} {r['time']:>10.1f}")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))

        # (a) Void surface σ_θθ for each E
        ax = axes[0]
        ax.plot(np.degrees(th_ana), stt_ana, 'k-', linewidth=2.5,
                label='Analytical (rigid-plastic)')
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, E in enumerate(E_values):
            r = results[E]
            order = np.argsort(r['theta'])
            ax.plot(np.degrees(r['theta'][order]), r['stt'][order],
                    'o', markersize=3, color=colors[i % len(colors)],
                    label=rf'$E/\tau_c = {E:.0f}$', alpha=0.7)

        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\sigma_{\theta\theta}(a, \theta) / \tau_{CRSS}$', fontsize=12)
        ax.set_title(r'(a) Void surface hoop stress: CPFEM $\to$ analytical', fontsize=12)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

        # (b) Error vs E
        ax = axes[1]
        E_arr = np.array(E_values)
        mean_errs = [results[E]['mean_err'] for E in E_values]
        max_errs = [results[E]['max_err'] for E in E_values]
        ax.loglog(E_arr, mean_errs, 'bo-', linewidth=2, markersize=8, label='Mean error')
        ax.loglog(E_arr, max_errs, 'rs-', linewidth=2, markersize=8, label='Max error')
        ax.set_xlabel(r'$E / \tau_{CRSS}$', fontsize=12)
        ax.set_ylabel(r'$|\sigma_{\theta\theta}^{FEM} - \sigma_{\theta\theta}^{analytical}| / \tau_{CRSS}$', fontsize=12)
        ax.set_title('(b) Convergence to rigid-plastic limit', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3, which='both')

        # (c) Fraction yielded vs E
        ax = axes[2]
        pcts = [100*results[E]['n_yielded']/results[E]['n_elems'] for E in E_values]
        ax.semilogx(E_arr, pcts, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel(r'$E / \tau_{CRSS}$', fontsize=12)
        ax.set_ylabel('Elements yielded (%)', fontsize=12)
        ax.set_title('(c) Plastic zone extent', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 100)

        plt.tight_layout()
        fig_path = 'figures/convergence_E_study.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")

    except ImportError:
        print("Matplotlib not available.")


if __name__ == "__main__":
    run_convergence_study()
