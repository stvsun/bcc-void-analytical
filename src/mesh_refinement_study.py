"""
Mesh refinement study: verify stress magnitude convergence at the void surface.

For P1 triangular elements, the stress error should converge as O(h) where
h is the element size. We refine the mesh near the void and track:
  1. σ_θθ at the void surface vs analytical
  2. Convergence rate

Strategy: keep the outer mesh coarse, but increase n_r (radial divisions)
with strong bias toward the void surface. This places the innermost element
centroid closer to r = a, reducing the extrapolation error.
"""

import numpy as np
import sys
import time

sys.path.insert(0, 'src')
from cpfem_bcc_void import generate_annular_mesh, analytical_void_surface_stress
from ultimate_algorithm import UltimateAlgorithmBCC, solve_cpfem_ultimate


def extract_void_surface_stress_polar(coords, elems, sigma_el, a=1.0):
    """
    Extract polar stress components at the innermost ring of elements.
    Returns element data sorted by angle.
    """
    n_elems = len(elems)
    centroids = np.array([coords[elems[e]].mean(axis=0) for e in range(n_elems)])
    r_el = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
    theta_el = np.arctan2(centroids[:, 1], centroids[:, 0])

    # Find the innermost ring: elements with r < median of first two radial layers
    r_sorted = np.sort(np.unique(np.round(r_el, 6)))
    if len(r_sorted) > 2:
        r_cut = (r_sorted[0] + r_sorted[1]) / 2
    else:
        r_cut = r_sorted[0] * 1.1

    mask = r_el <= r_cut

    r_sel = r_el[mask]
    theta_sel = theta_el[mask]
    r_mean = r_sel.mean()

    # Convert to polar stress
    sigma_rr = np.zeros(mask.sum())
    sigma_tt = np.zeros(mask.sum())
    sigma_rt = np.zeros(mask.sum())

    idx = 0
    for e in np.where(mask)[0]:
        s11, s22, s33, s12 = sigma_el[e]
        th = theta_el[e]
        c2, s2 = np.cos(2*th), np.sin(2*th)
        sigma_rr[idx] = (s11+s22)/2 + (s11-s22)/2*c2 + s12*s2
        sigma_tt[idx] = (s11+s22)/2 - (s11-s22)/2*c2 - s12*s2
        sigma_rt[idx] = -(s11-s22)/2*s2 + s12*c2
        idx += 1

    return theta_sel, r_sel, sigma_rr, sigma_tt, sigma_rt, r_mean


def run_single_mesh(n_r, n_theta, bias, a, b, E, nu, tau_crss, p_applied, n_steps):
    """Run a single CPFEM solve and return void-surface stress error."""
    coords, elems, bc_inner, bc_outer, r_vals, _ = \
        generate_annular_mesh(a, b, n_r, n_theta, bias=bias)

    # Characteristic element size at void surface
    h_void = r_vals[1] - r_vals[0]  # first radial spacing

    material = UltimateAlgorithmBCC(E=E, nu=nu, tau_crss=tau_crss, h=0.0)

    u, sigma_el = solve_cpfem_ultimate(
        coords, elems, bc_inner, bc_outer,
        material, p_applied, n_steps=n_steps
    )

    # Extract void surface stress
    theta_fem, r_fem, srr_fem, stt_fem, srt_fem, r_mean = \
        extract_void_surface_stress_polar(coords, elems, sigma_el, a)

    # Analytical at the centroid radius (NOT at r=a, but at the actual centroid)
    # This is the fair comparison: the FEM computes stress at the centroid,
    # so the analytical reference should be at the same radius.
    stt_ana = np.zeros_like(stt_fem)
    for i, (th, r) in enumerate(zip(theta_fem, r_fem)):
        th_fold = th % np.pi
        if th_fold < 0.01:
            th_fold = 0.01
        if th_fold > np.pi - 0.01:
            th_fold = np.pi - 0.01
        _, stt_void, _, _ = analytical_void_surface_stress(th_fold, tau_crss)
        # Interior field: σ_θθ(r, θ) = σ_θθ(a, θ) · [1 + ln(r/a)]
        stt_ana[i] = stt_void * (1.0 + np.log(r / a))

    error = np.abs(stt_fem - stt_ana)
    mean_err = error.mean()
    max_err = error.max()

    # Also compute error at void surface (r=a) by extrapolation
    # For the innermost elements, linearly extrapolate σ_θθ to r=a
    # using the known logarithmic form: σ_θθ(r) = A + B ln(r)
    # At r=a: σ_θθ(a) = A + B ln(a) = A (if a=1)

    # Check yielded fraction
    n_yielded = 0
    for e in range(len(elems)):
        for beta in range(material.N):
            if abs(material.alpha[beta] @ sigma_el[e]) > 0.95 * tau_crss:
                n_yielded += 1
                break

    return {
        'n_r': n_r,
        'n_theta': n_theta,
        'n_nodes': len(coords),
        'n_elems': len(elems),
        'h_void': h_void,
        'r_centroid': r_mean,
        'mean_err': mean_err,
        'max_err': max_err,
        'n_yielded': n_yielded,
        'theta': theta_fem,
        'stt_fem': stt_fem,
        'stt_ana': stt_ana,
        'r_fem': r_fem,
    }


def main():
    print("=" * 70)
    print("Mesh Refinement Study: Stress Convergence at Void Surface")
    print("=" * 70)

    a = 1.0
    b = 10.0
    tau_crss = 1.0
    E = 10000.0
    nu = 0.3
    p_applied = 2.5  # enough for significant yielding near void

    # Mesh refinement levels: increase n_r with high bias
    # The bias concentrates elements near r=a.
    # With bias=3.0 and n_r radial divisions, the first element has
    # width ≈ (b-a) * (1/n_r)^bias * bias

    configs = [
        # (n_r, n_theta, bias, n_steps)
        (15, 32, 3.0, 50),
        (25, 48, 3.0, 60),
        (40, 64, 3.0, 70),
        (60, 80, 3.0, 80),
        (80, 96, 3.0, 90),
    ]

    results = []

    for n_r, n_theta, bias, n_steps in configs:
        print(f"\n{'='*50}")
        print(f"n_r={n_r}, n_θ={n_theta}, bias={bias}")
        print(f"{'='*50}")

        t0 = time.time()
        res = run_single_mesh(n_r, n_theta, bias, a, b, E, nu, tau_crss,
                              p_applied, n_steps)
        elapsed = time.time() - t0

        res['time'] = elapsed
        results.append(res)

        pct = 100 * res['n_yielded'] / res['n_elems']
        print(f"  h_void = {res['h_void']:.6f}, r_centroid = {res['r_centroid']:.6f}")
        print(f"  Yielded: {res['n_yielded']}/{res['n_elems']} ({pct:.1f}%)")
        print(f"  Mean |σ_θθ error| = {res['mean_err']:.6f} τ_CRSS")
        print(f"  Max  |σ_θθ error| = {res['max_err']:.6f} τ_CRSS")
        print(f"  Time: {elapsed:.1f}s")

    # ---- Convergence table ----
    print(f"\n{'='*70}")
    print("CONVERGENCE TABLE")
    print(f"{'='*70}")
    print(f"{'n_r':>6s} {'n_θ':>6s} {'h_void':>10s} {'r_cent':>10s} "
          f"{'Mean err':>12s} {'Max err':>12s} {'Rate':>8s} {'Time':>8s}")
    print("-" * 80)

    for i, res in enumerate(results):
        if i > 0:
            # Convergence rate: error ∝ h^p → p = log(e1/e2) / log(h1/h2)
            h1 = results[i-1]['h_void']
            h2 = res['h_void']
            e1 = results[i-1]['mean_err']
            e2 = res['mean_err']
            if e2 > 1e-15 and h2 > 1e-15 and e1 > 1e-15:
                rate = np.log(e1/e2) / np.log(h1/h2)
                rate_str = f"{rate:.2f}"
            else:
                rate_str = "---"
        else:
            rate_str = "---"

        print(f"{res['n_r']:>6d} {res['n_theta']:>6d} {res['h_void']:>10.6f} "
              f"{res['r_centroid']:>10.6f} {res['mean_err']:>12.6f} "
              f"{res['max_err']:>12.6f} {rate_str:>8s} {res['time']:>7.1f}s")

    # ---- Plot ----
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(14, 11))

        # (a) σ_θθ at void surface for each mesh
        ax = axes[0, 0]
        th_ana = np.linspace(0.01, np.pi-0.01, 300)
        stt_ana_ref = np.array([analytical_void_surface_stress(t, tau_crss)[1]
                                for t in th_ana])
        ax.plot(np.degrees(th_ana), stt_ana_ref, 'k-', linewidth=2.5,
                label=r'Analytical at $r=a$')

        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']
        for i, res in enumerate(results):
            theta_fold = res['theta'] % np.pi
            order = np.argsort(theta_fold)
            # Plot analytical at centroid radius for this mesh
            if i == 0 or i == len(results) - 1:
                stt_ana_cent = np.array([
                    analytical_void_surface_stress(t if 0.01 < t < np.pi-0.01 else 0.01, tau_crss)[1]
                    * (1 + np.log(res['r_centroid'] / a))
                    for t in theta_fold[order]
                ])
                if i == 0:
                    ax.plot(np.degrees(theta_fold[order]), stt_ana_cent,
                            '--', color='gray', linewidth=1,
                            label=rf'Analytical at $r={res["r_centroid"]:.3f}a$')

            ax.plot(np.degrees(theta_fold[order]),
                    res['stt_fem'][order], 'o', markersize=2+i,
                    color=colors[i % len(colors)], alpha=0.6,
                    label=rf'$n_r={res["n_r"]}$, $h={res["h_void"]:.4f}$')

        ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
        ax.set_ylabel(r'$\sigma_{\theta\theta} / \tau_{CRSS}$', fontsize=12)
        ax.set_title(r'(a) Hoop stress convergence with mesh refinement', fontsize=12)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 180)

        # (b) Convergence plot: error vs h
        ax = axes[0, 1]
        h_arr = np.array([r['h_void'] for r in results])
        mean_arr = np.array([r['mean_err'] for r in results])
        max_arr = np.array([r['max_err'] for r in results])

        ax.loglog(h_arr, mean_arr, 'bo-', linewidth=2, markersize=8,
                  label='Mean error')
        ax.loglog(h_arr, max_arr, 'rs-', linewidth=2, markersize=8,
                  label='Max error')

        # Reference slopes
        h_ref = np.linspace(h_arr.min()*0.5, h_arr.max()*2, 50)
        if mean_arr[-1] > 1e-10:
            # O(h) reference
            c1 = mean_arr[-1] / h_arr[-1]
            ax.loglog(h_ref, c1 * h_ref, 'k--', linewidth=1, alpha=0.5,
                      label=r'$O(h^1)$ reference')
            # O(h^2) reference
            c2 = mean_arr[-1] / h_arr[-1]**2
            ax.loglog(h_ref, c2 * h_ref**2, 'k:', linewidth=1, alpha=0.5,
                      label=r'$O(h^2)$ reference')

        ax.set_xlabel(r'Element size $h$ at void surface', fontsize=12)
        ax.set_ylabel(r'$|\sigma_{\theta\theta}^{FEM} - \sigma_{\theta\theta}^{ana}| / \tau_{CRSS}$',
                      fontsize=12)
        ax.set_title('(b) Convergence rate', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        # (c) Centroid radius vs n_r
        ax = axes[1, 0]
        nr_arr = np.array([r['n_r'] for r in results])
        rc_arr = np.array([r['r_centroid'] for r in results])
        ax.plot(nr_arr, rc_arr - a, 'go-', linewidth=2, markersize=8)
        ax.set_xlabel(r'$n_r$ (radial divisions)', fontsize=12)
        ax.set_ylabel(r'$r_{centroid} - a$ (distance from void surface)', fontsize=12)
        ax.set_title('(c) Innermost centroid distance from void', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')

        # (d) Error at specific angles vs h
        ax = axes[1, 1]
        for th_target in [0, 45, 90]:
            errs_at_angle = []
            for res in results:
                theta_fold = res['theta'] % np.pi
                mask = np.abs(np.degrees(theta_fold) - th_target) < 8
                if mask.sum() > 0:
                    err_local = np.abs(res['stt_fem'][mask] - res['stt_ana'][mask]).mean()
                    errs_at_angle.append(err_local)
                else:
                    errs_at_angle.append(np.nan)

            errs_at_angle = np.array(errs_at_angle)
            valid = ~np.isnan(errs_at_angle)
            if valid.sum() > 1:
                ax.loglog(h_arr[valid], errs_at_angle[valid], 'o-',
                          markersize=6, linewidth=1.5,
                          label=rf'$\theta = {th_target}°$')

        if mean_arr[-1] > 1e-10:
            ax.loglog(h_ref, c1 * h_ref, 'k--', linewidth=1, alpha=0.4,
                      label=r'$O(h)$')

        ax.set_xlabel(r'Element size $h$', fontsize=12)
        ax.set_ylabel(r'Error at specific $\theta$', fontsize=12)
        ax.set_title(r'(d) Error at $\theta = 0°, 45°, 90°$', fontsize=12)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3, which='both')

        plt.tight_layout()
        fig_path = 'figures/mesh_refinement_study.png'
        plt.savefig(fig_path, dpi=150, bbox_inches='tight')
        print(f"\nFigure saved to: {fig_path}")

    except ImportError:
        print("Matplotlib not available.")

    print("\n" + "=" * 70)
    print("MESH REFINEMENT STUDY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
