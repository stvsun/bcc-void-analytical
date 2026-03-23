"""
Publication figure: CPFEM verification (3 panels).

(a) Active slip system map around void (100% sector match)
(b) Void surface hoop stress: CPFEM vs analytical
(c) Elastic-plastic boundary R(θ)/a
"""

import numpy as np
import sys
import time

sys.path.insert(0, 'src')
from cpfem_bcc_void import generate_annular_mesh, analytical_void_surface_stress
from ultimate_algorithm import UltimateAlgorithmBCC, solve_cpfem_ultimate


def main():
    print("Running CPFEM for Figure 8...")

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

    # Polar stress
    sigma_tt = np.zeros(n_elems)
    for e in range(n_elems):
        s11, s22, s33, s12 = sigma_el[e]
        th = theta_el[e]
        c2, s2 = np.cos(2*th), np.sin(2*th)
        sigma_tt[e] = (s11+s22)/2 - (s11-s22)/2*c2 - s12*s2

    # Yielded elements and active systems
    is_yielded = np.zeros(n_elems, dtype=bool)
    dom_sys = np.zeros(n_elems)
    for e in range(n_elems):
        max_tau = 0; best_k = 0
        for beta in range(material.N):
            tau_beta = abs(material.alpha[beta] @ sigma_el[e])
            if tau_beta > max_tau:
                max_tau = tau_beta; best_k = beta + 1
            if tau_beta > 0.95 * tau_crss:
                is_yielded[e] = True
        dom_sys[e] = best_k

    # Elastic-plastic boundary R(θ)
    n_bins = 36
    theta_bins = np.linspace(0, np.pi, n_bins + 1)
    theta_mid = (theta_bins[:-1] + theta_bins[1:]) / 2
    R_fem = np.zeros(n_bins)
    for k in range(n_bins):
        mask = (theta_el % np.pi >= theta_bins[k]) & \
               (theta_el % np.pi < theta_bins[k+1]) & is_yielded
        R_fem[k] = r_el[mask].max() if mask.sum() > 0 else a

    # Analytical R(θ)
    R_ana = np.zeros(n_bins)
    for k in range(n_bins):
        _, stt_a, _, _ = analytical_void_surface_stress(theta_mid[k], tau_crss)
        R_ana[k] = a * np.exp(p_applied / abs(stt_a)) if abs(stt_a) > 1e-10 else b

    # Near-void yielded elements
    near_void = is_yielded & (r_el < a * 1.5)

    # ---- Plot ----
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # (a) Active slip system map
    ax = axes[0]
    mask_y = is_yielded
    scatter = ax.scatter(centroids[mask_y, 0], centroids[mask_y, 1],
                         c=dom_sys[mask_y], cmap='tab10', s=2,
                         vmin=1, vmax=12, rasterized=True)
    cb = plt.colorbar(scatter, ax=ax, label='Dominant active system',
                      ticks=[1,3,4,5,6,11,12])
    void = plt.Circle((0, 0), a, fill=True, color='white', ec='black', lw=1.5)
    ax.add_patch(void)
    lim = R_fem.max() * 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_aspect('equal')
    ax.set_xlabel(r'$x_1/a$', fontsize=11)
    ax.set_ylabel(r'$x_2/a$', fontsize=11)
    ax.set_title('(a) Active slip systems (yielded zone)', fontsize=11)

    # Sector boundary lines
    for tb in [np.radians(35.26), np.radians(54.74)]:
        for sx, sy in [(1,1),(1,-1),(-1,1),(-1,-1)]:
            ax.plot([sx*a*np.cos(tb), sx*lim*np.cos(tb)],
                    [sy*a*np.sin(tb), sy*lim*np.sin(tb)],
                    'k--', lw=0.5, alpha=0.3)

    # (b) Void surface σ_θθ
    ax = axes[1]
    th_ana = np.linspace(0.01, np.pi-0.01, 200)
    stt_ana = np.array([analytical_void_surface_stress(t, tau_crss)[1] for t in th_ana])
    ax.plot(np.degrees(th_ana), stt_ana, 'r-', lw=2.5, label='Analytical (rigid-plastic)')

    if near_void.sum() > 0:
        theta_near = theta_el[near_void] % np.pi
        stt_near = sigma_tt[near_void]
        order = np.argsort(theta_near)
        ax.plot(np.degrees(theta_near[order]), stt_near[order],
                'bo', markersize=3, alpha=0.5, label='CPFEM (yielded, $r<1.5a$)')

    # Kirsch elastic reference
    ax.axhline(y=-2*p_applied, color='gray', ls='--', lw=1,
               alpha=0.6, label=rf'Elastic Kirsch ($-2p = {-2*p_applied:.0f}\,\tau$)')

    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=11)
    ax.set_ylabel(r'$\sigma_{\theta\theta}/\tau_{\mathrm{CRSS}}$', fontsize=11)
    ax.set_title(r'(b) Hoop stress at void surface', fontsize=11)
    ax.legend(fontsize=8, loc='lower center')
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)

    # (c) Elastic-plastic boundary
    ax = axes[2]
    ax.plot(np.degrees(theta_mid), R_fem, 'bo-', markersize=4, lw=1.5,
            label='CPFEM')
    R_ana_clip = np.clip(R_ana, a, b)
    ax.plot(np.degrees(theta_mid), R_ana_clip, 'r-', lw=2,
            label='Analytical')
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=11)
    ax.set_ylabel(r'$R(\theta)/a$', fontsize=11)
    ax.set_title('(c) Elastic-plastic boundary', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)

    plt.tight_layout()
    fig_path = 'figures/verification_yielded_zone.png'
    plt.savefig(fig_path, dpi=200, bbox_inches='tight')
    print(f"Saved: {fig_path}")


if __name__ == "__main__":
    main()
