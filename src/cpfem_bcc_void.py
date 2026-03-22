"""
Crystal Plasticity Finite Element (CPFEM) verification of the analytical
stress field around a cylindrical void in a BCC {110}<111> single crystal.

2D plane strain, elastic-perfectly plastic with high E/τ_CRSS (≈ rigid).
Rate-dependent (viscoplastic) flow rule for numerical regularization.

Components:
  1. Annular mesh (a < r < b) with graded refinement near void
  2. BCC {110}<111> crystal plasticity: 12 slip systems, Schmid law
  3. P1 triangular FEM, plane strain
  4. Incremental loading + Newton-Raphson with consistent tangent
  5. Post-processing: comparison with analytical solution
"""

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import spsolve
import time


# ================================================================
# 1. MESH GENERATION
# ================================================================

def generate_annular_mesh(a, b, n_r, n_theta, bias=2.0):
    """
    Generate a triangular mesh on an annular domain [a, b] x [0, 2π].

    Parameters
    ----------
    a : float — inner radius (void radius)
    b : float — outer radius
    n_r : int — number of radial divisions
    n_theta : int — number of circumferential divisions
    bias : float — radial grading (>1 concentrates near void)

    Returns
    -------
    coords : (n_nodes, 2) — nodal coordinates (x, y)
    elems : (n_elems, 3) — element connectivity (triangles)
    bc_inner : list of node indices on inner boundary (r = a)
    bc_outer : list of node indices on outer boundary (r = b)
    """
    # Biased radial distribution: finer near the void
    t = np.linspace(0, 1, n_r + 1)
    r_vals = a + (b - a) * t**bias

    theta_vals = np.linspace(0, 2 * np.pi, n_theta + 1)[:-1]  # periodic

    n_nodes = (n_r + 1) * n_theta
    coords = np.zeros((n_nodes, 2))

    for i, r in enumerate(r_vals):
        for j, th in enumerate(theta_vals):
            idx = i * n_theta + j
            coords[idx, 0] = r * np.cos(th)
            coords[idx, 1] = r * np.sin(th)

    # Triangulate: each quad → 2 triangles
    elems = []
    for i in range(n_r):
        for j in range(n_theta):
            n0 = i * n_theta + j
            n1 = i * n_theta + (j + 1) % n_theta
            n2 = (i + 1) * n_theta + j
            n3 = (i + 1) * n_theta + (j + 1) % n_theta
            elems.append([n0, n1, n2])
            elems.append([n1, n3, n2])

    elems = np.array(elems, dtype=int)

    # Boundary nodes
    bc_inner = list(range(n_theta))
    bc_outer = list(range(n_r * n_theta, (n_r + 1) * n_theta))

    return coords, elems, bc_inner, bc_outer, r_vals, theta_vals


# ================================================================
# 2. BCC CRYSTAL PLASTICITY CONSTITUTIVE MODEL
# ================================================================

class BCCCrystalPlasticity:
    """
    BCC {110}<111> crystal plasticity for 2D plane strain.

    Elastic-perfectly plastic with rate-dependent (viscoplastic) flow rule:
      γ̇^α = γ̇₀ · (|τ^α| / τ_CRSS)^(1/m) · sign(τ^α)

    Under plane strain, the full 3D stress σ = [σ11, σ22, σ33, σ12]
    is tracked (Voigt notation for symmetric 2D+out-of-plane).
    """

    def __init__(self, E=1000.0, nu=0.3, tau_crss=1.0,
                 gamma_dot_0=0.001, m=0.05):
        """
        Parameters
        ----------
        E : float — Young's modulus (>> τ_CRSS for rigid-plastic limit)
        nu : float — Poisson's ratio
        tau_crss : float — critical resolved shear stress
        gamma_dot_0 : float — reference slip rate
        m : float — rate sensitivity exponent (small → rate-independent)
        """
        self.E = E
        self.nu = nu
        self.tau_crss = tau_crss
        self.gamma_dot_0 = gamma_dot_0
        self.m = m

        # Plane strain elastic stiffness (4x4: σ11, σ22, σ33, σ12)
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.C = np.array([
            [lam + 2*mu, lam,       lam,       0],
            [lam,       lam + 2*mu, lam,       0],
            [lam,       lam,       lam + 2*mu, 0],
            [0,         0,         0,         mu],
        ])

        # Set up slip systems in the plane strain frame
        self._setup_slip_systems()

    def _setup_slip_systems(self):
        """Transform BCC {110}<111> systems to plane strain frame."""
        # Coordinate system: e1' = [001], e2' = [-110]/√2, e3' = [110]/√2
        R = np.array([
            [0, 0, 1],
            [-1/np.sqrt(2), 1/np.sqrt(2), 0],
            [1/np.sqrt(2), 1/np.sqrt(2), 0],
        ])

        slip_data = [
            ((1,1,0), (-1,1,1)),    ((1,1,0), (1,-1,1)),
            ((1,-1,0), (1,1,1)),    ((1,-1,0), (-1,-1,1)),
            ((1,0,1), (-1,1,1)),    ((1,0,1), (1,1,-1)),
            ((1,0,-1), (1,1,1)),    ((1,0,-1), (-1,1,-1)),
            ((0,1,1), (1,-1,1)),    ((0,1,1), (1,1,-1)),
            ((0,1,-1), (1,1,1)),    ((0,1,-1), (1,-1,-1)),
        ]

        self.n_systems = 12
        # Schmid tensor in Voigt: P^α · σ = P_11 σ_11 + P_22 σ_22 + P_33 σ_33 + 2 P_12 σ_12
        # (factor of 2 on shear because Voigt σ_4 = σ_12, not 2σ_12)
        self.P_voigt = np.zeros((12, 4))  # [P_11, P_22, P_33, 2*P_12]

        # Symmetric Schmid tensor for plastic strain:
        # dε^p_ij = Σ_α dγ^α P^α_ij
        # In Voigt: [dε11, dε22, dε33, 2*dε12] (engineering shear)
        self.P_strain = np.zeros((12, 4))

        for alpha, (n, s) in enumerate(slip_data):
            n_vec = np.array(n, dtype=float)
            s_vec = np.array(s, dtype=float)
            n_hat = n_vec / np.linalg.norm(n_vec)
            s_hat = s_vec / np.linalg.norm(s_vec)

            # Transform to primed coords
            n_p = R @ n_hat
            s_p = R @ s_hat

            # Full Schmid tensor P_ij = (s_i n_j + s_j n_i) / 2
            P_full = 0.5 * (np.outer(s_p, n_p) + np.outer(n_p, s_p))

            # Resolved shear stress: τ = P_ij σ_ij = P_11σ_11 + P_22σ_22 + P_33σ_33 + 2P_12σ_12
            self.P_voigt[alpha] = [P_full[0,0], P_full[1,1], P_full[2,2], 2*P_full[0,1]]

            # Plastic strain increment: dε_ij = dγ P_ij
            # In Voigt: [P_11, P_22, P_33, 2*P_12]
            self.P_strain[alpha] = [P_full[0,0], P_full[1,1], P_full[2,2], 2*P_full[0,1]]

    def stress_update(self, sigma_old, deps, dt):
        """
        Update stress given strain increment (viscoplastic, explicit).

        Parameters
        ----------
        sigma_old : (4,) — stress [σ11, σ22, σ33, σ12] at start of step
        deps : (4,) — strain increment [dε11, dε22, dε33=0, 2*dε12]
        dt : float — time step

        Returns
        -------
        sigma_new : (4,) — updated stress
        dgamma : (12,) — slip increments on each system
        """
        # Trial stress (elastic predictor)
        sigma_trial = sigma_old + self.C @ deps

        # Viscoplastic corrector: iterative
        sigma = sigma_trial.copy()
        dgamma = np.zeros(self.n_systems)

        n_iter = 20
        for iteration in range(n_iter):
            # Resolved shear stress on each system
            tau = self.P_voigt @ sigma  # (12,)

            # Slip rate: γ̇ = γ̇₀ |τ/τ_c|^(1/m) sign(τ)
            tau_ratio = np.abs(tau) / self.tau_crss
            # Clamp to avoid overflow
            tau_ratio_clamped = np.minimum(tau_ratio, 20.0)
            gamma_dot = self.gamma_dot_0 * tau_ratio_clamped**(1.0/self.m) * np.sign(tau)

            dgamma_iter = gamma_dot * dt

            # Plastic strain increment
            deps_p = np.zeros(4)
            for alpha in range(self.n_systems):
                deps_p += dgamma_iter[alpha] * self.P_strain[alpha]

            # Update stress
            sigma_new = sigma_trial - self.C @ deps_p
            dgamma += dgamma_iter

            # Check convergence
            dsigma = np.linalg.norm(sigma_new - sigma)
            sigma = sigma_new

            if dsigma < 1e-8 * self.tau_crss:
                break

        return sigma, dgamma

    def stress_update_implicit(self, sigma_old, deps, dt):
        """
        Implicit stress update with Newton-Raphson on the constitutive level.

        Returns
        -------
        sigma_new : (4,) — updated stress
        C_tang : (4,4) — consistent tangent modulus
        """
        sigma_trial = sigma_old + self.C @ deps
        sigma = sigma_trial.copy()
        total_deps_p = np.zeros(4)

        for iteration in range(50):
            tau = self.P_voigt @ sigma
            tau_ratio = np.clip(np.abs(tau) / self.tau_crss, 1e-12, 50.0)

            # Slip increments
            gamma_dot = self.gamma_dot_0 * tau_ratio**(1.0/self.m) * np.sign(tau)
            dgamma = gamma_dot * dt

            # Residual: R = σ - σ_trial + C : Σ dγ^α P^α
            deps_p = np.zeros(4)
            for alpha in range(self.n_systems):
                deps_p += dgamma[alpha] * self.P_strain[alpha]

            R = sigma - sigma_trial + self.C @ deps_p
            if np.linalg.norm(R) < 1e-10 * self.tau_crss:
                break

            # Jacobian: dR/dσ = I + C : Σ (∂dγ^α/∂σ) ⊗ P^α
            # ∂dγ^α/∂σ = dt · γ̇₀/(m·τ_c) · (|τ|/τ_c)^(1/m-1) · P_voigt^α
            J = np.eye(4)
            for alpha in range(self.n_systems):
                if tau_ratio[alpha] > 1e-12:
                    d_dgamma_d_sigma = (
                        dt * self.gamma_dot_0 / (self.m * self.tau_crss)
                        * tau_ratio[alpha]**(1.0/self.m - 1.0)
                        * self.P_voigt[alpha]
                    )
                    J += np.outer(self.C @ self.P_strain[alpha], d_dgamma_d_sigma)

            dsigma = np.linalg.solve(J, -R)
            sigma += dsigma

        # Consistent tangent
        C_tang = np.linalg.inv(J) @ self.C if np.linalg.det(J) > 1e-30 else self.C * 0.01

        return sigma, C_tang


# ================================================================
# 3. FEM SOLVER
# ================================================================

def tri_shape_grads(coords_el):
    """
    Shape function gradients for a P1 triangle.

    Parameters
    ----------
    coords_el : (3, 2) — nodal coordinates of the triangle

    Returns
    -------
    B : (4, 6) — strain-displacement matrix (plane strain Voigt)
    area : float — element area
    """
    x = coords_el[:, 0]
    y = coords_el[:, 1]

    # Area
    area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))

    if area < 1e-15:
        return np.zeros((4, 6)), 1e-15

    # Shape function gradients (constant over element)
    dN_dx = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*area)
    dN_dy = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*area)

    # B matrix: ε = [ε11, ε22, ε33, 2ε12] = B · u
    # u = [u1x, u1y, u2x, u2y, u3x, u3y]
    B = np.zeros((4, 6))
    for i in range(3):
        B[0, 2*i]   = dN_dx[i]   # ε11 = ∂u1/∂x1
        B[1, 2*i+1] = dN_dy[i]   # ε22 = ∂u2/∂x2
        # B[2, :] = 0             # ε33 = 0 (plane strain)
        B[3, 2*i]   = dN_dy[i]   # 2ε12 = ∂u1/∂x2 + ∂u2/∂x1
        B[3, 2*i+1] = dN_dx[i]

    return B, area


def solve_cpfem(coords, elems, bc_inner, bc_outer,
                material, p_applied, n_steps=20, dt=1.0):
    """
    Solve the void problem using incremental FEM with crystal plasticity.

    Uses elastic predictor + viscoplastic corrector at each Gauss point,
    with global Newton-Raphson iterations at each load step.

    Parameters
    ----------
    coords : (n_nodes, 2) — nodal coordinates
    elems : (n_elems, 3) — element connectivity
    bc_inner : list — inner boundary nodes (traction-free)
    bc_outer : list — outer boundary nodes (equibiaxial pressure)
    material : BCCCrystalPlasticity — constitutive model
    p_applied : float — total applied far-field pressure
    n_steps : int — number of load increments
    dt : float — pseudo-time step

    Returns
    -------
    u : (n_nodes, 2) — nodal displacements
    sigma_el : (n_elems, 4) — element stress [σ11, σ22, σ33, σ12]
    """
    n_nodes = len(coords)
    n_elems = len(elems)
    n_dof = 2 * n_nodes

    # Initialize
    u = np.zeros(n_dof)
    sigma_el = np.zeros((n_elems, 4))
    eps_el = np.zeros((n_elems, 4))  # total strain at each element

    # Precompute B matrices and areas
    B_all = []
    area_all = []
    for e in range(n_elems):
        el_nodes = elems[e]
        B, area = tri_shape_grads(coords[el_nodes])
        B_all.append(B)
        area_all.append(area)

    # Precompute outer boundary tributary lengths and normals
    outer_trib = []
    for i in range(len(bc_outer)):
        node = bc_outer[i]
        x, y = coords[node]
        r = np.sqrt(x**2 + y**2)
        nx, ny = x/r, y/r
        node_prev = bc_outer[(i - 1) % len(bc_outer)]
        node_next = bc_outer[(i + 1) % len(bc_outer)]
        dx1 = np.linalg.norm(coords[node_next] - coords[node])
        dx0 = np.linalg.norm(coords[node] - coords[node_prev])
        trib_len = (dx1 + dx0) / 2
        outer_trib.append((node, nx, ny, trib_len))

    # Pin rigid body modes: fix one inner node (translation) +
    # tangential DOF of another (rotation)
    pin_node = bc_inner[0]
    pin_node2 = bc_inner[len(bc_inner)//4]
    bc_dofs = [2*pin_node, 2*pin_node + 1, 2*pin_node2 + 1]

    # Incremental loading
    dp = p_applied / n_steps
    print(f"  CPFEM: {n_nodes} nodes, {n_elems} elements, {n_steps} load steps")

    for step in range(1, n_steps + 1):
        p_current = dp * step

        # Newton-Raphson iterations
        for nr_iter in range(40):
            # Assemble internal force and elastic stiffness
            rows, cols, vals = [], [], []
            f_int = np.zeros(n_dof)

            for e in range(n_elems):
                el_nodes = elems[e]
                B = B_all[e]
                area = area_all[e]
                dofs = np.array([2*el_nodes[0], 2*el_nodes[0]+1,
                                 2*el_nodes[1], 2*el_nodes[1]+1,
                                 2*el_nodes[2], 2*el_nodes[2]+1])

                # Current strain from displacements
                eps = B @ u[dofs]  # [ε11, ε22, 0, 2ε12]

                # Stress from total strain via viscoplastic constitutive law
                sigma_e, _ = material.stress_update(np.zeros(4), eps, dt * step)
                sigma_el[e] = sigma_e

                # Internal force
                f_el = B.T @ sigma_e * area
                for i in range(6):
                    f_int[dofs[i]] += f_el[i]

                # Elastic stiffness (secant approximation)
                K_el = B.T @ material.C @ B * area
                for i in range(6):
                    for j in range(6):
                        rows.append(dofs[i])
                        cols.append(dofs[j])
                        vals.append(K_el[i, j])

            # External force: pressure on outer boundary
            f_ext = np.zeros(n_dof)
            for node, nx, ny, trib_len in outer_trib:
                f_ext[2*node]   += (-p_current) * nx * trib_len
                f_ext[2*node+1] += (-p_current) * ny * trib_len

            # Residual
            residual = f_int - f_ext

            # Assemble sparse stiffness
            K = sparse.coo_matrix((vals, (rows, cols)),
                                  shape=(n_dof, n_dof)).tocsr()

            # Apply displacement BCs (pin rigid body modes)
            for d in bc_dofs:
                residual[d] = 0.0
                # Zero out row and column
                K = K.tolil()
                K[d, :] = 0
                K[:, d] = 0
                K[d, d] = 1.0
                K = K.tocsr()

            # Solve for displacement correction
            try:
                du = spsolve(K.tocsc(), -residual)
            except Exception:
                print(f"      Solver failed at step {step}, iter {nr_iter}")
                du = np.zeros(n_dof)

            u += du

            res_norm = np.linalg.norm(residual)
            du_norm = np.linalg.norm(du)

            if du_norm < 1e-8 and res_norm < 1e-6:
                break

        # Report
        if step % 5 == 0 or step == n_steps:
            max_tau = 0
            for e in range(n_elems):
                tau_all = material.P_voigt @ sigma_el[e]
                max_tau = max(max_tau, np.max(np.abs(tau_all)))
            print(f"    Step {step}/{n_steps}: p = {p_current:.3f}, "
                  f"max|τ/τ_c| = {max_tau:.4f}, "
                  f"|R| = {res_norm:.2e}, NR iters = {nr_iter+1}")

    return u, sigma_el


# ================================================================
# 4. ANALYTICAL SOLUTION (for comparison)
# ================================================================

def analytical_void_surface_stress(theta, tau_crss=1.0):
    """Exact void surface stress σ_θθ(a, θ) from the analytical derivation."""
    c2 = np.cos(2 * theta)
    s2 = np.sin(2 * theta)

    # Yield face coefficients
    schmid_sys = [
        (0, 0), (0, 0),
        (0, -np.sqrt(3)/3), (0, -np.sqrt(3)/3),
        (np.sqrt(6)/3, np.sqrt(3)/6), (-np.sqrt(6)/3, np.sqrt(3)/6),
        (-np.sqrt(6)/3, -np.sqrt(3)/6), (np.sqrt(6)/3, -np.sqrt(3)/6),
        (np.sqrt(6)/3, -np.sqrt(3)/6), (-np.sqrt(6)/3, -np.sqrt(3)/6),
        (-np.sqrt(6)/3, np.sqrt(3)/6), (np.sqrt(6)/3, np.sqrt(3)/6),
    ]

    R_min = float('inf')
    for k in range(2, 12):
        a_k, b_k = schmid_sys[k]
        denom = a_k * c2 + b_k * s2
        if abs(denom) < 1e-12:
            continue
        for sign in [+1, -1]:
            R = sign / denom
            if R > 1e-10:
                X_c = R * c2
                Y_c = R * s2
                ok = True
                for m in range(2, 12):
                    if abs(schmid_sys[m][0]*X_c + schmid_sys[m][1]*Y_c) > 1.0 + 1e-6:
                        ok = False
                        break
                if ok and R < R_min:
                    R_min = R

    X = R_min * c2
    Y = R_min * s2
    sigma_m = -(X * c2 + Y * s2) * tau_crss
    sigma_tt = 2 * sigma_m
    return sigma_m, sigma_tt, X * tau_crss, Y * tau_crss


def analytical_stress_field(r, theta, a=1.0, tau_crss=1.0):
    """
    Analytical stress field at (r, θ) in polar coordinates.

    σ_rr(r, θ) = σ_θθ(a, θ) · ln(r/a)
    σ_θθ(r, θ) = σ_θθ(a, θ) · [1 + ln(r/a)]
    """
    _, stt_void, _, _ = analytical_void_surface_stress(theta, tau_crss)
    ln_ra = np.log(r / a)
    sigma_rr = stt_void * ln_ra
    sigma_tt = stt_void * (1 + ln_ra)
    return sigma_rr, sigma_tt


# ================================================================
# 5. POST-PROCESSING AND COMPARISON
# ================================================================

def postprocess(coords, elems, sigma_el, a=1.0, tau_crss=1.0):
    """Compare CPFEM results with analytical solution."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_elems = len(elems)

    # Compute element centroids and polar coordinates
    centroids = np.zeros((n_elems, 2))
    for e in range(n_elems):
        centroids[e] = coords[elems[e]].mean(axis=0)

    r_el = np.sqrt(centroids[:, 0]**2 + centroids[:, 1]**2)
    theta_el = np.arctan2(centroids[:, 1], centroids[:, 0])
    theta_el = theta_el % np.pi  # fold to [0, π] using symmetry

    # Convert Cartesian stress to polar
    sigma_rr_fem = np.zeros(n_elems)
    sigma_tt_fem = np.zeros(n_elems)
    sigma_rt_fem = np.zeros(n_elems)

    for e in range(n_elems):
        s11, s22, s33, s12 = sigma_el[e]
        th = np.arctan2(centroids[e, 1], centroids[e, 0])
        c, s = np.cos(th), np.sin(th)
        c2, s2 = np.cos(2*th), np.sin(2*th)

        sigma_rr_fem[e] = (s11 + s22)/2 + (s11 - s22)/2 * c2 + s12 * s2
        sigma_tt_fem[e] = (s11 + s22)/2 - (s11 - s22)/2 * c2 - s12 * s2
        sigma_rt_fem[e] = -(s11 - s22)/2 * s2 + s12 * c2

    # Analytical solution at element centroids
    sigma_rr_ana = np.zeros(n_elems)
    sigma_tt_ana = np.zeros(n_elems)

    for e in range(n_elems):
        srr_a, stt_a = analytical_stress_field(r_el[e], theta_el[e], a, tau_crss)
        sigma_rr_ana[e] = srr_a
        sigma_tt_ana[e] = stt_a

    # ---- FIGURES ----
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))

    # (a) σ_rr along radial lines: CPFEM vs analytical
    ax = axes[0, 0]
    for th_target in [0, 35, 55, 90]:
        th_rad = np.radians(th_target)
        # Select elements near this angle (within ±5°)
        mask = np.abs(theta_el - th_rad) < np.radians(5)
        if mask.sum() < 3:
            continue
        r_sel = r_el[mask]
        srr_sel = sigma_rr_fem[mask]
        order = np.argsort(r_sel)
        ax.plot(r_sel[order], srr_sel[order], 'o', markersize=3,
                label=rf'CPFEM $\theta={th_target}°$')

        # Analytical
        r_ana = np.linspace(a * 1.01, r_sel.max(), 100)
        srr_ana = np.array([analytical_stress_field(ri, th_rad, a, tau_crss)[0]
                            for ri in r_ana])
        ax.plot(r_ana, srr_ana, '-', linewidth=1.5)

    ax.set_xlabel(r'$r/a$', fontsize=12)
    ax.set_ylabel(r'$\sigma_{rr} / \tau_{CRSS}$', fontsize=12)
    ax.set_title(r'(a) $\sigma_{rr}$ vs $r/a$ (dots=CPFEM, lines=analytical)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (b) σ_θθ along radial lines
    ax = axes[0, 1]
    for th_target in [0, 35, 55, 90]:
        th_rad = np.radians(th_target)
        mask = np.abs(theta_el - th_rad) < np.radians(5)
        if mask.sum() < 3:
            continue
        r_sel = r_el[mask]
        stt_sel = sigma_tt_fem[mask]
        order = np.argsort(r_sel)
        ax.plot(r_sel[order], stt_sel[order], 'o', markersize=3,
                label=rf'CPFEM $\theta={th_target}°$')

        r_ana = np.linspace(a * 1.01, r_sel.max(), 100)
        stt_ana = np.array([analytical_stress_field(ri, th_rad, a, tau_crss)[1]
                            for ri in r_ana])
        ax.plot(r_ana, stt_ana, '-', linewidth=1.5)

    ax.set_xlabel(r'$r/a$', fontsize=12)
    ax.set_ylabel(r'$\sigma_{\theta\theta} / \tau_{CRSS}$', fontsize=12)
    ax.set_title(r'(b) $\sigma_{\theta\theta}$ vs $r/a$', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # (c) σ_θθ at r ≈ a (void surface) vs θ
    ax = axes[0, 2]
    mask_void = (r_el < a * 1.3) & (r_el > a * 0.99)
    if mask_void.sum() > 5:
        order = np.argsort(theta_el[mask_void])
        ax.plot(np.degrees(theta_el[mask_void][order]),
                sigma_tt_fem[mask_void][order],
                'bo', markersize=4, label='CPFEM')

    th_ana = np.linspace(0.01, np.pi - 0.01, 200)
    stt_ana = np.array([analytical_void_surface_stress(t, tau_crss)[1] for t in th_ana])
    ax.plot(np.degrees(th_ana), stt_ana, 'r-', linewidth=2, label='Analytical')
    ax.set_xlabel(r'$\theta$ (degrees)', fontsize=12)
    ax.set_ylabel(r'$\sigma_{\theta\theta}(a, \theta) / \tau_{CRSS}$', fontsize=12)
    ax.set_title(r'(c) Hoop stress at void surface', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, 180)

    # (d) 2D contour: σ_rr from CPFEM
    ax = axes[1, 0]
    triang = plt.matplotlib.tri.Triangulation(centroids[:, 0], centroids[:, 1])
    levels = np.linspace(np.percentile(sigma_rr_fem, 2),
                         np.percentile(sigma_rr_fem, 98), 20)
    if len(levels) > 2:
        cs = ax.tricontourf(triang, sigma_rr_fem, levels=levels,
                            cmap='RdBu_r', extend='both')
        plt.colorbar(cs, ax=ax, label=r'$\sigma_{rr}/\tau_{CRSS}$')
    circle = plt.Circle((0, 0), a, fill=True, color='white', ec='black', lw=1.5)
    ax.add_patch(circle)
    ax.set_xlabel(r'$x_1/a$', fontsize=12)
    ax.set_ylabel(r'$x_2/a$', fontsize=12)
    ax.set_title(r'(d) $\sigma_{rr}$ contour (CPFEM)', fontsize=11)
    ax.set_aspect('equal')

    # (e) 2D contour: σ_θθ from CPFEM
    ax = axes[1, 1]
    levels = np.linspace(np.percentile(sigma_tt_fem, 2),
                         np.percentile(sigma_tt_fem, 98), 20)
    if len(levels) > 2:
        cs = ax.tricontourf(triang, sigma_tt_fem, levels=levels,
                            cmap='RdBu_r', extend='both')
        plt.colorbar(cs, ax=ax, label=r'$\sigma_{\theta\theta}/\tau_{CRSS}$')
    circle = plt.Circle((0, 0), a, fill=True, color='white', ec='black', lw=1.5)
    ax.add_patch(circle)
    ax.set_xlabel(r'$x_1/a$', fontsize=12)
    ax.set_ylabel(r'$x_2/a$', fontsize=12)
    ax.set_title(r'(e) $\sigma_{\theta\theta}$ contour (CPFEM)', fontsize=11)
    ax.set_aspect('equal')

    # (f) Error: |CPFEM - analytical| / τ_CRSS
    ax = axes[1, 2]
    error_rr = np.abs(sigma_rr_fem - sigma_rr_ana) / tau_crss
    error_tt = np.abs(sigma_tt_fem - sigma_tt_ana) / tau_crss

    # Plot error vs r/a
    for th_target in [0, 45, 90]:
        th_rad = np.radians(th_target)
        mask = np.abs(theta_el - th_rad) < np.radians(5)
        if mask.sum() < 3:
            continue
        r_sel = r_el[mask]
        err_sel = error_rr[mask]
        order = np.argsort(r_sel)
        ax.plot(r_sel[order], err_sel[order], 'o-', markersize=3,
                label=rf'$|\Delta\sigma_{{rr}}|$, $\theta={th_target}°$')

    ax.set_xlabel(r'$r/a$', fontsize=12)
    ax.set_ylabel(r'Error $/\tau_{CRSS}$', fontsize=12)
    ax.set_title('(f) CPFEM vs analytical error', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')

    plt.tight_layout()
    fig_path = 'figures/cpfem_verification.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"  Figure saved to: {fig_path}")

    # Summary statistics
    valid = r_el < 5 * a
    print(f"\n  Verification summary (r < 5a):")
    print(f"    Mean |σ_rr error| = {error_rr[valid].mean():.4f} τ_CRSS")
    print(f"    Mean |σ_θθ error| = {error_tt[valid].mean():.4f} τ_CRSS")
    print(f"    Max  |σ_rr error| = {error_rr[valid].max():.4f} τ_CRSS")
    print(f"    Max  |σ_θθ error| = {error_tt[valid].max():.4f} τ_CRSS")


# ================================================================
# MAIN
# ================================================================

def main():
    print("=" * 70)
    print("CPFEM Verification: BCC Void Problem")
    print("=" * 70)

    # Problem parameters
    a = 1.0           # void radius
    b = 10.0          # outer boundary (b/a = 10)
    tau_crss = 1.0    # normalize
    E = 500.0         # high E/τ for near-rigid response
    nu = 0.3
    p_applied = 2.0   # far-field pressure

    # Mesh
    n_r = 40           # radial divisions
    n_theta = 64       # circumferential divisions
    print(f"\nMesh: n_r={n_r}, n_θ={n_theta}, bias=2.5")
    t0 = time.time()
    coords, elems, bc_inner, bc_outer, r_vals, theta_vals = \
        generate_annular_mesh(a, b, n_r, n_theta, bias=2.5)
    print(f"  Generated: {len(coords)} nodes, {len(elems)} elements "
          f"({time.time()-t0:.2f}s)")

    # Material
    material = BCCCrystalPlasticity(
        E=E, nu=nu, tau_crss=tau_crss,
        gamma_dot_0=0.001, m=0.05
    )
    print(f"  Material: E={E}, ν={nu}, τ_CRSS={tau_crss}, m={material.m}")

    # Solve
    print(f"\nSolving with p = {p_applied} τ_CRSS...")
    t0 = time.time()
    u, sigma_el = solve_cpfem(
        coords, elems, bc_inner, bc_outer,
        material, p_applied,
        n_steps=20, dt=1.0
    )
    print(f"  Solve time: {time.time()-t0:.1f}s")

    # Post-process
    print("\nPost-processing...")
    postprocess(coords, elems, sigma_el, a, tau_crss)

    print("\n" + "=" * 70)
    print("CPFEM VERIFICATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
