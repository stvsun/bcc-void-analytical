"""
Ultimate Algorithm for rate-independent crystal plasticity (Borja, 2013, Box 8.1).

This implements the stress-point integration algorithm from Chapter 8 of
"Plasticity: Modeling & Computation" by R.I. Borja (Springer, 2013).

Key properties of the Ultimate Algorithm:
  (a) Unconditionally convergent
  (b) Exact for imposed crystal deformation varying as a ramp function
  (c) Unique crystal stress regardless of selected independent systems

The algorithm incrementally activates slip systems one by one via a
ramp parameter κ, identifying linearly independent active constraints
at each stage.

For the BCC void problem in plane strain:
  - 12 slip systems in {110}<111> family
  - Plane strain: ε33 = 0, but σ33 ≠ 0 (tracked)
  - 4 independent stress components: [σ11, σ22, σ33, σ12]
  - Max 3 independent active systems (since tr(ε^p) = 0 adds a constraint)
"""

import numpy as np


class UltimateAlgorithmBCC:
    """
    Rate-independent crystal plasticity using the Ultimate Algorithm.

    Infinitesimal strain formulation with Taylor hardening.
    """

    def __init__(self, E=1000.0, nu=0.3, tau_crss=1.0, h=0.0):
        """
        Parameters
        ----------
        E : float — Young's modulus
        nu : float — Poisson's ratio
        tau_crss : float — initial critical resolved shear stress (equal on all systems)
        h : float — Taylor hardening modulus (0 = perfect plasticity)
        """
        self.E = E
        self.nu = nu
        self.tau_y0 = tau_crss
        self.h = h

        # Elastic stiffness (4x4 Voigt: [σ11, σ22, σ33, σ12])
        lam = E * nu / ((1 + nu) * (1 - 2 * nu))
        mu = E / (2 * (1 + nu))
        self.mu_c = mu  # crystal shear modulus
        self.Ce = np.array([
            [lam + 2*mu, lam,       lam,       0],
            [lam,       lam + 2*mu, lam,       0],
            [lam,       lam,       lam + 2*mu, 0],
            [0,         0,         0,         mu],
        ])

        self._setup_slip_systems()

    def _setup_slip_systems(self):
        """Set up BCC {110}<111> Schmid tensors in the plane strain frame."""
        R = np.array([
            [0, 0, 1],
            [1/np.sqrt(2), -1/np.sqrt(2), 0],
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

        self.N = 12  # number of slip systems (N in the book's notation)
        # α^(β) Schmid tensors in Voigt: [α_11, α_22, α_33, α_12]
        # such that τ = α : σ = α_11 σ_11 + α_22 σ_22 + α_33 σ_33 + 2 α_12 σ_12
        self.alpha = np.zeros((self.N, 4))
        # For strain: dε^p = Σ dγ^β α^(β)  in Voigt [dε11, dε22, dε33, 2dε12]
        self.alpha_strain = np.zeros((self.N, 4))

        for beta, (n, s) in enumerate(slip_data):
            n_vec = np.array(n, dtype=float)
            s_vec = np.array(s, dtype=float)
            n_hat = n_vec / np.linalg.norm(n_vec)
            s_hat = s_vec / np.linalg.norm(s_vec)
            n_p = R @ n_hat
            s_p = R @ s_hat

            P = 0.5 * (np.outer(s_p, n_p) + np.outer(n_p, s_p))
            self.alpha[beta] = [P[0,0], P[1,1], P[2,2], 2*P[0,1]]
            self.alpha_strain[beta] = [P[0,0], P[1,1], P[2,2], 2*P[0,1]]

    def resolved_shear(self, sigma, beta):
        """Resolved shear stress on system β: τ = α^(β) : σ"""
        return self.alpha[beta] @ sigma

    def _g_matrix(self, active_set):
        """
        Compute the g_ij matrix (Eq. 8.29) for the active set.

        g_ij = {μ_c + h,                        if i = j
               {2μ_c ψ^(βi) α^(βi) : α^(βj) + h,  otherwise

        For perfect plasticity (h=0), off-diagonal terms involve
        the elastic interaction between slip systems.
        """
        m = len(active_set)
        g = np.zeros((m, m))

        for i in range(m):
            bi = active_set[i][0]  # system index
            psi_i = active_set[i][1]  # sign ψ = sign(τ)
            for j in range(m):
                bj = active_set[j][0]
                psi_j = active_set[j][1]
                if i == j:
                    g[i, j] = self.mu_c + self.h
                else:
                    # 2μ_c ψ^(βi) α^(βi) : Ce : ψ^(βj) α^(βj)  ... actually
                    # g_ij = 2μ_c (ψ^(βi) α^(βi)) : (ψ^(βj) α^(βj)) + h
                    # where the inner product is via Ce
                    ai = psi_i * self.alpha[bi]
                    aj = psi_j * self.alpha[bj]
                    # The interaction: α_i : Ce : α_j (using Voigt)
                    # For Voigt with [σ11, σ22, σ33, σ12]:
                    # α : Ce : α' = α^T Ce α'  (but need to account for
                    # the factor of 2 on shear: Ce acts on [ε11,ε22,ε33,2ε12])
                    # Actually: τ = α : σ = α · σ_voigt (with σ_voigt = [σ11,σ22,σ33,σ12])
                    # and ε^p_voigt = dγ α_strain
                    # so Ce : α_strain gives stress increment per unit slip
                    Ce_alpha_j = self.Ce @ (psi_j * self.alpha_strain[bj])
                    g[i, j] = (psi_i * self.alpha[bi]) @ Ce_alpha_j + self.h

        return g

    def stress_update(self, sigma_n, delta_eps, tau_y_n=None):
        """
        Ultimate Algorithm stress update (Box 8.1).

        Parameters
        ----------
        sigma_n : (4,) — stress at time t_n [σ11, σ22, σ33, σ12]
        delta_eps : (4,) — strain increment [Δε11, Δε22, Δε33=0, 2Δε12]
        tau_y_n : float or None — current yield stress (None → use tau_y0)

        Returns
        -------
        sigma : (4,) — updated stress
        tau_y : float — updated yield stress
        delta_gamma : dict — {system_index: slip_increment}
        C_tang : (4,4) — algorithmic tangent modulus
        """
        if tau_y_n is None:
            tau_y_n = self.tau_y0

        # Step 1: Initialize
        J_act = set()  # all potentially active systems (indices)
        J_bar_act = []  # linearly independent active systems [(index, sign)]
        sigma = sigma_n.copy()
        tau_y = tau_y_n
        delta_eps_remaining = delta_eps.copy()
        total_delta_gamma = {}  # accumulate slips

        # Elastic predictor: find κ to first yield
        # σ(t) = σ_n + κ Ce : Δε, where κ goes from 0 to 1
        sigma_trial_full = sigma_n + self.Ce @ delta_eps

        # Check if fully elastic
        max_f = -1e30
        for beta in range(self.N):
            tau_beta = abs(self.alpha[beta] @ sigma_trial_full)
            f_beta = tau_beta - tau_y
            if f_beta > max_f:
                max_f = f_beta

        if max_f < 1e-10 * self.tau_y0:
            # Fully elastic
            C_tang = self.Ce.copy()
            return sigma_trial_full, tau_y, {}, C_tang

        # Step 6 equivalent: find the first system to activate
        kappa_applied = 0.0  # how much of Δε has been applied

        for outer_iter in range(50):  # max activations
            # Current elastic predictor with remaining strain
            Ce_deps = self.Ce @ delta_eps_remaining

            # Find κ for next system activation (Step 5-6)
            kappa_next = 1.0 - kappa_applied  # remaining
            beta_next = -1

            # For each system NOT in J_bar_act, find κ where it activates
            active_indices = {item[0] for item in J_bar_act}

            for beta in range(self.N):
                if beta in active_indices:
                    continue
                # Current resolved shear stress
                psi_beta = np.sign(self.alpha[beta] @ (sigma + Ce_deps))
                if abs(psi_beta) < 0.5:
                    continue

                # τ^(β)(κ) = α^(β) : [σ + κ Ce : Δε_rem - Σ Δγ̃ ψ Ce α]
                # At what κ does |τ^(β)| = τ_y + h Σ Δγ̃?

                tau_current = self.alpha[beta] @ sigma
                d_tau = self.alpha[beta] @ Ce_deps

                # Account for slip corrections from currently active systems
                if len(J_bar_act) > 0:
                    g_mat = self._g_matrix(J_bar_act)
                    try:
                        g_inv = np.linalg.inv(g_mat)
                    except np.linalg.LinAlgError:
                        continue

                    # Compute how slips change with κ
                    # Δγ^(η) = κ Σ_η' g^{-1}_{ηη'} ψ^(η') α^(η') : Ce : Δε_rem
                    rhs = np.array([
                        J_bar_act[j][1] * self.alpha[J_bar_act[j][0]] @ Ce_deps
                        for j in range(len(J_bar_act))
                    ])
                    d_gamma_d_kappa = g_inv @ rhs  # dΔγ/dκ for active systems

                    # Effect on τ^(β):
                    correction = 0.0
                    for j in range(len(J_bar_act)):
                        bj, psi_j = J_bar_act[j]
                        correction += d_gamma_d_kappa[j] * (
                            self.alpha[beta] @ (self.Ce @ (psi_j * self.alpha_strain[bj]))
                        )
                    # Also hardening correction
                    h_correction = self.h * np.sum(np.abs(d_gamma_d_kappa))

                    d_tau_eff = d_tau - correction
                    d_tau_y = h_correction
                else:
                    d_tau_eff = d_tau
                    d_tau_y = 0.0

                # |tau_current + κ * d_tau_eff| = tau_y + κ * d_tau_y
                # Two cases: positive and negative τ
                for psi in [+1, -1]:
                    numerator = psi * tau_y - psi * tau_current
                    denominator = psi * d_tau_eff - d_tau_y
                    if abs(denominator) < 1e-15:
                        continue
                    kappa_beta = numerator / denominator
                    if kappa_beta > 1e-12 and kappa_beta < kappa_next - 1e-12:
                        kappa_next = kappa_beta
                        beta_next = beta

            # Apply strain up to kappa_next
            kappa_step = min(kappa_next, 1.0 - kappa_applied)

            if kappa_step < 1e-14:
                kappa_step = 1e-14

            # Update stress with current active set over this sub-increment
            sub_deps = kappa_step * delta_eps_remaining / (1.0 - kappa_applied) \
                if (1.0 - kappa_applied) > 1e-14 else np.zeros(4)

            sigma_trial = sigma + self.Ce @ sub_deps

            if len(J_bar_act) > 0:
                g_mat = self._g_matrix(J_bar_act)
                try:
                    g_inv = np.linalg.inv(g_mat)
                except np.linalg.LinAlgError:
                    break

                # Compute slips (Eq. 8.31)
                rhs = np.array([
                    J_bar_act[j][1] * self.alpha[J_bar_act[j][0]] @ (self.Ce @ sub_deps)
                    for j in range(len(J_bar_act))
                ])
                d_gamma = g_inv @ rhs

                # Check for deactivation (Step 4): Δγ̃ < 0
                deactivated = False
                for j in range(len(J_bar_act)):
                    if d_gamma[j] < -1e-12:
                        # Remove this system
                        removed = J_bar_act.pop(j)
                        deactivated = True
                        break

                if deactivated:
                    continue  # redo with updated active set

                # Apply plastic correction to stress
                deps_p = np.zeros(4)
                for j in range(len(J_bar_act)):
                    bj, psi_j = J_bar_act[j]
                    deps_p += d_gamma[j] * psi_j * self.alpha_strain[bj]

                    # Accumulate total slip
                    key = bj
                    total_delta_gamma[key] = total_delta_gamma.get(key, 0.0) + d_gamma[j]

                sigma = sigma_trial - self.Ce @ deps_p

                # Update yield stress (Taylor hardening)
                tau_y += self.h * np.sum(np.abs(d_gamma))
            else:
                sigma = sigma_trial

            kappa_applied += kappa_step

            # Add newly activated system (Step 8-9)
            if beta_next >= 0 and kappa_applied < 1.0 - 1e-12:
                psi_new = np.sign(self.alpha[beta_next] @ sigma)
                if abs(psi_new) < 0.5:
                    psi_new = 1.0
                J_bar_act.append((beta_next, int(psi_new)))

                # Check linear independence
                if len(J_bar_act) > 1:
                    A = np.array([
                        item[1] * self.alpha_strain[item[0]]
                        for item in J_bar_act
                    ])
                    rank = np.linalg.matrix_rank(A, tol=1e-8)
                    if rank < len(J_bar_act):
                        J_bar_act.pop()  # remove the redundant one

            if kappa_applied >= 1.0 - 1e-12:
                break

        # Algorithmic tangent (elastic for simplicity; can be improved)
        C_tang = self.Ce.copy()

        return sigma, tau_y, total_delta_gamma, C_tang


def solve_cpfem_ultimate(coords, elems, bc_inner, bc_outer,
                         material, p_applied, n_steps=50):
    """
    Solve the BCC void problem using FEM + Ultimate Algorithm.

    Uses displacement-controlled incremental loading with Newton-Raphson.
    """
    from scipy import sparse
    from scipy.sparse.linalg import spsolve

    n_nodes = len(coords)
    n_elems = len(elems)
    n_dof = 2 * n_nodes

    # Precompute B matrices and areas
    B_all = []
    area_all = []
    for e in range(n_elems):
        el_nodes = elems[e]
        x = coords[el_nodes, 0]
        y = coords[el_nodes, 1]
        area = 0.5 * abs((x[1]-x[0])*(y[2]-y[0]) - (x[2]-x[0])*(y[1]-y[0]))
        if area < 1e-15:
            area = 1e-15
        dN_dx = np.array([y[1]-y[2], y[2]-y[0], y[0]-y[1]]) / (2*area)
        dN_dy = np.array([x[2]-x[1], x[0]-x[2], x[1]-x[0]]) / (2*area)
        B = np.zeros((4, 6))
        for i in range(3):
            B[0, 2*i]   = dN_dx[i]
            B[1, 2*i+1] = dN_dy[i]
            B[3, 2*i]   = dN_dy[i]
            B[3, 2*i+1] = dN_dx[i]
        B_all.append(B)
        area_all.append(area)

    # Outer boundary tributary lengths and normals
    outer_trib = []
    for i in range(len(bc_outer)):
        node = bc_outer[i]
        x, y = coords[node]
        r = np.sqrt(x**2 + y**2)
        nx, ny = x/r, y/r
        n_prev = bc_outer[(i-1) % len(bc_outer)]
        n_next = bc_outer[(i+1) % len(bc_outer)]
        tl = (np.linalg.norm(coords[n_next] - coords[node]) +
              np.linalg.norm(coords[node] - coords[n_prev])) / 2
        outer_trib.append((node, nx, ny, tl))

    # Pin rigid body modes
    pin0 = bc_inner[0]
    pin1 = bc_inner[len(bc_inner)//4]
    bc_dofs = [2*pin0, 2*pin0+1, 2*pin1+1]

    # State variables
    u = np.zeros(n_dof)
    sigma_el = np.zeros((n_elems, 4))
    tau_y_el = np.full(n_elems, material.tau_y0)
    eps_el_old = np.zeros((n_elems, 4))

    dp = p_applied / n_steps
    print(f"  CPFEM (Ultimate): {n_nodes} nodes, {n_elems} elems, {n_steps} steps")

    for step in range(1, n_steps + 1):
        p_curr = dp * step

        for nr_iter in range(50):
            rows, cols, vals = [], [], []
            f_int = np.zeros(n_dof)

            for e in range(n_elems):
                B = B_all[e]
                area = area_all[e]
                nodes_e = elems[e]
                dofs = np.array([2*nodes_e[0], 2*nodes_e[0]+1,
                                 2*nodes_e[1], 2*nodes_e[1]+1,
                                 2*nodes_e[2], 2*nodes_e[2]+1])
                eps = B @ u[dofs]
                d_eps = eps - eps_el_old[e]

                sig, ty, _, Ct = material.stress_update(
                    sigma_el[e], d_eps, tau_y_el[e])

                f_el = B.T @ sig * area
                K_el = B.T @ Ct @ B * area

                for i in range(6):
                    f_int[dofs[i]] += f_el[i]
                    for j in range(6):
                        rows.append(dofs[i])
                        cols.append(dofs[j])
                        vals.append(K_el[i, j])

            # External force
            f_ext = np.zeros(n_dof)
            for node, nx, ny, tl in outer_trib:
                f_ext[2*node]   += (-p_curr) * nx * tl
                f_ext[2*node+1] += (-p_curr) * ny * tl

            residual = f_int - f_ext

            K = sparse.coo_matrix((vals, (rows, cols)),
                                  shape=(n_dof, n_dof)).tocsr()

            # Apply BCs
            K_lil = K.tolil()
            for d in bc_dofs:
                residual[d] = 0.0
                K_lil[d, :] = 0
                K_lil[:, d] = 0
                K_lil[d, d] = 1.0
            K = K_lil.tocsr()

            try:
                du = spsolve(K.tocsc(), -residual)
            except Exception:
                du = np.zeros(n_dof)

            u += du
            res_norm = np.linalg.norm(residual)

            if np.linalg.norm(du) < 1e-8 and res_norm < 1e-6:
                break

        # Commit: update state
        for e in range(n_elems):
            B = B_all[e]
            nodes_e = elems[e]
            dofs = np.array([2*nodes_e[0], 2*nodes_e[0]+1,
                             2*nodes_e[1], 2*nodes_e[1]+1,
                             2*nodes_e[2], 2*nodes_e[2]+1])
            eps = B @ u[dofs]
            d_eps = eps - eps_el_old[e]
            sig, ty, _, _ = material.stress_update(sigma_el[e], d_eps, tau_y_el[e])
            sigma_el[e] = sig
            tau_y_el[e] = ty
            eps_el_old[e] = eps.copy()

        if step % 10 == 0 or step == n_steps or step <= 5:
            max_tau_ratio = 0
            n_yielded = 0
            for e in range(n_elems):
                for beta in range(material.N):
                    tr = abs(material.alpha[beta] @ sigma_el[e]) / tau_y_el[e]
                    max_tau_ratio = max(max_tau_ratio, tr)
                    if tr > 0.99:
                        n_yielded += 1
                        break
            print(f"    Step {step}/{n_steps}: p={p_curr:.3f}, "
                  f"max|τ/τ_y|={max_tau_ratio:.4f}, "
                  f"yielded={n_yielded}/{n_elems}, "
                  f"|R|={res_norm:.2e}, NR={nr_iter+1}")

    return u, sigma_el


# ================================================================
# MAIN
# ================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, 'src')
    from cpfem_bcc_void import (generate_annular_mesh, analytical_void_surface_stress,
                                 analytical_stress_field, postprocess)
    import time

    print("=" * 70)
    print("CPFEM Verification with Ultimate Algorithm")
    print("=" * 70)

    a = 1.0
    b = 8.0
    tau_crss = 1.0
    E = 1000.0
    nu = 0.3
    p_applied = 1.5

    n_r = 30
    n_theta = 48

    print(f"\nMesh: n_r={n_r}, n_θ={n_theta}")
    coords, elems, bc_inner, bc_outer, _, _ = \
        generate_annular_mesh(a, b, n_r, n_theta, bias=2.0)
    print(f"  {len(coords)} nodes, {len(elems)} elements")

    material = UltimateAlgorithmBCC(E=E, nu=nu, tau_crss=tau_crss, h=0.0)
    print(f"  Material: E={E}, ν={nu}, τ_CRSS={tau_crss}, h={material.h}")

    print(f"\nSolving with p = {p_applied} τ_CRSS...")
    t0 = time.time()
    u, sigma_el = solve_cpfem_ultimate(
        coords, elems, bc_inner, bc_outer,
        material, p_applied, n_steps=30
    )
    print(f"  Solve time: {time.time()-t0:.1f}s")

    print("\nPost-processing...")
    postprocess(coords, elems, sigma_el, a, tau_crss)

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)
