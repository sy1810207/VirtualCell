"""
VirtualCell v0.85 — Force / Energy Computations
Implements Eq.1–6 from the paper with numpy vectorization.
"""
import numpy as np
from scipy.spatial import cKDTree
from config import SimConfig

EPS = 1e-7          # small number to avoid division by zero


# ═══════════════════════════════════════════════════════════════════
#  1. Membrane Forces (Eq.1)  — used for both cell and nucleus
# ═══════════════════════════════════════════════════════════════════

def membrane_forces(vertices, edges, faces, face_pairs, shared_edges,
                    S0, config: SimConfig):
    """
    Compute all membrane forces and total energy.

    Parameters
    ----------
    vertices : (N_v, 3)
    edges : (N_e, 2)
    faces : (N_f, 3)
    face_pairs : (N_p, 2)  — pairs of adjacent face indices
    shared_edges : (N_p, 2) — vertex indices of the shared edge per pair
    S0 : float — reference surface area
    config : SimConfig

    Returns (forces: (N_v, 3), energy: float)
    """
    N = len(vertices)
    forces = np.zeros((N, 3), dtype=float)
    energy = 0.0

    # ── Precompute edge vectors ──
    vi = vertices[edges[:, 0]]    # (N_e, 3)
    vj = vertices[edges[:, 1]]    # (N_e, 3)
    rij_vec = vj - vi             # (N_e, 3)
    rij = np.linalg.norm(rij_vec, axis=1)  # (N_e,)
    rij_hat = rij_vec / np.maximum(rij, EPS)[:, None]  # unit vectors

    # ── U_FENE+WCA (bond stretching + short-range repulsion) ──
    e_bond, f_bond = _fene_wca_potential(rij, config)
    energy += e_bond
    f_vec = f_bond[:, None] * rij_hat
    np.add.at(forces, edges[:, 0], -f_vec)
    np.add.at(forces, edges[:, 1],  f_vec)

    # ── U_curvature ──
    e_curv, f_curv = _curvature_forces(vertices, faces, face_pairs,
                                        shared_edges, config)
    energy += e_curv
    forces += f_curv

    # ── U_surface_area ──
    e_sa, f_sa = _surface_area_forces(vertices, faces, S0, config)
    energy += e_sa
    forces += f_sa

    return forces, energy


def _fene_wca_potential(rij, cfg):
    """
    FENE + WCA bond potential for membrane edges.

    U_FENE = -(k/2) * R_max^2 * ln(1 - ((r - r_eq) / R_max)^2)
    U_WCA  = 4ε[(σ/r)^12 - (σ/r)^6] + ε   for r < 2^(1/6)·σ
             0                               for r ≥ 2^(1/6)·σ

    Returns (total_energy, f_scalar) where f_scalar = -dU/dr (F along bond).
    """
    # --- FENE ---
    delta = rij - cfg.r_eq_bond
    ratio_sq = delta**2 / cfg.R_max**2
    # Clamp to prevent log(0) or log(negative)
    ratio_sq = np.minimum(ratio_sq, 0.999)

    U_fene = -0.5 * cfg.k_fene * cfg.R_max**2 * np.log(1.0 - ratio_sq)
    # F_FENE = -dU/dr = -k·(r-r_eq) / (1 - ((r-r_eq)/R_max)²)
    F_fene = -cfg.k_fene * delta / (1.0 - ratio_sq)

    e_fene = float(np.sum(U_fene))

    # --- WCA (shifted-truncated repulsive LJ) ---
    r_cut_wca = 2.0**(1.0/6.0) * cfg.sigma_wca

    U_wca = np.zeros_like(rij)
    F_wca = np.zeros_like(rij)

    mask = rij < r_cut_wca
    if np.any(mask):
        r_m = np.maximum(rij[mask], 0.4 * cfg.sigma_wca)  # hard floor
        sr6 = (cfg.sigma_wca / r_m)**6
        sr12 = sr6**2
        U_wca[mask] = 4.0 * cfg.epsilon_wca * (sr12 - sr6) + cfg.epsilon_wca
        # F_WCA = -dU/dr = 24ε/r [2(σ/r)^12 - (σ/r)^6]  (positive = repulsive)
        F_wca[mask] = 24.0 * cfg.epsilon_wca / r_m * (2.0 * sr12 - sr6)

    e_wca = float(np.sum(U_wca))

    energy = e_fene + e_wca
    f_scalar = F_fene + F_wca

    return energy, f_scalar


def _curvature_forces(vertices, faces, face_pairs, shared_edges, cfg):
    """
    U_curvature = (κ_curve/2) * Σ (1 - cos θ_ij)
    θ_ij = angle between normals of adjacent triangles sharing an edge.

    Fully vectorized analytical gradient computation.
    For each face pair sharing edge (e0, e1), we identify the opposite
    vertices (p_a for face_a, p_b for face_b) and compute the gradient
    of cos(θ) = n_a · n_b w.r.t. each of the 4 vertices.
    """
    N = len(vertices)
    forces = np.zeros((N, 3), dtype=float)

    if len(face_pairs) == 0:
        return 0.0, forces

    # ── Compute all face normals and areas ──
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross_all = np.cross(v1 - v0, v2 - v0)
    area_2 = np.linalg.norm(cross_all, axis=1)  # 2 * area
    area_2 = np.maximum(area_2, EPS)
    normals = cross_all / area_2[:, None]  # unit normals (N_f, 3)

    # ── Energy: sum over pairs ──
    n_a = normals[face_pairs[:, 0]]  # (N_p, 3)
    n_b = normals[face_pairs[:, 1]]  # (N_p, 3)
    cos_theta = np.sum(n_a * n_b, axis=1)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    energy = float(cfg.kappa_curve / 2.0 * np.sum(1.0 - cos_theta))

    # ── Identify opposite vertices for each face pair ──
    # For each pair, face_a and face_b share edge (e0, e1).
    # The opposite vertex of face_a is the vertex NOT on the shared edge.
    # The opposite vertex of face_b is the vertex NOT on the shared edge.
    fa_verts = faces[face_pairs[:, 0]]  # (N_p, 3) vertex indices of face a
    fb_verts = faces[face_pairs[:, 1]]  # (N_p, 3) vertex indices of face b
    se0 = shared_edges[:, 0]  # (N_p,) first vertex of shared edge
    se1 = shared_edges[:, 1]  # (N_p,) second vertex of shared edge

    # Opposite vertex = the one that's neither se0 nor se1
    def find_opposite(face_verts, e0, e1):
        """For each face, find the vertex that is not e0 or e1."""
        N_p = len(face_verts)
        opp = np.zeros(N_p, dtype=int)
        for col in range(3):
            v = face_verts[:, col]
            mask = (v != e0) & (v != e1)
            opp[mask] = v[mask]
        return opp

    pa_idx = find_opposite(fa_verts, se0, se1)  # opposite vertex of face a
    pb_idx = find_opposite(fb_verts, se0, se1)  # opposite vertex of face b

    # ── Compute gradient of cos(θ) analytically ──
    # For face_a with vertices (e0, e1, pa): normal n_a = (e1-e0)×(pa-e0) / |...|
    # The gradient of n_a · n_b w.r.t. vertex positions uses the chain rule
    # through the normalized cross product.
    #
    # For n = c / |c| where c = u × v:
    #   dn/dx = (I - n⊗n) / |c| · dc/dx
    #
    # For face_a: c_a = (e1-e0) × (pa-e0)
    #   dc_a/dpa = -(e1-e0)×  (skew matrix, result: -(e1-e0) × dpa)
    #   dc_a/de0 = (pa-e1)×   (combined from both edge vectors)
    #   dc_a/de1 = -(pa-e0)×  (from first edge vector)

    pos_e0 = vertices[se0]    # (N_p, 3)
    pos_e1 = vertices[se1]    # (N_p, 3)
    pos_pa = vertices[pa_idx] # (N_p, 3)
    pos_pb = vertices[pb_idx] # (N_p, 3)

    # Edge vectors for face_a
    ea1 = pos_e1 - pos_e0  # (N_p, 3)
    ea2 = pos_pa - pos_e0  # (N_p, 3)
    # Edge vectors for face_b
    eb1 = pos_e1 - pos_e0  # same shared edge
    eb2 = pos_pb - pos_e0  # (N_p, 3)

    # Areas (half cross product magnitude)
    ca = np.cross(ea1, ea2)  # (N_p, 3) unnormalized normal of face a
    cb = np.cross(eb1, eb2)  # (N_p, 3) unnormalized normal of face b
    ca_len = np.linalg.norm(ca, axis=1, keepdims=True)
    cb_len = np.linalg.norm(cb, axis=1, keepdims=True)
    ca_len = np.maximum(ca_len, EPS)
    cb_len = np.maximum(cb_len, EPS)
    na = ca / ca_len  # (N_p, 3) unit normal face a
    nb = cb / cb_len  # (N_p, 3) unit normal face b

    # Prefactor: -kappa/2 * d(1-cos)/dcos = kappa/2
    # Force = -dU/dv = -kappa/2 * (-dcos/dv) = kappa/2 * dcos/dv
    # But we compute dcos/dv and multiply by kappa/2, then negate for force
    # F = -dU/dv = kappa/2 * dcos/dv  (since U = kappa/2 * (1 - cos))
    prefactor = cfg.kappa_curve / 2.0  # scalar

    # Gradient of cos(θ) = na · nb w.r.t. each vertex:
    # dcos/dv = (dnb/dv)^T na + (dna/dv)^T nb
    # For opposite vertex pa (only affects na):
    #   dcos/dpa = (dna/dpa)^T nb
    #   dna/dpa = (I - na⊗na)/|ca| · dca/dpa
    #   dca/dpa = -skew(ea1) → dca/dpa · x = -(ea1 × x) → for component, = -ea1 × e_k
    #   So dna/dpa_k = (I - na⊗na)/|ca| · (-ea1 × e_k)
    #   dcos/dpa_k = nb · dna/dpa_k = nb · (I-na⊗na)/|ca| · (-ea1×e_k)
    #   Let q_a = (I-na⊗na)/|ca| nb = (nb - (na·nb)na)/|ca|
    #   dcos/dpa_k = q_a · (-ea1×e_k) = (ea1×q_a) · e_k  (vector triple product)
    #   So dcos/dpa = ea1 × q_a  (as a 3-vector)

    cos_th = np.sum(na * nb, axis=1, keepdims=True)  # (N_p, 1)
    qa = (nb - cos_th * na) / ca_len  # (N_p, 3)
    qb = (na - cos_th * nb) / cb_len  # (N_p, 3)

    # Gradient w.r.t. pa: dcos/dpa = ea1 × qa
    dcos_dpa = np.cross(ea1, qa)   # (N_p, 3)

    # Gradient w.r.t. pb: dcos/dpb = eb1 × qb
    dcos_dpb = np.cross(eb1, qb)   # (N_p, 3)

    # Gradient w.r.t. e1:
    #   From face_a: dca/de1 = -skew(ea2)^T → dca/de1 · x = ea2 × x
    #     Actually: ca = ea1 × ea2, dca/de1 = d(ea1)/de1 × ea2 = I × ea2
    #     So dca_k/de1 = e_k × ea2, meaning dca/de1 applied to direction = cross with ea2
    #     Wait: d(ea1 × ea2)/de1 where ea1 = e1 - e0
    #     = d(ea1)/de1 × ea2 = I × ea2 → not right notation
    #     Actually dca/de1_k = (e_k) × ea2 → as matrix: dca/de1 = -[ea2]_×
    #   From face_b: same shared edge, dcb/de1 = -[eb2]_×
    #   dcos/de1 = (dna/de1)^T nb + (dnb/de1)^T na
    #            = cross(-ea2, qa) + cross(-eb2, qb)
    #   Using same derivation: dna/de1_k = (I-na⊗na)/|ca| · (e_k × ea2)
    #   so dcos/de1_k = qa · (e_k × ea2) + qb · (e_k × eb2)
    #                 = (-ea2 × qa + -eb2 × qb) · e_k   (wait, need to be careful)
    #   Actually: qa · (e_k × ea2) = (ea2 × qa) · e_k  (by cyclic property of triple product: a·(b×c)=b·(c×a)=c·(a×b))
    #   Wait: a · (b × c) = b · (c × a). So qa · (e_k × ea2) = e_k · (ea2 × qa)
    #   Hmm, that gives the same sign? Let me redo:
    #   a · (b × c) = (a × b) · c  → qa · (e_k × ea2) = (qa × e_k) · ea2
    #   Alternatively: scalar triple product a·(b×c) = det[a,b,c]
    #   qa·(e_k × ea2) = det[qa, e_k, ea2] = e_k · (ea2 × qa)
    #   So dcos/de1 = ea2 × qa + eb2 × qb   (not negated!)
    #
    #   Actually I need to re-derive. ca = ea1 × ea2 where ea1 = e1 - e0.
    #   dca/de1 (as operator on perturbation δe1):  δca = δe1 × ea2
    #   dna/de1 · δe1 = (I - na⊗na)/|ca| · (δe1 × ea2)
    #   dcos/de1 · δe1 = nb · dna/de1·δe1 + na · dnb/de1·δe1
    #                   = qa · (δe1 × ea2) + qb · (δe1 × eb2)
    #   qa · (δe1 × ea2) = δe1 · (ea2 × qa)  (scalar triple product identity)
    #   So dcos/de1 = ea2 × qa + eb2 × qb
    dcos_de1 = np.cross(ea2, qa) + np.cross(eb2, qb)  # (N_p, 3)

    # Gradient w.r.t. e0:
    #   ca = (e1-e0) × (pa-e0), dca/de0 · δe0 = (-δe0) × ea2 + ea1 × (-δe0)
    #     = -(δe0 × ea2) - (ea1 × (-δe0)) ... wait
    #     = -δe0 × ea2 + ea1 × (-δe0)   -- no
    #   Let me be careful: ca = ea1 × ea2, ea1 = e1 - e0, ea2 = pa - e0
    #   dea1/de0 = -I, dea2/de0 = -I
    #   dca/de0 · δe0 = (-δe0) × ea2 + ea1 × (-δe0)
    #                  = -(δe0 × ea2) - (ea1 × δe0)
    #                  = -(δe0 × ea2) + (δe0 × ea1)   [since a×b = -b×a]
    #                  = δe0 × (ea1 - ea2)
    #   Similarly for face_b: dcb/de0 · δe0 = δe0 × (eb1 - eb2)
    #
    #   dcos/de0 · δe0 = qa · (δe0 × (ea1-ea2)) + qb · (δe0 × (eb1-eb2))
    #                  = δe0 · ((ea1-ea2) × qa) + δe0 · ((eb1-eb2) × qb)
    #   So dcos/de0 = (ea1-ea2) × qa + (eb1-eb2) × qb
    dcos_de0 = np.cross(ea1 - ea2, qa) + np.cross(eb1 - eb2, qb)  # (N_p, 3)

    # ── Scatter forces: F = -dU/dv = prefactor * dcos/dv ──
    f_pa = prefactor * dcos_dpa
    f_pb = prefactor * dcos_dpb
    f_e0 = prefactor * dcos_de0
    f_e1 = prefactor * dcos_de1

    np.add.at(forces, pa_idx, f_pa)
    np.add.at(forces, pb_idx, f_pb)
    np.add.at(forces, se0, f_e0)
    np.add.at(forces, se1, f_e1)

    return energy, forces


def _surface_area_forces(vertices, faces, S0, cfg):
    """
    U_surfacearea = (κ_s/2) * (S - S0)²
    """
    N = len(vertices)
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    e1 = v1 - v0
    e2 = v2 - v0
    cross = np.cross(e1, e2)             # (N_f, 3)
    area_2 = np.linalg.norm(cross, axis=1)  # 2 * area
    area_2 = np.maximum(area_2, EPS)
    S = float(np.sum(area_2) * 0.5)
    energy = cfg.kappa_s / 2.0 * (S - S0) ** 2

    # dS/dv_k for each vertex of each triangle
    # For triangle (v0, v1, v2), area = 0.5 * |e1 × e2|
    # dA/dv0 = -0.5 * (cross × (v2-v1)) / |cross|  ... etc.
    # Use: d|cross|/dv = (cross/|cross|) · d(cross)/dv
    forces = np.zeros((N, 3), dtype=float)
    prefactor = cfg.kappa_s * (S - S0)  # dU/dS

    cross_hat = cross / area_2[:, None]  # (N_f, 3)

    # d(e1 × e2)/dv0 = d((v1-v0)×(v2-v0))/dv0 = -(v2-v0)× + (v1-v0)× ...
    # Actually: d/dv0 [(v1-v0)×(v2-v0)] = -(something)
    # Let's use: d(cross)/dv0_k = -e_k × e2 - e1 × (-e_k) where e_k is unit along k
    # Simpler: dA/dv0 = 0.5 * cross_hat · d(cross)/dv0
    # d(cross)/dv0 = (v2-v1) ×   (but with sign from cross product derivative)
    # For area of triangle:
    #   dA/dv0 = 0.5 * (n × (v2 - v1))  where n = cross_hat
    #   dA/dv1 = 0.5 * (n × (v0 - v2))
    #   dA/dv2 = 0.5 * (n × (v1 - v0))
    dA_dv0 = 0.5 * np.cross(cross_hat, v2 - v1)
    dA_dv1 = 0.5 * np.cross(cross_hat, v0 - v2)
    dA_dv2 = 0.5 * np.cross(cross_hat, v1 - v0)

    # Force = -dU/dv = -prefactor * dA/dv
    np.add.at(forces, faces[:, 0], -prefactor * dA_dv0)
    np.add.at(forces, faces[:, 1], -prefactor * dA_dv1)
    np.add.at(forces, faces[:, 2], -prefactor * dA_dv2)

    return energy, forces


def volume_constraint_forces(vertices, faces, V0, cfg):
    """
    U_vol = (kappa_vol/2) * ((V - V0)/V0)^2
    Prevents volume loss during cell spreading.
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    # Signed volume via divergence theorem
    V = abs(float(np.sum(np.sum(v0 * np.cross(v1, v2), axis=1)) / 6.0))
    V = max(V, EPS)

    strain = (V - V0) / V0
    energy = cfg.kappa_vol / 2.0 * strain ** 2

    # dV/dv_k for each triangle vertex
    dV_dv0 = np.cross(v1, v2) / 6.0
    dV_dv1 = np.cross(v2, v0) / 6.0
    dV_dv2 = np.cross(v0, v1) / 6.0

    # F = -dU/dv = -kappa_vol * strain / V0 * dV/dv
    prefactor = -cfg.kappa_vol * strain / V0

    N = len(vertices)
    forces = np.zeros((N, 3), dtype=float)
    np.add.at(forces, faces[:, 0], prefactor * dV_dv0)
    np.add.at(forces, faces[:, 1], prefactor * dV_dv1)
    np.add.at(forces, faces[:, 2], prefactor * dV_dv2)

    return forces, energy


# ═══════════════════════════════════════════════════════════════════
#  2. Cytoplasm Forces (Eq.2-4)
# ═══════════════════════════════════════════════════════════════════

def cytoplasm_forces(positions, edges, eq_lengths, config: SimConfig):
    """
    Harmonic springs with viscoelastic equilibrium length evolution.
    Returns (forces, energy, d_eq_dt)
    """
    vi = positions[edges[:, 0]]
    vj = positions[edges[:, 1]]
    rij_vec = vj - vi
    rij = np.linalg.norm(rij_vec, axis=1)
    rij = np.maximum(rij, EPS)
    rij_hat = rij_vec / rij[:, None]

    delta_r = rij - eq_lengths  # extension beyond equilibrium
    energy_per = 0.5 * config.kappa_cyt * delta_r**2
    energy = float(np.sum(energy_per))

    # f_ij (scalar, force magnitude along bond direction)
    f_scalar = -config.kappa_cyt * delta_r  # negative = attractive when stretched

    N = len(positions)
    forces = np.zeros((N, 3), dtype=float)
    f_vec = f_scalar[:, None] * rij_hat
    np.add.at(forces, edges[:, 0], -f_vec)  # force on i
    np.add.at(forces, edges[:, 1],  f_vec)  # force on j

    # Viscoelastic: dr_eq/dt = f_ij / mu_0  (Eq.3)
    # f_ij here is the magnitude of force on the bond
    f_bond_mag = config.kappa_cyt * delta_r  # positive when stretched
    d_eq_dt = f_bond_mag / config.mu_0

    return forces, energy, d_eq_dt


# ═══════════════════════════════════════════════════════════════════
#  3. Chromatin Forces (Eq.5)
# ═══════════════════════════════════════════════════════════════════

def chromatin_forces(chains, config: SimConfig):
    """
    Bond springs + repulsive LJ between all bead pairs.
    chains: list of N_c arrays, each (beads_per_chain, 3)
    Returns (forces_list: same structure, energy: float)
    """
    all_beads = np.vstack(chains)  # (N_beads, 3)
    N = len(all_beads)
    forces = np.zeros_like(all_beads)
    energy = 0.0

    # ── Bond springs ──
    offset = 0
    for chain in chains:
        n = len(chain)
        if n < 2:
            offset += n
            continue
        dr = chain[1:] - chain[:-1]  # (n-1, 3)
        dist = np.linalg.norm(dr, axis=1)
        dist = np.maximum(dist, EPS)
        dr_hat = dr / dist[:, None]

        delta = dist - config.r0_ch
        e_bond = 0.5 * config.kappa_bonding_ch * delta**2
        energy += float(np.sum(e_bond))

        f_scalar = -config.kappa_bonding_ch * delta
        f_vec = f_scalar[:, None] * dr_hat

        # force on bead j from bond (j-1, j): +f_vec
        # force on bead j-1: -f_vec
        idx = np.arange(offset, offset + n - 1)
        np.add.at(forces, idx, -f_vec)
        np.add.at(forces, idx + 1, f_vec)
        offset += n

    # ── Repulsive LJ (excluded volume) ──
    # Only repulsive part: cut at r_cut = 2^(1/6) * sigma_ch
    r_cut = config.r_cut_ch
    sigma = config.sigma_ch
    eps_ch = config.epsilon_ch

    tree = cKDTree(all_beads)
    pairs = tree.query_pairs(r_cut, output_type='ndarray')

    if len(pairs) > 0:
        # Filter out bonded pairs (consecutive beads in same chain)
        # Build set of bonded pairs for quick lookup
        bonded = set()
        offset = 0
        for chain in chains:
            n = len(chain)
            for k in range(n - 1):
                bonded.add((offset + k, offset + k + 1))
            offset += n

        mask = np.array([
            (int(p[0]), int(p[1])) not in bonded and
            (int(p[1]), int(p[0])) not in bonded
            for p in pairs
        ])
        pairs = pairs[mask]

    if len(pairs) > 0:
        dr = all_beads[pairs[:, 1]] - all_beads[pairs[:, 0]]
        dist = np.linalg.norm(dr, axis=1)
        dist = np.maximum(dist, 0.3 * sigma)  # hard floor
        dr_hat = dr / dist[:, None]

        # U_LJ = 4ε [(σ/r)^12 - (σ/r)^6] + ε  (shifted)
        sr6 = (sigma / dist)**6
        sr12 = sr6**2
        U_lj = 4.0 * eps_ch * (sr12 - sr6) + eps_ch
        energy += float(np.sum(U_lj))

        # F = -dU/dr = 4ε [12σ^12/r^13 - 6σ^6/r^7] = 24ε/r [2(σ/r)^12 - (σ/r)^6]
        f_mag = 24.0 * eps_ch / dist * (2.0 * sr12 - sr6)
        f_vec = f_mag[:, None] * dr_hat  # repulsive: pushes apart

        np.add.at(forces, pairs[:, 0], -f_vec)
        np.add.at(forces, pairs[:, 1],  f_vec)

    # Split back into per-chain forces
    forces_list = []
    offset = 0
    for chain in chains:
        n = len(chain)
        forces_list.append(forces[offset:offset + n].copy())
        offset += n

    return forces_list, energy


# ═══════════════════════════════════════════════════════════════════
#  4. Cell-ECM Interaction (Eq.6)  — Phase 2 only
# ═══════════════════════════════════════════════════════════════════

def cell_ecm_forces(cell_vertices, cell_faces,
                    sub_centroids, sub_normals, sub_tree_2d, config: SimConfig):
    """
    Harmonic tethering of cell membrane to substrate surface.

    Each cell face within adhesion range is pulled toward a target point
    (sigma_perp above the local substrate surface along surface normal).
    Force direction is 3D (perpendicular to local surface), so on groove
    slopes the force has a lateral component guiding cells into grooves.

    Returns (forces_on_cell_vertices: (N_cv, 3), energy: float)
    """
    N_cv = len(cell_vertices)
    forces = np.zeros((N_cv, 3), dtype=float)
    energy = 0.0

    d_target = config.sigma_perp   # target height above substrate surface
    k_adhere = config.epsilon_ECM  # spring constant for adhesion

    # Compute cell face centroids
    c0 = cell_vertices[cell_faces[:, 0]]
    c1 = cell_vertices[cell_faces[:, 1]]
    c2 = cell_vertices[cell_faces[:, 2]]
    centroids = (c0 + c1 + c2) / 3.0  # (N_cf, 3)

    # Find nearest substrate face for each cell face
    _, nearest_idx = sub_tree_2d.query(centroids[:, :2], k=1)

    # Target point: substrate surface + d_target along surface normal
    target_points = sub_centroids[nearest_idx] + d_target * sub_normals[nearest_idx]

    # 3D displacement from cell centroid to target point
    disp_vec = target_points - centroids  # (N_cf, 3)
    disp_mag = np.linalg.norm(disp_vec, axis=1)

    # Only act on faces within adhesion range
    adhesion_cutoff = config.R_cell * 0.5  # half cell radius
    mask = (disp_mag > 1e-8) & (disp_mag < adhesion_cutoff)

    if np.any(mask):
        d = disp_mag[mask]

        # U = (k/2) * |displacement|^2
        U = 0.5 * k_adhere * d**2
        energy = float(np.sum(U))

        # F = k * displacement_vector (toward target point, 3D)
        f_vec = k_adhere * disp_vec[mask]

        # Distribute force from face centroid to its 3 vertices
        masked_faces = cell_faces[mask]
        f_per_vert = f_vec / 3.0
        for k in range(3):
            np.add.at(forces, masked_faces[:, k], f_per_vert)

    return forces, energy


# ═══════════════════════════════════════════════════════════════════
#  5. Active Force  — Phase 2 only
# ═══════════════════════════════════════════════════════════════════

def active_forces(cell_vertices, cell_faces, sub_centroids, sub_normals,
                  sub_tree_2d, config: SimConfig):
    """
    Push membrane nodes near substrate outward along curvature vector,
    projected onto the local substrate tangent plane so that spreading
    follows groove slopes rather than pushing cells out of grooves.
    Returns (forces: (N_cv, 3), )
    """
    N = len(cell_vertices)
    forces = np.zeros((N, 3), dtype=float)

    # Find substrate height below each vertex
    _, nearest_all = sub_tree_2d.query(cell_vertices[:, :2], k=1)
    sub_z = sub_centroids[nearest_all, 2]

    # Height above substrate — act on lower 75% of cell (most of the cell)
    height = cell_vertices[:, 2] - sub_z
    cell_z_min = cell_vertices[:, 2].min()
    cell_z_max = cell_vertices[:, 2].max()
    cell_height = max(cell_z_max - cell_z_min, EPS)
    # Act on all vertices below 75% of total cell height
    threshold = cell_z_min + 0.75 * cell_height

    # Select vertices in active zone
    near_mask = (height > 0) & (cell_vertices[:, 2] < threshold)
    near_idx = np.where(near_mask)[0]

    if len(near_idx) == 0:
        return forces

    # Compute vertex normals (average of adjacent face normals)
    v0 = cell_vertices[cell_faces[:, 0]]
    v1 = cell_vertices[cell_faces[:, 1]]
    v2 = cell_vertices[cell_faces[:, 2]]
    face_normals = np.cross(v1 - v0, v2 - v0)
    fn_norms = np.linalg.norm(face_normals, axis=1, keepdims=True)
    face_normals /= np.maximum(fn_norms, EPS)

    vert_normals = np.zeros((N, 3), dtype=float)
    vert_counts = np.zeros(N, dtype=float)
    for k in range(3):
        np.add.at(vert_normals, cell_faces[:, k], face_normals)
        np.add.at(vert_counts, cell_faces[:, k], 1.0)
    vert_counts = np.maximum(vert_counts, 1.0)
    vert_normals /= vert_counts[:, None]
    vn_norms = np.linalg.norm(vert_normals, axis=1, keepdims=True)
    vert_normals /= np.maximum(vn_norms, EPS)

    # Active force: outward along vertex normal, projected onto local
    # substrate tangent plane for vertices near substrate
    f_active_vec = config.F_active * vert_normals[near_idx].copy()
    h_near = height[near_idx]

    # Smooth transition: fully tangent-projected below 2*sigma_perp,
    # fully 3D above 6*sigma_perp
    z_blend = np.clip((h_near - 2.0 * config.sigma_perp) / (4.0 * config.sigma_perp), 0.0, 1.0)

    # Get local substrate surface normals for active vertices
    local_normals = sub_normals[nearest_all[near_idx]]

    # Project force onto substrate tangent plane: f_tangent = f - (f . n) * n
    dot_fn = np.sum(f_active_vec * local_normals, axis=1, keepdims=True)
    f_tangent = f_active_vec - dot_fn * local_normals

    # Blend: full 3D (far from substrate) ↔ tangent-projected (near substrate)
    f_active_vec = z_blend[:, None] * f_active_vec + (1.0 - z_blend[:, None]) * f_tangent

    # Renormalize to maintain force magnitude F_active
    f_mag = np.linalg.norm(f_active_vec, axis=1, keepdims=True)
    f_mag = np.maximum(f_mag, EPS)
    f_active_vec *= config.F_active / f_mag
    forces[near_idx] += f_active_vec

    return forces


# ═══════════════════════════════════════════════════════════════════
#  6. Steric Repulsion (penetration prevention)
# ═══════════════════════════════════════════════════════════════════

def steric_sphere_confinement(positions, center, radius, inside, config: SimConfig):
    """
    Confine points inside (or outside) a sphere using soft repulsion.

    inside=True: keep points inside sphere (e.g., chromatin in nucleus)
    inside=False: keep points outside sphere (e.g., cytoplasm outside nucleus)

    Returns forces: (N, 3)
    """
    N = len(positions)
    forces = np.zeros((N, 3), dtype=float)
    center = np.asarray(center)
    dr = positions - center
    dist = np.linalg.norm(dr, axis=1)
    dist = np.maximum(dist, EPS)
    dr_hat = dr / dist[:, None]

    if inside:
        # Push inward when dist > radius - d_steric
        penetration = dist - (radius - config.d_steric)
        mask = penetration > 0
        if np.any(mask):
            pen = penetration[mask]
            f_mag = -config.k_steric * pen
            forces[mask] += f_mag[:, None] * dr_hat[mask]
    else:
        # Push outward when dist < radius + d_steric
        penetration = (radius + config.d_steric) - dist
        mask = penetration > 0
        if np.any(mask):
            pen = penetration[mask]
            f_mag = config.k_steric * pen
            forces[mask] += f_mag[:, None] * dr_hat[mask]

    return forces


def steric_substrate_repulsion(cell_vertices, sub_centroids, sub_normals,
                               sub_tree_2d, config: SimConfig):
    """
    Prevent cell membrane from penetrating below substrate surface.
    Push along local surface normal (perpendicular to surface), so on
    groove slopes the repulsion has a lateral component into the groove.
    Returns forces on cell vertices.
    """
    N = len(cell_vertices)
    forces = np.zeros((N, 3), dtype=float)

    _, nearest_idx = sub_tree_2d.query(cell_vertices[:, :2], k=1)

    # Signed distance to substrate plane along surface normal
    disp_to_surface = cell_vertices - sub_centroids[nearest_idx]
    signed_dist = np.sum(disp_to_surface * sub_normals[nearest_idx], axis=1)

    # Penetration: vertex is within d_steric of (or below) the surface
    penetration = config.d_steric - signed_dist
    mask = penetration > 0
    if np.any(mask):
        f_mag = config.k_steric * penetration[mask]
        forces[mask] += f_mag[:, None] * sub_normals[nearest_idx[mask]]

    return forces


def steric_nucleus_in_cell(nuc_vertices, cell_vertices, config):
    """Keep nucleus vertices inside cell membrane using directional radius."""
    N = len(nuc_vertices)
    forces = np.zeros((N, 3), dtype=float)
    cell_center = cell_vertices.mean(axis=0)
    cell_rel = cell_vertices - cell_center  # (N_cv, 3)

    nuc_rel = nuc_vertices - cell_center
    nuc_dist = np.linalg.norm(nuc_rel, axis=1, keepdims=True)
    nuc_hat = nuc_rel / np.maximum(nuc_dist, EPS)

    # For each nuc vertex, find cell membrane radius in that direction
    dots = nuc_hat @ cell_rel.T  # (N_nuc, N_cv)
    # Maximum projection = cell radius in that direction
    max_dots = np.max(dots, axis=1, keepdims=True)  # (N_nuc, 1)

    # Push inward when nuc vertex exceeds 80% of cell radius in its direction
    limit = max_dots * 0.80
    penetration = nuc_dist - limit
    mask_flat = (penetration > 0).ravel()
    if np.any(mask_flat):
        pen = penetration[mask_flat]
        f_mag = -config.k_steric * pen  # (M, 1)
        forces[mask_flat] += f_mag * nuc_hat[mask_flat]

    return forces


def gravity_force(n_vertices, config: SimConfig, strength=0.5):
    """Gentle downward force to help cell settle onto substrate in Phase 2."""
    forces = np.zeros((n_vertices, 3), dtype=float)
    forces[:, 2] = -strength * config.kBT
    return forces
