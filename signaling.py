"""
VirtualCell v0.87 — YAP/TAZ Signaling Module
Steady-state algebraic cascade coupling cell geometry to YAP nuclear/cytoplasmic ratio.
Based on Sun-Spill-Zaman (2016) + Francis et al. (2025) curvature correction.
"""
import numpy as np
from config import SimConfig
from utils import compute_surface_area


# ═══════════════════════════════════════════════════════════════════
#  Function 1: Cotangent Mean Curvature
# ═══════════════════════════════════════════════════════════════════

def cotangent_mean_curvature(vertices, faces):
    """
    Compute per-vertex mean curvature scalar H using the cotangent Laplacian
    (Meyer-Desbrun-Schröder-Barr 2003).

    Parameters
    ----------
    vertices : (N_v, 3) float64
    faces    : (N_f, 3) int

    Returns
    -------
    H : (N_v,) float64, clamped to [0, 20]
    """
    N_v = len(vertices)

    # Step 1: Build edge-to-faces mapping and per-face cotangent weights
    # For each face, compute cotangent of each interior angle
    v0 = vertices[faces[:, 0]]  # (N_f, 3)
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]

    # Edge vectors for each face
    e01 = v1 - v0  # opposite to vertex 2
    e02 = v2 - v0  # opposite to vertex 1
    e12 = v2 - v1  # opposite to vertex 0

    # Cotangent of angle at vertex 0 (between edges e01, e02)
    cos0 = np.sum(e01 * e02, axis=1)
    sin0 = np.linalg.norm(np.cross(e01, e02), axis=1)
    cot0 = cos0 / np.maximum(sin0, 1e-6)

    # Cotangent of angle at vertex 1 (between edges -e01, e12)
    cos1 = np.sum(-e01 * e12, axis=1)
    sin1 = np.linalg.norm(np.cross(-e01, e12), axis=1)
    cot1 = cos1 / np.maximum(sin1, 1e-6)

    # Cotangent of angle at vertex 2 (between edges -e02, -e12)
    cos2 = np.sum(-e02 * -e12, axis=1)
    sin2 = np.linalg.norm(np.cross(-e02, -e12), axis=1)
    cot2 = cos2 / np.maximum(sin2, 1e-6)

    # Step 3: Mixed Voronoi area (approximate with 1/3 of triangle area)
    triangle_areas = 0.5 * np.linalg.norm(np.cross(e01, e02), axis=1)  # (N_f,)
    A_mixed = np.zeros(N_v, dtype=np.float64)
    for k in range(3):
        np.add.at(A_mixed, faces[:, k], triangle_areas / 3.0)
    A_mixed = np.maximum(A_mixed, 1e-10)

    # Step 4: Curvature normal vector K(i) = (1/2A_i) * sum_j (cot_alpha + cot_beta) * (x_j - x_i)
    # For edge (i0, i1) in face with vertex i2: cot at i2 is the weight
    # For edge (i1, i2) in face with vertex i0: cot at i0 is the weight
    # For edge (i0, i2) in face with vertex i1: cot at i1 is the weight
    K = np.zeros((N_v, 3), dtype=np.float64)

    # Edge (faces[:,0], faces[:,1]): opposite angle is at vertex 2 → weight = cot2
    diff_01 = v1 - v0  # x_j - x_i for edge (0→1)
    w_01 = cot2[:, None] * diff_01
    np.add.at(K, faces[:, 0], w_01)
    np.add.at(K, faces[:, 1], -w_01)

    # Edge (faces[:,1], faces[:,2]): opposite angle is at vertex 0 → weight = cot0
    diff_12 = v2 - v1
    w_12 = cot0[:, None] * diff_12
    np.add.at(K, faces[:, 1], w_12)
    np.add.at(K, faces[:, 2], -w_12)

    # Edge (faces[:,0], faces[:,2]): opposite angle is at vertex 1 → weight = cot1
    diff_02 = v2 - v0
    w_02 = cot1[:, None] * diff_02
    np.add.at(K, faces[:, 0], w_02)
    np.add.at(K, faces[:, 2], -w_02)

    # Normalize by 1/(2*A_mixed)
    K /= (2.0 * A_mixed[:, None])

    # Step 5-6: Scalar curvature H = 0.5 * ||K||, clamped to [0, 20]
    H = 0.5 * np.linalg.norm(K, axis=1)
    H = np.clip(np.abs(H), 0.0, 20.0)

    return H


# ═══════════════════════════════════════════════════════════════════
#  Function 2: Identify Contact Nodes
# ═══════════════════════════════════════════════════════════════════

def identify_contact_nodes(sim, contact_threshold=2.5):
    """
    Identify cell membrane vertices in contact with the substrate.

    Uses precomputed sim.sub_tree_2d and sim.sub_centroids (v0.87).
    Does NOT construct its own cKDTree.

    Parameters
    ----------
    sim : VirtualCellSimulation instance
    contact_threshold : float
        Multiplier of sigma_perp for z-distance contact criterion.

    Returns
    -------
    mask : (N_v,) bool array
    """
    N_v = len(sim.cell_vertices)

    if sim.sub_centroids is None:
        return np.zeros(N_v, dtype=bool)

    # Find nearest substrate face centroid in XY plane for each cell vertex
    _, nearest_idx = sim.sub_tree_2d.query(sim.cell_vertices[:, :2], k=1)

    # Contact condition: z_cell - z_substrate < threshold * sigma_perp
    z_diff = sim.cell_vertices[:, 2] - sim.sub_centroids[nearest_idx, 2]
    mask = z_diff < contact_threshold * sim.cfg.sigma_perp

    return mask


# ═══════════════════════════════════════════════════════════════════
#  Function 3: Extract Geometric Features
# ═══════════════════════════════════════════════════════════════════

def extract_geometric_features(sim):
    """
    Extract all geometric quantities needed by the signaling module.

    All quantities in simulation α units (no unit conversion).

    Parameters
    ----------
    sim : VirtualCellSimulation instance

    Returns
    -------
    features : dict
    """
    # Cell membrane area
    A_PM = compute_surface_area(sim.cell_vertices, sim.cell_faces)

    # Cell volume (divergence theorem)
    v0 = sim.cell_vertices[sim.cell_faces[:, 0]]
    v1 = sim.cell_vertices[sim.cell_faces[:, 1]]
    v2 = sim.cell_vertices[sim.cell_faces[:, 2]]
    V_cyto = abs(float(np.sum(v0 * np.cross(v1, v2)) / 6.0))

    # Nuclear envelope area
    A_NE = compute_surface_area(sim.nuc_vertices, sim.nuc_faces)

    # Nuclear volume
    nv0 = sim.nuc_vertices[sim.nuc_faces[:, 0]]
    nv1 = sim.nuc_vertices[sim.nuc_faces[:, 1]]
    nv2 = sim.nuc_vertices[sim.nuc_faces[:, 2]]
    V_nuc = abs(float(np.sum(nv0 * np.cross(nv1, nv2)) / 6.0))

    # Contact nodes
    contact_mask = identify_contact_nodes(sim)
    n_contact = int(contact_mask.sum())

    # Contact area: triangles with >= 2 contact vertices
    A_contact = 0.0
    if n_contact > 0:
        face_contact_count = (contact_mask[sim.cell_faces[:, 0]].astype(int) +
                              contact_mask[sim.cell_faces[:, 1]].astype(int) +
                              contact_mask[sim.cell_faces[:, 2]].astype(int))
        contact_faces_mask = face_contact_count >= 2
        if np.any(contact_faces_mask):
            cf = sim.cell_faces[contact_faces_mask]
            cv0 = sim.cell_vertices[cf[:, 0]]
            cv1 = sim.cell_vertices[cf[:, 1]]
            cv2 = sim.cell_vertices[cf[:, 2]]
            A_contact = float(np.sum(0.5 * np.linalg.norm(
                np.cross(cv1 - cv0, cv2 - cv0), axis=1)))

    # Curvature
    H_all = cotangent_mean_curvature(sim.cell_vertices, sim.cell_faces)

    # Contact region curvature stats
    if n_contact > 0:
        H_contact = H_all[contact_mask]
        H_mean_contact = float(np.mean(H_contact))
        H_std_contact = float(np.std(H_contact))
    else:
        H_mean_contact = 0.0
        H_std_contact = 0.0

    # High-curvature edge nodes (H > 0.5)
    high_curv_mask = H_all > 0.5
    if np.any(high_curv_mask):
        H_edge_mean = float(np.mean(H_all[high_curv_mask]))
    else:
        H_edge_mean = 0.0

    # Nucleus PCA: aspect ratio, flattening, and principal-axis orientation
    nuc_centered = sim.nuc_vertices - sim.nuc_vertices.mean(axis=0)
    cov = np.cov(nuc_centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)  # ascending
    eigvals = np.maximum(eigvals, 1e-12)
    nuc_aspect_ratio = float(np.sqrt(eigvals[2] / eigvals[0]))
    nuc_flattening = float(1.0 - np.sqrt(eigvals[0] / eigvals[1]))

    # Principal-axis alignment with groove (Y) axis
    principal_axis = eigvecs[:, 2]
    cos_theta = abs(float(principal_axis[1])) / (np.linalg.norm(principal_axis) + 1e-12)
    cos_theta = min(max(cos_theta, 0.0), 1.0)
    theta_align = float(np.degrees(np.arccos(cos_theta)))

    # Directional nuclear AR on world axes: std(y)/std(x)
    sx = float(np.std(nuc_centered[:, 0]))
    sy = float(np.std(nuc_centered[:, 1]))
    nuc_AR_YX = sy / max(sx, 1e-9)

    # Nuclear envelope stretch ratio (kept for logging; no longer drives k_in_stretch)
    nuc_A_stretch = A_NE / sim.nuc_S0

    # Cell and nucleus center of mass z
    cell_cm = sim.cell_vertices.mean(axis=0)
    nuc_cm = sim.nuc_vertices.mean(axis=0)
    cell_cm_z = float(cell_cm[2])
    nuc_cm_z = float(nuc_cm[2])

    # Local groove depth under cell center
    if sim.sub_centroids is not None:
        _, idx = sim.sub_tree_2d.query([[cell_cm[0], cell_cm[1]]], k=1)
        groove_depth_local = float(sim.sub_centroids[idx[0], 2])
    else:
        groove_depth_local = 0.0

    return {
        'A_PM': A_PM,
        'V_cyto': V_cyto,
        'A_NE': A_NE,
        'V_nuc': V_nuc,
        'A_contact': A_contact,
        'n_contact': n_contact,
        'H_mean_contact': H_mean_contact,
        'H_std_contact': H_std_contact,
        'H_edge_mean': H_edge_mean,
        'nuc_aspect_ratio': nuc_aspect_ratio,
        'nuc_flattening': nuc_flattening,
        'nuc_A_stretch': nuc_A_stretch,
        'nuc_AR_YX': nuc_AR_YX,
        'theta_align': theta_align,
        'cell_cm_z': cell_cm_z,
        'nuc_cm_z': nuc_cm_z,
        'groove_depth_local': groove_depth_local,
    }


# ═══════════════════════════════════════════════════════════════════
#  Class: YAP Signaling Module
# ═══════════════════════════════════════════════════════════════════

class YAPSignalingModule:
    """
    Steady-state algebraic cascade for YAP/TAZ nuclear-cytoplasmic ratio.
    Sun-Spill-Zaman (2016) + Francis et al. (2025) curvature correction.
    """

    def __init__(self, cfg: SimConfig, E_substrate_kPa: float = 40.0):
        self.E_substrate_kPa = E_substrate_kPa
        self.params = {
            'H0': 5.0 * cfg.a,
            'k_f': 0.01,
            'k_sf': 0.12,
            'k_dFAK': 0.1,
            'FAK_tot': 1.0,
            'C': 5.0,
            'E_ref': 40.0,
            'k_rho': 1.0,
            'k_dRho': 0.5,
            'RhoA_tot': 1.0,
            'k_myo': 1.0,
            'k_dMyo': 0.3,
            'Myo_tot': 1.0,
            'k_act': 1.5,
            'k_dAct': 0.4,
            'G_actin': 1.0,
            'n': 2.6,
            'K_SF': 0.3,
            'alpha0': 2.5,
            'k_in_base': 0.8,
            'k_out': 0.1,
            'k_ly': 0.05,
            'LATS_base': 1.0,
        }

    def compute(self, features: dict) -> dict:
        """
        Run the 6-step algebraic cascade.

        Parameters
        ----------
        features : dict from extract_geometric_features()

        Returns
        -------
        result : dict with YAP_NC and all intermediates
        """
        p = self.params
        H_mean_contact = features.get('H_mean_contact', 0.0)
        nuc_A_stretch = features.get('nuc_A_stretch', 1.0)
        nuc_AR_YX = features.get('nuc_AR_YX', 1.0)
        A_NE = features.get('A_NE', 200.0)

        # Step 1: Curvature-corrected integrin density
        rho_I = np.exp(-H_mean_contact / p['H0'])

        # Step 2: FAK phosphorylation
        E_norm = self.E_substrate_kPa / p['E_ref']
        k_FAK = rho_I * (p['k_f'] + p['k_sf'] * E_norm / (p['C'] + E_norm))
        FAK_p = k_FAK * p['FAK_tot'] / (k_FAK + p['k_dFAK'])

        # Step 3: RhoA-GTP
        RhoA_GTP = (p['k_rho'] * FAK_p * p['RhoA_tot'] /
                    (p['k_rho'] * FAK_p + p['k_dRho']))

        # Step 4: F-actin and activated myosin
        Myo_A = (p['k_myo'] * RhoA_GTP * p['Myo_tot'] /
                 (p['k_myo'] * RhoA_GTP + p['k_dMyo']))
        F_actin = (p['k_act'] * RhoA_GTP * p['G_actin'] /
                   (p['k_act'] * RhoA_GTP + p['k_dAct']))

        # Step 5: Stress fibers → YAP free fraction
        SF = F_actin * Myo_A
        phi_free = SF ** p['n'] / (p['K_SF'] ** p['n'] + SF ** p['n'] + 1e-12)

        # Step 6: NPC stretch enhancement driven by directional Y-axis stretch
        # (Francis et al.: NPC preferentially opens along stretch direction, not area)
        alpha = nuc_AR_YX
        k_in_stretch = np.exp((alpha - 1.0) / p['alpha0'])
        k_in_eff = p['k_in_base'] * k_in_stretch
        YAP_NC = (phi_free * k_in_eff * A_NE /
                  (p['k_out'] * A_NE + p['k_ly'] * p['LATS_base'] + 1e-12))

        return {
            'YAP_NC': float(YAP_NC),
            'FAK_p': float(FAK_p),
            'RhoA_GTP': float(RhoA_GTP),
            'F_actin': float(F_actin),
            'Myo_A': float(Myo_A),
            'phi_free': float(phi_free),
            'k_in_stretch': float(k_in_stretch),
            'A_PM': features.get('A_PM', 0.0),
            'A_contact': features.get('A_contact', 0.0),
            'n_contact': features.get('n_contact', 0),
            'H_mean_contact': float(H_mean_contact),
            'H_edge_mean': features.get('H_edge_mean', 0.0),
            'nuc_aspect_ratio': features.get('nuc_aspect_ratio', 1.0),
            'nuc_A_stretch': float(nuc_A_stretch),
            'nuc_AR_YX': float(nuc_AR_YX),
            'theta_align': features.get('theta_align', 90.0),
            'groove_depth_local': features.get('groove_depth_local', 0.0),
        }

    def summary_str(self, result: dict) -> str:
        """One-line summary for logging."""
        return (f"YAP N/C={result['YAP_NC']:.3f} | "
                f"FAK={result['FAK_p']:.3f} | "
                f"F-actin={result['F_actin']:.3f} | "
                f"k_in_str={result['k_in_stretch']:.4f} | "
                f"H_ct={result['H_mean_contact']:.3f}/α | "
                f"AR_YX={result['nuc_AR_YX']:.2f} | "
                f"θ={result['theta_align']:.0f}°")
