"""
VirtualCell v0.85 — Utility Functions
Logging, geometry helpers, contact probability, checkpoint I/O.
"""
import logging
import os
import numpy as np
from datetime import datetime
from config import SimConfig


# ═══════════════════════════════════════════════════════════════════
#  Logging
# ═══════════════════════════════════════════════════════════════════

def setup_logger(log_dir='.', name='virtualcell'):
    """Create a logger that writes to a timestamped file and console."""
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    logs_dir = os.path.join(log_dir, 'logs')
    os.makedirs(logs_dir, exist_ok=True)
    filename = os.path.join(logs_dir, f'virtualcell_{timestamp}.log')

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    # Remove existing handlers
    logger.handlers.clear()

    fh = logging.FileHandler(filename, encoding='utf-8')
    fh.setLevel(logging.DEBUG)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    fmt = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s',
                            datefmt='%H:%M:%S')
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    logger.info(f'Log file: {filename}')
    return logger


# ═══════════════════════════════════════════════════════════════════
#  Geometry Helpers
# ═══════════════════════════════════════════════════════════════════

def compute_aspect_ratio(vertices):
    """
    Compute aspect ratio of a point cloud using PCA.
    Returns ratio of largest to smallest principal axis length.
    """
    centered = vertices - vertices.mean(axis=0)
    cov = np.cov(centered.T)
    eigenvalues = np.linalg.eigvalsh(cov)
    eigenvalues = np.maximum(eigenvalues, 1e-12)
    return float(np.sqrt(eigenvalues.max() / eigenvalues.min()))


def compute_surface_area(vertices, faces):
    """Sum of triangle areas."""
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    return float(np.sum(areas))


def compute_volume(vertices, faces):
    """
    Compute volume of closed triangulated surface using divergence theorem.
    V = (1/6) * |Σ (v0 · (v1 × v2))|
    """
    v0 = vertices[faces[:, 0]]
    v1 = vertices[faces[:, 1]]
    v2 = vertices[faces[:, 2]]
    vol = np.sum(v0 * np.cross(v1, v2)) / 6.0
    return abs(float(vol))


def compute_mean_radius(vertices, center=None):
    """Mean distance of vertices from center."""
    if center is None:
        center = vertices.mean(axis=0)
    dists = np.linalg.norm(vertices - center, axis=1)
    return float(np.mean(dists))


def bond_length_stats(vertices, edges):
    """Compute min, max, mean bond lengths."""
    diffs = vertices[edges[:, 1]] - vertices[edges[:, 0]]
    lengths = np.linalg.norm(diffs, axis=1)
    return float(lengths.min()), float(lengths.max()), float(lengths.mean())


# ═══════════════════════════════════════════════════════════════════
#  Contact Probability Matrix (SI Eq. S.2, S.3)
# ═══════════════════════════════════════════════════════════════════

def contact_probability_matrix(chains, config: SimConfig):
    """
    P_ij = (1/N_c) * Σ_p Θ(d0 - d_ij^p)
    For a single snapshot, each chain is one conformation.
    """
    all_beads = np.vstack(chains)
    N = len(all_beads)
    d0 = 1.5 * config.sigma_ch  # contact threshold

    # Pairwise distances
    from scipy.spatial.distance import pdist, squareform
    dist_matrix = squareform(pdist(all_beads))

    # Contact: d < d0
    contact = (dist_matrix < d0).astype(float)
    np.fill_diagonal(contact, 0.0)  # no self-contacts

    return contact


def dissimilarity_distance(P_A, P_B):
    """D^{A,B} = sqrt(Σ (P_ij^B - P_ij^A)²)  (SI Eq. S.3)"""
    return float(np.sqrt(np.sum((P_B - P_A)**2)))


# ═══════════════════════════════════════════════════════════════════
#  Checkpoint I/O
# ═══════════════════════════════════════════════════════════════════

def save_checkpoint(filepath, cell_vertices, cell_faces, cell_edges,
                    nuc_vertices, nuc_faces, nuc_edges,
                    cyt_positions, cyt_edges, cyt_eq_lengths,
                    chains, step, energy,
                    cell_S0, nuc_S0,
                    cell_anchor_indices, nuc_anchor_indices):
    """Save simulation state to .npz file."""
    # Flatten chains for storage
    chain_lengths = [len(c) for c in chains]
    chains_flat = np.vstack(chains)

    np.savez(filepath,
             cell_vertices=cell_vertices,
             cell_faces=cell_faces,
             cell_edges=cell_edges,
             nuc_vertices=nuc_vertices,
             nuc_faces=nuc_faces,
             nuc_edges=nuc_edges,
             cyt_positions=cyt_positions,
             cyt_edges=cyt_edges,
             cyt_eq_lengths=cyt_eq_lengths,
             chains_flat=chains_flat,
             chain_lengths=np.array(chain_lengths),
             step=step,
             energy=energy,
             cell_S0=cell_S0,
             nuc_S0=nuc_S0,
             cell_anchor_keys=np.array(list(cell_anchor_indices.keys())),
             cell_anchor_vals=np.array(list(cell_anchor_indices.values())),
             nuc_anchor_keys=np.array(list(nuc_anchor_indices.keys())),
             nuc_anchor_vals=np.array(list(nuc_anchor_indices.values())))


def load_checkpoint(filepath):
    """Load simulation state from .npz file. Returns dict."""
    data = np.load(filepath, allow_pickle=False)

    # Reconstruct chains
    chains_flat = data['chains_flat']
    chain_lengths = data['chain_lengths']
    chains = []
    offset = 0
    for n in chain_lengths:
        chains.append(chains_flat[offset:offset + n].copy())
        offset += n

    # Reconstruct anchor dicts
    cell_anchor_indices = dict(zip(
        data['cell_anchor_keys'].astype(int),
        data['cell_anchor_vals'].astype(int)))
    nuc_anchor_indices = dict(zip(
        data['nuc_anchor_keys'].astype(int),
        data['nuc_anchor_vals'].astype(int)))

    return {
        'cell_vertices': data['cell_vertices'].copy(),
        'cell_faces': data['cell_faces'].copy(),
        'cell_edges': data['cell_edges'].copy(),
        'nuc_vertices': data['nuc_vertices'].copy(),
        'nuc_faces': data['nuc_faces'].copy(),
        'nuc_edges': data['nuc_edges'].copy(),
        'cyt_positions': data['cyt_positions'].copy(),
        'cyt_edges': data['cyt_edges'].copy(),
        'cyt_eq_lengths': data['cyt_eq_lengths'].copy(),
        'chains': chains,
        'step': int(data['step']),
        'energy': float(data['energy']),
        'cell_S0': float(data['cell_S0']),
        'nuc_S0': float(data['nuc_S0']),
        'cell_anchor_indices': cell_anchor_indices,
        'nuc_anchor_indices': nuc_anchor_indices,
    }
