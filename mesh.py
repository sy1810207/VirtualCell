"""
VirtualCell v0.85 — Mesh Generation
Icosphere membrane, tetrahedral cytoplasm, chromatin bead-spring chains.
"""
import numpy as np
from scipy.spatial import Delaunay, cKDTree
from config import SimConfig


# ═══════════════════════════════════════════════════════════════════
#  Icosphere Mesh
# ═══════════════════════════════════════════════════════════════════

class IcosphereMesh:
    """Triangulated sphere by recursive subdivision of an icosahedron."""

    def __init__(self, radius: float, subdivisions: int, center=None):
        self.radius = radius
        self.center = np.zeros(3) if center is None else np.asarray(center, dtype=float)
        self.vertices, self.faces = self._build(subdivisions)
        self.vertices = self.vertices * radius + self.center
        self.edges, self.edge_to_faces = self._build_topology()
        self.S0 = self._compute_surface_area()

    # ── Icosahedron base ──
    @staticmethod
    def _icosahedron():
        phi = (1.0 + np.sqrt(5.0)) / 2.0  # golden ratio
        verts = np.array([
            [-1,  phi, 0], [ 1,  phi, 0], [-1, -phi, 0], [ 1, -phi, 0],
            [ 0, -1,  phi], [ 0,  1,  phi], [ 0, -1, -phi], [ 0,  1, -phi],
            [ phi, 0, -1], [ phi, 0,  1], [-phi, 0, -1], [-phi, 0,  1],
        ], dtype=float)
        norms = np.linalg.norm(verts, axis=1, keepdims=True)
        verts /= norms
        faces = np.array([
            [0,11,5],[0,5,1],[0,1,7],[0,7,10],[0,10,11],
            [1,5,9],[5,11,4],[11,10,2],[10,7,6],[7,1,8],
            [3,9,4],[3,4,2],[3,2,6],[3,6,8],[3,8,9],
            [4,9,5],[2,4,11],[6,2,10],[8,6,7],[9,8,1],
        ], dtype=int)
        return verts, faces

    def _build(self, subdivisions):
        vertices, faces = self._icosahedron()
        vlist = list(vertices)
        midpoint_cache = {}

        def get_midpoint(i, j):
            key = (min(i, j), max(i, j))
            if key in midpoint_cache:
                return midpoint_cache[key]
            mid = (vlist[i] + vlist[j]) / 2.0
            mid /= np.linalg.norm(mid)
            idx = len(vlist)
            vlist.append(mid)
            midpoint_cache[key] = idx
            return idx

        for _ in range(subdivisions):
            new_faces = []
            midpoint_cache = {}
            for tri in faces:
                a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
                ab = get_midpoint(a, b)
                bc = get_midpoint(b, c)
                ca = get_midpoint(c, a)
                new_faces.extend([
                    [a, ab, ca], [b, bc, ab], [c, ca, bc], [ab, bc, ca]
                ])
            faces = np.array(new_faces, dtype=int)

        return np.array(vlist, dtype=float), faces

    def _build_topology(self):
        edge_set = set()
        edge_to_faces = {}
        for fi, face in enumerate(self.faces):
            for k in range(3):
                i, j = int(face[k]), int(face[(k + 1) % 3])
                edge = (min(i, j), max(i, j))
                edge_set.add(edge)
                edge_to_faces.setdefault(edge, []).append(fi)
        edges = np.array(sorted(edge_set), dtype=int)
        return edges, edge_to_faces

    def _compute_surface_area(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        areas = 0.5 * np.linalg.norm(cross, axis=1)
        return float(np.sum(areas))

    def face_normals(self):
        v0 = self.vertices[self.faces[:, 0]]
        v1 = self.vertices[self.faces[:, 1]]
        v2 = self.vertices[self.faces[:, 2]]
        cross = np.cross(v1 - v0, v2 - v0)
        norms = np.linalg.norm(cross, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        return cross / norms


# ═══════════════════════════════════════════════════════════════════
#  Adjacent-face pair array (for curvature computation)
# ═══════════════════════════════════════════════════════════════════

def build_adjacent_face_pairs(edge_to_faces):
    """Return (N_pairs, 2) array of face index pairs sharing an edge,
    plus the shared edge vertex indices (N_pairs, 2)."""
    pairs = []
    shared_edges = []
    for edge, flist in edge_to_faces.items():
        if len(flist) == 2:
            pairs.append(flist)
            shared_edges.append(edge)
    return np.array(pairs, dtype=int), np.array(shared_edges, dtype=int)


# ═══════════════════════════════════════════════════════════════════
#  Cytoplasm Tetrahedral Mesh
# ═══════════════════════════════════════════════════════════════════

def generate_cytoplasm_mesh(cell_mesh: IcosphereMesh,
                            nuc_mesh: IcosphereMesh,
                            config: SimConfig):
    """
    Fill the shell between nucleus and cell membrane with a tetrahedral mesh.
    Returns:
        cyt_positions: (N_cyt, 3)
        cyt_edges: (N_edges, 2)
        cyt_eq_lengths: (N_edges,)
        cell_anchor_indices: dict mapping cytoplasm index → cell membrane vertex index
        nuc_anchor_indices:  dict mapping cytoplasm index → nucleus membrane vertex index
    """
    R_cell = config.R_cell
    R_nuc = config.R_nucleus
    center = nuc_mesh.center

    # 1) Anchor points: every 4th membrane vertex
    cell_anchor_verts = cell_mesh.vertices[::4].copy()
    nuc_anchor_verts = nuc_mesh.vertices[::4].copy()
    n_cell_anchors = len(cell_anchor_verts)
    n_nuc_anchors = len(nuc_anchor_verts)

    # 2) Interior points via rejection sampling
    margin = 0.3  # stay away from boundaries
    interior_pts = []
    rng = np.random.default_rng(42)
    attempts = 0
    target = config.n_cyt_points
    while len(interior_pts) < target and attempts < target * 100:
        pt = rng.uniform(-R_cell, R_cell, size=3) + center
        r = np.linalg.norm(pt - center)
        if (R_nuc + margin) < r < (R_cell - margin):
            interior_pts.append(pt)
        attempts += 1
    interior_pts = np.array(interior_pts, dtype=float)

    # 3) Combine all points
    all_pts = np.vstack([cell_anchor_verts, nuc_anchor_verts, interior_pts])
    n_total = len(all_pts)

    # Build anchor index mappings (cytoplasm idx → membrane vertex idx)
    cell_anchor_indices = {}
    for i in range(n_cell_anchors):
        cell_anchor_indices[i] = i * 4  # every 4th vertex in cell mesh
    nuc_anchor_indices = {}
    for i in range(n_nuc_anchors):
        nuc_anchor_indices[n_cell_anchors + i] = i * 4

    # 4) Delaunay tetrahedralization
    tri = Delaunay(all_pts)

    # 5) Filter tetrahedra whose centroid is inside shell
    centroids = all_pts[tri.simplices].mean(axis=1)
    r_centroids = np.linalg.norm(centroids - center, axis=1)
    mask = (r_centroids > R_nuc + 0.1) & (r_centroids < R_cell - 0.1)
    good_tets = tri.simplices[mask]

    # 6) Extract unique edges
    edge_set = set()
    for tet in good_tets:
        for i in range(4):
            for j in range(i + 1, 4):
                a, b = int(tet[i]), int(tet[j])
                edge_set.add((min(a, b), max(a, b)))
    cyt_edges = np.array(sorted(edge_set), dtype=int)

    # 7) Compute initial equilibrium lengths
    diffs = all_pts[cyt_edges[:, 1]] - all_pts[cyt_edges[:, 0]]
    cyt_eq_lengths = np.linalg.norm(diffs, axis=1)

    return all_pts, cyt_edges, cyt_eq_lengths, cell_anchor_indices, nuc_anchor_indices


# ═══════════════════════════════════════════════════════════════════
#  Chromatin Bead-Spring Chains
# ═══════════════════════════════════════════════════════════════════

def generate_chromatin_chains(config: SimConfig, center=None):
    """
    Generate N_c random-walk chains confined in nucleus.
    Returns list of N_c arrays, each (beads_per_chain, 3).
    """
    if center is None:
        center = np.zeros(3)
    center = np.asarray(center, dtype=float)
    R = config.R_nucleus - config.sigma_ch  # keep beads away from wall
    r0 = config.r0_ch  # equilibrium bond length = 2*sigma_ch
    bpc = config.beads_per_chain
    rng = np.random.default_rng(123)

    chains = []
    for _ in range(config.N_c):
        beads = np.zeros((bpc, 3))
        # first bead: random inside nucleus
        beads[0] = _random_point_in_sphere(R, rng) + center
        for k in range(1, bpc):
            for _attempt in range(1000):
                direction = rng.standard_normal(3)
                direction /= np.linalg.norm(direction)
                candidate = beads[k - 1] + direction * r0
                if np.linalg.norm(candidate - center) < R:
                    beads[k] = candidate
                    break
            else:
                # fallback: place at slightly smaller radius
                beads[k] = beads[k - 1] * 0.95
        chains.append(beads)
    return chains


def _random_point_in_sphere(R, rng):
    while True:
        p = rng.uniform(-R, R, size=3)
        if np.linalg.norm(p) < R:
            return p


# ═══════════════════════════════════════════════════════════════════
#  LINC bonds (nucleus apex ↔ cell membrane)
# ═══════════════════════════════════════════════════════════════════

def generate_linc_bonds(cell_vertices, nuc_vertices, config: SimConfig):
    """
    Build LINC anchor pairs between the upper hemisphere of the nuclear
    envelope (perinuclear actin cap region) and their nearest cell-membrane
    vertices within a biological search radius.

    Parameters
    ----------
    cell_vertices : (N_c, 3)
    nuc_vertices  : (N_n, 3)
    config        : SimConfig (uses config.a and config.n_linc_bonds)

    Returns
    -------
    linc_cell_idx  : (N_linc,) int  — cell vertex indices
    linc_nuc_idx   : (N_linc,) int  — nucleus vertex indices
    linc_eq_lengths: (N_linc,) float — initial pair distances (equilibrium)
    """
    nuc_cm_z = float(nuc_vertices[:, 2].mean())
    upper_mask = nuc_vertices[:, 2] > nuc_cm_z
    upper_idx = np.where(upper_mask)[0]

    if len(upper_idx) == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0, dtype=float))

    search_radius = 3.0 * config.a
    cell_tree = cKDTree(cell_vertices)

    pairs = []
    for nj in upper_idx:
        d, ci = cell_tree.query(nuc_vertices[nj], k=1,
                                 distance_upper_bound=search_radius)
        if np.isfinite(d) and ci < len(cell_vertices):
            pairs.append((d, int(ci), int(nj)))

    pairs.sort(key=lambda t: t[0])
    pairs = pairs[:config.n_linc_bonds]

    if len(pairs) == 0:
        return (np.zeros(0, dtype=int), np.zeros(0, dtype=int),
                np.zeros(0, dtype=float))

    linc_eq_lengths = np.array([p[0] for p in pairs], dtype=float)
    linc_cell_idx = np.array([p[1] for p in pairs], dtype=int)
    linc_nuc_idx = np.array([p[2] for p in pairs], dtype=int)
    return linc_cell_idx, linc_nuc_idx, linc_eq_lengths
