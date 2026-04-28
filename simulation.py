"""
VirtualCell v0.85 — Simulation Orchestrator
Phase 1: suspended cell equilibration
Phase 2: cell on grooved substrate
"""
import os
import numpy as np
from config import SimConfig
from mesh import (IcosphereMesh, build_adjacent_face_pairs,
                  generate_cytoplasm_mesh, generate_chromatin_chains)
from forces import (membrane_forces, cytoplasm_forces, chromatin_forces,
                    cell_ecm_forces, active_forces,
                    steric_sphere_confinement, steric_substrate_repulsion,
                    steric_nucleus_in_cell, gravity_force,
                    volume_constraint_forces, linc_forces)
from scipy.spatial import cKDTree
from dynamics import OverdampedLangevin
from substrate import generate_grooved_substrate
from visualization import generate_html_visualization, RealtimeVisualizer
from utils import (setup_logger, compute_aspect_ratio, bond_length_stats,
                   compute_mean_radius, compute_surface_area, save_checkpoint,
                   load_checkpoint, contact_probability_matrix)

# Alias for clarity in init code
compute_surface_area_from_verts = compute_surface_area


class VirtualCellSimulation:

    def __init__(self, config: SimConfig, output_dir='.', realtime_vis=True):
        self.cfg = config
        self.output_dir = output_dir
        self.logger = setup_logger(output_dir)
        self.integrator = OverdampedLangevin(config)
        self.should_stop = False
        self.realtime_vis = realtime_vis
        # Substrate data (initialized in run_phase2)
        self.sub_centroids = None
        self.sub_normals = None
        self.sub_tree_2d = None
        # LINC bonds (initialized in run_phase2 if kappa_linc > 0)
        self.linc_cell_idx = None
        self.linc_nuc_idx = None
        self.linc_eq_lengths = None

        self._init_meshes()
        self.logger.info('VirtualCell simulation initialized')
        self.logger.info(f'  Cell membrane: {len(self.cell_mesh.vertices)} verts, '
                         f'{len(self.cell_mesh.faces)} faces')
        self.logger.info(f'  Nuclear envelope: {len(self.nuc_mesh.vertices)} verts, '
                         f'{len(self.nuc_mesh.faces)} faces')
        self.logger.info(f'  Cytoplasm: {len(self.cyt_positions)} nodes, '
                         f'{len(self.cyt_edges)} edges')
        self.logger.info(f'  Chromatin: {self.cfg.N_c} chains × '
                         f'{self.cfg.beads_per_chain} beads')

    # ─────────────────────────────────────────────────
    #  Initialization
    # ─────────────────────────────────────────────────

    def _init_meshes(self):
        cfg = self.cfg

        # Cell membrane
        self.cell_mesh = IcosphereMesh(cfg.R_cell, cfg.ico_subdiv_cell)
        self.cell_vertices = self.cell_mesh.vertices.copy()
        self.cell_faces = self.cell_mesh.faces.copy()
        self.cell_edges = self.cell_mesh.edges.copy()

        # Compute actual 'a' = mean edge length and scale all parameters
        _, _, a_mean = bond_length_stats(self.cell_vertices, self.cell_edges)
        cfg.scale_by_a(a_mean)
        self.logger.info(f'  Unit length a = {a_mean:.4f}')
        self.logger.info(f'  Scaled: l_min={cfg.l_min:.4f}, l_max={cfg.l_max:.4f}, '
                         f'sigma_ch={cfg.sigma_ch:.4f}')

        # Pre-relax cell membrane: bond forces only, constrained to sphere
        self.cell_vertices = self._pre_relax_membrane(
            self.cell_vertices, self.cell_edges,
            cfg.R_cell, self.cell_mesh.center, n_steps=2000)
        mn, mx, avg = bond_length_stats(self.cell_vertices, self.cell_edges)
        self.logger.info(f'  Cell post-relax: bonds [{mn:.4f}, {mx:.4f}] mean={avg:.4f}')

        self.cell_face_pairs, self.cell_shared_edges = \
            build_adjacent_face_pairs(self.cell_mesh.edge_to_faces)
        self.cell_S0 = compute_surface_area_from_verts(
            self.cell_vertices, self.cell_faces)
        self.cell_V0 = self._compute_volume(self.cell_vertices, self.cell_faces)

        # Nuclear envelope
        self.nuc_mesh = IcosphereMesh(cfg.R_nucleus, cfg.ico_subdiv_nucleus)
        self.nuc_vertices = self.nuc_mesh.vertices.copy()
        self.nuc_faces = self.nuc_mesh.faces.copy()
        self.nuc_edges = self.nuc_mesh.edges.copy()

        # Pre-relax nucleus membrane
        self.nuc_vertices = self._pre_relax_membrane(
            self.nuc_vertices, self.nuc_edges,
            cfg.R_nucleus, self.nuc_mesh.center, n_steps=2000)
        mn, mx, avg = bond_length_stats(self.nuc_vertices, self.nuc_edges)
        self.logger.info(f'  Nuc post-relax: bonds [{mn:.4f}, {mx:.4f}] mean={avg:.4f}')

        self.nuc_face_pairs, self.nuc_shared_edges = \
            build_adjacent_face_pairs(self.nuc_mesh.edge_to_faces)
        self.nuc_S0 = compute_surface_area_from_verts(
            self.nuc_vertices, self.nuc_faces)
        self.nuc_V0 = self._compute_volume(self.nuc_vertices, self.nuc_faces)

        # Cytoplasm (use post-relax meshes)
        self.cell_mesh.vertices = self.cell_vertices.copy()
        self.nuc_mesh.vertices = self.nuc_vertices.copy()
        (self.cyt_positions, self.cyt_edges, self.cyt_eq_lengths,
         self.cell_anchor_indices, self.nuc_anchor_indices) = \
            generate_cytoplasm_mesh(self.cell_mesh, self.nuc_mesh, cfg)

        # Chromatin
        self.chains = generate_chromatin_chains(cfg)

        # Substrate (set later in Phase 2)
        self.sub_vertices = None
        self.sub_faces = None

    @staticmethod
    def _compute_volume(vertices, faces):
        """Compute signed volume of a triangulated mesh."""
        v0 = vertices[faces[:, 0]]
        v1 = vertices[faces[:, 1]]
        v2 = vertices[faces[:, 2]]
        return abs(float(np.sum(np.sum(v0 * np.cross(v1, v2), axis=1)) / 6.0))

    def _pre_relax_membrane(self, vertices, edges, radius, center,
                            n_steps=2000):
        """
        Relax membrane bonds on fixed sphere surface.
        Uses only bonding + repulsion forces (Eq.1 terms 1-2),
        no curvature or surface area. Projects back to sphere each step.
        """
        from forces import _fene_wca_potential
        cfg = self.cfg
        center = np.asarray(center)

        for step in range(n_steps):
            vi = vertices[edges[:, 0]]
            vj = vertices[edges[:, 1]]
            dr = vj - vi
            rij = np.linalg.norm(dr, axis=1)
            rij = np.maximum(rij, 1e-10)
            dr_hat = dr / rij[:, None]

            _, f_scalar = _fene_wca_potential(rij, cfg)

            forces = np.zeros_like(vertices)
            f_vec = f_scalar[:, None] * dr_hat
            np.add.at(forces, edges[:, 0], -f_vec)
            np.add.at(forces, edges[:, 1], f_vec)

            # Adaptive step: cap displacement
            f_mag = np.linalg.norm(forces, axis=1, keepdims=True)
            max_disp = 0.02 * cfg.a  # small controlled steps
            scale = np.minimum(1.0, max_disp / np.maximum(f_mag, 1e-12))
            displacement = forces * scale * 0.01  # conservative factor

            vertices = vertices + displacement

            # Project back to sphere of fixed radius
            dr_from_center = vertices - center
            dist = np.linalg.norm(dr_from_center, axis=1, keepdims=True)
            vertices = center + dr_from_center * (radius / np.maximum(dist, 1e-12))

        return vertices

    # ─────────────────────────────────────────────────
    #  Force computation
    # ─────────────────────────────────────────────────

    def _compute_forces_phase1(self):
        """All internal forces for suspended cell."""
        cfg = self.cfg
        total_energy = 0.0

        # Cell membrane
        f_cell, e_cell = membrane_forces(
            self.cell_vertices, self.cell_edges, self.cell_faces,
            self.cell_face_pairs, self.cell_shared_edges,
            self.cell_S0, cfg)
        total_energy += e_cell

        # Cell volume constraint
        f_cell_vol, e_cell_vol = volume_constraint_forces(
            self.cell_vertices, self.cell_faces, self.cell_V0, cfg)
        f_cell += f_cell_vol
        total_energy += e_cell_vol

        # Nuclear envelope
        f_nuc, e_nuc = membrane_forces(
            self.nuc_vertices, self.nuc_edges, self.nuc_faces,
            self.nuc_face_pairs, self.nuc_shared_edges,
            self.nuc_S0, cfg)
        total_energy += e_nuc

        # Nucleus volume constraint
        f_nuc_vol, e_nuc_vol = volume_constraint_forces(
            self.nuc_vertices, self.nuc_faces, self.nuc_V0, cfg)
        f_nuc += f_nuc_vol
        total_energy += e_nuc_vol

        # Cytoplasm
        f_cyt, e_cyt, self._d_eq_dt = cytoplasm_forces(
            self.cyt_positions, self.cyt_edges, self.cyt_eq_lengths, cfg)
        total_energy += e_cyt

        # Chromatin
        f_chains, e_ch = chromatin_forces(self.chains, cfg)
        total_energy += e_ch

        # Steric: chromatin confined inside nucleus
        nuc_center = self.nuc_vertices.mean(axis=0)
        nuc_radius = compute_mean_radius(self.nuc_vertices, nuc_center)
        for i, chain in enumerate(self.chains):
            f_steric = steric_sphere_confinement(
                chain, nuc_center, nuc_radius, inside=True, config=cfg)
            f_chains[i] += f_steric

        # Steric: cytoplasm outside nucleus
        f_cyt += steric_sphere_confinement(
            self.cyt_positions, nuc_center, nuc_radius, inside=False, config=cfg)

        # Steric: cytoplasm inside cell (spherical approximation for forces;
        # hard directional projection applied in _update_positions)
        cell_center = self.cell_vertices.mean(axis=0)
        cell_radius = compute_mean_radius(self.cell_vertices, cell_center)
        f_cyt += steric_sphere_confinement(
            self.cyt_positions, cell_center, cell_radius, inside=True, config=cfg)

        # Steric: nucleus inside cell (spherical for forces;
        # hard directional projection in _update_positions)
        f_nuc += steric_sphere_confinement(
            self.nuc_vertices, cell_center, cell_radius, inside=True, config=cfg)

        # Transfer forces from anchored cytoplasm nodes to membrane nodes
        f_cell, f_nuc = self._transfer_anchor_forces(
            f_cell, f_nuc, f_cyt)

        # Zero net force on each component (internal forces should not translate)
        f_cell -= f_cell.mean(axis=0)
        f_nuc -= f_nuc.mean(axis=0)
        f_cyt -= f_cyt.mean(axis=0)
        for i in range(len(f_chains)):
            f_chains[i] -= f_chains[i].mean(axis=0)

        return f_cell, f_nuc, f_cyt, f_chains, total_energy

    def _compute_forces_phase2(self):
        """Phase 1 forces + ECM interaction + active force + substrate steric + gravity."""
        f_cell, f_nuc, f_cyt, f_chains, total_energy = self._compute_forces_phase1()
        cfg = self.cfg

        # Gravity: downward force for spreading and groove entry
        grav = 1.5 * cfg.kBT
        f_cell[:, 2] -= grav
        f_nuc[:, 2] -= grav
        f_cyt[:, 2] -= grav
        for i in range(len(f_chains)):
            f_chains[i][:, 2] -= grav

        # Cell-ECM adhesion (3D force toward target on surface)
        f_ecm, e_ecm = cell_ecm_forces(
            self.cell_vertices, self.cell_faces,
            self.sub_centroids, self.sub_normals, self.sub_tree_2d, cfg)
        f_cell += f_ecm
        total_energy += e_ecm

        # Active force (spreading along substrate tangent plane)
        f_act = active_forces(
            self.cell_vertices, self.cell_faces,
            self.sub_centroids, self.sub_normals, self.sub_tree_2d, cfg)
        f_cell += f_act

        # Steric: cell membrane above substrate (along surface normal)
        f_sub_steric = steric_substrate_repulsion(
            self.cell_vertices, self.sub_centroids, self.sub_normals,
            self.sub_tree_2d, cfg)
        f_cell += f_sub_steric

        # Steric: nucleus above substrate (along surface normal)
        f_nuc_sub = steric_substrate_repulsion(
            self.nuc_vertices, self.sub_centroids, self.sub_normals,
            self.sub_tree_2d, cfg)
        f_nuc += f_nuc_sub

        # LINC coupling: nucleus apex ↔ cell membrane (F-actin-modulated)
        if cfg.kappa_linc > 0.0 and self.linc_cell_idx is not None \
                and len(self.linc_cell_idx) > 0:
            myo_act = self._linc_myo_activation()
            f_linc_cell, f_linc_nuc = linc_forces(
                self.cell_vertices, self.nuc_vertices,
                self.linc_cell_idx, self.linc_nuc_idx, self.linc_eq_lengths,
                cfg, myo_act)
            f_cell += f_linc_cell
            f_nuc += f_linc_nuc

        return f_cell, f_nuc, f_cyt, f_chains, total_energy

    def _linc_myo_activation(self):
        """Current myosin activation for LINC stiffness modulation.
        Priority: latest signaling Myo_A; fallback: contact-node fraction."""
        if hasattr(self, '_signaling_history') and len(self._signaling_history) > 0:
            return float(self._signaling_history[-1].get('Myo_A', 0.5))
        if self.sub_centroids is not None:
            _, near = self.sub_tree_2d.query(self.cell_vertices[:, :2], k=1)
            z_diff = self.cell_vertices[:, 2] - self.sub_centroids[near, 2]
            return float(np.mean(z_diff < 3.0 * self.cfg.a))
        return 0.0

    @staticmethod
    def _directional_confinement(positions, cell_vertices, cfg, fraction=0.90):
        """Push points inward when they exceed `fraction` of cell radius in their direction."""
        N = len(positions)
        forces = np.zeros((N, 3), dtype=float)
        cell_center = cell_vertices.mean(axis=0)
        cell_rel = cell_vertices - cell_center

        pos_rel = positions - cell_center
        pos_dist = np.linalg.norm(pos_rel, axis=1, keepdims=True)
        pos_hat = pos_rel / np.maximum(pos_dist, 1e-10)

        # Cell radius in each point's direction
        dots = pos_hat @ cell_rel.T  # (N, N_cv)
        max_dots = np.max(dots, axis=1, keepdims=True)

        limit = max_dots * fraction
        penetration = pos_dist - limit
        mask_flat = (penetration > 0).ravel()
        if np.any(mask_flat):
            pen = penetration[mask_flat]
            f_mag = -cfg.k_steric * pen  # (M, 1)
            forces[mask_flat] += f_mag * pos_hat[mask_flat]

        return forces

    def _transfer_anchor_forces(self, f_cell, f_nuc, f_cyt):
        """Transfer forces between anchored cytoplasm nodes and membrane nodes."""
        # Cell anchors: cytoplasm node i is anchored to cell membrane vertex j
        for cyt_idx, mem_idx in self.cell_anchor_indices.items():
            if cyt_idx < len(f_cyt) and mem_idx < len(f_cell):
                # Bidirectional coupling: share force
                shared = 0.5 * (f_cyt[cyt_idx] + f_cell[mem_idx])
                f_cyt[cyt_idx] = shared
                f_cell[mem_idx] = shared

        # Nucleus anchors
        for cyt_idx, mem_idx in self.nuc_anchor_indices.items():
            if cyt_idx < len(f_cyt) and mem_idx < len(f_nuc):
                shared = 0.5 * (f_cyt[cyt_idx] + f_nuc[mem_idx])
                f_cyt[cyt_idx] = shared
                f_nuc[mem_idx] = shared

        return f_cell, f_nuc

    # ─────────────────────────────────────────────────
    #  Position updates
    # ─────────────────────────────────────────────────

    def _update_positions(self, f_cell, f_nuc, f_cyt, f_chains):
        """Integrate positions one timestep."""
        integ = self.integrator

        self.cell_vertices = integ.step(self.cell_vertices, f_cell)
        self.nuc_vertices = integ.step(self.nuc_vertices, f_nuc)
        self.cyt_positions = integ.step(self.cyt_positions, f_cyt)

        for i, chain in enumerate(self.chains):
            self.chains[i] = integ.step(chain, f_chains[i])

        # Update cytoplasm equilibrium lengths (viscoelastic)
        if hasattr(self, '_d_eq_dt'):
            self.cyt_eq_lengths = integ.update_eq_lengths(
                self.cyt_eq_lengths, self._d_eq_dt)
            # Clamp eq lengths to reasonable range
            self.cyt_eq_lengths = np.clip(self.cyt_eq_lengths, 0.3, 5.0)

        # Synchronize anchored cytoplasm nodes with membrane nodes
        for cyt_idx, mem_idx in self.cell_anchor_indices.items():
            if cyt_idx < len(self.cyt_positions) and mem_idx < len(self.cell_vertices):
                avg = 0.5 * (self.cyt_positions[cyt_idx] + self.cell_vertices[mem_idx])
                self.cyt_positions[cyt_idx] = avg
                self.cell_vertices[mem_idx] = avg

        for cyt_idx, mem_idx in self.nuc_anchor_indices.items():
            if cyt_idx < len(self.cyt_positions) and mem_idx < len(self.nuc_vertices):
                avg = 0.5 * (self.cyt_positions[cyt_idx] + self.nuc_vertices[mem_idx])
                self.cyt_positions[cyt_idx] = avg
                self.nuc_vertices[mem_idx] = avg

        # Project chromatin beads back inside nucleus (hard constraint)
        nuc_center = self.nuc_vertices.mean(axis=0)
        nuc_radius = compute_mean_radius(self.nuc_vertices, nuc_center)
        for i, chain in enumerate(self.chains):
            dr = chain - nuc_center
            dist = np.linalg.norm(dr, axis=1, keepdims=True)
            outside = dist > nuc_radius * 0.95
            if np.any(outside):
                scale = np.where(outside,
                                 nuc_radius * 0.9 / np.maximum(dist, 1e-10),
                                 1.0)
                self.chains[i] = nuc_center + dr * scale

        # ── Directional confinement: keep nucleus and cytoplasm inside cell ──
        # Use actual cell shape (not spherical approximation)
        cell_center = self.cell_vertices.mean(axis=0)
        cell_rel = self.cell_vertices - cell_center

        # Nucleus: confine within 75% of cell radius in each direction
        nuc_rel = self.nuc_vertices - cell_center
        nuc_dist = np.linalg.norm(nuc_rel, axis=1, keepdims=True)
        nuc_hat = nuc_rel / np.maximum(nuc_dist, 1e-10)
        dots_nuc = nuc_hat @ cell_rel.T
        max_nuc = np.max(dots_nuc, axis=1, keepdims=True)
        limit_nuc = max_nuc * 0.75
        too_far_nuc = nuc_dist > limit_nuc
        if np.any(too_far_nuc):
            scale_nuc = np.where(too_far_nuc,
                                 limit_nuc * 0.95 / np.maximum(nuc_dist, 1e-10),
                                 1.0)
            self.nuc_vertices = cell_center + nuc_rel * scale_nuc

        # Cytoplasm: confine within 90% of cell radius in each direction
        cyt_rel = self.cyt_positions - cell_center
        cyt_dist = np.linalg.norm(cyt_rel, axis=1, keepdims=True)
        cyt_hat = cyt_rel / np.maximum(cyt_dist, 1e-10)
        dots_cyt = cyt_hat @ cell_rel.T
        max_cyt = np.max(dots_cyt, axis=1, keepdims=True)
        limit_cyt = max_cyt * 0.90
        too_far_cyt = cyt_dist > limit_cyt
        if np.any(too_far_cyt):
            scale_cyt = np.where(too_far_cyt,
                                 limit_cyt * 0.95 / np.maximum(cyt_dist, 1e-10),
                                 1.0)
            self.cyt_positions = cell_center + cyt_rel * scale_cyt

        # Hard floor: keep nucleus, cytoplasm, and chromatin above substrate
        if self.sub_centroids is not None:
            floor_margin = self.cfg.sigma_perp * 0.5

            # Nucleus vertices
            _, nidx = self.sub_tree_2d.query(self.nuc_vertices[:, :2], k=1)
            nuc_floor = self.sub_centroids[nidx, 2] + floor_margin
            below = self.nuc_vertices[:, 2] < nuc_floor
            if np.any(below):
                self.nuc_vertices[below, 2] = nuc_floor[below]

            # Cytoplasm nodes
            _, cidx = self.sub_tree_2d.query(self.cyt_positions[:, :2], k=1)
            cyt_floor = self.sub_centroids[cidx, 2] + floor_margin
            below_c = self.cyt_positions[:, 2] < cyt_floor
            if np.any(below_c):
                self.cyt_positions[below_c, 2] = cyt_floor[below_c]

            # Chromatin beads
            for i, chain in enumerate(self.chains):
                _, kidx = self.sub_tree_2d.query(chain[:, :2], k=1)
                ch_floor = self.sub_centroids[kidx, 2] + floor_margin
                below_ch = chain[:, 2] < ch_floor
                if np.any(below_ch):
                    self.chains[i][below_ch, 2] = ch_floor[below_ch]

    # ─────────────────────────────────────────────────
    #  Monitoring & Safety Checks
    # ─────────────────────────────────────────────────

    def _monitor(self, step, energy, phase='Phase1'):
        """Check for instabilities. Returns True if should stop."""
        cfg = self.cfg
        log = self.logger

        # Bond lengths
        cell_min, cell_max, cell_mean = bond_length_stats(
            self.cell_vertices, self.cell_edges)
        nuc_min, nuc_max, nuc_mean = bond_length_stats(
            self.nuc_vertices, self.nuc_edges)

        # Aspect ratios
        cell_ar = compute_aspect_ratio(self.cell_vertices)
        nuc_ar = compute_aspect_ratio(self.nuc_vertices)

        # Mean radii
        cell_center = self.cell_vertices.mean(axis=0)
        nuc_center = self.nuc_vertices.mean(axis=0)
        cell_r = compute_mean_radius(self.cell_vertices, cell_center)
        nuc_r = compute_mean_radius(self.nuc_vertices, nuc_center)

        if step % cfg.log_interval == 0:
            log.info(f'[{phase}] Step {step:>6d} | E={energy:>10.2f} | '
                     f'Cell bonds [{cell_min:.3f}, {cell_max:.3f}] AR={cell_ar:.3f} R={cell_r:.2f} | '
                     f'Nuc bonds [{nuc_min:.3f}, {nuc_max:.3f}] AR={nuc_ar:.3f} R={nuc_r:.2f} | '
                     f'Center=({cell_center[0]:.2f}, {cell_center[1]:.2f}, {cell_center[2]:.2f})')

        # ── Critical checks ──

        # NaN / Inf
        if not np.isfinite(energy):
            log.warning(f'[{phase}] STOP: Energy is {energy} at step {step}')
            return True

        # Energy explosion — use relative growth from initial energy
        if not hasattr(self, '_initial_energy'):
            self._initial_energy = abs(energy)
        threshold = max(self._initial_energy * 10.0, 1e15)
        if abs(energy) > threshold:
            log.warning(f'[{phase}] STOP: Energy explosion ({energy:.2e}) at step {step}')
            return True

        # Bond length critical bounds — FENE limits at r_eq ± R_max
        fene_max = cfg.r_eq_bond + cfg.R_max * 0.99  # 99% of FENE limit
        fene_min = cfg.r_eq_bond - cfg.R_max * 0.99
        fene_min = max(fene_min, 0.3 * cfg.a)  # absolute floor
        if cell_min < fene_min:
            log.warning(f'[{phase}] STOP: Cell bond too short ({cell_min:.4f}, '
                        f'limit={fene_min:.4f}) at step {step}')
            return True
        if cell_max > fene_max:
            log.warning(f'[{phase}] STOP: Cell bond too long ({cell_max:.4f}, '
                        f'limit={fene_max:.4f}) at step {step}')
            return True

        # Bond warning zone — 93% of FENE limit
        warn_max = cfg.r_eq_bond + cfg.R_max * 0.93
        warn_min = cfg.r_eq_bond - cfg.R_max * 0.93
        if cell_min < warn_min:
            log.warning(f'[{phase}] WARNING: Cell bond short ({cell_min:.4f})')
        if cell_max > warn_max:
            log.warning(f'[{phase}] WARNING: Cell bond long ({cell_max:.4f})')

        # Collapse detection (relaxed in Phase 2 — cell flattens when spreading)
        collapse_thresh = 0.3 if phase == 'Phase2' else 0.5
        if cell_r < cfg.R_cell * collapse_thresh:
            log.warning(f'[{phase}] STOP: Cell collapsing (R={cell_r:.2f}, '
                        f'initial={cfg.R_cell:.2f}) at step {step}')
            return True
        if nuc_r < cfg.R_nucleus * 0.3:
            log.warning(f'[{phase}] STOP: Nucleus collapsing (R={nuc_r:.2f}) at step {step}')
            return True

        # Phase 1: sphericity check
        if phase == 'Phase1' and cell_ar > 1.5:
            log.warning(f'[{phase}] WARNING: Cell losing sphericity (AR={cell_ar:.3f})')

        return False

    # ─────────────────────────────────────────────────
    #  Phase 1: Suspended Cell Equilibration
    # ─────────────────────────────────────────────────

    def run_phase1(self):
        """Relax suspended cell to internal equilibrium. No substrate."""
        cfg = self.cfg
        log = self.logger
        log.info('=' * 60)
        log.info('PHASE 1: Suspended cell equilibration')
        log.info(f'  Steps: {cfg.n_steps_phase1}, dt={cfg.dt}')
        log.info('=' * 60)

        vis = RealtimeVisualizer(enabled=self.realtime_vis)
        energy_prev = None

        for step in range(cfg.n_steps_phase1):
            if self.should_stop or vis.stopped:
                log.info(f'Phase 1 stopped by user at step {step}')
                break

            # Compute forces
            f_cell, f_nuc, f_cyt, f_chains, energy = self._compute_forces_phase1()

            # Integrate
            self._update_positions(f_cell, f_nuc, f_cyt, f_chains)

            # Monitor
            if step % cfg.check_interval == 0:
                should_stop = self._monitor(step, energy, 'Phase1')
                if should_stop:
                    self.should_stop = True
                    break

            # Real-time visualization
            if step % cfg.vis_interval == 0:
                vis.update(self.cell_vertices, self.cell_faces,
                           self.nuc_vertices, self.nuc_faces,
                           step, energy, phase='Phase 1')

            # Equilibrium check (energy convergence)
            if energy_prev is not None and step > 10000 and step % 5000 == 0:
                dE = abs(energy - energy_prev)
                if dE < 0.01 * abs(energy_prev) + 0.1:
                    log.info(f'Phase 1 reached equilibrium at step {step} '
                             f'(ΔE={dE:.4f})')

            if step % 5000 == 0:
                energy_prev = energy

        # Save checkpoint
        checkpoint_path = os.path.join(self.output_dir, 'phase1_checkpoint.npz')
        save_checkpoint(
            checkpoint_path,
            self.cell_vertices, self.cell_faces, self.cell_edges,
            self.nuc_vertices, self.nuc_faces, self.nuc_edges,
            self.cyt_positions, self.cyt_edges, self.cyt_eq_lengths,
            self.chains, step, energy,
            self.cell_S0, self.nuc_S0,
            self.cell_anchor_indices, self.nuc_anchor_indices)
        log.info(f'Phase 1 checkpoint saved: {checkpoint_path}')

        # Generate Phase 1 HTML
        html_path = generate_html_visualization(
            self.cell_vertices, self.cell_faces,
            self.nuc_vertices, self.nuc_faces,
            self.chains,
            cyt_positions=self.cyt_positions,
            output_dir=self.output_dir,
            version='v085_phase1')
        log.info(f'Phase 1 HTML: {html_path}')

        vis.close()
        return energy

    # ─────────────────────────────────────────────────
    #  Phase 2: Cell on Grooved Substrate
    # ─────────────────────────────────────────────────

    def run_phase2(self, groove_depth=None, cell_x=None, cell_y=None):
        """Drop equilibrated cell onto grooved substrate."""
        cfg = self.cfg
        log = self.logger

        if groove_depth is not None:
            # groove_depth is in units of a; scale to absolute if already scaled
            cfg.h = groove_depth * cfg.a if cfg._scaled else groove_depth

        log.info('=' * 60)
        log.info('PHASE 2: Cell on grooved substrate')
        log.info(f'  Groove: w={cfg.w}a, h={cfg.h}a')
        log.info(f'  Steps: {cfg.n_steps_phase2}')
        log.info('=' * 60)

        # Load checkpoint if available
        checkpoint_path = os.path.join(self.output_dir, 'phase1_checkpoint.npz')
        if os.path.exists(checkpoint_path):
            log.info('Loading Phase 1 checkpoint...')
            state = load_checkpoint(checkpoint_path)
            self.cell_vertices = state['cell_vertices']
            self.cell_faces = state['cell_faces']
            self.cell_edges = state['cell_edges']
            self.nuc_vertices = state['nuc_vertices']
            self.nuc_faces = state['nuc_faces']
            self.nuc_edges = state['nuc_edges']
            self.cyt_positions = state['cyt_positions']
            self.cyt_edges = state['cyt_edges']
            self.cyt_eq_lengths = state['cyt_eq_lengths']
            self.chains = state['chains']
            self.cell_S0 = state['cell_S0']
            self.nuc_S0 = state['nuc_S0']
            self.cell_anchor_indices = state['cell_anchor_indices']
            self.nuc_anchor_indices = state['nuc_anchor_indices']

            # Rebuild face pairs (not stored in checkpoint)
            cell_etf = self.cell_mesh.edge_to_faces
            self.cell_face_pairs, self.cell_shared_edges = \
                build_adjacent_face_pairs(cell_etf)
            nuc_etf = self.nuc_mesh.edge_to_faces
            self.nuc_face_pairs, self.nuc_shared_edges = \
                build_adjacent_face_pairs(nuc_etf)

        # Generate substrate
        self.sub_vertices, self.sub_faces = generate_grooved_substrate(cfg)
        log.info(f'  Substrate: {len(self.sub_vertices)} verts, '
                 f'{len(self.sub_faces)} faces')

        # Precompute substrate face centroids, normals, and 2D KD-tree
        # (substrate is rigid, so these never change)
        v0s = self.sub_vertices[self.sub_faces[:, 0]]
        v1s = self.sub_vertices[self.sub_faces[:, 1]]
        v2s = self.sub_vertices[self.sub_faces[:, 2]]
        self.sub_centroids = (v0s + v1s + v2s) / 3.0
        raw_normals = np.cross(v1s - v0s, v2s - v0s)
        norms = np.linalg.norm(raw_normals, axis=1, keepdims=True)
        self.sub_normals = raw_normals / np.maximum(norms, 1e-7)
        # Ensure all normals point upward (substrate surface faces up)
        flip = self.sub_normals[:, 2] < 0
        self.sub_normals[flip] *= -1.0
        self.sub_tree_2d = cKDTree(self.sub_centroids[:, :2])

        # Position cell above the groove floor
        target_x = cell_x if cell_x is not None else -5.0
        target_y = cell_y if cell_y is not None else 0.0
        sub_z = self.sub_vertices[:, 2]
        groove_floor_z = sub_z.min()
        log.info(f'  Groove floor z={groove_floor_z:.2f}, target x={target_x:.2f}, target y={target_y:.2f}')

        d_eq = 2.0 * cfg.sigma_perp  # above equilibrium → ECM pulls down
        cell_center = self.cell_vertices.mean(axis=0)
        cell_bottom_z = self.cell_vertices[:, 2].min()
        cell_R_actual = cell_center[2] - cell_bottom_z
        target_z = groove_floor_z + cell_R_actual + d_eq

        offset = np.array([target_x - cell_center[0],
                           target_y - cell_center[1],
                           target_z - cell_center[2]])
        log.info(f'  Cell center before offset: {cell_center}')
        log.info(f'  Offset applied: {offset}')
        self.cell_vertices += offset
        self.nuc_vertices += offset
        self.cyt_positions += offset
        for i in range(len(self.chains)):
            self.chains[i] += offset
        # Verify and force-correct: cell center must be at target
        actual_center = self.cell_vertices.mean(axis=0)
        log.info(f'  Cell center after offset: {actual_center}')
        correction = np.array([target_x - actual_center[0],
                               target_y - actual_center[1],
                               0.0])  # keep z as computed
        if abs(correction[0]) > 0.1 or abs(correction[1]) > 0.1:
            log.warning(f'  Position mismatch! Applying correction: {correction}')
            self.cell_vertices += correction
            self.nuc_vertices += correction
            self.cyt_positions += correction
            for i in range(len(self.chains)):
                self.chains[i] += correction
        final_center = self.cell_vertices.mean(axis=0)
        log.info(f'  Cell placed at x={final_center[0]:.2f}, '
                 f'y={final_center[1]:.2f}, z={final_center[2]:.2f}')

        if cfg.kappa_linc > 0.0:
            log.info(f'  LINC enabled: κ_linc={cfg.kappa_linc:.3g}, '
                     f'β_linc={cfg.beta_linc:.2f}, target {cfg.n_linc_bonds} pairs, '
                     f'activation at step {cfg.linc_activation_step}')

        # Reset integrator
        self.integrator.reset_dt()
        self.should_stop = False

        if cfg.enable_signaling:
            from signaling import YAPSignalingModule, extract_geometric_features
            signaling_module = YAPSignalingModule(cfg, E_substrate_kPa=cfg.E_substrate_kPa)
            self._signaling_history = []

        vis = RealtimeVisualizer(enabled=self.realtime_vis)
        energy = 0.0

        for step in range(cfg.n_steps_phase2):
            if self.should_stop or vis.stopped:
                log.info(f'Phase 2 stopped by user at step {step}')
                break

            # Deferred LINC activation: generate bonds once cell has spread
            if (cfg.kappa_linc > 0.0 and self.linc_cell_idx is None
                    and step >= cfg.linc_activation_step):
                from mesh import generate_linc_bonds
                self.linc_cell_idx, self.linc_nuc_idx, self.linc_eq_lengths = \
                    generate_linc_bonds(self.cell_vertices, self.nuc_vertices, cfg)
                n_pairs = len(self.linc_cell_idx)
                if n_pairs < cfg.n_linc_bonds:
                    log.warning(f'  LINC bonds: only {n_pairs}/{cfg.n_linc_bonds} pairs '
                                f'found at step {step} (search radius 3a={3*cfg.a:.2f})')
                else:
                    log.info(f'  LINC bonds: {n_pairs} pairs generated at step {step}')

            # Compute forces
            f_cell, f_nuc, f_cyt, f_chains, energy = self._compute_forces_phase2()

            # Integrate (displacement-capped)
            self._update_positions(f_cell, f_nuc, f_cyt, f_chains)

            # Monitor
            if step % cfg.check_interval == 0:
                should_stop = self._monitor(step, energy, 'Phase2')
                if should_stop:
                    self.should_stop = True
                    break

                # Additional Phase 2 checks: penetration
                self._check_penetration(step)

            # YAP/TAZ signaling evaluation
            if cfg.enable_signaling and step % cfg.signaling_interval == 0 and step > 0:
                features = extract_geometric_features(self)
                result = signaling_module.compute(features)
                log.info(f'[Signal] Step {step}: ' + signaling_module.summary_str(result))
                self._signaling_history.append({'step': step, **result})

            # Real-time visualization
            if step % cfg.vis_interval == 0:
                vis.update(self.cell_vertices, self.cell_faces,
                           self.nuc_vertices, self.nuc_faces,
                           step, energy,
                           sub_vertices=self.sub_vertices,
                           sub_faces=self.sub_faces,
                           phase='Phase 2')

        # Save signaling results
        if cfg.enable_signaling and hasattr(self, '_signaling_history'):
            import json
            from datetime import datetime
            sig_dir = os.path.join(self.output_dir, 'signaling')
            os.makedirs(sig_dir, exist_ok=True)
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            sig_path = os.path.join(sig_dir, f'signaling_results_{timestamp}.json')
            with open(sig_path, 'w') as f:
                json.dump(self._signaling_history, f, indent=2)
            log.info(f'Signaling results saved: {sig_path}')

        # Compute contact probability
        P = contact_probability_matrix(self.chains, cfg)
        n_contacts = int(np.sum(P > 0.5))
        log.info(f'Chromatin contacts (P>0.5): {n_contacts}')

        # Generate final HTML
        html_path = generate_html_visualization(
            self.cell_vertices, self.cell_faces,
            self.nuc_vertices, self.nuc_faces,
            self.chains,
            cyt_positions=self.cyt_positions,
            sub_vertices=self.sub_vertices,
            sub_faces=self.sub_faces,
            output_dir=self.output_dir,
            version='v085_phase2')
        log.info(f'Phase 2 HTML: {html_path}')

        vis.close()
        return energy

    def _check_penetration(self, step):
        """Check for membrane-substrate and nucleus-cell penetration."""
        log = self.logger

        if self.sub_centroids is not None:
            # Cell below substrate
            _, nearest = self.sub_tree_2d.query(self.cell_vertices[:, :2], k=1)
            sub_z = self.sub_centroids[nearest, 2]
            n_below = int(np.sum(self.cell_vertices[:, 2] < sub_z - 0.1))
            if n_below > 0:
                log.warning(f'[Phase2] {n_below} cell vertices below substrate at step {step}')

        # Nucleus outside cell (approximate: check mean radius)
        cell_center = self.cell_vertices.mean(axis=0)
        cell_r = compute_mean_radius(self.cell_vertices, cell_center)
        nuc_dists = np.linalg.norm(self.nuc_vertices - cell_center, axis=1)
        n_outside = int(np.sum(nuc_dists > cell_r + 0.5))
        if n_outside > 0:
            log.warning(f'[Phase2] {n_outside} nucleus vertices outside cell at step {step}')
