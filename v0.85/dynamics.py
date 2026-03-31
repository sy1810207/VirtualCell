"""
VirtualCell v0.85 — Time Integration
Overdamped Langevin dynamics with adaptive timestep.
"""
import numpy as np
from config import SimConfig


class OverdampedLangevin:
    """
    dr = (F / γ) * dt + sqrt(2 * kBT * dt / γ) * η
    """

    def __init__(self, config: SimConfig):
        self.cfg = config
        self.dt = config.dt
        self.base_dt = config.dt
        self.rng = np.random.default_rng(0)

    def step(self, positions, forces):
        """
        Integrate one timestep. Returns new positions.
        Separates center-of-mass (rigid body) motion from internal deformation
        so that external forces (ECM, gravity) are not swamped by large but
        nearly-canceling internal forces.
        """
        dt = self.dt
        N = len(positions)

        # Compute raw displacement
        raw_disp = (forces / self.cfg.gamma) * dt

        # Separate center-of-mass motion from internal deformation
        com_disp = raw_disp.mean(axis=0)  # net displacement of the body
        internal_disp = raw_disp - com_disp  # zero-mean internal deformation

        # Adaptive dt for internal deformation only
        max_internal = np.max(np.linalg.norm(internal_disp, axis=1))
        max_disp = 0.1  # cap per vertex
        if max_internal > max_disp:
            scale_int = max_disp / max_internal
            internal_disp *= scale_int
            # Also scale dt for noise calculation
            dt_effective = dt * scale_int
        else:
            dt_effective = dt

        # Cap COM displacement at 0.05a per step
        com_mag = np.linalg.norm(com_disp)
        max_com = 0.05
        if com_mag > max_com:
            com_disp *= max_com / com_mag

        # Thermal noise (uses effective dt)
        noise_amp = np.sqrt(2.0 * self.cfg.kBT * dt_effective / self.cfg.gamma)
        noise = noise_amp * self.rng.standard_normal(positions.shape)

        displacement = com_disp + internal_disp + noise

        self.dt = dt  # store base dt (not effective)
        return positions + displacement

    def update_eq_lengths(self, eq_lengths, bond_forces_mag, dt=None):
        """
        Viscoelastic equilibrium length evolution (Eq.3):
        dr_eq/dt = f / μ_0
        """
        if dt is None:
            dt = self.dt
        return eq_lengths + (bond_forces_mag / self.cfg.mu_0) * dt

    def reset_dt(self):
        self.dt = self.base_dt
