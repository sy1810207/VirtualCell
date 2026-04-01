"""
VirtualCell v0.85 — Simulation Configuration
All parameters from Table S1 of the paper (DOI: 10.1021/acsnano.7b03732).
"""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class SimConfig:
    # ── Energy unit ──
    kBT: float = 1.0

    # ── Membrane bond parameters (FENE + WCA) ──
    k_fene: float = 50.0      # FENE spring constant [kBT/a^2]
    R_max: float = 1.10       # max extension from equilibrium [a]
    r_eq_bond: float = 1.0    # equilibrium bond length [a]
    epsilon_wca: float = 1.0   # WCA repulsion depth [kBT]
    sigma_wca: float = 0.67    # WCA repulsion length [a] (= l_min)
    l_max: float = 2.10      # max bond length for safety checks [a]
    l_min: float = 0.67      # min bond length for safety checks [a]
    kappa_curve: float = 1.0    # bending rigidity [kBT] (low to allow flattening)
    kappa_s: float = 0.001     # surface area constraint [kBT/a^4] (very low for spreading)
    kappa_vol: float = 500000.0  # volume constraint [kBT] (dimensionless strain)

    # ── Cytoplasm / cytoskeleton (Eq.2-4) ──
    kappa_0: float = 1.0       # elasticity unit [kBT/a^2]
    kappa_cyt: float = 1.0     # cytoplasm linkage stiffness [kappa_0]
    # mu_0 not given in paper; controls viscoelastic relaxation rate
    mu_0: float = 50.0         # viscous coefficient [kBT·step/a^2]

    # ── Chromatin fibers (Eq.5) ──
    kappa_bonding_ch: float = 50.0   # chromatin bond stiffness [kBT/a^2] (not in Table S1)
    kappa_bending: float = 0.0       # explicitly zero in paper
    sigma_ch: float = 0.6            # LJ characteristic length [a]
    epsilon_ch: float = 1.0          # LJ depth [kBT] (not in Table S1)
    N_c: int = 5                     # number of chromatin chains
    N_beads: int = 500               # total beads across all chains

    # ── Geometry ──
    R_cell: float = 11.0       # cell radius [a]
    R_nucleus: float = 3.4     # nucleus radius [a]

    # ── ECM / substrate (Eq.6) ──
    epsilon_ECM: float = 12.0  # adhesion spring constant [kBT/a²]
    sigma_perp: float = 0.2   # target height above substrate [a]
    w: float = 10.0            # groove width [a]
    h: float = 1.0             # groove depth [a]

    # ── Active force ──
    F_active: float = 60.0     # [kBT/a]

    # ── Dynamics ──
    dt: float = 0.0005         # timestep
    gamma: float = 1.0         # friction coefficient for overdamped Langevin
    T: float = 1.0             # temperature [kBT]

    # ── Simulation length ──
    n_steps_phase1: int = 100_000
    n_steps_phase2: int = 150_000

    # ── Mesh resolution ──
    ico_subdiv_cell: int = 3     # icosphere subdivisions for cell (~642 verts)
    ico_subdiv_nucleus: int = 1  # icosphere subdivisions for nucleus (~42 verts)
    # Nucleus uses level 1 so edge length ≈ cell's edge length a
    # (paper uses same membrane parameters for both; edges must be same scale)
    n_cyt_points: int = 600      # target cytoplasm interior points

    # ── Monitoring ──
    check_interval: int = 500     # steps between safety checks
    vis_interval: int = 2000      # steps between visualization updates
    log_interval: int = 1000      # steps between log entries

    # ── Steric repulsion (penetration prevention) ──
    k_steric: float = 1000.0     # steric spring constant [kBT/a^2]
    d_steric: float = 0.3        # shell thickness for steric ramp [a]

    # ── Derived quantities (computed at init) ──
    a: float = field(init=False, default=1.0)            # unit length (set from mesh)
    r0_ch: float = field(init=False, default=1.2)        # chromatin equilibrium bond = 2*sigma_ch
    r_cut_ch: float = field(init=False, default=0.0)     # LJ cutoff

    # Flag to avoid double-scaling
    _scaled: bool = field(init=False, default=False)

    def __post_init__(self):
        self.r0_ch = 2.0 * self.sigma_ch          # = 1.2a
        self.r_cut_ch = 2.0**(1.0/6.0) * self.sigma_ch  # ≈ 0.673a

    def scale_by_a(self, a_actual: float):
        """
        Scale all length parameters from units-of-a to absolute simulation units.
        Must be called once after computing the actual edge length from the mesh.
        Table S1 defines everything in units of 'a' (initial triangle edge length).
        """
        if self._scaled:
            return
        self.a = a_actual
        # Length parameters: multiply by a
        self.l_max *= a_actual
        self.l_min *= a_actual
        self.R_max *= a_actual
        self.r_eq_bond *= a_actual
        self.sigma_wca *= a_actual
        self.sigma_ch *= a_actual
        self.sigma_perp *= a_actual
        self.d_steric *= a_actual
        self.w *= a_actual
        self.h *= a_actual
        # Re-derive
        self.r0_ch = 2.0 * self.sigma_ch
        self.r_cut_ch = 2.0**(1.0/6.0) * self.sigma_ch
        # Stiffness parameters: [kBT/a^n] → divide by a^n
        self.kappa_s /= a_actual**4
        self.kappa_0 /= a_actual**2
        self.kappa_cyt /= a_actual**2
        self.kappa_bonding_ch /= a_actual**2
        self.k_steric /= a_actual**2
        self.k_fene /= a_actual**2
        # F_active [kBT/a] → divide by a
        self.F_active /= a_actual
        self._scaled = True

    @property
    def beads_per_chain(self) -> int:
        return self.N_beads // self.N_c  # 100

    @property
    def noise_amplitude(self) -> float:
        return np.sqrt(2.0 * self.kBT * self.dt / self.gamma)
