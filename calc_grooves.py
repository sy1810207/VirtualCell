import sys
sys.path.insert(0, 'D:/LJL_MLearning_Projects/claude/VirtualCell/v0.85')
from config import SimConfig
from mesh import IcosphereMesh
from utils import bond_length_stats

cfg = SimConfig()
mesh = IcosphereMesh(cfg.R_cell, cfg.ico_subdiv_cell)
_, _, a = bond_length_stats(mesh.vertices, mesh.edges)
cfg.scale_by_a(a)

w = cfg.w
h = cfg.h
n_periods = 8
period = 2.0 * w
x_start = -n_periods * period / 2.0
slope_dx = h * 0.4

print(f"a={a:.4f}, w={w:.2f}, h={h:.2f}, period={period:.2f}")
print(f"x_start={x_start:.2f}, slope_dx={slope_dx:.2f}")
print()
for p in range(n_periods):
    x_base = x_start + p * period
    gf0 = x_base + w + slope_dx
    gf1 = x_base + 2 * w - slope_dx
    gc = (gf0 + gf1) / 2
    print(f"P{p}: Ridge[{x_base:.1f},{x_base+w:.1f}] Groove[{gf0:.1f},{gf1:.1f}] center={gc:.1f}")
