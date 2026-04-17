"""
VirtualCell v0.85 — Entry Point
Usage:
    python run.py                         # run both phases
    python run.py --phase 1               # phase 1 only
    python run.py --phase 2 --groove-depth 0.5  # phase 2 only
    python run.py --no-realtime-vis       # disable real-time viewer
"""
import argparse
import sys
import os

# Ensure the script's directory is on the path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import SimConfig
from simulation import VirtualCellSimulation


def main():
    parser = argparse.ArgumentParser(
        description='VirtualCell v0.85 — Multicomponent Cell Simulation')
    parser.add_argument('--phase', type=int, default=0,
                        choices=[0, 1, 2],
                        help='0=both phases, 1=phase1 only, 2=phase2 only')
    parser.add_argument('--groove-depth', type=float, default=1.0,
                        help='Groove depth h in units of a')
    parser.add_argument('--steps1', type=int, default=100000,
                        help='Number of steps for Phase 1')
    parser.add_argument('--steps2', type=int, default=150000,
                        help='Number of steps for Phase 2')
    parser.add_argument('--dt', type=float, default=0.0005,
                        help='Timestep')
    parser.add_argument('--no-realtime-vis', action='store_true',
                        help='Disable real-time matplotlib visualization')
    parser.add_argument('--cell-x', type=float, default=-5.0,
                        help='Initial cell X coordinate for Phase 2')
    parser.add_argument('--cell-y', type=float, default=0.0,
                        help='Initial cell Y coordinate for Phase 2')
    parser.add_argument('--output-dir', type=str, default='.',
                        help='Output directory for checkpoints and HTML')
    args = parser.parse_args()

    config = SimConfig(
        h=args.groove_depth,
        n_steps_phase1=args.steps1,
        n_steps_phase2=args.steps2,
        dt=args.dt,
    )

    sim = VirtualCellSimulation(
        config,
        output_dir=args.output_dir,
        realtime_vis=not args.no_realtime_vis,
    )

    if args.phase in (0, 1):
        sim.run_phase1()

    if args.phase in (0, 2):
        sim.run_phase2(groove_depth=args.groove_depth,
                       cell_x=args.cell_x, cell_y=args.cell_y)

    # Print final statistics
    from utils import compute_aspect_ratio
    print('=== FINAL RESULTS ===')
    print(f'Cell AR: {compute_aspect_ratio(sim.cell_vertices):.3f}')
    print(f'Nuc AR:  {compute_aspect_ratio(sim.nuc_vertices):.3f}')
    print(f'Cell z:  [{sim.cell_vertices[:,2].min():.2f}, {sim.cell_vertices[:,2].max():.2f}]')
    print(f'Nuc z:   [{sim.nuc_vertices[:,2].min():.2f}, {sim.nuc_vertices[:,2].max():.2f}]')
    if sim.sub_vertices is not None:
        print(f'Sub z:   [{sim.sub_vertices[:,2].min():.2f}, {sim.sub_vertices[:,2].max():.2f}]')
    print('Done.')


if __name__ == '__main__':
    main()
