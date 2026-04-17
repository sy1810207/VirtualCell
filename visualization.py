"""
VirtualCell v0.85 — Visualization
Plotly HTML 3D output + matplotlib real-time viewer with stop button.
"""
import numpy as np
import os
from datetime import datetime


# ═══════════════════════════════════════════════════════════════════
#  HTML 3D Visualization (Plotly)
# ═══════════════════════════════════════════════════════════════════

def generate_html_visualization(cell_vertices, cell_faces,
                                nuc_vertices, nuc_faces,
                                chains,
                                cyt_positions=None,
                                sub_vertices=None, sub_faces=None,
                                output_dir='.', version='v085'):
    """Create interactive 3D HTML file with Plotly."""
    import plotly.graph_objects as go

    fig = go.Figure()

    # Cell membrane — semi-transparent blue
    fig.add_trace(go.Mesh3d(
        x=cell_vertices[:, 0],
        y=cell_vertices[:, 1],
        z=cell_vertices[:, 2],
        i=cell_faces[:, 0],
        j=cell_faces[:, 1],
        k=cell_faces[:, 2],
        opacity=0.2,
        color='#7d7df0',
        name='Cell Membrane',
        flatshading=True,
    ))

    # Nuclear envelope — gold
    fig.add_trace(go.Mesh3d(
        x=nuc_vertices[:, 0],
        y=nuc_vertices[:, 1],
        z=nuc_vertices[:, 2],
        i=nuc_faces[:, 0],
        j=nuc_faces[:, 1],
        k=nuc_faces[:, 2],
        opacity=0.5,
        color='gold',
        name='Nucleus',
        flatshading=True,
    ))

    # Chromatin chains — colored lines
    colors = ['red', 'green', 'blue', 'purple', 'orange',
              'cyan', 'magenta', 'lime', 'pink', 'yellow']
    for i, chain in enumerate(chains):
        fig.add_trace(go.Scatter3d(
            x=chain[:, 0], y=chain[:, 1], z=chain[:, 2],
            mode='lines+markers',
            marker=dict(size=1.5, color=colors[i % len(colors)]),
            line=dict(width=2, color=colors[i % len(colors)]),
            name=f'Chromatin {i+1}',
        ))

    # Cytoplasm nodes — small translucent dots
    if cyt_positions is not None:
        fig.add_trace(go.Scatter3d(
            x=cyt_positions[:, 0],
            y=cyt_positions[:, 1],
            z=cyt_positions[:, 2],
            mode='markers',
            marker=dict(size=1, color='gray', opacity=0.15),
            name='Cytoplasm',
        ))

    # Substrate — gray mesh
    if sub_vertices is not None and sub_faces is not None:
        fig.add_trace(go.Mesh3d(
            x=sub_vertices[:, 0],
            y=sub_vertices[:, 1],
            z=sub_vertices[:, 2],
            i=sub_faces[:, 0],
            j=sub_faces[:, 1],
            k=sub_faces[:, 2],
            opacity=0.6,
            color='lightgray',
            name='Substrate',
            flatshading=True,
        ))

    fig.update_layout(
        title='VirtualCell v0.85 — 3D Visualization',
        scene=dict(
            aspectmode='data',
            xaxis_title='X [a]',
            yaxis_title='Y [a]',
            zaxis_title='Z [a]',
        ),
        legend=dict(x=0.02, y=0.98),
        margin=dict(l=0, r=0, t=40, b=0),
    )

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    html_dir = os.path.join(output_dir, 'html')
    os.makedirs(html_dir, exist_ok=True)
    filename = f'VirtualCell_{version}_{timestamp}.html'
    filepath = os.path.join(html_dir, filename)
    fig.write_html(filepath)
    return filepath


# ═══════════════════════════════════════════════════════════════════
#  Real-time Matplotlib Viewer with Stop Button
# ═══════════════════════════════════════════════════════════════════

class RealtimeVisualizer:
    """
    Matplotlib-based real-time 3D viewer with energy plot and stop button.
    Uses non-blocking mode (plt.ion).
    """

    def __init__(self, enabled=True):
        self.enabled = enabled
        self.stopped = False
        self.energy_history = []
        self.step_history = []
        self.fig = None
        self.ax3d = None
        self.ax_energy = None

        if not enabled:
            return

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Button
        self.plt = plt
        self.Button = Button

        plt.ion()
        self.fig = plt.figure(figsize=(14, 6))
        self.fig.canvas.manager.set_window_title('VirtualCell v0.85 — Real-time')

        self.ax3d = self.fig.add_subplot(121, projection='3d')
        self.ax_energy = self.fig.add_subplot(122)

        # Stop button
        stop_ax = self.fig.add_axes([0.85, 0.01, 0.12, 0.05])
        self.stop_btn = Button(stop_ax, 'STOP', color='#ff6666', hovercolor='#ff3333')
        self.stop_btn.on_clicked(self._on_stop)

        self.fig.tight_layout(rect=[0, 0.06, 1, 1])
        plt.show(block=False)
        plt.pause(0.01)

    def _on_stop(self, event):
        self.stopped = True

    def update(self, cell_vertices, cell_faces,
               nuc_vertices, nuc_faces,
               step, energy,
               sub_vertices=None, sub_faces=None,
               phase='Phase 1'):
        """Update the visualization."""
        if not self.enabled or self.fig is None:
            return

        self.energy_history.append(energy)
        self.step_history.append(step)

        try:
            # 3D view
            self.ax3d.clear()
            self.ax3d.set_title(f'{phase} — Step {step}')

            # Cell membrane wireframe
            self._draw_wireframe(self.ax3d, cell_vertices, cell_faces,
                                 color='steelblue', alpha=0.15)
            # Nucleus wireframe
            self._draw_wireframe(self.ax3d, nuc_vertices, nuc_faces,
                                 color='goldenrod', alpha=0.4)

            # Substrate
            if sub_vertices is not None and sub_faces is not None:
                self._draw_wireframe(self.ax3d, sub_vertices, sub_faces,
                                     color='gray', alpha=0.3)

            self.ax3d.set_xlabel('X')
            self.ax3d.set_ylabel('Y')
            self.ax3d.set_zlabel('Z')

            # Energy plot
            self.ax_energy.clear()
            self.ax_energy.plot(self.step_history, self.energy_history,
                                'b-', linewidth=0.8)
            self.ax_energy.set_xlabel('Step')
            self.ax_energy.set_ylabel('Total Energy [kBT]')
            self.ax_energy.set_title('Energy History')
            self.ax_energy.grid(True, alpha=0.3)

            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
        except Exception:
            pass  # don't crash sim if visualization fails

    def _draw_wireframe(self, ax, vertices, faces, color='blue', alpha=0.3):
        """Draw triangulated surface as wireframe."""
        from mpl_toolkits.mplot3d.art3d import Poly3DCollection
        # Sample faces to avoid overwhelming matplotlib
        max_faces = 500
        if len(faces) > max_faces:
            idx = np.linspace(0, len(faces) - 1, max_faces, dtype=int)
            faces_draw = faces[idx]
        else:
            faces_draw = faces

        polys = vertices[faces_draw]
        collection = Poly3DCollection(polys, alpha=alpha,
                                       facecolor=color,
                                       edgecolor=color,
                                       linewidth=0.2)
        ax.add_collection3d(collection)

        # Auto-scale
        all_pts = vertices
        for dim, setter in zip(range(3), [ax.set_xlim, ax.set_ylim, ax.set_zlim]):
            lo, hi = all_pts[:, dim].min(), all_pts[:, dim].max()
            margin = max((hi - lo) * 0.1, 1.0)
            setter(lo - margin, hi + margin)

    def close(self):
        if self.enabled and self.fig is not None:
            try:
                self.plt.close(self.fig)
            except Exception:
                pass
