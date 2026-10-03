"""Optional live viewers. Importing this module does not import graphics packages.

Snapshots are NumPy copies, so a viewer never owns or modifies solver state.
Call ``poll()`` regularly, including while paused; call ``update()`` only at the
desired display frequency. Scalar limits stay fixed after the initial snapshot.
"""

from __future__ import annotations

from pathlib import Path
import threading
from typing import Any

import numpy as np


FIELDS = ("eta", "depth", "speed")
FIELD_LABELS = {"eta": "Surface elevation (m)", "depth": "Depth (m)", "speed": "Speed (m/s)"}


def _scalar(snapshot: dict, field: str) -> np.ndarray:
    if field == "eta":
        return np.asarray(snapshot["surface_height"])
    if field == "depth":
        return np.asarray(snapshot["depth"])
    if field == "speed":
        return np.linalg.norm(snapshot["velocity"], axis=1)
    raise ValueError(f"field must be one of {FIELDS}, got {field!r}")


def _limits(values: np.ndarray) -> tuple[float, float]:
    lo, hi = float(np.min(values)), float(np.max(values))
    if hi - lo < 1.0e-10 * max(1.0, abs(lo), abs(hi)):
        padding = max(1.0e-3, 0.01 * abs(lo))
        lo, hi = lo - padding, hi + padding
    return lo, hi


def _validate_clim(clim: tuple[float, float] | None) -> tuple[float, float] | None:
    if clim is None:
        return None
    if len(clim) != 2 or not np.all(np.isfinite(clim)) or clim[0] >= clim[1]:
        raise ValueError("clim must contain two finite values with min < max")
    return float(clim[0]), float(clim[1])


def _field_limits(snapshot: dict) -> dict[str, tuple[float, float]]:
    ranges = {f: _limits(_scalar(snapshot, f)) for f in FIELDS}
    for field in ("depth", "speed"):
        lo, hi = ranges[field]
        ranges[field] = (max(0.0, lo), max(hi, 1.0e-3))
    return ranges


def _bed_geometry(model: Any, resolution: int = 81) -> tuple[np.ndarray, np.ndarray]:
    """Sample a static display-only bed, including dry islands and shoreline.

    This rectangular display grid is independent of the solver's wet material
    mesh. Its piecewise-linear interpolation is used only for drawing.
    """
    lower = np.min(model.points, axis=0)
    upper = np.max(model.points, axis=0)
    xx, yy = np.meshgrid(np.linspace(lower[0], upper[0], resolution),
                         np.linspace(lower[1], upper[1], resolution))
    xy = np.column_stack((xx.ravel(), yy.ravel()))
    elevation, _ = model.bottom.evaluate(xy)
    first = np.arange(resolution * resolution).reshape(resolution, resolution)[:-1, :-1].ravel()
    triangles = np.vstack((np.column_stack((first, first + 1, first + resolution + 1)),
                           np.column_stack((first, first + resolution + 1, first + resolution))))
    return np.column_stack((xy, elevation)), triangles


def _colors(values: np.ndarray, limits: tuple[float, float]) -> np.ndarray:
    # A compact blue-cyan-yellow ramp, shared by both viewers. No Matplotlib
    # dependency is needed for headless or browser use.
    stops = np.array([[25, 60, 150], [30, 190, 210], [250, 220, 70]])
    q = np.clip((values - limits[0]) / (limits[1] - limits[0]), 0.0, 1.0)
    return np.column_stack(
        [np.interp(q, [0.0, 0.5, 1.0], stops[:, c]) for c in range(3)]
    ).astype(np.uint8)


def _display_points(points: np.ndarray, vertical_scale: float) -> np.ndarray:
    """Exaggerate geometry for display without changing solver snapshots."""
    displayed = np.array(points, dtype=np.float32, copy=True)
    displayed[:, 2] *= vertical_scale
    return displayed


def _validate_vertical_scale(value: float) -> float:
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError("vertical_scale must be finite and positive")
    return value


def _status(snapshot: dict, diagnostics: dict, paused: bool) -> str:
    return (
        f"t = {float(snapshot['time']):.4f} s   "
        f"{'PAUSED' if paused else 'RUNNING'}\n"
        f"min J = {diagnostics.get('min_j', float('nan')):.4g}   "
        f"min h = {diagnostics.get('min_depth', float('nan')):.4g} m\n"
        f"max speed = {diagnostics.get('max_speed', float('nan')):.4g} m/s   "
        f"volume = {diagnostics.get('volume', float('nan')):.6g} m^3"
    )


class NullViewer:
    """Control interface used by all viewers, also useful for headless callers."""

    def __init__(self, *, paused: bool = False, **_: Any):
        self._lock = threading.Lock()
        self._running = True
        self._paused = bool(paused)
        self._step_requests = 0
        self._reset_requested = False

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def paused(self) -> bool:
        with self._lock:
            return self._paused

    @paused.setter
    def paused(self, value: bool) -> None:
        with self._lock:
            self._paused = bool(value)

    def toggle_pause(self) -> None:
        with self._lock:
            self._paused = not self._paused

    def request_step(self) -> None:
        with self._lock:
            self._paused = True
            self._step_requests += 1

    def request_reset(self) -> None:
        with self._lock:
            self._reset_requested = True
            self._step_requests = 0

    def consume_step_request(self) -> bool:
        with self._lock:
            if self._step_requests:
                self._step_requests -= 1
                return True
        return False

    def consume_reset_request(self) -> bool:
        with self._lock:
            requested = self._reset_requested
            self._reset_requested = False
            return requested

    def poll(self) -> None:
        """Process UI events without requesting a new solver snapshot."""

    def update(self, snapshot: dict, diagnostics: dict) -> None:
        pass

    def close(self) -> None:
        with self._lock:
            self._running = False


class PyVistaViewer(NullViewer):
    """Plain VTK desktop window; no Qt event loop or fancy water rendering."""

    def __init__(
        self, model: Any, *, field: str = "eta", paused: bool = False,
        off_screen: bool = False, clim: tuple[float, float] | None = None,
        vertical_scale: float = 1.0, wireframe: bool = False,
        **_: Any,
    ):
        super().__init__(paused=paused)
        if field not in FIELDS:
            raise ValueError(f"field must be one of {FIELDS}")
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "PyVista viewer is optional; install with pip install -e '.[desktop]'"
            ) from exc
        self.field = field
        self._vertical_scale = _validate_vertical_scale(vertical_scale)
        self._clim = _validate_clim(clim)
        self._limits_by_field: dict[str, tuple[float, float]] = {}
        self._snapshot: dict | None = None
        self._diagnostics: dict = {}
        self._off_screen = off_screen
        self._mesh = None
        self._actor = None
        self._shown = False
        self._wireframe = bool(wireframe)
        self._faces = np.column_stack(
            (np.full(len(model.triangles), 3), model.triangles)
        ).ravel()
        self._pv = pv
        self.plotter = pv.Plotter(off_screen=off_screen, window_size=(1100, 760))
        self.plotter.set_background("#edf2f5")
        bed_vertices, bed_triangles = _bed_geometry(model)
        bed_faces = np.column_stack((np.full(len(bed_triangles), 3), bed_triangles)).ravel()
        bed_mesh = pv.PolyData(_display_points(bed_vertices, self._vertical_scale), bed_faces)
        self.plotter.add_mesh(bed_mesh, color="#b7aa91", lighting=False)
        self.plotter.add_axes()
        self.plotter.add_text(
            "Space: pause  N: step  R: reset  1/2/3: eta/depth/speed\n"
            "W: wireframe  T: top view  I: 3D view  Q: quit",
            position="lower_left", font_size=10, color="#202b35",
        )
        for key in ("space",):
            self.plotter.add_key_event(key, self.toggle_pause)
        for key, callback in (
            ("n", self.request_step), ("r", self.request_reset),
            ("q", self.close), ("w", self._toggle_edges),
            ("t", self.plotter.view_xy), ("i", self.plotter.view_isometric),
        ):
            # Remove PyVista's built-in r/q/w actions before assigning ours.
            self.plotter.clear_events_for_key(key)
            self.plotter.add_key_event(key, callback)
        for key, selected in zip(("1", "2", "3"), FIELDS):
            self.plotter.add_key_event(key, lambda selected=selected: self._set_field(selected))
        if not off_screen:
            self.plotter.iren.add_observer("ExitEvent", lambda *_: self.close())

    def _set_field(self, field: str) -> None:
        self.field = field
        if self._snapshot is not None:
            self.update(self._snapshot, self._diagnostics)

    def _toggle_edges(self) -> None:
        self._wireframe = not self._wireframe
        if self._actor is not None:
            self._actor.prop.style = "wireframe" if self._wireframe else "surface"

    def update(self, snapshot: dict, diagnostics: dict) -> None:
        if not self.is_running:
            return
        self._snapshot, self._diagnostics = snapshot, diagnostics
        if not self._limits_by_field:
            self._limits_by_field = _field_limits(snapshot)
        limits = self._clim or self._limits_by_field[self.field]
        values = _scalar(snapshot, self.field)
        if self._mesh is None:
            self._mesh = self._pv.PolyData(
                _display_points(snapshot["points"], self._vertical_scale), self._faces)
            self._mesh.point_data["field"] = values
            self._actor = self.plotter.add_mesh(
                self._mesh, scalars="field", clim=limits, lighting=False,
                cmap=[f"#{r:02x}{g:02x}{b:02x}" for r, g, b in
                      _colors(np.linspace(0.0, 1.0, 256), (0.0, 1.0))],
                scalar_bar_args={"title": "Field value"},
                show_edges=False, edge_color="#526979",
                style="wireframe" if self._wireframe else "surface",
            )
            self.plotter.view_isometric()
            self.plotter.reset_camera()
        else:
            self._mesh.points = _display_points(snapshot["points"], self._vertical_scale)
            self._mesh.point_data["field"] = values
            self._actor.mapper.scalar_range = limits
        self.plotter.add_text(
            _status(snapshot, diagnostics, self.paused)
            + f"\nVertical scale: {self._vertical_scale:g}x (display only)"
            + f"\n{FIELD_LABELS[self.field]}: fixed limits [{limits[0]:.4g}, {limits[1]:.4g}]",
            name="status", position="upper_left", font_size=11, color="#202b35",
        )
        if not self._shown:
            self.plotter.show(auto_close=False, interactive_update=not self._off_screen)
            self._shown = True
        elif self._off_screen:
            self.plotter.render()
        else:
            self.poll()

    def poll(self) -> None:
        if self.is_running and self._shown and not self._off_screen:
            # The window is closed through VTK or our q callback. Never create
            # a new render window after close.
            if self.plotter.iren is None or self.plotter.render_window is None:
                super().close()
                return
            self.plotter.update(stime=1, force_redraw=False)

    def screenshot(self, path: str | Path) -> None:
        if not self._shown or not self.is_running:
            raise RuntimeError("Update a running viewer before taking a screenshot")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.plotter.screenshot(str(path))

    def close(self) -> None:
        if self.is_running:
            super().close()
            self.plotter.close()


class ViserViewer(NullViewer):
    """Browser mesh geometry with a colored nodal field overlay.

    Viser's simple mesh has a uniform material color; scalar values are shown
    explicitly at nodes with a persistent point cloud, not a rebuilt GLB mesh.
    """

    def __init__(
        self, model: Any, *, field: str = "eta", paused: bool = False,
        host: str = "127.0.0.1", port: int = 8080,
        clim: tuple[float, float] | None = None,
        vertical_scale: float = 1.0, show_nodes: bool = True,
        wireframe: bool = False, **_: Any,
    ):
        super().__init__(paused=paused)
        if field not in FIELDS:
            raise ValueError(f"field must be one of {FIELDS}")
        try:
            import viser
        except ImportError as exc:
            raise ImportError(
                "Browser viewer is optional; install with pip install -e '.[web]'"
            ) from exc
        self._clim = _validate_clim(clim)
        self._vertical_scale = _validate_vertical_scale(vertical_scale)
        self._limits_by_field: dict[str, tuple[float, float]] = {}
        self._mesh = None
        self._nodes = None
        self._faces = np.asarray(model.triangles, dtype=np.uint32)
        self._span = max(float(np.ptp(model.points, axis=0).max()), 1.0)
        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self._field = self.server.gui.add_dropdown("Nodal color", options=FIELDS, initial_value=field)
        self._pause_control = self.server.gui.add_checkbox("Paused", initial_value=paused)
        self._nodes_control = self.server.gui.add_checkbox("Show nodes", initial_value=show_nodes)
        self._edges_control = self.server.gui.add_checkbox("Wireframe", initial_value=wireframe)
        step_button = self.server.gui.add_button("Single step")
        reset_button = self.server.gui.add_button("Reset")
        stop_button = self.server.gui.add_button("Stop server")
        self._info = self.server.gui.add_markdown("Waiting for initial state.")
        self._pause_control.on_update(lambda _: setattr(self, "paused", self._pause_control.value))
        step_button.on_click(lambda _: self.request_step())
        reset_button.on_click(lambda _: self.request_reset())
        # Stop on the simulation/main thread, not inside Viser's callback pool.
        stop_button.on_click(lambda _: NullViewer.close(self))
        self._server_closed = False
        bed_vertices, bed_triangles = _bed_geometry(model)
        self.server.scene.add_mesh_simple(
            "/bottom", vertices=_display_points(bed_vertices, self._vertical_scale),
            faces=bed_triangles.astype(np.uint32), color=(183, 170, 145), side="double",
            cast_shadow=False, receive_shadow=False,
        )
        centre = np.mean(model.points, axis=0)
        elevation = float((np.min(bed_vertices[:, 2]) + np.max(bed_vertices[:, 2])) / 2) * self._vertical_scale

        @self.server.on_client_connect
        def _set_camera(client: Any) -> None:
            client.camera.position = (centre[0] + 0.75 * self._span,
                                      centre[1] - 0.75 * self._span,
                                      elevation + 0.65 * self._span)
            client.camera.look_at = (centre[0], centre[1], elevation)
            client.camera.up_direction = (0.0, 0.0, 1.0)

        self.url = f"http://{host}:{self.server.get_port()}"

    def poll(self) -> None:
        if self._pause_control.value != self.paused:
            self._pause_control.value = self.paused
        if self._mesh is not None:
            self._mesh.wireframe = self._edges_control.value
        if self._nodes is not None:
            self._nodes.visible = self._nodes_control.value
        self._field.disabled = not self._nodes_control.value

    def update(self, snapshot: dict, diagnostics: dict) -> None:
        if not self.is_running:
            return
        if not self._limits_by_field:
            self._limits_by_field = _field_limits(snapshot)
        field = self._field.value
        limits = self._clim or self._limits_by_field[field]
        colors = _colors(_scalar(snapshot, field), limits)
        vertices = _display_points(snapshot["points"], self._vertical_scale)
        nodal_vertices = vertices.copy()
        # Only the display overlay is offset to prevent z-fighting.
        nodal_vertices[:, 2] += 1.0e-4 * self._span
        with self.server.atomic():
            if self._mesh is None:
                self._mesh = self.server.scene.add_mesh_simple(
                    "/water", vertices=vertices, faces=self._faces, color=(72, 142, 172),
                    side="double", cast_shadow=False, receive_shadow=False,
                    wireframe=self._edges_control.value,
                )
                self._nodes = self.server.scene.add_point_cloud(
                    "/nodal_field", points=nodal_vertices, colors=colors,
                    point_size=0.006 * self._span, point_shape="circle",
                    visible=self._nodes_control.value,
                )
            else:
                self._mesh.vertices = vertices
                self._nodes.points = nodal_vertices
                self._nodes.colors = colors
            self._info.content = (
                f"**{'Paused' if self.paused else 'Running'}** · "
                f"t = {float(snapshot['time']):.4f} s\n\n"
                f"- Minimum J: {diagnostics.get('min_j', float('nan')):.4g}\n"
                f"- Minimum depth: {diagnostics.get('min_depth', float('nan')):.4g} m\n"
                f"- Maximum speed: {diagnostics.get('max_speed', float('nan')):.4g} m/s\n"
                f"- Volume: {diagnostics.get('volume', float('nan')):.6g} m³\n\n"
                f"Mesh: **{len(vertices):,} vertices / {len(self._faces):,} triangles**\n\n"
                f"Vertical scale: **{self._vertical_scale:g}×** (display only; values below are physical).\n\n"
                + (f"**{FIELD_LABELS[field]} at nodes** — fixed blue → yellow range: "
                   f"`{limits[0]:.4g}` → `{limits[1]:.4g}`. Values outside this range clip in color."
                   if self._nodes_control.value else "Node markers hidden; mesh geometry only.")
            )
        self.poll()
        self.server.flush()

    def close(self) -> None:
        super().close()
        if not self._server_closed:
            self._server_closed = True
            self.server.stop()


def create_viewer(kind: str, model: Any, **options: Any) -> NullViewer:
    """Construct a viewer without importing any unused graphics backend."""
    if kind == "none":
        return NullViewer(**options)
    if kind == "pyvista":
        return PyVistaViewer(model, **options)
    if kind == "viser":
        return ViserViewer(model, **options)
    raise ValueError("viewer must be 'none', 'pyvista', or 'viser'")
