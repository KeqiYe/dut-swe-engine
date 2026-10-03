"""Small, headless Python API example.

Run after installing the package: ``python examples/basic.py``.
This application defines its own scene and time loop using only the public API.
For an optional viewer see ``examples/standing_wave.py``.
"""

from pathlib import Path

import numpy as np
import swe


def main() -> None:
    # 1. Application inputs: horizontal mesh, bottom, water level and gravity.
    mesh = swe.Mesh.rectangle(nx=40, ny=4, size=(10.0, 1.0), walls=True)
    builder = swe.ModelBuilder(gravity=9.81, density=1000.0)
    builder.add_water(
        mesh=mesh,
        bathymetry=swe.FlatBottom(z=0.0),
        surface_height=lambda xy: 1.0 + 0.001 * np.cos(np.pi * xy[:, 0] / 10.0),
        velocity=(0.0, 0.0),
    )
    model = builder.finalize(device="auto")  # CUDA when available, otherwise CPU.

    # 2. Application-owned time loop; the engine advances exactly one dt.
    state, next_state = model.state(), model.state()
    solver = swe.DisplacementSolver(model)
    dt = 0.005

    for _ in range(100):
        solver.step(state, next_state, dt)
        state, next_state = next_state, state

    # 3. Read results. Device-to-host copies happen explicitly here.
    snapshot = state.to_numpy()
    print(f"t={state.time:.4f} s, surface height range:",
          snapshot["surface_height"].min(), snapshot["surface_height"].max())
    print(model.diagnostics(state))
    output = Path("outputs/basic_final.npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    state.save(output)
    print(f"Saved {output.resolve()}")


if __name__ == "__main__":
    main()
