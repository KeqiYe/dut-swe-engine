"""Small, headless Python API example.

Run after installing the package: ``python examples/basic.py``.
For interactive controls use ``swe --case wave --viewer pyvista --hold`` or
``swe --case wave --viewer viser --paused --hold``.
"""

from pathlib import Path

from swe import DisplacementSolver
from swe.cases import standing_wave


def main() -> None:
    model, settings = standing_wave(device="auto")
    state, next_state = model.state(), model.state()
    solver = DisplacementSolver(model)

    for _ in range(100):
        solver.step(state, next_state, settings["dt"])
        state, next_state = next_state, state

    # Device-to-host copies happen explicitly here, not inside every time step.
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
