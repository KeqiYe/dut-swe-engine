# First release scope

Python engine, CUDA priority with CPU fallback, configurable gravity,
bathymetry, triangular mesh, initial surface/velocity and dt. Validate island,
smooth waves and sloshing; target about 50,000 triangles (50 × the original 1,006).
Optional simple PyVista desktop and Viser browser viewers, separate from physics.

Numerical formulation: nondispersive shallow water using horizontal
Lagrangian P1 finite elements, three-point triangle quadrature, row-sum
lumped mass and explicit Verlet integration. Bed elevation is positive upwards.
Use float64 initially. No independent vertical velocity, dispersive extension,
shock regularization, arbitrary wet/dry topology changes or rigid-body coupling.

Validation must distinguish numerical stability, temporal/spatial convergence,
linear analytic approximations, and benchmark throughput. Mass consistency is
built into the formulation and is not an independent accuracy proof.

Runtime design: retain current state and optional output snapshots, not full
per-step histories. Fixed walls constrain horizontal displacement components.
An invalid step must not overwrite the accepted state. Only initialization may
automatically fall back from unavailable CUDA. Runtime numerical errors surface.

Public release scope: the Python engine, examples, tests, and generated benchmark
data are distributed under the repository's MIT license.
See [LICENSE](../LICENSE) and [NOTICE.md](../NOTICE.md) for the license and
method references. Papers and third-party dependencies retain their own
copyrights and licenses.
