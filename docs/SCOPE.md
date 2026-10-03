# First release scope

User-approved: Python engine, CUDA priority with CPU fallback, configurable gravity,
bathymetry, triangular mesh, initial surface/velocity and dt. Validate island,
smooth waves and sloshing; target about 50,000 triangles (50 × the original 1,006).
Optional simple PyVista desktop and Viser browser viewers, separate from physics.

Numerical baseline: the local `../swe3d_minimal` nondispersive horizontal
Lagrangian P1 finite-element solver, three-point triangle quadrature, row-sum
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

Original MATLAB files and papers remain outside this new Git repository.
No public upload or license grant is implied by this local development task.
