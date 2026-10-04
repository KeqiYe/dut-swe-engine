# Method and benchmark references

This repository contains the Python displacement shallow-water engine, its
NumPy and Warp backends, examples, tests, and benchmark fixtures.

## Mathematical method

Feng Wu and Wanxie Zhong, “On displacement shallow water wave equation and
symplectic solution,” *Computer Methods in Applied Mechanics and Engineering*
318 (2017), 431–455. <https://doi.org/10.1016/j.cma.2017.01.040>.

This implementation follows the **non-dispersive, depth-averaged** branch with
horizontal Lagrangian P1 triangles, three-point quadrature, lumped mass, and
explicit Verlet integration. It does not implement every model in the paper,
and it is not an official release by the paper's authors. Please cite the
paper when describing the numerical method and identify the software revision
when reporting results obtained with this implementation.

## Adaptive artificial viscosity

Keqi Ye, Yuelin Zhao, Feng Wu, and Wanxie Zhong, “An adaptive artificial
viscosity for the displacement shallow water wave equation,” *Applied
Mathematics and Mechanics* 43(2) (2022), 247–262.
<https://doi.org/10.1007/s10483-022-2815-7>.

This study introduces adaptive artificial viscosity (AAV) to suppress numerical
oscillations. It is relevant to extending this engine's treatment of shocks
and wet-dry interfaces.
**AAV is not implemented in the current Python solver.** The optional Manning
bottom-friction term (zero by default) is a different mechanism. Cite this
paper when discussing or adopting its AAV method; its inclusion here does not
establish that the current solver reproduces the paper's capabilities.

## Independent benchmark reference

The independent parabolic-basin sloshing benchmark uses the class of exact
solutions described by W. C. Thacker, “Some exact solutions to the nonlinear
shallow-water wave equations,” *Journal of Fluid Mechanics* 107 (1981),
499–508. <https://doi.org/10.1017/S0022112081001882>.

Machine-readable references are provided in [references.bib](references.bib).

## Included data

- `src/swe/data/island_mesh.npz`: a fixed 547-node, 1,006-triangle island mesh
  with zero-based connectivity and component-wise wall constraints.
- `src/swe/data/matlab_reference.npz` and
  `validation/matlab_reference.mat`: frozen displacement, velocity, and energy
  records for the 10-second island regression. Agreement with a baseline using
  the same numerical method is not independent evidence of physical accuracy.
- `validation/`: measurements, logs, and scripts documenting specific tested
  configurations. Normal installation and Python tests use the included
  fixtures without requiring external simulation software.

The MIT license applies to the material distributed in this repository. It
does not grant rights to journal papers or third-party dependencies.
Dependencies retain their own licenses.
