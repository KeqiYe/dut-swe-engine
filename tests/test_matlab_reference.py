from importlib.resources import files
import numpy as np
from swe import DisplacementSolver
from swe.cases import island


def test_original_island_matches_matlab_over_ten_seconds():
    with np.load(files("swe").joinpath("data/matlab_reference.npz")) as data:
        ref={k:data[k] for k in data.files}
    m,meta=island(device="cpu"); a,b=m.state(),m.state(); solver=DisplacementSolver(m)
    wanted=np.rint(ref["time"]/float(ref["dt"])).astype(int)
    j=0
    for step in range(int(wanted[-1])+1):
        if step==wanted[j]:
            expected=np.column_stack((ref["u"][:,j],ref["v"][:,j],ref["du"][:,j],ref["dv"][:,j]))
            actual=np.column_stack((a.displacement,a.velocity))
            relative=np.linalg.norm(actual-expected)/max(1.,np.linalg.norm(expected))
            assert relative<1e-9
            assert abs(m.diagnostics(a)["energy"]/ref["energy"][j]-1)<1e-11
            j+=1
        if step<wanted[-1]: solver.step(a,b,float(ref["dt"])); a,b=b,a
