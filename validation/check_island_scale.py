"""Investigate stable time steps without altering the historical equations."""
import json,time
from pathlib import Path
from swe import DisplacementSolver,NumericalError
from swe.cases import island

results=[]
for refinement,factor in ((1,1),(2,1),(3,1),(3,.5),(3,.25)):
    m,info=island(device="cuda",refinements=refinement)
    dt=info["dt"]*factor; a,b=m.state(),m.state(); solver=DisplacementSolver(m)
    start=time.perf_counter(); error=None; minj=1.
    for step in range(round(10/dt)):
        try: solver.step(a,b,dt)
        except NumericalError as exc:
            error=str(exc); break
        a,b=b,a
        if step%200==0: minj=min(minj,m.diagnostics(a)["min_j"])
    row=dict(refinement=refinement,dt=dt,elapsed=time.perf_counter()-start,
             completed_seconds=a.time,steps=a.step_index,error=error,min_j_sampled=minj,
             final=m.diagnostics(a),suggested_dt=m.suggested_dt(a))
    results.append(row); print(json.dumps(row),flush=True)
    Path("validation/island_scale_investigation.json").write_text(json.dumps(results,indent=2)+"\n")
    if error is not None: a.save(f"validation/island_failed_r{refinement}_f{factor}.npz")
