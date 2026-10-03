"""Reproducible physical checks and synchronized throughput measurements.

python validation/run_validation.py --device cuda --scale --output validation/cuda.json
Compilation and model construction are excluded from timed stepping. Every
step validates the candidate on device. Diagnostic extrema below are sampled.
"""
import argparse
from datetime import datetime,timezone
from importlib.resources import files
import json
import math
from pathlib import Path
import platform
import time
import numpy as np
from swe import DisplacementSolver
from swe.cases import island,standing_wave,sloshing


def run(model,dt,duration,reference=None,reference_steps=None):
    solver=DisplacementSolver(model); a,b=model.state(),model.state()
    solver.step(a,b,dt)  # warm native kernel compilation before timing
    a.reset(); b.reset()
    initial=model.diagnostics(a); start=time.perf_counter()
    steps=round(duration/dt)
    if not math.isclose(steps*dt,duration,rel_tol=1e-10,abs_tol=1e-12):
        steps=math.ceil(duration/dt); dt=duration/steps
    stride=max(1,steps//100)
    minimum_j=1.; energy_drift=0.; maximum_error=0.
    checkpoints={int(s):j for j,s in enumerate(reference_steps)} if reference_steps is not None else {}
    for step in range(steps+1):
        if step%stride==0 or step==steps:
            d=model.diagnostics(a)
            minimum_j=min(minimum_j,d["min_j"])
            energy_drift=max(energy_drift,abs(d["energy"]-initial["energy"])/max(abs(initial["energy"]),1))
        if step in checkpoints:
            j=checkpoints[step]
            actual=np.column_stack((model.backend.numpy(a.displacement),model.backend.numpy(a.velocity)))
            expected=np.column_stack([reference[k][:,j] for k in ("u","v","du","dv")])
            maximum_error=max(maximum_error,float(np.linalg.norm(actual-expected)/max(np.linalg.norm(expected),1)))
        if step<steps: solver.step(a,b,dt); a,b=b,a
    elapsed=time.perf_counter()-start
    result=dict(device=model.device,nodes=len(model.points),triangles=len(model.triangles),
                simulation_seconds=duration,steps=steps,dt=dt,wall_seconds=elapsed,
                physical_seconds_per_wall_second=duration/elapsed,
                min_j_sampled=minimum_j,max_relative_energy_drift_sampled=energy_drift,
                final=model.diagnostics(a))
    if reference is not None:
        result["max_matlab_relative_state_error"]=maximum_error
        if maximum_error>=1e-9: raise AssertionError(f"MATLAB mismatch {maximum_error}")
    return a,result


def main():
    p=argparse.ArgumentParser(); p.add_argument("--device",default="cpu")
    p.add_argument("--scale",action="store_true")
    p.add_argument("--output",type=Path,default=Path("validation/results.json"))
    args=p.parse_args()
    report=dict(timestamp_utc=datetime.now(timezone.utc).isoformat(),platform=platform.platform(),
                python=platform.python_version(),numpy=np.__version__,requested_device=args.device,
                timing="warm stepping plus ~100 diagnostics, no viewer; compilation/model setup excluded",cases={})
    def save():
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.write_text(json.dumps(report,indent=2)+"\n")
    with np.load(files("swe").joinpath("data/matlab_reference.npz")) as data:
        ref={k:data[k] for k in data.files}
    m,info=island(device=args.device)
    _,report["cases"]["island_matlab"]=run(m,info["dt"],10,ref,np.rint(ref["time"]/ref["dt"]))
    print("PASS island MATLAB 10 s",flush=True); save()
    wave_errors=[]
    for nx in (16,32,64):
        m,info=standing_wave(device=args.device,nx=nx,ny=4,amplitude=1e-4)
        state,result=run(m,info["dt"],info["period"])
        snapshot=state.to_numpy(); truth=info["analytic"](snapshot["points"][:,:2],state.time)
        error=float(np.sqrt(np.mean((snapshot["surface_height"]-truth["surface_height"])**2))/info["amplitude"])
        result["relative_surface_rms"]=error; wave_errors.append(error)
        report["cases"][f"standing_wave_nx{nx}"]=result
    assert wave_errors[2]<wave_errors[1]<wave_errors[0] and wave_errors[-1]<.005
    print("PASS standing-wave spatial trend",wave_errors,flush=True); save()
    m,info=sloshing(device=args.device)
    state,result=run(m,info["dt"],3*info["period"])
    snapshot=state.to_numpy(); truth=info["analytic"](m.points,state.time)
    result["material_displacement_rms"]=float(np.sqrt(np.mean((snapshot["displacement"]-truth["displacement"])**2)))
    result["material_displacement_vector_rms"]=result["material_displacement_rms"]*np.sqrt(2.)
    assert result["material_displacement_vector_rms"]/info["amplitude"]<.08
    report["cases"]["sloshing_three_periods"]=result
    print("PASS sloshing three periods",flush=True); save()
    if args.scale:
        m,info=island(device=args.device,refinements=3)
        _,report["cases"]["island_64k"]=run(m,info["dt"],3)
        report["limitations"]=["Refined island validated only to 3 s: dt, dt/2, dt/4 reject at approximately 3.96, 4.03, 4.08 s. See island_scale_investigation.json; not a validated 10-second large-grid case."]
        print("PASS island 64,384 elements / 3 s (10 s remains unsupported)",flush=True); save()
        m,info=standing_wave(device=args.device,nx=250,ny=100,amplitude=1e-4)
        state,result=run(m,info["dt"],info["period"])
        snap=state.to_numpy(); truth=info["analytic"](snap["points"][:,:2],state.time)
        result["relative_surface_rms"]=float(np.sqrt(np.mean((snap["surface_height"]-truth["surface_height"])**2))/info["amplitude"])
        assert result["relative_surface_rms"]<.005
        report["cases"]["standing_wave_50k"]=result
        print("PASS standing wave 50,000 elements / one period",flush=True); save()
        m,info=sloshing(device=args.device,rings=64,sectors=400)
        state,result=run(m,info["dt"],info["period"])
        snap=state.to_numpy(); truth=info["analytic"](m.points,state.time)
        result["material_displacement_rms"]=float(np.sqrt(np.mean((snap["displacement"]-truth["displacement"])**2)))
        result["material_displacement_vector_rms"]=result["material_displacement_rms"]*np.sqrt(2.)
        assert result["material_displacement_vector_rms"]/info["amplitude"]<.01
        report["cases"]["sloshing_50k"]=result
        print("PASS sloshing 50,800 elements / one period",flush=True); save()
    report["status"]="passed"; save()
    print(args.output)


if __name__=="__main__": main()
