import numpy as np
import pytest
from swe import Mesh,ModelBuilder,FlatBottom,GridBottom,DisplacementSolver,NumericalError


def tank():
    return ModelBuilder().add_water(mesh=Mesh.rectangle(8,4),surface_height=1).finalize(device="cpu")


def test_lake_at_rest_and_wall_constraints():
    m=tank(); a,b=m.state(),m.state(); solver=DisplacementSolver(m)
    for _ in range(100):
        solver.step(a,b,.002); a,b=b,a
    s=a.to_numpy()
    assert np.max(abs(s["velocity"]))<1e-13
    assert np.max(abs(s["surface_height"]-1))<1e-13
    assert np.all(s["displacement"][m.fixed]==0)
    assert m.diagnostics(a)["relative_volume_error"]<1e-14


def test_checkpoint_restart_and_model_mismatch(tmp_path):
    from swe.cases import standing_wave
    m,meta=standing_wave(device="cpu",nx=12,ny=2)
    solver=DisplacementSolver(m); a,b=m.state(),m.state()
    for _ in range(5): solver.step(a,b,meta["dt"]); a,b=b,a
    path=tmp_path/"state.npz"; a.save(path); restored=m.load_state(path)
    solver.step(a,b,meta["dt"])
    c=m.state(); solver.step(restored,c,meta["dt"])
    np.testing.assert_array_equal(b.to_numpy()["displacement"],c.to_numpy()["displacement"])
    assert b.time==c.time
    with pytest.raises(ValueError,match="different model"): tank().load_state(path)


def test_rejected_step_is_transactional():
    m=tank(); a,b=m.state(),m.state(); a.velocity[:,0]=-100*m.points[:,0]
    # Large compression causes an inverted candidate. Output retains sentinel.
    b.displacement[:]=123.; b.velocity[:]=456.; b.time=7.
    before=a.to_numpy()["displacement"].copy()
    with pytest.raises(NumericalError): DisplacementSolver(m).step(a,b,.1)
    assert np.all(b.displacement==123) and np.all(b.velocity==456) and b.time==7
    np.testing.assert_array_equal(a.displacement,before)


def test_refine_preserves_domain_without_fixing_interior_diagonal():
    mesh=Mesh.rectangle(1,1).refined()
    midpoint=np.argmin(np.linalg.norm(mesh.points-[5,1],axis=1))
    assert not mesh.fixed[midpoint].any()
    m=ModelBuilder().add_water(mesh=mesh).finalize(device="cpu")
    assert len(mesh.triangles)==8
    assert m.area.sum()==pytest.approx(20)


def test_sampled_function_and_gradient():
    bottom=GridBottom.from_function(lambda p:2*p[:,0]-3*p[:,1]+1,(0,2,-1,3),(7,9))
    xy=np.array([[.2,1.2],[2,3],[0,-1.]])
    z,g=bottom.evaluate(xy)
    np.testing.assert_allclose(z,2*xy[:,0]-3*xy[:,1]+1)
    np.testing.assert_allclose(g,np.tile([2,-3],(3,1)))
    with pytest.raises(ValueError,match="outside"): bottom.evaluate([[3,1]])


def test_gravity_scales_acceleration():
    mesh=Mesh.rectangle(5,2)
    def build(g):
        return ModelBuilder(gravity=g).add_water(mesh=mesh,surface_height=lambda p:1+.01*np.cos(np.pi*p[:,0]/10)).finalize(device="cpu")
    a,b=build(5),build(10)
    fa=a.backend.force(a.state().displacement,a.state().velocity)
    fb=b.backend.force(b.state().displacement,b.state().velocity)
    np.testing.assert_allclose(fb,2*fa)


def test_bad_inputs_fail_before_simulation():
    with pytest.raises(ValueError): ModelBuilder(gravity=-1)
    with pytest.raises(ValueError): ModelBuilder().add_water(mesh=Mesh.rectangle(2,2),surface_height=-1).finalize(device="cpu")
    with pytest.raises(ValueError): Mesh([[0,0],[0,1],[1,0]],[[0,1,2]])
    m=tank(); a,b=m.state(),m.state()
    with pytest.raises(ValueError): DisplacementSolver(m).step(a,b,0)
    with pytest.raises(ValueError): DisplacementSolver(m).step(a,a,.01)


def test_advance_hits_requested_time():
    m=tank(); s=m.state()
    steps=DisplacementSolver(m).advance(s,.105,.01)
    assert steps==11
    assert s.time==pytest.approx(.105)


def test_finalized_model_and_terrain_are_immutable():
    bottom=GridBottom([0,5,10],[0,1,2],np.zeros((3,3)))
    m=ModelBuilder().add_water(mesh=Mesh.rectangle(4,2),bathymetry=bottom).finalize(device="cpu")
    with pytest.raises(AttributeError): m.gravity=1
    with pytest.raises(AttributeError): m.bottom=FlatBottom(4)
    with pytest.raises(ValueError): m.bottom.values[:]=1
    with pytest.raises(AttributeError): bottom.values=np.ones((3,3))
    assert m.bottom is not bottom


def test_invalid_substeps_and_unrepresentable_time_are_rejected():
    m=tank(); a,b=m.state(),m.state(); solver=DisplacementSolver(m)
    with pytest.raises(ValueError): solver.advance(a,1,.01,max_subdivisions=-1)
    a.time=1e6
    with pytest.raises(ValueError,match="advance"): solver.step(a,b,1e-12)
    assert b.time==0
    with pytest.raises(ValueError,match="advance"): solver.advance(a,1e-12,1e-12)


def test_modified_mesh_is_revalidated():
    mesh=Mesh.rectangle(3,3)
    builder=ModelBuilder().add_water(mesh=mesh)
    mesh.triangles[4]=mesh.triangles[4,::-1]
    with pytest.raises(ValueError,match="counterclockwise"): builder.finalize(device="cpu")


def test_invalid_output_array_does_not_partially_commit():
    m=tank(); a,b=m.state(),m.state(); b.displacement[:]=123
    b.velocity=np.zeros((len(m.points),3))
    with pytest.raises(ValueError): DisplacementSolver(m).step(a,b,.01)
    assert np.all(b.displacement==123) and b.time==0


def test_small_scale_nonuniform_raster_is_rejected():
    with pytest.raises(ValueError,match="uniform"):
        GridBottom([0,1e-9,3e-9],[0,1],np.zeros((3,2)))


def test_self_contained_checkpoint_restores_sampled_fields(tmp_path):
    from swe import load_checkpoint
    from swe.cases import island
    m,meta=island(device="cpu"); state=m.state()
    DisplacementSolver(m).advance(state,.025,meta["dt"])
    path=tmp_path/"portable.npz"; state.save(path)
    restored_model,restored=load_checkpoint(path,device="cpu")
    assert restored_model.fingerprint==m.fingerprint
    assert restored.time==state.time
    np.testing.assert_array_equal(restored.to_numpy()["displacement"],state.to_numpy()["displacement"])
    a,b=m.state(),restored_model.state()
    DisplacementSolver(m).step(state,a,meta["dt"])
    DisplacementSolver(restored_model).step(restored,b,meta["dt"])
    np.testing.assert_array_equal(a.velocity,b.velocity)
