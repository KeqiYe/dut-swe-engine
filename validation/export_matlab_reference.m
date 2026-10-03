% Reproduce the untouched historical numerical kernel in the sibling project.
project = fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(fileparts(project),'swe3d_minimal'));
r = run_swe3d(0.00625,10,false);
indices = [1,2,17,161,801,1601];
time = r.t(indices);
u = r.u(:,indices); v = r.v(:,indices);
du = r.du(:,indices); dv = r.dv(:,indices);
energy = r.H(indices); dt = r.dt;
matlab_version = version;
save(fullfile(project,'validation','matlab_reference.mat'), ...
    'time','u','v','du','dv','energy','dt','matlab_version','-v7');
