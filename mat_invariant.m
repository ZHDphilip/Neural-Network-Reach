clear all 
clc

% Q = Polyhedron([1 -1;0.5 -2;-1 0.4; -1 -2],[1;2;3;4])
% 
% sys1 = LTISystem('A', 0.5, 'f', 0);
% sys1.setDomain('x', Polyhedron('lb',0, 'ub', 1));
% 
% sys2 = LTISystem('A', -0.5, 'f', 0);
% sys2.setDomain('x', Polyhedron('lb',-1, 'ub', 0));
% dd = [sys1, sys2];
% 
% for iter = 1:5
%     array(iter) = sys1;
% end
% 
% pwa = PWASystem([sys1, sys2])
% S = pwa.invariantSet()
% plot(S)

% Load in A,b, C,d cell arrays
load('models/vanderpol/vanderpol_pwa.mat');
load('models/vanderpol/vanderpol_seed.mat');
P_seed = Polyhedron(A_roa, b_roa);

num_regions = length(A);
for i = 1:num_regions
    P_i = Polyhedron(A{i}, b{i});
    C_i = C{i};
    d_i = d{i};
    system_i = LTISystem('A', C_i, 'f', d_i);
    system_i.setDomain('x', P_i);
    systems(i) = system_i;
end

pwa = PWASystem(systems);
S = pwa.invariantSet('X', P_seed, 'maxIterations', 10); % Can add arguments of 'X' (seed set) and maxIterations
plot(S)
xlim([-2.5 2.5])
ylim([-3 3])
