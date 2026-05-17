function traj = main_bb_reverse_as_func(seed)
%MAIN_BB_REVERSE_AS_FUNC Run BB reverse simulation without plotting or saving.

if nargin < 1
    seed = 0;
end
rng(seed);

project_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));

%% Dependencies
addpath(genpath(fullfile(project_root, 'controllers', 'ac', 'controller_cmpc_2d_quadprog')));
addpath(genpath(fullfile(project_root, 'controllers', 'bc', 'safety_controller_quadprog')));
addpath(genpath(fullfile(project_root, 'controllers', 'PrSBC_filter')));
addpath(genpath(fullfile(project_root, 'decision_module')));
addpath(genpath(fullfile(project_root, 'extended_BBS')));
addpath(fullfile(project_root, 'common'));
addpath(genpath(fullfile(project_root, 'experiment', 'dynamics')));
addpath(genpath(fullfile(project_root, 'experiment', 'utilities')));

%% params
params.n = 15;
params.dt = 0.3;
params.ct = 0.3;
params.h_ac = 10;
params.h_bc = 10;
params.steps = 80;

params.amax = 1.5;
params.a_max = params.amax;
params.vmax = 2;
params.v_max = params.vmax;

% Safety property
params.dmin = 1.7;
params.R_safe = 1.7;

% initial configuration
params.diameter = 15;
params.switch_step = 1;

params.ws = 10000;
params.wt = 10;

params.ws_bb = 3000;
params.w_orient = 20;

% Unmodeled params in true_dynamics
params.acc_scale = 0.95;
params.acc_bias = 0;
params.damping = 0.03;
params.nonlinear_drift = false;
params.noise_std = 0;

% Process noise parameters
params.epsilon_w_pos = 0.2 * params.vmax * params.dt;
params.sigma_obs_pos = params.epsilon_w_pos / 3;

% Sensor noise parameters
params.epsilon_v_pos = 0.2 * params.dmin;
params.sigma_obs_pos = params.epsilon_v_pos / 3;
params.epsilon_v_vel = 0.2 * params.vmax;
params.sigma_obs_vel = params.epsilon_v_vel / 3;

% DM safety check parameters
params.confidence = 0.9;
params.gamma = 0.4;
params.sensing_range = 4.0;

%% Predator params
params.predator = 0;
params.pFactor = 1.40;
params.pred_radius = 6;
params.wp = 500;

%% Results structure
result.is_switch = false;

%% quadprog Optimizer Settings
opt = optimoptions('quadprog');
opt.Algorithm = 'interior-point-convex';
opt.Display = 'off';
opt.OptimalityTolerance = 1e-6;
opt.ConstraintTolerance = 1e-6;
opt.StepTolerance = 1e-10;
opt.LinearSolver = 'sparse';

%% Init
acc = zeros(2, params.n);
controller_run = params.ct / params.dt;
params.switch_step = params.steps;

[posi, veli, params.tgt] = gen_init_bb(params);

x = zeros([params.steps, params.n]);
y = zeros([params.steps, params.n]);
vx = zeros([params.steps, params.n]);
vy = zeros([params.steps, params.n]);
ax = zeros([params.steps + 1, params.n]);
ay = zeros([params.steps + 1, params.n]);
mpc_cost = zeros(1, params.steps);
bb_sp = zeros(1, params.steps);
bb_orient = zeros(1, params.steps);
policy = ones(1, params.steps);
mde = 1;

exit_flag_optimizer = zeros(1, params.steps);

% a_sequence = zeros(params.steps, params.n, 2, params.h_bc+1);
a_sequence = zeros(params.steps, params.n, 2, params.h_bc);

a_ac_traj = zeros(2, params.n, params.steps);

x(1,:) = posi(1,:);
y(1,:) = posi(2,:);
vx(1,:) = veli(1,:);
vy(1,:) = veli(2,:);

decision = false;
bc_counter = 1;
rslt = [];
prev_seq = [];

pos = posi;
vel = veli;
prev_sol = zeros(2 * params.n * params.h_ac, 1);
a_h = 0;

for t = 1:params.steps
    for i = 1:params.n
        v_i_pos = params.sigma_obs_pos * randn(2, 1);
        v_i_vel = params.sigma_obs_vel * randn(2, 1);
        hat_pos(:, i) = pos(:, i) + v_i_pos;
        hat_vel(:, i) = vel(:, i) + v_i_vel;
    end

    if mod(t - 1, controller_run) == 0
        [a_ac, fval, e_flag, prev_sol, history] = controller_cmpc_2d_quadprog_soft(hat_pos, hat_vel, params, opt);
        [next_pos, next_vel] = next_state(hat_pos, hat_vel, a_ac, params);
        [~, ~, ~, ~, ~, a_h] = controller_safety_bb_quadprog_soft(next_pos, next_vel, params, opt);
        [decision, result] = decison_module(hat_pos, hat_vel, params, a_ac, a_h);

        if mde == 1
            if decision
                mde = 2;
                action_number = min(bc_counter, size(prev_seq, 3));
                [acc, prev_seq] = resolve_collision(result, pos, vel, params, prev_seq, a_ac, a_h, action_number, t);
                bc_counter = bc_counter + 1;
                params.switch_step = t;
                result.is_switch = decision;
                result.switch_step = t;
                rslt = [rslt result];
            else
                prev_seq = a_h;
                acc = a_ac;
            end
        elseif mde == 2
            if decision
                action_number = min(bc_counter, size(prev_seq, 3));
                [acc, prev_seq] = resolve_collision(result, pos, vel, params, prev_seq, a_ac, a_h, action_number, t);
                bc_counter = bc_counter + 1;
            else
                mde = 1;
                bc_counter = 1;
                prev_seq = a_h;
                acc = a_ac;
            end
        end
    else
        acc = [ax(t, :); ay(t, :)];
    end

    [pos, vel] = plant_dynamics(pos, vel, acc, params);
    x(t+1,:) = pos(1,:);
    y(t+1,:) = pos(2,:);
    vx(t+1,:) = vel(1,:);
    vy(t+1,:) = vel(2,:);
    ax(t+1,:) = acc(1,:);
    ay(t+1,:) = acc(2,:);
    a_ac_traj(:, :, t+1) = a_ac;
    policy(t) = mde;
    a_sequence(t+1,:,:,:) = a_h;
    mpc_cost(t) = fval;
    exit_flag_optimizer(t) = e_flag;
    %#ok<NASGU>
end

traj.x = x;
traj.y = y;
traj.vx = vx;
traj.vy = vy;
traj.ax = ax(1:params.steps,:);
traj.ay = ay(1:params.steps,:);
traj.a_ac = a_ac_traj;
traj.mpc_cost = mpc_cost;
traj.fitness = [];
traj.a_sequence = a_sequence;
traj.exit_flag = exit_flag_optimizer;
traj.params = params;
traj.bb_sp = bb_sp;
traj.bb_orient = bb_orient;
traj.result = rslt;
traj.policy = policy;
end
