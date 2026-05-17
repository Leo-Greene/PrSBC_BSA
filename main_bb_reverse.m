clc
clear
clear functions
close all
rng(42); % For reproducibility of random initial conditions

project_root = fileparts(mfilename('fullpath'));

%% Dependencies
% root = '/Users/Usama/UmehmoodGoogle/Work/Code/';

% addpath(genpath(fullfile(project_root, 'controllers', 'ac', 'controller_cmpc_2d')));
% addpath(genpath(fullfile(project_root, 'controllers', 'bc', 'safety_controller')));
addpath(genpath(fullfile(project_root, 'controllers', 'ac', 'controller_cmpc_2d_quadprog')));
addpath(genpath(fullfile(project_root, 'controllers', 'bc', 'safety_controller_quadprog')));
addpath(genpath(fullfile(project_root, 'controllers', 'PrSBC_filter')));
addpath(genpath(fullfile(project_root, 'decision_module')));
addpath(genpath(fullfile(project_root, 'extended_BBS')));
addpath(fullfile(project_root, 'common'));
addpath(genpath(fullfile(project_root, 'experiment', 'dynamics')));
addpath(genpath(fullfile(project_root, 'experiment', 'utilities')));


% addpath([root 'Swarms New/m-functions']);

%% params

params.n = 15;
params.dt = 0.3;
params.ct = 0.3;
params.h_ac = 10;
params.h_bc = 10;
params.steps = 80;

params.amax = 1.5; %1.5
params.a_max = params.amax;
params.vmax = 2;%2
params.v_max = params.vmax;

% Safety property
params.dmin = 1.7;
params.R_safe = 1.9;

% inital configuration
params.diameter = 15;
params.switch_step = 1;

params.ws = 10000; %1800
params.wt = 10;

params.ws_bb = 3000; %1800
params.w_orient = 20;

% Unmodeled params in true_dynamics
params.acc_scale = 0.95;
params.acc_bias = 0;
params.damping = 0.03;
params.nonlinear_drift = false;
params.noise_std = 0;

% Process noise parameters
params.epsilon_w_pos = 0.2*params.vmax*params.dt; % 位置的过程噪声边界 (米)：最大速度下百分之五的误差
params.sigma_obs_pos = params.epsilon_w_pos / 3;  % 假设 epsilon_w_pos 对应 3-sigma 边界 (99.7% 置信度)

% Sensor noise parameters: 满足高斯噪声分布，通过估计得到的最大噪声边界估算得到高斯分布标准差
params.epsilon_v_pos = 0.2*params.dmin; % 观测位置噪声边界 (米)：机器人最小距离百分之五的误差
params.sigma_obs_pos = params.epsilon_v_pos / 3;  % 假设 epsilon_v_pos 对应 3-sigma 边界 (99.7% 置信度)
params.epsilon_v_vel = 0.2*params.vmax; % 观测速度噪声标准差 (米/秒)：最大速度的百分之五
params.sigma_obs_vel = params.epsilon_v_vel / 3;  % 假设 epsilon_v_vel 对应 3-sigma 边界 (99.7% 置信度)

% DM安全检查参数
params.confidence = 0.9;
params.gamma = 0.4;
params.sensing_range = 4.0; % DM的感知截断距离：超过此距离认为绝对安全，不施加防碰撞检查

%% Predator params
params.predator = 0;
params.pFactor = 1.40;
params.pred_radius = 6;
params.wp = 500;

%% Results structure
result.is_switch = false;

%% Legacy fmincon Optimizer Settings
% opt = optimoptions('fmincon');
% opt.Display = 'off';
% % opt.Algorithm = 'active-set';
% opt.MaxIterations = 8000;
% opt.MaxFunctionEvaluations = 12000;
% % opt.StepTolerance = 1e-12;
% % Stopping criteria if change in function value is lesser than this:
% opt.FunctionTolerance = 1e-7;

%% quadprog Optimizer Settings
opt = optimoptions('quadprog');
opt.Algorithm = 'interior-point-convex'; % 显式指定内点法，适合大规模约束
opt.Display = 'off';                     % 关闭输出，加快仿真速度

% 容差设置：
% 适当放宽容差可以显著提升在软约束环境下的求解速度，避免数值抖动
opt.OptimalityTolerance = 1e-6; 
opt.ConstraintTolerance = 1e-6; % 关键：配合 1e6 的 rho 惩罚项，防止因数值精度报错
opt.StepTolerance = 1e-10;

% 性能与稳定性：
% opt.MaxIterations = 8000;       % 2000 次迭代足够内点法收敛
opt.LinearSolver = 'sparse';    % 启用稀疏矩阵求解器，提升计算效率
% opt.FunValCheck = 'on';       % 调试时可开启，正式运行建议关闭以提升性能



%%
% u = zeros(2*params.n*params.h + 1,1);
indexes = 1:params.n;
acc = zeros(2, params.n);
controller_run = params.ct / params.dt; %
zero_vec = zeros(params.n , 1) ;
params.switch_step = params.steps;

cpu_time = 0;
function_calls = 0;
tic;

%% Initial configuration

[posi, veli, params.tgt] = gen_init_bb(params);

% scatter(posi(1,:), posi(2,:),'.', 'MarkerEdgeColor', 'r', 'SizeData', 200);
% hold on
% scatter(params.tgt(1,:), params.tgt(2,:), 'x', 'MarkerEdgeColor', 'b', 'SizeData', 100);
% axis equal
%
% run controller_cmpc_2d/common/create_mex.m;
% run safety_controller/common/create_mex.m;

%%
x = zeros([params.steps, params.n]);
y = zeros([params.steps, params.n]);
vx = zeros([params.steps, params.n]);
vy = zeros([params.steps, params.n]);
ax = zeros([params.steps+1, params.n]);
ay = zeros([params.steps+1, params.n]);
bd = zeros([params.n, params.steps, params.n]);
mpc_cost = zeros(1, params.steps);
% f = zeros(1, params.steps);
bb_sp = zeros(1, params.steps);
bb_orient = zeros(1, params.steps);
policy = ones(1, params.steps);
mde = 1;

exit_flag_optimizer = zeros(1, params.steps);

% 暂时不考虑让最后一步指令为零，检验决策模块和安全控制器的功能
% a_sequence = zeros(params.steps, params.n, 2, params.h_bc+1);
a_sequence = zeros(params.steps, params.n, 2, params.h_bc);

a_ac_traj = zeros(2, params.n, params.steps);

x(1,:) = posi(1,:);
y(1,:) = posi(2,:);
vx(1,:) = veli(1,:);
vy(1,:) = veli(2,:);
% f(1) = fitness(posi, params);
decision = false;
bc_counter = 1;
rslt = [];
prev_seq = []; % 初始化以防第一步就切换到 BC 模式导致报错
%% Controller and Dynamics:
pos = posi; vel = veli;
prev_sol = zeros(2*params.n*params.h_ac,1);
a_h = 0;
tStart = tic;
for t = 1:params.steps  % 1) run optimizer 2) update instruction 3) run dynamics
    if mod(t,5) == 0
        e = round(toc, 1);
        disp(['step: ' num2str(t) '/' num2str(params.steps) ', Time:' num2str(e) 's']);
    end

    % Sensor observation: controllers and decision module see noisy states.
    for i = 1:params.n
        % --- 生成 2D 高斯位置噪声 v_i ---
        % randn(2,1) 会直接生成 2x1 的标准正态分布 [N(0,1); N(0,1)]
        % 乘以 sigma_obs 后，就变成了协方差矩阵为 sigma_obs^2 * I 的 2D 高斯噪声！
        v_i_pos = params.sigma_obs_pos * randn(2, 1); 
        v_i_vel = params.sigma_obs_vel * randn(2, 1);
        
        % 将观测噪声叠加到真实位置上，形成观测位置
        hat_pos(:, i) = pos(:, i) + v_i_pos;
        hat_vel(:, i) = vel(:, i) + v_i_vel;
    end
    % % 暂时不考虑噪声，检验决策模块和安全控制器的功能
    % hat_pos = pos;
    % hat_vel = vel;
    
    if mod(t - 1, controller_run) == 0  % run optimizer every controller_run steps
        
        %[a_ac, fval, e_flag, prev_sol, history] = controller_cmpc_2d(hat_pos, hat_vel, params, opt);
        % [a_ac, fval, e_flag, prev_sol, history] = controller_cmpc_2d_quadprog(hat_pos, hat_vel, params, opt);
        [a_ac, fval, e_flag, prev_sol, history] = controller_cmpc_2d_quadprog_soft(hat_pos, hat_vel, params, opt);

        [next_pos, next_vel] = next_state(hat_pos, hat_vel, a_ac, params);
        % [~, ~, ~, ~, ~, a_h] = controller_safety_bb(next_pos, next_vel, params, opt);
        % [~, ~, ~, ~,~, a_h] = controller_safety_bb_quadprog(next_pos, next_vel, params, opt);
        [~, ~, ~, ~, ~, a_h] = controller_safety_bb_quadprog_soft(next_pos, next_vel, params, opt);
        
        [decision, result] = decison_module(hat_pos, hat_vel, params, a_ac, a_h);
        if decision
            if isfield(result, 'cause') && isfield(result, 'pair')
                disp(['[DM] Decision rejected at step ' num2str(t) ', cause=' num2str(result.cause) ', pair=' mat2str(result.pair)]);
            else
                disp(['[DM] Decision rejected at step ' num2str(t) ', result is incomplete.']);
            end
        end
%         display(result)

        % prev_seq = a_h; % by sanaz; in some instances the first reference to prev_seq it is empty

        if mde == 1 % controlled by AC
            if decision % switch to BC: 1) update instruction 2) record switch

                mde = 2;
                action_number = min(bc_counter, size(prev_seq, 3)); % by lzj: saved instructions shouldn't be empty. If it is empty, run the program again

                %acc = prev_seq(:,:,action_number);                    
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
                
                %acc = prev_seq(:,:,action_number)';
                [acc, prev_seq] = resolve_collision(result, pos, vel, params, prev_seq, a_ac, a_h, action_number, t);

                bc_counter = bc_counter + 1;

                % if bc_counter > size(prev_seq, 3)
                %     warning('BC counter exceeded the length of the previous sequence. The acceleration command will be set to zero to avoid index out of bounds. This may indicate that the decision module is not resolving the collision properly.');
                %     acc = zeros(2, params.n);
                % else
                %     action_number = min(bc_counter, size(prev_seq, 3));
                
                %     %acc = prev_seq(:,:,action_number)';
                %     [acc, prev_seq] = resolve_collision(result, pos, vel, params, prev_seq, a_ac, a_h, action_number, t);
                % end
                
                % bc_counter = bc_counter + 1;

            else
                mde = 1;
                bc_counter = 1;
                prev_seq = a_h;
                acc = a_ac;
            end
        end

%         if ~decision
%             
%             if decision
%                 params.switch_step = t;
%                 result.is_switch = decision;
%                 result.switch_step = t;
%             end
%             if ~decision
%                 prev_seq = a_h;
%                 acc = a_ac;
%             end
%         end
%         
%         if decision
%             action_number = min(bc_counter, size(prev_seq, 3));
%             acc = prev_seq(:,:,action_number)';
%             bc_counter = bc_counter + 1;
%         end
    else
        acc = [ax(t, :); ay(t, :)];
    end

    % % 1. 将 DM (船长) 选出的名义指令作为期望目标传入 params
    % params.u_cmd = acc; 
    
    % % 2. 召唤舵手：在满足绝对安全的空间内，寻找离 u_cmd 最近的动作
    % % 注意这里调用的是我们上一轮写好的单步预测过滤器
    % [acc_safe, ~, filter_exit_flag] = prcbc_filter(pos, vel, params, []);
    
    % % 3. 结果核验与赋值
    % if filter_exit_flag >= 0
    %     acc = acc_safe; % 正常微调，应用安全的指令
    % else
    %     warning('第 %d 步 PrSBC Filter 求解失败! 维持原指令或采取紧急刹车.', t);
    %     % 此处如果无解(由于软约束存在，几乎不可能发生)，你可以选择保持原 acc，或者 acc = zeros(2, n);
    % end
    %[pos, vel] = stochastic_dynamics(pos, vel, acc, params, 0.02, 0.05);
    %[pos, vel] = true_dynamics(pos, vel, acc, params);
    % [pos, vel] = dynamics(pos, vel, acc, params);
    [pos, vel] = plant_dynamics(pos, vel, acc, params);
    x(t+1,:) = pos(1,:);
    y(t+1,:) = pos(2,:);
    vx(t+1,:) = vel(1,:);
    vy(t+1,:) = vel(2,:);
    ax(t+1,:) = acc(1,:);
    ay(t+1,:) = acc(2,:);
    a_ac_traj(:, :, t+1) = a_ac;
    policy(t) = mde;
    %     if t >= params.switch_step
    a_sequence(t+1,:,:,:) = a_h;
    %     end
    mpc_cost(t) = fval;
    exit_flag_optimizer(t) = e_flag;
    % f(t+1) = fitness(pos, params);
    % [~, bb_sp(t+1), bb_orient(t+1)] = fitness_bb(pos, vel, params);
end
tEnd = toc(tStart);
display('Runtime: ');
display(tEnd);

%% Store output.
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

displayTraj(x,y,vx,vy,policy); title('Black-Box Simplex', 'FontSize', 17);
set(gcf, 'Position', get(0, 'Screensize'));
% saveas(gcf, ['Images/trajComparison_' num2str(i) '.jpg']);


%% Save output

date_string = datestr(datetime,' [yyyy-mm-dd]');
out_traj_name = ['traj_bb'  date_string];

dir_path = 'traj/';

[dest_path, out_traj_name] = create_dir(dir_path, out_traj_name);
mkdir([dest_path '/Results']);

fprintf(['\nOUTPUT:\n' dest_path '\n']);
fprintf(['switch: ' num2str(params.switch_step) '\n']);

save([dest_path '/' out_traj_name '.mat'], 'traj');

for ii = 1:numel(rslt)
    disp(result_message(rslt(ii), params.dt));
end

for ii = 1:numel(traj.result)
    disp(result_message(traj.result(ii), params.dt));
end
%%
figure;
plot(traj.bb_orient)





