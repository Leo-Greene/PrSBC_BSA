clc;
clear;
close all;

% run_collect_step3
% 必要的运行条件：
%   1. 确保当前工作目录位于项目根目录。
%   2. 需要安装并配置 Parallel Computing Toolbox 以支持并行运行（若 force_parallel=true）。
%   3. 确保 controllers/、decision_module/、dynamics/ 等依赖路径在 MATLAB 路径中。
%   4. 数据将保存至 traj/step3_collect/ 下的唯一时间戳文件夹内。

% 48-case collection plan:
%   1-30: default distribution (seed changes only)
%   31-38: init perturbations (small position/heading/velocity changes)
%   39-44: scene geometry perturbations (diameter/target rotation)
%   45-48: boundary-focused cases (near tighter spacing + tiny control noise)

num_cases = 48;
base_seed = 260318;
stamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
output_root = fullfile('traj', 'step3_collect', stamp);
force_parallel = true;
requested_workers = 0; % 0 means use current/default pool size.

use_parallel = force_parallel && license('test', 'Distrib_Computing_Toolbox');

% Make the perturbation plan deterministic.
rng(base_seed, 'twister');

cases = repmat(struct(), num_cases, 1);

for i = 1:num_cases
    cases(i).case_id = i;
    cases(i).seed = base_seed + i;
    cases(i).save_mat = true;
    cases(i).enable_plot = false; % 采样时不绘图，之后统一调用作图脚本
    cases(i).output_root = output_root;
    cases(i).params_overrides = struct();
end

% --- Step 1: 先定义实验条件分类 (Tags) ---

% 1) Default cases (1-30): no extra changes.
for i = 1:30
    cases(i).tag = 'default';
end

% 2) Init perturbation cases (31-38).
for i = 31:38
    cases(i).tag = 'init_perturb';

    p = struct();
    p.init_theta_offset_deg = -12 + 24 * rand();
    p.init_pos_jitter_std = 0.12;
    p.init_vel_noise_std = 0.05;

    cases(i).params_overrides = p;
end

% 3) Geometry perturbation cases (39-44).
for i = 39:44
    cases(i).tag = 'geometry_perturb';

    p = struct();
    p.diameter = 9.0 + 2.0 * rand();
    p.target_rotation_deg = 170 + 20 * rand();

    cases(i).params_overrides = p;
end

% 4) Boundary-focused cases (45-48).
for i = 45:48
    cases(i).tag = 'boundary_focus';

    p = struct();
    p.diameter = 8.0;
    p.dmin = 1.8;
    p.init_pos_jitter_std = 0.10;
    p.init_theta_offset_deg = -8 + 16 * rand();
    p.control_noise_std = 0.03;

    cases(i).params_overrides = p;
end

% --- Step 2: 实现分层抽样 (Stratified Sampling) 确保 Split 分布均匀 ---

unique_tags = {'default', 'init_perturb', 'geometry_perturb', 'boundary_focus'};
train_ratio = 0.80;
val_ratio = 0.20;

for t = 1:numel(unique_tags)
    tag_name = unique_tags{t};
    % 找到属于当前 tag 的所有索引，并在组内打散
    idx = find(strcmp({cases.tag}, tag_name));
    idx = idx(randperm(numel(idx)));
    n_tag = numel(idx);
    
    % 计算该组内的切分点
    n_train = round(n_tag * train_ratio);
    n_val = round(n_tag * val_ratio);
    if n_train + n_val > n_tag
        n_val = max(0, n_tag - n_train);
    end
    
    % 分配数据子集
    for k = 1:n_tag
        case_idx = idx(k);
        if k <= n_train
            cases(case_idx).split = 'train';
        elseif k <= (n_train + n_val)
            cases(case_idx).split = 'val';
        else
            cases(case_idx).split = 'test';
        end
    end
end

if ~exist(output_root, 'dir')
    mkdir(output_root);
end

manifest = repmat(struct( ...
    'case_id', 0, ...
    'seed', 0, ...
    'split', '', ...
    'tag', '', ...
    'runtime_s', NaN, ...
    'status', '', ...
    'message', '', ...
    'output_file', ''), num_cases, 1);

fprintf('Collection started | parallel=%d | num_cases=%d\n', use_parallel, num_cases);

if use_parallel
    pool = gcp('nocreate');
    if isempty(pool)
        if requested_workers > 0
            parpool(requested_workers);
        else
            parpool;
        end
    end

    manifest_local = repmat(manifest(1), num_cases, 1);
    parfor i = 1:num_cases
        manifest_local(i) = execute_case(cases(i));
    end
    manifest = manifest_local;
else
    for i = 1:num_cases
        fprintf('Running case %d/%d | tag=%s | split=%s\n', i, num_cases, cases(i).tag, cases(i).split);
        manifest(i) = execute_case(cases(i));
    end
end

manifest_mat = fullfile(output_root, 'manifest.mat');
save(manifest_mat, 'manifest', 'cases');

manifest_csv = fullfile(output_root, 'manifest.csv');
T = struct2table(manifest);
writetable(T, manifest_csv);

fprintf('\nCollection finished.\n');
fprintf('Manifest MAT: %s\n', manifest_mat);
fprintf('Manifest CSV: %s\n', manifest_csv);

function row = execute_case(case_cfg)
row = struct( ...
    'case_id', case_cfg.case_id, ...
    'seed', case_cfg.seed, ...
    'split', case_cfg.split, ...
    'tag', case_cfg.tag, ...
    'runtime_s', NaN, ...
    'status', '', ...
    'message', '', ...
    'output_file', '');

try
    [~, info] = run_bb_reverse_once(case_cfg);
    row.runtime_s = info.runtime_s;
    row.status = 'ok';
    row.message = '';
    if isfield(info, 'output_file')
        row.output_file = info.output_file;
    end
catch ME
    row.status = 'failed';
    row.message = ME.message;
end
end
