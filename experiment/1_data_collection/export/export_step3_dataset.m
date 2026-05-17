clc;
clear;
close all;

% export_step3_dataset
% 必要的运行条件：
%   1. 需要先运行 run_collect_step3.m 并成功生成数据。
%   2. 脚本将自动在 traj/step3_collect/ 下搜索最新的 manifest.mat 文件。
%   3. 导出的数据文件将保存在该采集文件夹中。
%   4. 输出文件包括 dataset_all.mat, dataset_train.mat, dataset_val.mat, dataset_test.mat。

% Export transitions for deterministic residual learning:
%   Input  : (x_k, u_k)
%   Target : residual_k = x_true_{k+1} - f_nominal(x_k, u_k)
%
% Output MAT files under output_root:
%   - dataset_all_*.mat
%   - dataset_train_*.mat
%   - dataset_val_*.mat
%   - dataset_test_*.mat

addpath(fullfile(pwd, 'common'));
addpath(fullfile(pwd, '..', 'dynamics'));

output_root = fullfile('traj', 'step3_collect');

% 自动逻辑：寻找最新的单个文件夹
files = dir(fullfile(output_root, '**', 'manifest.mat'));
if isempty(files)
    error('No manifest files found under %s. Please run run_collect_step3.m first.', output_root);
end

% 按日期排序
[~, idx] = max([files.datenum]);
manifest_mat = fullfile(files(idx).folder, files(idx).name);
[export_output_root, ~, ~] = fileparts(manifest_mat);

fprintf('--> Using latest manifest: %s\n', manifest_mat);

S = load(manifest_mat);
if ~isfield(S, 'manifest')
    error('Manifest file does not contain variable ''manifest''.');
end
manifest = S.manifest;

X = [];
U = [];
X_next_true = [];
X_next_nominal = [];
R_label = [];
case_id = [];
step_id = [];
episode_id = [];
is_BC_active = [];
split = {};
tag = {};

for i = 1:numel(manifest)
    if ~strcmp(manifest(i).status, 'ok'), continue; end
    
    data_file = manifest(i).output_file;
    if ~exist(data_file, 'file')
        % 尝试解决相对路径问题
        [m_dir, ~, ~] = fileparts(manifest_mat);
        [~, d_name, d_ext] = fileparts(data_file);
        data_file = fullfile(m_dir, [d_name, d_ext]);
    end

    if ~exist(data_file, 'file')
        warning('Skip case %d: output file missing.', manifest(i).case_id);
        continue;
    end

    T = load(data_file);
    if ~isfield(T, 'traj'), continue; end
    traj = T.traj;

    [Xi, Ui, Xni_true, Xni_nom, Ri, cid, sid, eid, bc] = trajectory_to_transitions(traj, manifest(i).case_id);

    n_i = size(Xi, 1);
    X = [X; Xi];
    U = [U; Ui];
    X_next_true = [X_next_true; Xni_true];
    X_next_nominal = [X_next_nominal; Xni_nom];
    R_label = [R_label; Ri];
    case_id = [case_id; cid];
    step_id = [step_id; sid];
    episode_id = [episode_id; eid];
    is_BC_active = [is_BC_active; bc];
    split = [split; repmat({manifest(i).split}, n_i, 1)];
    tag = [tag; repmat({manifest(i).tag}, n_i, 1)];
end

dataset = struct();
dataset.X = X;
dataset.U = U;
dataset.X_next_true = X_next_true;
dataset.X_next_nominal = X_next_nominal;
dataset.R_label = R_label;
dataset.case_id = case_id;
dataset.step_id = step_id;
dataset.episode_id = episode_id;
dataset.is_BC_active = is_BC_active;
dataset.split = split;
dataset.tag = tag;
dataset.feature_layout.state = '[x_obs(1..n), y_obs(1..n), vx_obs(1..n), vy_obs(1..n)]';
dataset.feature_layout.action = '[ax(1..n), ay(1..n)]';

% 保存文件
all_file = fullfile(export_output_root, 'dataset_all.mat');
save(all_file, 'dataset', '-v7.3');

save_split(dataset, 'train', export_output_root);
save_split(dataset, 'val', export_output_root);
save_split(dataset, 'test', export_output_root);

fprintf('\nExport finished.\n');
fprintf('All dataset: %s\n', all_file);
fprintf('Samples total: %d\n', size(dataset.X, 1));

function [X, U, X_next_true, X_next_nominal, R, cid, sid, eid, bc] = trajectory_to_transitions(traj, case_id_value)
params = traj.params;
if isfield(traj, 'x_obs')
    x_obs = traj.x_obs; y_obs = traj.y_obs; vx_obs = traj.vx_obs; vy_obs = traj.vy_obs;
else
    x_obs = traj.x; y_obs = traj.y; vx_obs = traj.vx; vy_obs = traj.vy;
end
x = traj.x; y = traj.y; vx = traj.vx; vy = traj.vy;

if isfield(traj, 'ax_applied') && isfield(traj, 'ay_applied')
    ax = traj.ax_applied; ay = traj.ay_applied;
else
    ax = traj.ax; ay = traj.ay;
end

T = min([size(x, 1) - 1, size(ax, 1)]);

X = zeros(T, 4 * params.n);
U = zeros(T, 2 * params.n);
X_next_true = zeros(T, 4 * params.n);
X_next_nominal = zeros(T, 4 * params.n);
R = zeros(T, 4 * params.n);

for k = 1:T
    pos_k_obs = [x_obs(k, :); y_obs(k, :)]; vel_k_obs = [vx_obs(k, :); vy_obs(k, :)];
    pos_k = [x(k, :); y(k, :)]; vel_k = [vx(k, :); vy(k, :)]; acc_k = [ax(k, :); ay(k, :)];
    pos_k1_true = [x(k + 1, :); y(k + 1, :)]; vel_k1_true = [vx(k + 1, :); vy(k + 1, :)];
    [pos_k1_nom, vel_k1_nom] = dynamics(pos_k, vel_k, acc_k, params);
    
    X(k, :) = [pos_k_obs(1, :), pos_k_obs(2, :), vel_k_obs(1, :), vel_k_obs(2, :)];
    U(k, :) = [acc_k(1, :), acc_k(2, :)];
    X_next_true(k, :) = [pos_k1_true(1, :), pos_k1_true(2, :), vel_k1_true(1, :), vel_k1_true(2, :)];
    X_next_nominal(k, :) = [pos_k1_nom(1, :), pos_k1_nom(2, :), vel_k1_nom(1, :), vel_k1_nom(2, :)];
    R(k, :) = X_next_true(k, :) - X_next_nominal(k, :);
end
cid = case_id_value * ones(T, 1); sid = (1:T)';
if isfield(traj, 'episode_id')
    eid = traj.episode_id(1:T)';
else
    eid = cid;
end
if isfield(traj, 'is_BC_active')
    bc = traj.is_BC_active(1:T)';
else
    bc = false(T, 1);
end
end

function save_split(dataset, split_name, output_root)
mask = strcmp(dataset.split, split_name);

split_ds = struct();
split_ds.X = dataset.X(mask, :);
split_ds.U = dataset.U(mask, :);
split_ds.X_next_true = dataset.X_next_true(mask, :);
split_ds.X_next_nominal = dataset.X_next_nominal(mask, :);
split_ds.R_label = dataset.R_label(mask, :);
split_ds.case_id = dataset.case_id(mask, :);
split_ds.step_id = dataset.step_id(mask, :);
split_ds.episode_id = dataset.episode_id(mask, :);
split_ds.is_BC_active = dataset.is_BC_active(mask, :);
split_ds.split = dataset.split(mask, :);
split_ds.tag = dataset.tag(mask, :);
split_ds.feature_layout = dataset.feature_layout;

out_file = fullfile(output_root, ['dataset_' split_name '.mat']);
save(out_file, 'split_ds', '-v7.3');

fprintf('%s samples: %d -> %s\n', split_name, size(split_ds.X, 1), out_file);
end
