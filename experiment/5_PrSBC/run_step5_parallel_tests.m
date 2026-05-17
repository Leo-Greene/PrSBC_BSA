%% PrSBC Step 5: 并行批处理测试脚本
% 作用：并行运行带有 PrSBC 过滤器和不带过滤器的 Simplex 架构，对比其安全性与性能。

clc; clear; close all;
num_runs = 16; % 设置测试轮数，建议根据 CPU 核心数调整
project_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));

% 定义路径
traj_dir = fullfile(project_root, 'traj', 'step5_prsbc');
result_dir = fullfile(project_root, 'experiment', '5_PrSBC', 'result');
if ~exist(traj_dir, 'dir'), mkdir(traj_dir); end
if ~exist(result_dir, 'dir'), mkdir(result_dir); end

run_stamp = datestr(now, 'yyyy-mm-dd_HHMMSS');
run_traj_dir = fullfile(traj_dir, run_stamp);
if ~exist(run_traj_dir, 'dir')
    mkdir(run_traj_dir);
end

% 添加路径依赖
addpath(genpath(fullfile(project_root, 'experiment', '5_PrSBC', 'example')));
addpath(genpath(fullfile(project_root, 'common')));

% 初始化统计变量
stats_none = struct('collisions', 0, 'min_dist', [], 'cost', []);
stats_prsbc = struct('collisions', 0, 'min_dist', [], 'cost', []);

fprintf('开始并行测试，总轮数: %d...\n', num_runs);
tStart = tic;

% 启动并行池 (如果未启动)
if isempty(gcp('nocreate')), parpool; end

% 并行循环
results = cell(num_runs, 2); % 存储每轮的结果：[None_Result, PrSBC_Result]

parfor i = 1:num_runs
    fprintf('正在运行第 %d 轮测试...\n', i);
    
    % --- 1. 运行无过滤器版本 (None) ---
    % 注意：为了并行，这里需要确保 main_bb_reverse 能作为函数调用并返回 traj
    % 假设我们将脚本修改为了函数格式：[traj] = main_bb_reverse_func(seed)
    res_none = run_simulation_instance('none', i);
    
    % --- 2. 运行 PrSBC 过滤器版本 ---
    res_prsbc = run_simulation_instance('prsbc', i);
    
    % 保存轨迹文件 (参考 Step 3/4 命名规则)
    save_name_none = fullfile(run_traj_dir, sprintf('traj_step5_none_run%d.mat', i));
    save_name_prsbc = fullfile(run_traj_dir, sprintf('traj_step5_prsbc_run%d.mat', i));
    
    save_parfor(save_name_none, res_none);
    save_parfor(save_name_prsbc, res_prsbc);
    
    results(i, :) = {res_none, res_prsbc};
end

tEnd = toc(tStart);
fprintf('测试完成！总耗时: %.2f 秒。\n', tEnd);

%% 2. 处理统计数据并生成报告
[report_text, summary_data] = analyze_results(results, num_runs);

% 输出到文件
report_file = fullfile(result_dir, ['Test_Report_', run_stamp, '.txt']);
fid = fopen(report_file, 'w');
if fid < 0
    error('无法创建报告文件: %s', report_file);
end
fprintf(fid, '%s', report_text);
fclose(fid);

fprintf('报告已生成至: %s\n', report_file);

%% 辅助函数：保存数据 (因为 parfor 内不能直接 save)
function save_parfor(fname, data)
    save(fname, 'data');
end

%% 辅助函数：运行单次仿真实例
function traj = run_simulation_instance(type, seed)
    % 这里需要根据你的 example/main_*.m 内容进行微调
    % 建议将 main 脚本改写成接受 seed 并关闭图形显示的函数
    if strcmp(type, 'none')
        traj = main_bb_reverse_as_func(seed); 
    else
        traj = main_prsbc_bb_reverse_as_func(seed);
    end
end