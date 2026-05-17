function [residual, J_u] = predict_residual_and_jacobian(x_obs, u_des, cfg)
% predict_residual_and_jacobian Run ONNX residual model and optional Jacobian.
%
% Inputs:
%   x_obs: observed state, either 4n vector or 2xn pos/vel stacked as [x;y;vx;vy].
%   u_des: desired action, either 2n vector or 2xn [ax;ay].
%   cfg (optional): struct with fields
%       - onnx_path: path to residual_model.onnx (required if not cached).
%       - scaling_stats_path: path to scaling_stats.json (optional).
%       - stats: struct with X_mean, X_std, U_mean, U_std, R_mean, R_std (optional).
%       - compute_jacobian: true/false (default false).
%       - jacobian_eps: finite difference step (default 1e-3).
%       - output_format: 'vector' (default) or 'matrix'.
%
% Outputs:
%   residual: 4n vector (or 2xn matrix if output_format='matrix').
%   J_u: 4n x 2n Jacobian of residual w.r.t. u_des (empty if not requested).

if nargin < 3
    cfg = struct();
end

if ~isfield(cfg, 'compute_jacobian')
    cfg.compute_jacobian = false;
end
if ~isfield(cfg, 'jacobian_eps')
    cfg.jacobian_eps = 1e-3;
end
if ~isfield(cfg, 'output_format')
    cfg.output_format = 'vector';
end

persistent net_cached stats_cached onnx_cached

if isfield(cfg, 'onnx_path')
    onnx_path = cfg.onnx_path;
else
    onnx_path = '';
end

if isempty(net_cached) || (~isempty(onnx_path) && ~strcmp(onnx_cached, onnx_path))
    if isempty(onnx_path)
        error('predict_residual_and_jacobian:MissingONNX', 'onnx_path is required for first load.');
    end
    net_cached = importONNXNetwork(onnx_path, "InputDataFormats", "BC");
    onnx_cached = onnx_path;
    stats_cached = [];
end

if ~isempty(cfg) && isfield(cfg, 'stats')
    stats_cached = cfg.stats;
elseif ~isempty(cfg) && isfield(cfg, 'scaling_stats_path')
    stats_cached = load_scaling_stats(cfg.scaling_stats_path);
elseif isempty(stats_cached)
    stats_cached = [];
end

[x_vec, u_vec, n] = flatten_inputs(x_obs, u_des);

[x_in, u_in] = maybe_normalize(x_vec, u_vec, stats_cached);
residual_norm = predict(net_cached, {single(x_in), single(u_in)});
residual_vec = maybe_denormalize(residual_norm, stats_cached);

if strcmpi(cfg.output_format, 'matrix')
    residual = reshape_residual(residual_vec, n);
else
    residual = residual_vec;
end

J_u = [];
if cfg.compute_jacobian
    J_u = zeros(numel(residual_vec), numel(u_vec));
    for j = 1:numel(u_vec)
        u_pert = u_vec;
        u_pert(j) = u_pert(j) + cfg.jacobian_eps;
        [x_in_p, u_in_p] = maybe_normalize(x_vec, u_pert, stats_cached);
        r_norm_p = predict(net_cached, {single(x_in_p), single(u_in_p)});
        r_p = maybe_denormalize(r_norm_p, stats_cached);
        J_u(:, j) = (r_p - residual_vec) / cfg.jacobian_eps;
    end
end

end

function stats = load_scaling_stats(json_path)
raw = fileread(json_path);
stats = jsondecode(raw);
end

function [x_vec, u_vec, n] = flatten_inputs(x_obs, u_des)
if isvector(x_obs)
    x_vec = x_obs(:)';
else
    x_vec = [x_obs(1, :), x_obs(2, :), x_obs(3, :), x_obs(4, :)];
end

if isvector(u_des)
    u_vec = u_des(:)';
else
    u_vec = [u_des(1, :), u_des(2, :)];
end

n = numel(x_vec) / 4;
end

function [x_in, u_in] = maybe_normalize(x_vec, u_vec, stats)
if isempty(stats)
    x_in = x_vec; u_in = u_vec; return;
end
x_in = (x_vec - stats.X_mean) ./ stats.X_std;
u_in = (u_vec - stats.U_mean) ./ stats.U_std;
end

function r_vec = maybe_denormalize(r_norm, stats)
if isempty(stats)
    r_vec = double(r_norm);
    return;
end
r_vec = double(r_norm) .* stats.R_std + stats.R_mean;
end

function r_mat = reshape_residual(r_vec, n)
r_mat = [r_vec(1:n); r_vec(n+1:2*n); r_vec(2*n+1:3*n); r_vec(3*n+1:4*n)];
end
