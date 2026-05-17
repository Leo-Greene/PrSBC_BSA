%% compute_PrSBC_constraints
% Computes the constraints for a probabilistic Control Barrier Function (PrSBC) for a group of agents.

function [A_sbc, b_sbc] = compute_PrSBC_constraints(pos, vel, params)
    % pos: 2 x n 当前位置
    % vel: 2 x n 当前速度

    n = params.n;
    h_ac = params.h_ac;
    dim = 2;
    N_vars = dim * n * h_ac; % U 的总长度

    % --- 1. 离散动力学系统矩阵 (半隐式欧拉) ---
    dt = params.dt;
    F_sys = [ 1, 0, dt, 0;  
              0, 1, 0, dt;  
              0, 0, 1, 0;   
              0, 0, 0, 1 ]; 
              
    G_sys = [ dt^2,   0;    
              0,    dt^2;   
              dt,     0;    
              0,     dt  ]; 
    
    % 提取 G 矩阵关于位置的前两行
    G_p = G_sys(1:2, :); 

    % --- 2. 初始化 QP 约束矩阵 ---
    num_constraints = nchoosek(n, 2);
    A_sbc = zeros(num_constraints, N_vars);
    b_sbc = zeros(num_constraints, 1);

    % --- 3. 配置参数 ---
    gamma = params.gamma;
    Confidence = params.confidence;
    
    % 感知截断距离：超过此距离认为绝对安全，不施加防撞约束
    Sensing_Range = params.sensing_range;

    count = 1;
    for i = 1:n-1
        for j = i+1:n
            % 提取当前状态
            p_i = pos(:, i);
            p_j = pos(:, j);
            v_i = vel(:, i);
            v_j = vel(:, j);

            % =========================================================
            % 步骤 0: 感知截断 (Sensing Cutoff) - 性能优化与解耦
            % =========================================================
            dist_current = norm(p_i - p_j);
            
            % 如果两辆车当前相距甚远，直接给一个极大安全裕度并跳过
            if dist_current > Sensing_Range
                A_sbc(count, :) = 0;
                b_sbc(count) = 1e6; % 给一个极大的安全裕度
                count = count + 1;
                continue;
            end

            % 计算当前时刻的安全函数 h(t)
            % 注意：这里 h(t) 的基准依然是未膨胀的物理半径
            h_t = dist_current^2 - params.R_safe^2;

            % =========================================================
            % 步骤 A: 名义动力学预测 (标称状态)
            % =========================================================
            x_i_t = [p_i; v_i];
            x_j_t = [p_j; v_j];

            % 一步名义预测 (假设 u = 0)
            x_i_next_base = F_sys * x_i_t;
            x_j_next_base = F_sys * x_j_t;

            % 提取位置部分
            p_i_next_base = x_i_next_base(1:2);
            p_j_next_base = x_j_next_base(1:2);

            % 预测的相对位置
            p_next_base = p_i_next_base - p_j_next_base;
            norm_p_next = norm(p_next_base);
            
            % 防止重合导致的除零错误
            if norm_p_next < 1e-4
                norm_p_next = 1e-4;
                p_next_base = [1e-4; 0];
            end

            % =========================================================
            % 步骤 B: 噪声参数与概率边界计算
            % =========================================================
            % sigma_obs = params.sigma_obs_pos;  
            % epsilon_w = params.epsilon_w_pos;  

            % sigma_total_sq = 2 * sigma_obs^2 + epsilon_w^2;
            % sigma_total = sqrt(sigma_total_sq);

            % % 瑞利分布逆函数：计算出最坏情况下的概率边界半径
            % epsilon_total = sigma_total * sqrt(-2 * log(1 - Confidence));
            sigma_obs = params.sigma_obs_pos; 
            sigma_vel = params.sigma_obs_vel; 
            epsilon_w = params.epsilon_w_pos;  

            % 两个独立高斯分布相减，方差相加
            sigma_total_sq = 2 * (sigma_obs^2 + (sigma_vel * dt)^2 + epsilon_w^2);
            sigma_total = sqrt(sigma_total_sq);

            % --- 破除保守性：1D 单边高斯投影 ---
            % 我们只关心噪声将两车"推近"的那一个维度的分量 (1D Gaussian)
            % 使用标准正态分布的逆函数计算单边置信度 (One-sided confidence interval)
            % erfinv(2*p - 1) * sqrt(2) 是 MATLAB 中计算正态分位数的标准写法，无需统计工具箱
            % 如果 Confidence = 0.95， quantile_1D 约为 1.645 (远小于原先的 2.447)
            quantile_1D = sqrt(2) * erfinv(2 * Confidence - 1);
            
            epsilon_total = sigma_total * quantile_1D;

            % ---------------------------------------------------------
            % (原有的 noise_penalty 柯西-施瓦茨惩罚被移除)
            % noise_penalty = 2 * norm_p_next * epsilon_total;
            % ---------------------------------------------------------

            % =========================================================
            % 步骤 D: 构造 A 矩阵和 b 向量 (三角不等式修正)
            % =========================================================
            idx_i = (2*i - 1) : (2*i);
            idx_j = (2*j - 1) : (2*j);

            % A 矩阵系数：利用名义预测方向投影
            a_coeff = -2 * p_next_base' * G_p; 
            
            A_sbc(count, idx_i) = a_coeff;
            A_sbc(count, idx_j) = -a_coeff; 

            % b 向量：利用三角不等式，直接将 epsilon_total 融合进安全半径中
            % 这是优雅的闵可夫斯基和膨胀 (Minkowski Sum Expansion)
            b_val = norm_p_next^2 - (params.R_safe + epsilon_total)^2 - (1 - gamma) * h_t;
            b_sbc(count) = b_val;
            
            count = count + 1;
        end
    end
end