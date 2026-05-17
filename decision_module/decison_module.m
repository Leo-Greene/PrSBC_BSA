% function [decision, result] = decison_module(pos, vel, params, a_ac, a_seq)

%     decision = false;
%     result = [];
%     tol = 1e-3; % 允许极小的数值误差计算宽容度

%     % =========================================================================
%     % 0. 提取当前状态下的 PrSBC 安全约束
%     % =========================================================================
%     [A_sbc, b_sbc] = compute_PrSBC_constraints(pos, vel, params);
    
%     % 为了适配 A_sbc 的维度 (dim * n * h_ac)，需要对单步指令 a_ac 进行补零扩展
%     N_vars = 2 * params.n * params.h_ac;
%     u_ac_vec = zeros(N_vars, 1);
%     u_ac_vec(1 : 2*params.n) = a_ac(:); % 仅把第1步指令填入头部

%     % =========================================================================
%     % 1. One step sanity check for AC using PrSBC (概率安全验证)
%     % =========================================================================
%     % 计算约束违背量：如果 A*u - b > 0，说明违背了概率安全证书
%     violations_ac = A_sbc * u_ac_vec - b_sbc;
    
%     if any(violations_ac > tol)
%         disp('[DM] [AC PrSBC Violation] AC command rejected due to probabilistic noise boundary.');
        
%         % === Pro Fix: 精准定位最危险的机器人对 ===
%         [max_violation, max_idx] = max(violations_ac); % 找到违规最严重的一行
%         [r, c] = get_agent_pair_from_row(max_idx, params.n); % 反推机器人编号
        
%         % 计算当前距离、相对速度和相对加速度
%         dist_approx = norm(pos(:, r) - pos(:, c));
%         rel_v = vel(:, r) - vel(:, c);
%         rel_a = a_ac(:, r) - a_ac(:, c);
        
%         fprintf('[DM] [AC PrSBC Violation] Info: Agents [%d, %d], Dist: %.4f, RelV: [%.4f, %.4f], RelA: [%.4f, %.4f]\n', ...
%             r, c, dist_approx, rel_v(1), rel_v(2), rel_a(1), rel_a(2));
            
%         decision = true;
%         result.cause = 1;
        
%         result.pair = [r, c, dist_approx];
%         return;
%     end

%     % =========================================================================
%     % 2. m step sanity check for BC sequence
%     % =========================================================================
%     seq_len = size(a_seq, 3);
%     current_pos = pos;
%     current_vel = vel;

%     for i = 1:seq_len
%         a_bc_step = a_seq(:,:,i)';

%         if i == 1
%             % --- 第 1 步：使用高精度的 PrSBC 检查 BC 的起步指令 ---
%             u_bc_vec = zeros(N_vars, 1);
%             u_bc_vec(1 : 2*params.n) = a_bc_step(:);
%             violations_bc = A_sbc * u_bc_vec - b_sbc;

%             if any(violations_bc > tol)
%                 disp('[DM] [BC PrSBC Violation] BC command sequence rejected at step 1.');
%                 decision = true;
%                 result.cause = 2;
                
%                 % === Pro Fix: 精准定位最危险的机器人对 ===
%                 [max_violation, max_idx] = max(violations_bc);
%                 [r, c] = get_agent_pair_from_row(max_idx, params.n);
                
%                 dist_approx = norm(pos(:, r) - pos(:, c));
                
%                 % 计算相对速度和相对加速度
%                 rel_v = vel(:, r) - vel(:, c);
%                 rel_a = a_bc_step(:, r) - a_bc_step(:, c);
                
%                 fprintf('[DM] [BC PrSBC Violation] Info: Agents [%d, %d], Dist: %.4f, RelV: [%.4f, %.4f] (norm: %.4f), RelA: [%.4f, %.4f]\n', ...
%                     r, c, dist_approx, rel_v(1), rel_v(2), norm(rel_v), rel_a(1), rel_a(2));
                
%                 result.pair = [r, c, dist_approx, i];
%                 return;
%             end
            
%             % 更新下一时刻的标称状态
%             [~, current_pos, current_vel, ~, ~, ~] = check_next_state(current_pos, current_vel, a_bc_step, params);
            
%         else
%             % --- 第 2~m 步：使用确定性预测 + 膨胀安全边界 (Tube-based Expansion) ---
%             % 膨胀量 = i * 过程噪声边界
%             margin = i * params.epsilon_w_pos; 
%             inflated_dmin = params.dmin + margin;

%             temp_params = params;
%             temp_params.dmin = inflated_dmin;

%             [is_collision, current_pos, current_vel, r, c, d] = check_next_state(current_pos, current_vel, a_bc_step, temp_params);
            
%             if is_collision
%                 % 计算碰撞时刻的相对运动状态
%                 rel_v = current_vel(:, r(1)) - current_vel(:, c(1));
%                 rel_a = a_bc_step(:, r(1)) - a_bc_step(:, c(1));
                
%                 fprintf('[DM] [BC Future Collision] Info: Agents [%d, %d] at step %d, Dist: %.4f, RelV: [%.4f, %.4f] (norm: %.4f), RelA: [%.4f, %.4f]\n', ...
%                     r(1), c(1), i, d, rel_v(1), rel_v(2), norm(rel_v), rel_a(1), rel_a(2));
                
%                 disp(['[DM] [BC Future Collision] Collision detected at step ' num2str(i) ' with inflated safe distance ' num2str(d)]);
%                 decision = true;
%                 result.cause = 2;
%                 result.pair = [r(1), c(1), d, i];
%                 return;
%             end
%         end
%     end

%     % =========================================================================
%     % 3. Divergence check for the final state
%     % =========================================================================
%     [is_converging, r, c] = check_divergence_simple(current_pos, current_vel);
%     if ~is_converging
%        disp(['[DM] Soft prompt: final velocities are not diverging between agents ' num2str(r(1)) ' and ' num2str(c(1))]);
%     end

%     % % =========================================================================
%     % % 3. Divergence & Terminal Invariant Set Check
%     % % =========================================================================
%     % % 获取预测终点的最大速度（绝对值）
%     % max_vel_at_end = max(abs(current_vel(:))); 
    
%     % % 如果所有车辆速度都接近于 0，说明成功进入了无穷期安全的“控制不变集”
%     % if max_vel_at_end < 1e-2
%     %     disp('[DM] 🟢 绝对安全保证达成！车辆序列已成功规划至安全刹停状态。');
%     % else
%     %     % 如果还没刹停（比如采用了 AC 的指令），则继续检查是否在相互发散
%     %     [is_converging, r, c] = check_divergence_simple(current_pos, current_vel);
%     %     if ~is_converging
%     %        disp(['[DM] 🟡 Soft prompt: final velocities are not diverging between agents ' num2str(r(1)) ' and ' num2str(c(1))]);
%     %     end
%     % end
% end

% % =========================================================================
% % 辅助函数：根据 A_sbc 的行号，反推是哪两个机器人 (i, j) 产生了约束
% % =========================================================================
% function [agent_i, agent_j] = get_agent_pair_from_row(row_idx, n)
%     % A_sbc 的行是按照 i=1...n-1, j=i+1...n 的两层循环生成的组合数排列的
%     count = 1;
%     for i = 1:n-1
%         for j = i+1:n
%             if count == row_idx
%                 agent_i = i;
%                 agent_j = j;
%                 return;
%             end
%             count = count + 1;
%         end
%     end
%     % Fallback 保护，理论上不会走到这里
%     agent_i = 1;
%     agent_j = 2; 
% end




function [decision, result] = decison_module(pos, vel, params, a_ac, a_seq)

decision = false;
result = [];

%% One step sanity check for advanced controller command
[is_collision, next_pos, next_vel, r, c, d] = check_next_state(pos, vel, a_ac, params);
if is_collision
    disp(['[DM] [AC collision] Immediate collision detected between agents ' num2str(r(1)) ' and ' num2str(c(1)) ' with distance ' num2str(d)]);
    decision = is_collision;
    result.cause = 1;
    result.pair = [r(1), c(1), d];
    return;
end

%% m step sanity check for basleine controller command sequence
seq_len = size(a_seq, 3);
for i = 1:seq_len
    [is_collision, next_pos, next_vel, r, c, d] = check_next_state(next_pos, next_vel, a_seq(:,:,i)', params);
    if is_collision
        % disp(['[DM] [BC collision] Collision detected at step ' num2str(i) ' between agents ' num2str(r(1)) ' and ' num2str(c(1)) ' with distance ' num2str(d)]);
        decision = is_collision;
        result.cause = 2;
        result.pair = [r(1), c(1), d, i];
        return;
    end
end

%% divergence check for the final state
[is_converging, r, c] = check_divergence_simple(next_pos, next_vel);
if is_converging
    % disp(['[DM] Soft prompt: final velocities are not diverging between agents ' num2str(r(1)) ' and ' num2str(c(1))]);
    % Soft prompt only: keep the divergence warning for diagnostics,
    % but do not hard-reject the AC command sequence here.
    % decision = is_converging;
    % result.pair = [r(1), c(1)];
    % result.cause = 3;
end

end


