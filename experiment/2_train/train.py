"""
Step 3: Deterministic Residual Learning - Robust Multi-Step Training
==================================================================
改进点：
1. Trajectory-Aware Dataset: 确保 Rollout 窗口不跨越不同轨迹的边界。通过位置跳变自动识别轨迹切换。
2. Mixed Loss: 结合 One-step Residual Loss (作为锚点防止变坏) 和 Multi-step Rollout Loss (作为长时优化目标)。
3. Input Augmentation Fix: 只对初始输入 x0 加噪以增强鲁棒性，绝对不污染未来的真值标签 (x1...xH)。
4. Curriculum Learning: beta 随训练轮次逐步增加，前期先练好单步偏差，后期再攻克多步累积。
"""

import os
import json
import copy
import h5py
import numpy as np
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# ============================================================
# Global Configuration
# ============================================================
GLOBAL_HORIZON = 10

# ============================================================
# Dataset (轨迹感知 + 预归一化优化)
# ============================================================

class TrajectoryDataset(Dataset):
    """
    按轨迹加载数据，并在初始化时完成所有数据的预归一化，显著提升训练速度。
    """
    def __init__(self, mat_file, horizon=GLOBAL_HORIZON, stats=None):
        self.mat_file = mat_file
        self.horizon = horizon
        self.physical_params = None

        # 尝试从数据集所在目录提取物理参数
        try:
            import scipy.io as sio
            data_dir = os.path.dirname(mat_file)
            import glob
            # 搜索数据集目录下任意包含 traj_case_*.mat 的子文件夹
            case_files = glob.glob(os.path.join(data_dir, 'case_*', 'traj_case_*.mat'))
            if case_files:
                sample_file = case_files[0]
                sample_m = sio.loadmat(sample_file, squeeze_me=True)
                if 'traj' in sample_m and 'params' in sample_m['traj'].dtype.names:
                    p_struct = sample_m['traj']['params'].item()
                    self.physical_params = {n: p_struct[n] for n in p_struct.dtype.names}
        except Exception as e:
            print(f"--> Warning: Could not extract physical params from dataset: {e}")

        with h5py.File(mat_file, 'r') as f:
            root = f['split_ds'] if 'split_ds' in f else f['dataset']
            X_all = np.array(root['X']).T.astype(np.float32)
            U_all = np.array(root['U']).T.astype(np.float32)
            R_all = np.array(root['R_label']).T.astype(np.float32)
            
            # 使用 case_id 识别轨迹边界 (更健壮)
            if 'case_id' in root:
                case_id = np.array(root['case_id']).flatten()
                boundaries = np.where(case_id[1:] != case_id[:-1])[0] + 1
            else:
                # 备选方案：通过位置跳变识别轨迹边界
                pos = X_all[:, :30]
                dist_sq = np.sum((pos[1:] - pos[:-1])**2, axis=1)
                boundaries = np.where(dist_sq > 25.0)[0] + 1
            boundary_count = int(boundaries.shape[0])
            boundary_preview = boundaries[:10].tolist() if boundary_count > 0 else []
            print(f"--> Trajectory boundaries detected: {boundary_count} | preview={boundary_preview}")
            
            # 计算或接收统计量
            if stats is None:
                self.stats = {
                    'X_mean': np.mean(X_all, axis=0), 'X_std': np.std(X_all, axis=0) + 1e-6,
                    'U_mean': np.mean(U_all, axis=0), 'U_std': np.std(U_all, axis=0) + 1e-6,
                    'R_mean': np.mean(R_all, axis=0), 'R_std': np.std(R_all, axis=0) + 1e-6,
                }
            else:
                self.stats = stats
            
            # 关键：预归一化全部数据 (CPU 运行一次即可)
            self.X_norm = (X_all - self.stats['X_mean']) / self.stats['X_std']
            self.U_norm = (U_all - self.stats['U_mean']) / self.stats['U_std']
            self.R_norm = (R_all - self.stats['R_mean']) / self.stats['R_std']

            self.trajs_X = np.split(self.X_norm, boundaries)
            self.trajs_U = np.split(self.U_norm, boundaries)
            self.trajs_R = np.split(self.R_norm, boundaries)
            
        self.valid_indices = []
        for t_idx, t_data in enumerate(self.trajs_X):
            T = t_data.shape[0]
            if T > self.horizon:
                for s in range(T - self.horizon):
                    self.valid_indices.append((t_idx, s))
        print(f"--> Loaded {len(self.valid_indices)} valid training samples from {mat_file}")

    def __len__(self): return len(self.valid_indices)

    def __getitem__(self, idx):
        t_idx, s_idx = self.valid_indices[idx]
        # 直接切片已归一化的数据，无需重复计算
        x_seq = self.trajs_X[t_idx][s_idx : s_idx + self.horizon + 1]
        u_seq = self.trajs_U[t_idx][s_idx : s_idx + self.horizon]
        r_seq = self.trajs_R[t_idx][s_idx : s_idx + self.horizon]

        return (
            torch.from_numpy(x_seq), 
            torch.from_numpy(u_seq), 
            torch.from_numpy(r_seq)
        )

# ============================================================
# Model
# ============================================================

class ResidualBlock(nn.Module):
    def __init__(self, dim, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.LayerNorm(dim), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(dim, dim), nn.LayerNorm(dim)
        )
        self.act = nn.SiLU()
    def forward(self, x): return self.act(x + self.net(x))

class ResidualMLP(nn.Module):
    def __init__(self, input_dim=90, output_dim=60, hidden_dim=256, depth=3):
        super().__init__()
        self.input_proj = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResidualBlock(hidden_dim) for _ in range(depth)])
        self.head = nn.Linear(hidden_dim, output_dim)
    def forward(self, x, u):
        h = self.input_proj(torch.cat([x, u], dim=-1))
        return self.head(self.blocks(h))

# ============================================================
# Ops (核心提速点：减少冗余张量创建和多步优化)
# ============================================================

def denormalize(val, m_t, s_t):
    """直接使用已在 device 上的 tensor，消除函数闭包/临时张量创建开销"""
    return val * s_t + m_t

def normalize(val, m_t, s_t):
    return (val - m_t) / s_t

def nominal_dynamics_torch(x, u, dt, vmax, pFactor, predator, n):
    p, v, a = x[:, :2*n], x[:, 2*n:], u
    new_v = v + a * dt
    
    # 物理限速 (vmax 截断)
    vx = new_v[:, :n]
    vy = new_v[:, n:]
    v_mag = torch.sqrt(vx**2 + vy**2 + 1e-8)
    
    # 计算每个 agent 的最大速度限制
    vmax_tensor = torch.full_like(v_mag, vmax)
    if predator > 0:
        # 假设最后一个 agent (index n-1) 是 predator
        vmax_tensor[:, -1] = vmax * pFactor
        
    scale = torch.clamp(vmax_tensor / v_mag, max=1.0)
    new_v_clipped = torch.cat([vx * scale, vy * scale], dim=-1)
    
    new_p = p + new_v_clipped * dt # Semi-implicit Euler: 使用更新后的速度计算位移
    return torch.cat([new_p, new_v_clipped], dim=-1)

def compute_mixed_loss(model, x_seq, u_seq, r_lbl_seq, stats_t, horizon, alpha, beta, x0_override, vmax, pFactor, predator, dt, n):
    """
    stats_t: 必须是已经 .to(device) 的张量字典。
    x0_override: 提供用于加噪训练的初始状态，避免 clone 整个 x_seq。
    """
    x_curr_norm = x0_override if x0_override is not None else x_seq[:, 0, :]
    
    # One-step Residual Anchor (锚点)
    r_pred_1 = model(x_curr_norm, u_seq[:, 0, :])
    anchor_loss = torch.mean((r_pred_1 - r_lbl_seq[:, 0, :])**2)
    
    # 获取统计量
    xm, xs = stats_t['X_mean'], stats_t['X_std']
    rm, rs = stats_t['R_mean'], stats_t['R_std']
    
    # Multi-step Rollout
    rollout_loss = 0
    for t in range(horizon):
        u_t_norm = u_seq[:, t, :]
        r_p_norm = model(x_curr_norm, u_t_norm)
        
        # 物理演化 (通过已在设备上的 stats_t 进行反归一化)
        x_c = denormalize(x_curr_norm, xm, xs)
        u_c = denormalize(u_t_norm, stats_t['U_mean'], stats_t['U_std'])
        r_p = denormalize(r_p_norm, rm, rs)
        
        x_next_pred = nominal_dynamics_torch(x_c, u_c, dt=dt, vmax=vmax, pFactor=pFactor, predator=predator, n=n) + r_p
        
        # 对比物理空间真值 (真值在 Dataset 预归一化时已就绪)
        x_true_next = denormalize(x_seq[:, t+1, :], xm, xs)
        step_err = torch.mean((x_next_pred - x_true_next)**2)
        rollout_loss += step_err * (1.0 + t/horizon)
        
        # 重新归一化进入下一步循环
        x_curr_norm = normalize(x_next_pred, xm, xs)

    return alpha * anchor_loss + beta * (rollout_loss / horizon)

# ============================================================
# Train
# ============================================================

def train():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--horizon', type=int, default=None)
    parser.add_argument('--beta_fixed', type=float, default=None)
    parser.add_argument('--x0_noise_std', type=float, default=0.005)
    args = parser.parse_args()

    TRAIN_ROOT = os.path.dirname(os.path.abspath(__file__))
    PROJECT_ROOT = os.path.abspath(os.path.join(TRAIN_ROOT, '..', '..'))
    COLLECT_ROOT = os.path.join(PROJECT_ROOT, 'traj', 'step3_collect')
    
    import glob
    manifests = glob.glob(os.path.join(COLLECT_ROOT, "**", "manifest.mat"), recursive=True)
    if not manifests:
        print("Error: No data found."); return
    # 按修改时间排序找到最新的数据集目录
    DATA_DIR = os.path.dirname(max(manifests, key=os.path.getmtime))

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
    RUN_OUT_DIR = os.path.join(TRAIN_ROOT, 'out')
    BACKUP_DIR = os.path.join(RUN_OUT_DIR, timestamp)
    os.makedirs(BACKUP_DIR, exist_ok=True)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    HORIZON = args.horizon if args.horizon is not None else GLOBAL_HORIZON
    BATCH_SIZE, EPOCHS, LR = args.batch_size, args.epochs, args.lr

    # 加载数据集
    train_ds = TrajectoryDataset(os.path.join(DATA_DIR, 'dataset_train.mat'), horizon=HORIZON)
    val_ds = TrajectoryDataset(os.path.join(DATA_DIR, 'dataset_val.mat'), horizon=HORIZON, stats=train_ds.stats)
    
    # 启用多线程读取和锁页内存
    num_workers = 4 if os.name != 'nt' else 0 # Windows 下 num_workers > 0 容易报错，小心使用
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True, num_workers=num_workers)

    print(f"--> Starting Optimized Robust Training on {DEVICE}")
    vmax_val = 2.0
    pFactor_val = 1.40
    predator_val = 0
    dt_val = 0.3
    n_val = 15
    if hasattr(train_ds, 'physical_params') and train_ds.physical_params:
        vmax_val = float(train_ds.physical_params.get('vmax', 2.0))
        pFactor_val = float(train_ds.physical_params.get('pFactor', 1.40))
        predator_val = int(train_ds.physical_params.get('predator', 0))
        dt_val = float(train_ds.physical_params.get('dt', 0.3))
        n_val = int(train_ds.physical_params.get('n', 15))

    # 准备统计量 Tensor (核心改进：只创建一次，并传给 loss 函数)
    stats_t = {k: torch.tensor(v, device=DEVICE) for k, v in train_ds.stats.items()}

    model = ResidualMLP(input_dim=6*n_val, output_dim=4*n_val).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=2e-3)
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == 'cuda')) # 开启混合精度
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=20)

    best_val = float('inf')
    PATIENCE_LIMIT = 25
    patience_counter = 0

    best_val_loss = float('inf')  # 初始最佳 loss 设为正无穷
    patience_counter = 0          # 初始化耐心计数器（用于早停）

    def beta_for_epoch(epoch_idx):
        if args.beta_fixed is not None:
            return float(args.beta_fixed)
        if epoch_idx <= 20:
            return 0.0
        if epoch_idx <= 70:
            return (epoch_idx - 20) / 50.0
        return 1.0

    print(
        "--> Train config | horizon=%d | beta_fixed=%s | x0_noise_std=%.4f" % (
            HORIZON,
            "None" if args.beta_fixed is None else f"{args.beta_fixed:.3f}",
            args.x0_noise_std,
        )
    )

    for epoch in range(EPOCHS):
        model.train()
        train_l = 0
        beta_curr = beta_for_epoch(epoch)

        for x_seq, u_seq, r_lbl in train_loader:
            x_seq, u_seq, r_lbl = x_seq.to(DEVICE), u_seq.to(DEVICE), r_lbl.to(DEVICE)
            
            # 使用高效加噪：只对 x0 掩码加噪，不 clone 整个序列
            noise = torch.randn_like(x_seq[:, 0, :]) * args.x0_noise_std
            x0_aug = x_seq[:, 0, :] + noise

            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == 'cuda')):
                loss = compute_mixed_loss(
                    model, x_seq, u_seq, r_lbl, stats_t, HORIZON, 
                    alpha=1.0, beta=beta_curr, x0_override=x0_aug, 
                    vmax=vmax_val, pFactor=pFactor_val, predator=predator_val, dt=dt_val, n=n_val
                )
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            train_l += loss.item()

        model.eval()
        val_l = 0
        with torch.no_grad():
            for x_seq, u_seq, r_lbl in val_loader:
                x_seq, u_seq, r_lbl = x_seq.to(DEVICE), u_seq.to(DEVICE), r_lbl.to(DEVICE)
                v_loss = compute_mixed_loss(
                    model, x_seq, u_seq, r_lbl, stats_t, HORIZON, 
                    alpha=1.0, beta=beta_curr, x0_override=None,
                    vmax=vmax_val, pFactor=pFactor_val, predator=predator_val, dt=dt_val, n=n_val
                )
                val_l += v_loss.item()
        
        avg_train = train_l / len(train_loader); avg_val = val_l / len(val_loader)
        scheduler.step(avg_val)
        
        # 检查是否进入平稳期 (假设你的 beta 最大值是 1.0)
        is_curriculum_stable = (beta_curr >= 1.0) 

        # 只要还在上难度（非平稳期），或者损失真的下降了，就保存并重置耐心值
        if (not is_curriculum_stable) or (avg_val < best_val_loss):
            best_val_loss = avg_val
            patience_counter = 0
            best_state_dict = copy.deepcopy(model.state_dict())
            
            for save_dir in [RUN_OUT_DIR, BACKUP_DIR]:
                # 1. 存模型权重
                torch.save(best_state_dict, os.path.join(save_dir, 'residual_model.pt'))
                
                # 2. 存归一化参数
                with open(os.path.join(save_dir, 'scaling_stats.json'), 'w') as f:
                    json.dump({k: v.tolist() for k, v in train_ds.stats.items()}, f)
                
                # ---------------------------------------------------------
                # 3. [新增] 存验证集残差方差 (供 MATLAB PrSBC 自动缩紧安全边界使用)
                # ---------------------------------------------------------
                metrics = {"residual_variance": float(best_val_loss)}
                with open(os.path.join(save_dir, 'validation_metrics.json'), 'w') as f:
                    json.dump(metrics, f, indent=4)
                # ---------------------------------------------------------

                # 4. 导出物理参数以供验证 (使用 NumpyEncoder 处理 ndarray)
                if hasattr(train_ds, 'physical_params') and train_ds.physical_params:
                    import numpy as np
                    class NumpyEncoder(json.JSONEncoder):
                        def default(self, obj):
                            if isinstance(obj, np.ndarray): return obj.tolist()
                            if isinstance(obj, np.generic): return obj.item()
                            return super().default(obj)
                    with open(os.path.join(save_dir, 'physical_params.json'), 'w') as f:
                        json.dump(train_ds.physical_params, f, indent=4, cls=NumpyEncoder)
        else:
            # 只有在平稳期且 Loss 上升时，才累加耐心值
            patience_counter += 1

        # 打印日志 (已修复变量名报错，并增加 Patience 显示)
        if (epoch+1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:03d} | Train: {avg_train:.6f} | Val: {avg_val:.6f} | B: {beta_curr:.2f} | Best: {best_val_loss:.6f} | Patience: {patience_counter}")

        # 早停触发
        if patience_counter >= PATIENCE_LIMIT:
            print(f"--> Early Stopping at epoch {epoch+1}. Best Val: {best_val_loss:.6f}")
            break

if __name__ == "__main__":
    train()
