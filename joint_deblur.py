"""
joint_deblur.py — 用关节角数据去模糊
========================================
核心功能：从机器人关节角推算 PSF，再反卷积去模糊。

两部分：
  1. 机器人运动学：关节角 → 相机速度 → 像素位移 → PSF
  2. 非盲反卷积：已知 PSF → Wiener / RL 去模糊

用法示例：
  from joint_deblur import compute_psf, wiener_deconvolution
  
  psf, (du, dv) = compute_psf(q, q_dot, depth=0.5, fx=500, fy=500)
  deblurred = wiener_deconvolution(blurred_img, psf, K=0.01)
"""

import numpy as np
import math
from apply_blur import pad_psf, create_motion_psf


# ============================================================
# 第一部分：Franka Panda 机器人运动学
# ============================================================
# D-H 参数（修正版，与 v7.2.3 notebook 一致）

DH_A = np.array([0, 0, 0, 0.0825, -0.0825, 0, 0])
DH_ALPHA = np.array([-np.pi/2, np.pi/2, np.pi/2, -np.pi/2, np.pi/2, np.pi/2, 0])
DH_D = np.array([0.333, 0, 0.316, 0, 0.384, 0, 0.088])
DH_THETA_OFFSET = np.array([0, -np.pi/2, 0, -np.pi/2, 0, np.pi/2, np.pi/4])


def dh_matrix(a, alpha, d, theta):
    """标准 D-H 参数 → 4×4 齐次变换矩阵"""
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([
        [ct, -st*ca,  st*sa,  a*ct],
        [st,  ct*ca, -ct*sa,  a*st],
        [0,   sa,     ca,     d   ],
        [0,   0,      0,      1   ]
    ])


def forward_kinematics(q):
    """
    正运动学：7 个关节角 → 末端位姿。
    返回 4×4 齐次变换矩阵 T_ee。
    """
    T = np.eye(4)
    for i in range(7):
        T_i = dh_matrix(DH_A[i], DH_ALPHA[i], DH_D[i], q[i] + DH_THETA_OFFSET[i])
        T = T @ T_i
    return T


def get_geometric_jacobian(q):
    """
    几何雅可比矩阵 J(q)：关节角速度 → 末端速度。
    返回 (J, T_list)，其中 T_list 存储每步的变换矩阵。
    """
    T_list = [np.eye(4)]
    for i in range(7):
        T_list.append(T_list[-1] @ dh_matrix(
            DH_A[i], DH_ALPHA[i], DH_D[i], q[i] + DH_THETA_OFFSET[i]))
    
    T_ee = T_list[-1]
    o_n = T_ee[:3, 3]
    z_prev = [T[:3, 2] for T in T_list[:-1]]
    o_prev = [T[:3, 3] for T in T_list[:-1]]
    
    J = np.zeros((6, 7))
    for i in range(7):
        J[:3, i] = np.cross(z_prev[i], o_n - o_prev[i])
        J[3:, i] = z_prev[i]
    return J, T_list


def get_camera_velocity(q, q_dot):
    """
    关节角速度 → 相机在基座系中的速度。
    
    相机安装在 Franka 末端，有固定偏移 t_he。
    旋转矩阵 R_he 描述相机坐标系与末端坐标系的相对姿态。
    
    相机速度 = R_he^T * v_ee + omega_ee × (R_he^T * t_he)
    """
    J, _ = get_geometric_jacobian(q)
    v_ee = J @ q_dot
    
    # 手眼标定参数：相机相对于末端法兰的姿态和位置
    R_he = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    t_he = np.array([0, 0, 0.05])
    
    w = R_he.T @ v_ee[3:]
    v = R_he.T @ v_ee[:3] + np.cross(w, R_he.T @ t_he)
    return np.concatenate([v, w])


# ============================================================
# 第二部分：交互矩阵 — 相机速度 → 像素速度
# ============================================================

def compute_interaction_matrix(u, v, Z, fx, fy, cx, cy):
    """
    速度-像素雅可比（交互矩阵）。
    
    公式：
    [du/dt]   [ -fx/Z   0       u_c/Z    u_c*v_c/fx   -(1+u_c^2)*fx   v_c*fx  ]   [ v_x ]
    [dv/dt] = [ 0      -fy/Z   v_c/Z   (1+v_c^2)*fy  -u_c*v_c/fy   -u_c*fy ] * [ omega ]
    
    其中 u_c = (u-cx)/fx, v_c = (v-cy)/fy 是归一化像素坐标。
    """
    xn = (u - cx) / fx
    yn = (v - cy) / fy
    
    L = np.array([
        [-fx/Z, 0, xn*fx/Z, xn*yn*fx, -(1+xn*xn)*fx, yn*fx],
        [0, -fy/Z, yn*fy/Z, (1+yn*yn)*fy, -xn*yn*fy, -xn*fy]
    ])
    return L


# ============================================================
# 第三部分：从关节角到 PSF
# ============================================================

def compute_psf(q, q_dot, depth, fx=500, fy=500, cx=None, cy=None,
               exposure_time=0.03, noise_level=0.0):
    """
    核心函数：从机器人关节角推算运动模糊 PSF。
    
    完整计算链：
    q, q_dot → 正运动学 → 雅可比 → 末端速度
             → 手眼变换 → 相机速度 v_cam
             → 交互矩阵 → 像素速度 (du/dt, dv/dt)
             → 积分 → 像素位移 (du, dv)
             → 创建 PSF 核
    
    参数:
        q (7,): 关节角（弧度）
        q_dot (7,): 关节角速度（弧度/秒）
        depth (float): 物距（米）
        fx, fy: 焦距（像素单位）
        cx, cy: 主点坐标（默认图像中心）
        exposure_time (float): 曝光时间（秒）
        noise_level (float): 模拟传感器噪声标准差
    
    返回:
        psf (ksize×ksize): 归一化 PSF 核
        (du, dv): 像素位移量
    """
    if cx is None or cy is None:
        raise ValueError("cx, cy must be provided or set to image center")
    
    # 步骤 1-2: 关节角 → 相机速度
    v_cam = get_camera_velocity(q, q_dot)
    
    # 步骤 3: 交互矩阵
    L = compute_interaction_matrix(cx, cy, depth, fx, fy, cx, cy)
    
    # 步骤 4: 像素速度 → 积分 → 像素位移
    du_dt, dv_dt = L @ v_cam
    du = du_dt * exposure_time
    dv = dv_dt * exposure_time
    
    # 可选：加入噪声模拟传感器不精确
    if noise_level > 0:
        du += np.random.randn() * noise_level
        dv += np.random.randn() * noise_level
    
    # 步骤 5: 从位移创建 PSF
    return create_motion_psf(du, dv), (du, dv)


def compute_psf_map(q, q_dot, depth, H, W, fx=500, fy=500,
                    exposure_time=0.03, grid_rows=4, grid_cols=4):
    """
    计算空间变化 PSF 地图。
    
    对图像的不同区域分别计算 PSF，因为旋转运动时
    图像不同位置的模糊方向和长度不同。
    
    返回:
        psf_map: (grid_rows, grid_cols) 的 PSF 数组
        du_grid, dv_grid: 每个网格的像素位移
    """
    v_cam = get_camera_velocity(q, q_dot)
    psf_map = [[None] * grid_cols for _ in range(grid_rows)]
    du_grid = np.zeros((grid_rows, grid_cols))
    dv_grid = np.zeros((grid_rows, grid_cols))
    
    for r in range(grid_rows):
        for c in range(grid_cols):
            u = int((c + 0.5) * W / grid_cols)
            v = int((r + 0.5) * H / grid_rows)
            
            L = compute_interaction_matrix(u, v, depth, fx, fy, W/2, H/2)
            du_dt, dv_dt = L @ v_cam
            
            du_grid[r, c] = du_dt * exposure_time
            dv_grid[r, c] = dv_dt * exposure_time
            
            psf_map[r][c] = create_motion_psf(du_grid[r, c], dv_grid[r, c])
    
    return psf_map, (du_grid, dv_grid)


# ============================================================
# 第四部分：非盲反卷积（去模糊）
# ============================================================

def wiener_deconvolution(blurred, psf, K=0.01):
    """
    维纳滤波去卷积。
    
    频域公式：
    F(u,v) = H* / (|H|^2 + K) * G
    
    其中 H = FFT(PSF), G = FFT(blurred)
    K 控制去模糊强度与噪声放大的平衡。
    
    参数:
        blurred: 模糊图像 (H, W)
        psf: PSF 核
        K: 噪声-信号比，默认 0.01
           (K 小→去模糊强，K 大→去噪好)
    """
    h, w = blurred.shape[:2]
    H = np.fft.fft2(pad_psf(psf, (h, w)))
    B = np.fft.fft2(blurred.astype(np.float64))
    result = np.fft.ifft2(B * np.conj(H) / (np.abs(H)**2 + K)).real
    return np.clip(result, 0, 255).astype(np.uint8)


def richardson_lucy(blurred, psf, iterations=30):
    """
    Richardson-Lucy 迭代反卷积。
    
    基于泊松噪声模型的最大似然估计：
    x_{k+1} = x_k * (conj(H) * (b / (H * x_k)))
    
    步数越多细节越锐利，但振铃越严重。
    
    参数:
        blurred: 模糊图像 (H, W)
        psf: PSF 核
        iterations: 迭代次数
    """
    b = blurred.astype(np.float64)
    h, w = b.shape[:2]
    
    H = np.fft.fft2(pad_psf(psf, (h, w)))
    Hc = np.conj(H)
    
    bound = 1e10
    x = b.copy()
    
    for _ in range(iterations):
        # 前向投影: H * x_k
        Hx = np.fft.ifft2(np.fft.fft2(x) * H).real
        Hx = np.maximum(Hx, 1e-10)
        
        # 比值: b / (H * x_k)
        ratio = np.clip(b / Hx, 0, bound)
        
        # 反向投影: conj(H) * ratio
        upd = np.clip(np.fft.ifft2(np.fft.fft2(ratio) * Hc).real, 0, bound)
        
        # 更新
        x = np.clip(x * upd, 0, None)
    
    return np.clip(x, 0, 255).astype(np.uint8)
