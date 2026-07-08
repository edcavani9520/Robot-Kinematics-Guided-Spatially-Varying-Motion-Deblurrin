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
from robot_configs import RobotConfig, HandEyeCalib, PANDA, PANDA_HAND_EYE_SIMPLE, DROID_HAND_EYE_LEFT, DROID_HAND_EYE_RIGHT, get_configs
# ============================================================
# 第一部分：机器人运动学 — 从 D-H 到正运动学 / 雅可比
# ============================================================


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


def forward_kinematics(q, robot=None):
    """
    正运动学：7 个关节角 → 末端位姿。
    
    参数:
        q (7,): 关节角（弧度）
        robot (RobotConfig): 机器人 D-H 配置，默认 PANDA
    
    返回:
        T_ee (4×4): 末端齐次变换矩阵
    """
    if robot is None:
        robot = PANDA
    T = np.eye(4)
    for i in range(7):
        T_i = dh_matrix(robot.a[i], robot.alpha[i], robot.d[i], q[i] + robot.theta_offset[i])
        T = T @ T_i
    return T


def get_geometric_jacobian(q, robot=None):
    """
    几何雅可比矩阵 J(q)：关节角速度 → 末端速度。
    
    参数:
        q (7,): 关节角（弧度）
        robot (RobotConfig): 机器人 D-H 配置，默认 PANDA
    
    返回:
        J (6×7): 几何雅可比矩阵
        T_list: 每步齐次变换矩阵列表
    """
    if robot is None:
        robot = PANDA
    T_list = [np.eye(4)]
    for i in range(7):
        T_list.append(T_list[-1] @ dh_matrix(
            robot.a[i], robot.alpha[i], robot.d[i], q[i] + robot.theta_offset[i]))
    
    T_ee = T_list[-1]
    o_n = T_ee[:3, 3]
    z_prev = [T[:3, 2] for T in T_list[:-1]]
    o_prev = [T[:3, 3] for T in T_list[:-1]]
    
    J = np.zeros((6, 7))
    for i in range(7):
        J[:3, i] = np.cross(z_prev[i], o_n - o_prev[i])
        J[3:, i] = z_prev[i]
    return J, T_list


def get_camera_velocity(q, q_dot, hand_eye=None, robot=None):
    """
    关节角速度 → 相机在基座系中的速度。
    
    相机安装在末端法兰上，通过手眼标定参数 R_he / t_he
    将末端速度转换到相机坐标系。
    
    公式（三步推导）：
      v_ee_ee = R_ee^T · v_ee,   ω_ee_ee = R_ee^T · ω_ee      （基座系 → 末端系）
      v_cam_ee = v_ee_ee + ω_ee_ee × t_he                        （附加线速度）
      v_cam = R_he^T · v_cam_ee,   ω_cam = R_he^T · ω_ee_ee     （末端系 → 相机系）
    
    参数:
        q (7,): 关节角（弧度）
        q_dot (7,): 关节角速度（弧度/秒）
        hand_eye (HandEyeCalib): 手眼标定，默认 PANDA_HAND_EYE_SIMPLE
        robot (RobotConfig): 机器人 D-H 配置，默认 PANDA
    
    返回:
        v_cam (6,): [v_x, v_y, v_z, ω_x, ω_y, ω_z] 相机速度
    """
    if hand_eye is None:
        hand_eye = PANDA_HAND_EYE_SIMPLE
    if robot is None:
        robot = PANDA
    
    J, T_list = get_geometric_jacobian(q, robot=robot)
    v_ee = J @ q_dot

    # 第1步：从基座系变换到末端系（R_ee^T）
    R_ee = T_list[-1][:3, :3]  # 末端姿态旋转矩阵
    w_ee = R_ee.T @ v_ee[3:]   # 角速度 → 末端系
    v_ee_local = R_ee.T @ v_ee[:3]  # 线速度 → 末端系

    R_he, t_he = hand_eye.R, hand_eye.t

    # 第2步：末端系内，角速度引起的附加线速度
    v_cam_ee = v_ee_local + np.cross(w_ee, t_he)

    # 第3步：从末端系到相机系（手眼标定 R_he^T）
    w_cam = R_he.T @ w_ee
    v_cam = R_he.T @ v_cam_ee
    return np.concatenate([v_cam, w_cam])


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
               exposure_time=0.03, hand_eye=None, robot=None):
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
        hand_eye (HandEyeCalib): 手眼标定，默认 PANDA_HAND_EYE_SIMPLE
        robot (RobotConfig): 机器人 D-H 配置，默认 PANDA
    
    返回:
        psf (ksize×ksize): 归一化 PSF 核
        (du, dv): 像素位移量
    """
    if cx is None or cy is None:
        raise ValueError("cx, cy must be provided or set to image center")
    
    # 步骤 1-2: 关节角 → 相机速度
    v_cam = get_camera_velocity(q, q_dot, hand_eye=hand_eye, robot=robot)
    
    # 步骤 3: 交互矩阵
    L = compute_interaction_matrix(cx, cy, depth, fx, fy, cx, cy)
    
    # 步骤 4: 像素速度 → 积分 → 像素位移
    du_dt, dv_dt = L @ v_cam
    du = du_dt * exposure_time
    dv = dv_dt * exposure_time
    
    # 步骤 5: 从位移创建 PSF
    return create_motion_psf(du, dv), (du, dv)


def compute_psf_map(q, q_dot, depth, H, W, fx=500, fy=500,
                    exposure_time=0.03, grid_rows=4, grid_cols=4,
                    hand_eye=None, robot=None):
    """
    计算空间变化 PSF 地图。
    
    对图像的不同区域分别计算 PSF，因为旋转运动时
    图像不同位置的模糊方向和长度不同。
    
    参数:
        q (7,): 关节角（弧度）
        q_dot (7,): 关节角速度（弧度/秒）
        depth (float): 物距（米）
        H, W: 图像高宽（像素）
        fx, fy: 焦距（像素单位）
        exposure_time (float): 曝光时间（秒）
        grid_rows, grid_cols: 网格划分
        hand_eye (HandEyeCalib): 手眼标定
        robot (RobotConfig): 机器人 D-H 配置
    
    返回:
        psf_map: (grid_rows, grid_cols) 的 PSF 数组
        du_grid, dv_grid: 每个网格的像素位移
    """
    v_cam = get_camera_velocity(q, q_dot, hand_eye=hand_eye, robot=robot)
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
# 第三部分（续）：空间变化反卷积（重叠 patch + cosine blending）
# ============================================================


# ============================================================
# PSF utility functions (moved from apply_blur.py)
# ============================================================


def create_motion_psf(du, dv):
    """
    Create a motion blur PSF kernel from pixel displacement (du, dv).

    Draws a line along the motion direction in the spatial domain,
    then normalizes so total energy = 1.

    Parameters:
        du, dv: pixel displacement in x/y during exposure
    Returns:
        (ksize, ksize) normalized PSF kernel
    """
    ksize = max(2 * int(abs(du)), 2 * int(abs(dv))) + 1
    ksize = max(ksize, 3)
    ksize = ksize if ksize % 2 == 1 else ksize + 1

    psf = np.zeros((ksize, ksize), dtype=np.float32)
    c = ksize // 2
    steps = max(int(math.hypot(du, dv) * 5), 400)

    for t in np.linspace(0, 1, steps):
        x = int(round(c + t * dv))
        y = int(round(c + t * du))
        if 0 <= x < ksize and 0 <= y < ksize:
            psf[x, y] += 1.0

    s = psf.sum()
    return psf / s if s > 0 else psf


def pad_psf(psf, shape):
    """
    Pad PSF to image size and shift its centroid to (0,0).

    This is critical for FFT convolution: the PSF centroid must be
    at (0,0), otherwise the convolution result will have a shift.

    Parameters:
        psf: PSF kernel (kh, kw)
        shape: target (h, w)
    Returns:
        padded and rolled PSF (h, w)
    """
    h, w = shape[:2]
    kh, kw = psf.shape
    cy, cx = np.unravel_index(psf.argmax(), psf.shape)

    pad = np.zeros((h, w), dtype=np.float64)
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -cy, axis=0)
    pad = np.roll(pad, -cx, axis=1)
    return pad


def spatial_wiener_deconvolution(blurred, psf_map, grid_rows, grid_cols,
                                 K=0.01, overlap=0.25):
    """Spatially-varying Wiener deconvolution with overlapping patches and cosine blending."""
    H, W = blurred.shape[:2]
    result = np.zeros_like(blurred, dtype=np.float64)
    weight = np.zeros_like(blurred, dtype=np.float64)
    cell_h = H / grid_rows
    cell_w = W / grid_cols
    overlap_h = int(cell_h * overlap + 0.5)
    overlap_w = int(cell_w * overlap + 0.5)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = max(0, int(r * cell_h) - overlap_h)
            y1 = min(H, int((r + 1) * cell_h) + overlap_h)
            x0 = max(0, int(c * cell_w) - overlap_w)
            x1 = min(W, int((c + 1) * cell_w) + overlap_w)
            patch = blurred[y0:y1, x0:x1]
            psf = psf_map[r][c]
            deblurred_patch = wiener_deconvolution(patch, psf, K=K)
            wy = 0.5 - 0.5 * np.cos(np.pi * np.arange(y1 - y0) / (y1 - y0))
            wx = 0.5 - 0.5 * np.cos(np.pi * np.arange(x1 - x0) / (x1 - x0))
            w = np.outer(wy, wx)
            result[y0:y1, x0:x1] += deblurred_patch.astype(np.float64) * w
            weight[y0:y1, x0:x1] += w
    weight = np.maximum(weight, 1e-10)
    return np.clip(result / weight, 0, 255).astype(np.uint8)


def spatial_richardson_lucy(blurred, psf_map, grid_rows, grid_cols,
                            iterations=30, overlap=0.25):
    """Spatially-varying RL deconvolution with overlapping patches and cosine blending."""
    H, W = blurred.shape[:2]
    result = np.zeros_like(blurred, dtype=np.float64)
    weight = np.zeros_like(blurred, dtype=np.float64)
    cell_h = H / grid_rows
    cell_w = W / grid_cols
    overlap_h = int(cell_h * overlap + 0.5)
    overlap_w = int(cell_w * overlap + 0.5)
    for r in range(grid_rows):
        for c in range(grid_cols):
            y0 = max(0, int(r * cell_h) - overlap_h)
            y1 = min(H, int((r + 1) * cell_h) + overlap_h)
            x0 = max(0, int(c * cell_w) - overlap_w)
            x1 = min(W, int((c + 1) * cell_w) + overlap_w)
            patch = blurred[y0:y1, x0:x1]
            psf = psf_map[r][c]
            deblurred_patch = richardson_lucy(patch, psf, iterations=iterations)
            wy = 0.5 - 0.5 * np.cos(np.pi * np.arange(y1 - y0) / (y1 - y0))
            wx = 0.5 - 0.5 * np.cos(np.pi * np.arange(x1 - x0) / (x1 - x0))
            w = np.outer(wy, wx)
            result[y0:y1, x0:x1] += deblurred_patch.astype(np.float64) * w
            weight[y0:y1, x0:x1] += w
    weight = np.maximum(weight, 1e-10)
    return np.clip(result / weight, 0, 255).astype(np.uint8)


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
