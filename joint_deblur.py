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

MAX_PSF_KERNEL_SIZE = 4095


def _require_finite_positive(name, value):
    if isinstance(value, bool):
        raise ValueError(f"{name} must be finite and greater than zero")
    try:
        valid = np.isfinite(value) and value > 0
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ValueError(f"{name} must be finite and greater than zero")
    return float(value)


def _require_finite_vector(name, value, length):
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector with shape ({length},)")
    return array


def _validate_psf_array(psf):
    array = np.asarray(psf, dtype=np.float64)
    if (
        array.ndim != 2
        or array.size == 0
        or not np.isfinite(array).all()
        or np.any(array < 0)
        or array.sum() <= 0
    ):
        raise ValueError("psf must be a finite, non-negative, non-empty 2-D kernel")
    return array / array.sum()

# === ADDED: Euler ZYX deg -> rotation matrix ===
def euler_zyx_to_rotmat(rpy_deg):
    """ZYX Euler angles (deg) -> 3x3 rotation matrix."""
    r, p, y = np.deg2rad(rpy_deg)
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,           cp*cr]
    ])

# === ADDED: PSF from tool_pose + tool_twist (bypasses FK+Jacobian) ===
def compute_psf_from_pose(tool_pose=None, tool_twist=None, depth=0.5, fx=None, fy=None,
                           cx=None, cy=None, exposure_time=0.01, hand_eye=None,
                           v_cam_6d=None):
    """Compute PSF from h5 tool_pose + tool_twist or direct v_cam_6d."""
    _require_finite_positive("depth", depth)
    _require_finite_positive("fx", fx)
    _require_finite_positive("fy", fy)
    _require_finite_positive("exposure_time", exposure_time)
    if cx is None or cy is None:
        raise ValueError("cx, cy required")
    if not np.isfinite([cx, cy]).all():
        raise ValueError("cx and cy must be finite")
    if v_cam_6d is not None:
        v_cam_6d = _require_finite_vector("v_cam_6d", v_cam_6d, 6)
    elif tool_twist is not None and tool_pose is not None:
        tool_pose = _require_finite_vector("tool_pose", tool_pose, 6)
        tool_twist = _require_finite_vector("tool_twist", tool_twist, 6)
        R_ee = euler_zyx_to_rotmat(tool_pose[3:])
        v_ee_base = tool_twist[:3]
        w_ee_base = tool_twist[3:]
        v_ee_ee = R_ee.T @ v_ee_base
        w_ee_ee = R_ee.T @ w_ee_base
        if hand_eye is not None:
            R_he, t_he = hand_eye.R, hand_eye.t
            w_cam = R_he.T @ w_ee_ee
            v_cam = R_he.T @ v_ee_ee + np.cross(w_cam, R_he.T @ t_he)
        else:
            v_cam, w_cam = v_ee_ee, w_ee_ee
        v_cam_6d = np.concatenate([v_cam, w_cam])
    else:
        raise ValueError("Either v_cam_6d or (tool_pose + tool_twist) must be provided")
    L = compute_interaction_matrix(cx, cy, depth, fx, fy, cx, cy)
    du_dt, dv_dt = L @ v_cam_6d
    du = du_dt * exposure_time
    dv = dv_dt * exposure_time
    return create_motion_psf(du, dv), (du, dv)
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
    q = _require_finite_vector("q", q, 7)
    q_dot = _require_finite_vector("q_dot", q_dot, 7)
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
    _require_finite_positive("depth", Z)
    _require_finite_positive("fx", fx)
    _require_finite_positive("fy", fy)
    if not np.isfinite([u, v, cx, cy]).all():
        raise ValueError("u, v, cx, and cy must be finite")
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

def compute_psf(q, q_dot, depth, fx=None, fy=None, cx=None, cy=None,
                exposure_time=0.01, hand_eye=None, robot=None,
                tool_pose=None, tool_twist=None):
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
    
    # 如果提供了 tool_pose + tool_twist，跳过 FK+Jacobian，直接用真值
    if tool_pose is not None and tool_twist is not None:
        return compute_psf_from_pose(
            tool_pose, tool_twist, depth,
            fx=fx, fy=fy, cx=cx, cy=cy,
            exposure_time=exposure_time,
            hand_eye=hand_eye
        )
    
    # 否则用 FK + Jacobian（Panda 等标准 DH 机器人）
    v_cam = get_camera_velocity(q, q_dot, hand_eye=hand_eye, robot=robot)
    L = compute_interaction_matrix(cx, cy, depth, fx, fy, cx, cy)
    du_dt, dv_dt = L @ v_cam
    du = du_dt * exposure_time
    dv = dv_dt * exposure_time
    return create_motion_psf(du, dv), (du, dv)


def compute_psf_map(q, q_dot, depth, H, W, fx=None, fy=None,
                    exposure_time=0.01, grid_rows=4, grid_cols=4,
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
    Create a centered subpixel motion PSF using bilinear trajectory splatting.

    The exposure path is sampled at at most quarter-pixel spacing. Each sample
    distributes its energy over four neighboring pixels, so displacements below
    one pixel remain represented instead of being rounded to a delta kernel.
    Parameters:
        du, dv: pixel displacement in x/y during exposure
    Returns:
        (ksize, ksize) normalized PSF kernel
    """
    if not np.isfinite([du, dv]).all():
        raise ValueError("du and dv must be finite")
    du, dv = float(du), float(dv)
    length = math.hypot(du, dv)
    radius = max(1, int(math.ceil(max(abs(du), abs(dv)) / 2.0)) + 1)
    ksize = 2 * radius + 1
    if ksize > MAX_PSF_KERNEL_SIZE:
        raise ValueError(
            f"motion PSF kernel size {ksize} exceeds safety limit "
            f"{MAX_PSF_KERNEL_SIZE}"
        )
    psf = np.zeros((ksize, ksize), dtype=np.float64)
    center = float(radius)
    if length < 1e-12:
        psf[radius, radius] = 1.0
        return psf

    sample_count = max(2, int(math.ceil(length / 0.25)) + 1)
    for t in np.linspace(-0.5, 0.5, sample_count):
        row = center + t * dv
        col = center + t * du
        row0, col0 = math.floor(row), math.floor(col)
        row_fraction = row - row0
        col_fraction = col - col0
        for rr, row_weight in (
            (row0, 1.0 - row_fraction),
            (row0 + 1, row_fraction),
        ):
            for cc, col_weight in (
                (col0, 1.0 - col_fraction),
                (col0 + 1, col_fraction),
            ):
                if 0 <= rr < ksize and 0 <= cc < ksize:
                    psf[rr, cc] += row_weight * col_weight
    return psf / psf.sum()


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
    if kh > h or kw > w:
        raise ValueError(f"PSF shape {(kh, kw)} exceeds image shape {(h, w)}")
    # The trajectory PSF is centered geometrically; use that center for FFT.
    cy, cx = kh // 2, kw // 2

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
            blend = w[..., None] if blurred.ndim == 3 else w
            result[y0:y1, x0:x1] += deblurred_patch.astype(np.float64) * blend
            weight[y0:y1, x0:x1] += blend
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
            blend = w[..., None] if blurred.ndim == 3 else w
            result[y0:y1, x0:x1] += deblurred_patch.astype(np.float64) * blend
            weight[y0:y1, x0:x1] += blend
    weight = np.maximum(weight, 1e-10)
    return np.clip(result / weight, 0, 255).astype(np.uint8)


# ============================================================
# 第四部分：非盲反卷积（去模糊）
# ============================================================

def _apply_to_image_channels(image, operation):
    """Apply a two-dimensional operation to grayscale or every RGB channel."""
    image = np.asarray(image)
    if image.ndim == 2:
        return operation(image)
    if image.ndim == 3 and image.shape[2] == 3:
        return np.stack(
            [operation(image[..., channel]) for channel in range(3)], axis=2
        )
    raise ValueError(
        "deconvolution input must be a 2-D image or an RGB image with shape "
        f"(H, W, 3); got {image.shape}"
    )


def _wiener_deconvolution_2d(blurred, psf, K=0.01):
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
            (K 小->去模糊强，K 大->去噪好)
    """
    _require_finite_positive("K", K)
    psf = _validate_psf_array(psf)
    h, w = blurred.shape[:2]
    H = np.fft.fft2(pad_psf(psf, (h, w)))
    B = np.fft.fft2(blurred.astype(np.float64))
    result = np.fft.ifft2(B * np.conj(H) / (np.abs(H)**2 + K)).real
    # DC gain compensation: restore average brightness
    result *= (1.0 + K)
    return np.clip(result, 0, 255).astype(np.uint8)


def wiener_deconvolution(blurred, psf, K=0.01):
    """Wiener deconvolution for a 2-D image or independent RGB channels."""
    return _apply_to_image_channels(
        blurred, lambda channel: _wiener_deconvolution_2d(channel, psf, K=K)
    )


def _richardson_lucy_2d(blurred, psf, iterations=30):
    """Richardson-Lucy iterative deconvolution.
    x_{k+1} = x_k * (H^T * (b / (H * x_k)))
    """
    if isinstance(iterations, bool) or not isinstance(iterations, (int, np.integer)) or iterations < 1:
        raise ValueError("iterations must be an integer at least 1")
    psf = _validate_psf_array(psf)
    b = blurred.astype(np.float64)
    h, w = b.shape[:2]
    H = np.fft.fft2(pad_psf(psf, (h, w)))
    Hc = np.conj(H)
    bound = 1e10
    x = b.copy()
    for _ in range(iterations):
        Hx = np.fft.ifft2(np.fft.fft2(x) * H).real
        Hx = np.maximum(Hx, 1e-10)
        ratio = np.clip(b / Hx, 0, bound)
        upd = np.clip(np.fft.ifft2(np.fft.fft2(ratio) * Hc).real, 0, bound)
        x = np.clip(x * upd, 0, None)
    result = x
    return np.clip(result, 0, 255).astype(np.uint8)


def richardson_lucy(blurred, psf, iterations=30):
    """Richardson-Lucy deconvolution for 2-D or independent RGB channels."""
    return _apply_to_image_channels(
        blurred,
        lambda channel: _richardson_lucy_2d(
            channel, psf, iterations=iterations
        ),
    )


def _tv_deconv_2d(blurred, psf, lam=0.002, rho=None, max_iter=30, tol=1e-4):
    """TV-L2 deconvolution via ADMM.
    
    Solves: min_x  ||h*x - b||^2 + lam * ||grad(x)||_1
    
    ADMM splits z = grad(x):
      x: FFT-based least squares
      z: soft thresholding (shrinkage)
      u: dual variable update
    
    Parameters:
        blurred: blurry image (H, W)
        psf: PSF kernel
        lam: TV regularization strength (default 0.04)
        rho: ADMM penalty (default lam * 10)
        max_iter: max iterations (default 30)
        tol: convergence tolerance (default 1e-4)
    Returns:
        deblurred image (H, W), uint8
    """
    _require_finite_positive("lam", lam)
    if rho is None:
        rho = lam * 10
    _require_finite_positive("rho", rho)
    if isinstance(max_iter, bool) or not isinstance(max_iter, (int, np.integer)) or max_iter < 1:
        raise ValueError("max_iter must be an integer at least 1")
    _require_finite_positive("tol", tol)
    psf = _validate_psf_array(psf)
    
    b = blurred.astype(np.float64)
    h, w = b.shape
    
    # PSF in frequency domain
    H_fft = np.fft.fft2(pad_psf(psf, (h, w)))
    HH = np.conj(H_fft) * H_fft
    
    # Gradient operators in frequency domain
    # Dx = 1 - exp(-2*pi*j*u/W),  Dy = 1 - exp(-2*pi*j*v/H)
    u = np.fft.fftfreq(w).reshape(1, -1)
    v = np.fft.fftfreq(h).reshape(-1, 1)
    Dx = 1 - np.exp(-2j * np.pi * u)
    Dy = 1 - np.exp(-2j * np.pi * v)
    DD = np.abs(Dx)**2 + np.abs(Dy)**2
    
    denom = HH + rho * DD
    denom = np.maximum(denom, 1e-10)
    
    # Load B
    B_fft = np.fft.fft2(b)
    
    # ADMM variables
    x = b.copy()
    zx = np.zeros_like(x)
    zy = np.zeros_like(x)
    ux = np.zeros_like(x)
    uy = np.zeros_like(x)
    t = lam / rho
    
    for it in range(max_iter):
        x_old = x
        
        # x-update: FFT
        rhs = np.conj(H_fft) * B_fft + rho * (
            np.conj(Dx) * np.fft.fft2(zx - ux) +
            np.conj(Dy) * np.fft.fft2(zy - uy))
        x = np.fft.ifft2(rhs / denom).real
        
        # Gradients of new x (forward diff, periodic BC)
        gx = np.roll(x, -1, axis=1) - x
        gy = np.roll(x, -1, axis=0) - x
        
        # z-update: soft thresholding
        vx = gx + ux
        vy = gy + uy
        vn = np.sqrt(vx**2 + vy**2 + 1e-10)
        scale = np.maximum(1 - t / vn, 0)
        zx = scale * vx
        zy = scale * vy
        
        # Dual update
        ux += (gx - zx)
        uy += (gy - zy)
        
        # Check convergence
        diff = np.linalg.norm(x - x_old) / max(np.linalg.norm(x), 1)
        if diff < tol:
            break
    
    return np.clip(x, 0, 255).astype(np.uint8)


def tv_deconv(blurred, psf, lam=0.002, rho=None, max_iter=30, tol=1e-4):
    """TV-L2 deconvolution for a 2-D image or independent RGB channels."""
    return _apply_to_image_channels(
        blurred,
        lambda channel: _tv_deconv_2d(
            channel,
            psf,
            lam=lam,
            rho=rho,
            max_iter=max_iter,
            tol=tol,
        ),
    )
