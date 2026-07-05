"""
apply_blur.py — 给图片加运动模糊
===================================
功能：用关节角数据或手动指定的 PSF 给图像加运动模糊。

核心函数：
  create_motion_psf(du, dv)         — 从像素位移生成 PSF 核
  apply_motion_blur(img, psf)       — 用 FFT 卷积加均匀模糊
  apply_spatial_blur(img, psf_map)  — 对图像不同区域加不同模糊
"""

import numpy as np
import math


def create_motion_psf(du, dv):
    """
    从像素位移 (du, dv) 创建运动模糊 PSF 核。
    
    原理：在空域沿运动方向画一条线段，归一化使总能量 = 1。
    
    参数:
        du, dv: 曝光时间内 x/y 方向的像素位移量
    返回:
        (ksize, ksize) 的归一化 PSF 核
    """
    ksize = max(2 * int(abs(du)), 2 * int(abs(dv))) + 1
    ksize = max(ksize, 3)
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    
    psf = np.zeros((ksize, ksize), dtype=np.float32)
    c = ksize // 2
    steps = max(int(math.hypot(du, dv) * 2.5), 200)
    
    for t in np.linspace(0, 1, steps):
        x = int(round(c + t * dv))
        y = int(round(c + t * du))
        if 0 <= x < ksize and 0 <= y < ksize:
            psf[x, y] += 1.0
    
    s = psf.sum()
    return psf / s if s > 0 else psf


def pad_psf(psf, shape):
    """
    将 PSF 填充到图像大小，并将 PSF 中心对齐到 (0,0)。
    
    这是 FFT 卷积的关键步骤：PSF 的质心必须放在 (0,0)，
    否则卷积结果会有平移偏移。
    """
    h, w = shape[:2]
    kh, kw = psf.shape
    cy, cx = np.unravel_index(psf.argmax(), psf.shape)
    
    pad = np.zeros((h, w), dtype=np.float64)
    pad[:kh, :kw] = psf
    pad = np.roll(pad, -cy, axis=0)
    pad = np.roll(pad, -cx, axis=1)
    return pad


def apply_motion_blur(img, psf):
    """
    用 FFT 卷积对图像施加运动模糊。
    
    退化模型: g = f * h (空域)
    频域实现: G = FFT(g), H = FFT(padded_psf)
              g = IFFT(G * H)
    
    使用 FFT 而不是空域 filter2D 是为了确保
    加模糊和去模糊使用相同的卷积模型。
    """
    H = np.fft.fft2(pad_psf(psf, img.shape[:2]))
    result = np.fft.ifft2(np.fft.fft2(img.astype(np.float64)) * H).real
    return np.clip(result, 0, 255).astype(np.uint8)


def apply_spatial_blur(img, psf_map, grid_rows=4, grid_cols=4, overlap=0.25):
    """
    对图像施加空间变化的运动模糊。
    
    不同图像区域使用不同的 PSF（因为旋转运动时
    图像不同位置的模糊方向/长度不同）。
    
    参数:
        img: 输入图像 (H, W)
        psf_map: (grid_rows, grid_cols) 的 PSF 核数组
        grid_rows, grid_cols: 网格划分数量
    """
    H, W = img.shape[:2]
    out = np.zeros_like(img, dtype=np.float64)
    wt = np.zeros_like(img, dtype=np.float64)
    
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
            
            patch = img[y0:y1, x0:x1]
            blurred_patch = apply_motion_blur(patch, psf_map[r][c])
            
            wy = 0.5 - 0.5 * np.cos(np.pi * np.arange(y1 - y0) / (y1 - y0))
            wx = 0.5 - 0.5 * np.cos(np.pi * np.arange(x1 - x0) / (x1 - x0))
            w = np.outer(wy, wx)
            
            out[y0:y1, x0:x1] += blurred_patch.astype(np.float64) * w
            wt[y0:y1, x0:x1] += w
    
    return (out / np.maximum(wt, 1e-10)).astype(np.uint8)
