"""
evaluate.py — 去模糊效果评估
==============================
包含 PSNR、SSIM、直方图匹配、含配准的评估函数。
"""

import numpy as np
import math


def psnr(img1, img2):
    """
    PSNR = 20 * log10(255 / sqrt(MSE))
    
    衡量两幅图像的像素级差异。
    >40dB 极好, 30-40dB 好, 20-30dB 可接受
    """
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    mse = np.mean((i1 - i2) ** 2)
    if mse < 1e-10:
        return float("inf")
    return 20 * math.log10(255 / math.sqrt(mse))


def ssim(img1, img2):
    """
    SSIM(x,y) = (2*ux*uy+C1)(2*sig_xy+C2) / ((ux^2+uy^2+C1)(sig_x^2+sig_y^2+C2))
    
    衡量结构相似度，范围 [0, 1]，越接近 1 越好。
    比 PSNR 更符合人眼感知。
    """
    i1 = img1.astype(np.float64)
    i2 = img2.astype(np.float64)
    
    mu1 = np.mean(i1)
    mu2 = np.mean(i2)
    s1 = np.var(i1)
    s2 = np.var(i2)
    s12 = np.mean((i1 - mu1) * (i2 - mu2))
    
    if s1 + s2 == 0:
        return 1.0
    
    C1 = (0.01 * 255) ** 2
    C2 = (0.03 * 255) ** 2
    
    numerator = (2 * mu1 * mu2 + C1) * (2 * s12 + C2)
    denominator = (mu1**2 + mu2**2 + C1) * (s1 + s2 + C2)
    return numerator / denominator

def laplacian_sharpness(img):
    import cv2
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def compare_sharpness(img_before, img_after):
    b = laplacian_sharpness(img_before)
    a = laplacian_sharpness(img_after)
    ch = a - b
    return {
        "before": b,
        "after": a,
        "change": ch,
        "improved": a > b,
        "judgment": "++ SHARPER" if ch > 0 else "-- BLURRIER" if ch < 0 else "   SAME",
    }


def match_histogram(source, reference):
    """
    直方图匹配：将 source 的灰度分布映射到 reference 的分布。
    
    用于去模糊评估：去模糊可能改变图像的整体亮度/对比度，
    在评估前做直方图匹配可以消除这种影响，
    更准确地衡量结构恢复效果。
    """
    # 计算直方图
    src_hist = np.zeros(256, dtype=np.float64)
    ref_hist = np.zeros(256, dtype=np.float64)
    
    for v in source.ravel():
        src_hist[v] += 1
    for v in reference.ravel():
        ref_hist[v] += 1
    
    # 累计分布函数
    src_cdf = np.cumsum(src_hist) / source.size
    ref_cdf = np.cumsum(ref_hist) / reference.size
    
    # 映射表：对 source 的每个灰度值，找到 ref 中 CDF 最接近的值
    mapping = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        mapping[i] = j
    
    return mapping[source.ravel()].reshape(source.shape)


def evaluate(original, processed):
    """
    完整评估：PSNR + SSIM + 直方图匹配后 PSNR/SSIM。
    
    返回字典包含四个指标：
      PSNR_raw:    直接比较
      SSIM_raw:    直接比较
      PSNR_matched: 直方图匹配后比较
      SSIM_matched: 直方图匹配后比较
    """
    p_raw = psnr(original, processed)
    s_raw = ssim(original, processed)
    
    matched = match_histogram(processed, original)
    p_matched = psnr(original, matched)
    s_matched = ssim(original, matched)
    
    return {
        "PSNR_raw": p_raw,
        "SSIM_raw": s_raw,
        "PSNR_matched": p_matched,
        "SSIM_matched": s_matched,
    }, matched
