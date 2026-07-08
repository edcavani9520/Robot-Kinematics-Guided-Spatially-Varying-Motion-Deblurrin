"""
evaluate.py — 去模糊效果评估
==============================
包含 PSNR、SSIM、直方图匹配、Tenengrad、Laplacian 锐度等评估函数。
所有评估统一通过 evaluate() 入口调用。
"""

import numpy as np
import math
import cv2


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
    """拉普拉斯方差锐度评估，值越大越清晰。"""
    import cv2
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    lap = cv2.Laplacian(img, cv2.CV_64F)
    return float(lap.var())


def tenengrad(img):
    """
    Tenengrad 清晰度指标：使用 Sobel 算子计算梯度幅值平方和。

    值越大表示图像越清晰，常用于无参考图像质量评估。
    """
    import cv2
    if img.dtype != np.uint8:
        img = np.clip(img, 0, 255).astype(np.uint8)
    gx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx**2 + gy**2)
    return float(np.sum(mag**2))


def match_histogram(source, reference):
    """
    直方图匹配：将 source 的灰度分布映射到 reference 的分布。

    用于去模糊评估：去模糊可能改变图像的整体亮度/对比度，
    在评估前做直方图匹配可以消除这种影响，
    更准确地衡量结构恢复效果。
    """
    src_hist = np.zeros(256, dtype=np.float64)
    ref_hist = np.zeros(256, dtype=np.float64)

    for v in source.ravel():
        src_hist[v] += 1
    for v in reference.ravel():
        ref_hist[v] += 1

    src_cdf = np.cumsum(src_hist) / source.size
    ref_cdf = np.cumsum(ref_hist) / reference.size

    mapping = np.zeros(256, dtype=np.uint8)
    j = 0
    for i in range(256):
        while j < 255 and ref_cdf[j] < src_cdf[i]:
            j += 1
        mapping[i] = j

    return mapping[source.ravel()].reshape(source.shape)


def evaluate(original, processed):
    """
    完整评估：PSNR + SSIM + 直方图匹配后 PSNR/SSIM + Tenengrad + Laplacian 锐度。

    返回字典包含指标：
      PSNR_raw / SSIM_raw               — 直接比较
      PSNR_matched / SSIM_matched       — 直方图匹配后比较
      tenengrad_before / after / change / improved   — Tenengrad 锐度
      laplacian_before / after / change / improved   — Laplacian 锐度
    """
    p_raw = psnr(original, processed)
    s_raw = ssim(original, processed)

    matched = match_histogram(processed, original)
    p_matched = psnr(original, matched)
    s_matched = ssim(original, matched)

    t_before = tenengrad(original)
    t_after = tenengrad(processed)
    t_change = t_after - t_before

    l_before = laplacian_sharpness(original)
    l_after = laplacian_sharpness(processed)
    l_change = l_after - l_before

    return {
        "PSNR_raw": p_raw,
        "SSIM_raw": s_raw,
        "PSNR_matched": p_matched,
        "SSIM_matched": s_matched,
        "tenengrad_before": t_before,
        "tenengrad_after": t_after,
        "tenengrad_change": t_change,
        "tenengrad_improved": t_change > 0,
        "laplacian_before": l_before,
        "laplacian_after": l_after,
        "laplacian_change": l_change,
        "laplacian_improved": l_change > 0,
    }, matched


def compare_sharpness(img_before, img_after):
    """通过 evaluate 进行锐度对比（使用 Tenengrad 指标）。"""
    result, _ = evaluate(img_before, img_after)
    ch = result["tenengrad_change"]
    return {
        "before": result["tenengrad_before"],
        "after": result["tenengrad_after"],
        "change": ch,
        "improved": result["tenengrad_improved"],
        "judgment": (
            "++ SHARPER (Tenengrad)" if ch > 0
            else "-- BLURRIER (Tenengrad)" if ch < 0
            else "   SAME (Tenengrad)"
        ),
    }


def laplacian_variance(img):
    """Laplacian variance sharpness metric. Higher = sharper."""
    lap = cv2.Laplacian(img.astype(np.float64), cv2.CV_64F)
    return float(lap.var())


def image_stats(img):
    """Return dict of basic image statistics."""
    flat = img.ravel()
    hist, _ = np.histogram(flat, bins=256, range=(0, 256))
    hist = hist / flat.size
    entropy = -np.sum(hist * np.log2(hist + 1e-10))
    return {
        "mean": float(img.mean()),
        "std": float(img.std()),
        "min": int(img.min()),
        "max": int(img.max()),
        "median": float(np.median(img)),
        "entropy": float(entropy),
    }


def full_evaluate(original, processed):
    """Complete evaluation: returns all metrics in one dict.
    
    Returns:
        dict with keys: psnr_raw, ssim_raw, psnr_matched, ssim_matched,
                        laplacian_before, laplacian_after,
                        tenengrad_before, tenengrad_after,
                        stats_before, stats_after
    """
    results, matched = evaluate(original, processed)
    lap_b = laplacian_variance(original)
    lap_a = laplacian_variance(processed)
    ten_b = tenengrad(original)
    ten_a = tenengrad(processed)
    stats_b = image_stats(original)
    stats_a = image_stats(processed)
    
    return {
        **results,
        "laplacian_before": lap_b,
        "laplacian_after": lap_a,
        "laplacian_change": lap_a - lap_b,
        "laplacian_improved": lap_a > lap_b,
        "tenengrad_before": ten_b,
        "tenengrad_after": ten_a,
        "tenengrad_change": ten_a - ten_b,
        "tenengrad_improved": ten_a > ten_b,
        "stats_before": stats_b,
        "stats_after": stats_a,
        "matched_image": matched,
    }
