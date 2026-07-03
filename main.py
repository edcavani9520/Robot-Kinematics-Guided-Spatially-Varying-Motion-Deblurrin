"""
main.py — 主函数：逐帧去模糊 pipeline
=========================================
功能：
  1. 读取视频和同步的关节角数据
  2. 逐帧计算 PSF → 去模糊
  3. 保存去模糊后的视频
  4. 如果有清晰真值视频则做评估

输入数据格式（二选一）：
  A. CSV 文件：timestamp, q1..q7, qd1..qd7（每帧一行）
  B. HDF5 文件：来自仿真或 DROID 等数据集的格式

用法示例：
  python main.py --video input.mp4 --joints joint_data.csv
  
  # 带真值评估：
  python main.py --video blurry.mp4 --joints joints.csv --ground_truth clean.mp4
"""

import numpy as np
import cv2
import os, sys, csv, time, argparse, json
from datetime import datetime

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from joint_deblur import compute_psf, wiener_deconvolution, richardson_lucy
from apply_blur import apply_motion_blur, create_motion_psf
from evaluate import evaluate, psnr, ssim


# ============================================================
# 第一部分：数据加载器
# ============================================================

def load_joint_csv(csv_path):
    """
    从 CSV 文件加载关节角数据。
    
    期望格式（每行一帧）：
      timestamp, q1, q2, q3, q4, q5, q6, q7, qd1, qd2, qd3, qd4, qd5, qd6, qd7
    
    返回:
        timestamps: 每个测量时刻的时间戳（秒）
        joint_data: (N, 7) 关节角数组
        joint_vel_data: (N, 7) 关节角速度数组
    """
    timestamps = []
    q_list = []
    qd_list = []
    
    with open(csv_path, "r") as f:
        reader = csv.reader(f)
        header = next(reader, None)  # 跳过表头
        
        for row in reader:
            if len(row) < 15:
                continue
            try:
                t = float(row[0])
                q = np.array([float(row[i]) for i in range(1, 8)])
                qd = np.array([float(row[i]) for i in range(8, 15)])
                timestamps.append(t)
                q_list.append(q)
                qd_list.append(qd)
            except (ValueError, IndexError):
                continue
    
    print("Loaded %d joint states from %s" % (len(timestamps), csv_path))
    if len(timestamps) == 0:
        raise ValueError("No valid joint data found in %s" % csv_path)
    
    return np.array(timestamps), np.array(q_list), np.array(qd_list)


def find_nearest_joint(frame_timestamp, joint_timestamps, q_list, qd_list):
    """
    找到与视频帧时间戳最接近的关节角数据。
    
    对于每个视频帧，在关节角时间序列中找
    时间差最小的那一帧作为匹配。
    """
    idx = np.argmin(np.abs(joint_timestamps - frame_timestamp))
    return q_list[idx], qd_list[idx], joint_timestamps[idx]


# ============================================================
# 第二部分：摄像头实时采集模式
# ============================================================

def load_camera_or_video(video_path=None):
    """
    打开视频文件或摄像头。
    
    如果 video_path 是 None 或 "camera", 则打开默认摄像头。
    否则打开视频文件。
    """
    if video_path is None or video_path.lower() == "camera":
        cap = cv2.VideoCapture(0)
        print("Opening default camera")
    else:
        cap = cv2.VideoCapture(video_path)
        print("Opening video: %s" % video_path)
    
    if not cap.isOpened():
        raise IOError("Cannot open video source: %s" % video_path)
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print("  FPS: %.1f, Size: %dx%d, Frames: %d" % (fps, w, h, total))
    return cap, fps, w, h, total


# ============================================================
# 第三部分：核心去模糊循环
# ============================================================

def compute_median_depth_from_frame(frame):
    """
    从图像中估计场景的中位深度。
    
    这个方法返回一个默认值 0.5m，因为单目 RGB 图像
    无法直接获取深度。实际使用时可以从深度图传感器读取，
    或使用单目深度估计模型（如 MiDaS）。
    
    如果你有深度图，可以直接替换这个函数。
    """
    return 0.5  # 默认物距 0.5m


def process_frame(frame_gray, q, q_dot, params):
    """
    对单帧图像执行去模糊。
    
    参数:
        frame_gray: 灰度图像 (H, W)
        q, q_dot: 该帧对应的关节角和角速度
        params: 相机参数字典
    
    返回:
        deblurred: 去模糊后的图像
        psf: 使用的 PSF 核
        metrics: 评估结果（如果没有真值则为 None）
    """
    h, w = frame_gray.shape
    
    # 从关节角计算 PSF
    psf, (du, dv) = compute_psf(
        q, q_dot,
        depth=params.get("depth", 0.5),
        fx=params.get("fx", 500),
        fy=params.get("fy", 500),
        cx=w // 2,
        cy=h // 2,
        exposure_time=params.get("exposure_time", 0.03),
        noise_level=params.get("noise_level", 0.0)
    )
    
    # 选择去模糊方法
    method = params.get("method", "wiener")
    
    if method == "wiener":
        deblurred = wiener_deconvolution(frame_gray, psf, K=params.get("K", 0.01))
    elif method == "rl":
        deblurred = richardson_lucy(frame_gray, psf, iterations=params.get("rl_iters", 30))
    else:
        raise ValueError("Unknown method: %s" % method)
    
    return deblurred, psf, (du, dv)


# ============================================================
# 第四部分：主函数 — 逐帧去模糊
# ============================================================

def run_deblur_pipeline(video_path, joint_csv_path, output_dir, 
                         ground_truth_path=None, camera_params=None,
                         method="wiener", K=0.01, rl_iters=30,
                         max_frames=None, skip_blank=True):
    """
    完整 pipeline：逐帧去模糊 + 评估。
    
    参数:
        video_path: 输入视频路径（或 "camera"）
        joint_csv_path: 关节角 CSV 路径
        output_dir: 输出目录
        ground_truth_path: 清晰真值视频路径（可选）
        camera_params: 相机参数字典
        method: "wiener" 或 "rl"
        K: 维纳滤波参数
        rl_iters: RL 迭代次数
        max_frames: 最大处理帧数（None=全部）
        skip_blank: 是否跳过全黑或全白帧
    
    输出:
        output_dir/deblurred_video.mp4 — 去模糊后的视频
        output_dir/summary.json — 逐帧评估结果
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 默认相机参数
    if camera_params is None:
        camera_params = {}
    camera_params.setdefault("fx", 500)
    camera_params.setdefault("fy", 500)
    camera_params.setdefault("depth", 0.5)
    camera_params.setdefault("exposure_time", 0.03)
    camera_params.setdefault("noise_level", 0.0)
    camera_params["method"] = method
    camera_params["K"] = K
    camera_params["rl_iters"] = rl_iters
    
    print("=" * 60)
    print("  Deblur Pipeline — Frame by Frame")
    print("=" * 60)
    print()
    print("Camera params:", camera_params)
    print("Output dir:   ", output_dir)
    print()
    
    # 加载关节角数据
    joint_timestamps, q_all, qd_all = load_joint_csv(joint_csv_path)
    
    # 加载视频
    cap, fps, w, h, total = load_camera_or_video(video_path)
    
    # 加载真值视频（可选）
    gt_cap = None
    if ground_truth_path:
        gt_cap = cv2.VideoCapture(ground_truth_path)
        print("Loaded ground truth video: %s" % ground_truth_path)
    
    # 创建输出视频 writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out_path = os.path.join(output_dir, "deblurred_video.mp4")
    writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h), isColor=False)
    print("Writing deblurred video to: %s" % out_path)
    
    # 逐帧处理
    frame_idx = 0
    start_time = time.time()
    
    # 记录每帧结果
    results = {
        "params": {
            "method": method,
            "K": K,
            "rl_iters": rl_iters,
            "camera_params": {k: float(v) if isinstance(v, (int, float)) else v 
                             for k, v in camera_params.items()}
        },
        "frames": []
    }
    
    save_interval = max(1, total // 10) if total > 0 else 10
    
    while True:
        ret, frame_bgr = cap.read()
        if not ret:
            break
        
        frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        # 跳过无效帧
        if skip_blank and frame_gray.std() < 2:
            frame_idx += 1
            continue
        
        # 获取该帧时间戳（用帧率估算）
        frame_timestamp = frame_idx / fps
        
        # 找最近的关节角数据
        q, q_dot, matched_t = find_nearest_joint(
            frame_timestamp, joint_timestamps, q_all, qd_all)
        
        # 去模糊
        deblurred, psf, (du, dv) = process_frame(
            frame_gray, q, q_dot, camera_params)
        
        # 写输出视频
        writer.write(deblurred)
        
        # 读真值（如果有）
        gt_gray = None
        if gt_cap:
            ret_gt, gt_bgr = gt_cap.read()
            if ret_gt:
                gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
        
        # 评估
        frame_result = {
            "frame": frame_idx,
            "timestamp": frame_timestamp,
            "joint_timestamp": float(matched_t),
            "pixel_displacement_du": float(du),
            "pixel_displacement_dv": float(dv),
            "psf_size": psf.shape[0],
        }
        
        if gt_gray is not None:
            m, _ = evaluate(gt_gray, deblurred)
            frame_result["PSNR_raw"] = round(m["PSNR_raw"], 2)
            frame_result["SSIM_raw"] = round(m["SSIM_raw"], 4)
            frame_result["PSNR_matched"] = round(m["PSNR_matched"], 2)
            
            # 也评估模糊前的帧（如果视频本身就是模糊的）
            m_blur, _ = evaluate(gt_gray, frame_gray)
            frame_result["blur_PSNR_raw"] = round(m_blur["PSNR_raw"], 2)
            frame_result["improvement"] = round(
                m["PSNR_raw"] - m_blur["PSNR_raw"], 2)
        
        results["frames"].append(frame_result)
        
        # 打印进度
        if frame_idx % save_interval == 0:
            elapsed = time.time() - start_time
            fps_proc = (frame_idx + 1) / elapsed if elapsed > 0 else 0
            info = "Frame %d | du=%.1f dv=%.1f" % (frame_idx, du, dv)
            if gt_gray is not None:
                info += " | PSNR: blur=%.1f deblur=%.1f" % (
                    frame_result.get("blur_PSNR_raw", 0),
                    frame_result["PSNR_raw"])
                info += " | Improv: +%.1f" % frame_result["improvement"]
            info += " | %.1f fps" % fps_proc
            print(info)
        
        frame_idx += 1
        if max_frames and frame_idx >= max_frames:
            break
    
    # 清理
    cap.release()
    writer.release()
    if gt_cap:
        gt_cap.release()
    
    total_time = time.time() - start_time
    print()
    print("Done. Processed %d frames in %.1f seconds (%.1f fps)" % (
        frame_idx, total_time, frame_idx / total_time))
    
    # 汇总统计
    if results["frames"]:
        psnrs = [f.get("PSNR_raw", 0) for f in results["frames"] if "PSNR_raw" in f]
        imprs = [f.get("improvement", 0) for f in results["frames"] if "improvement" in f]
        if psnrs:
            results["summary"] = {
                "total_frames": len(results["frames"]),
                "avg_deblur_PSNR": round(np.mean(psnrs), 2),
                "std_deblur_PSNR": round(np.std(psnrs), 2),
                "min_deblur_PSNR": round(min(psnrs), 2),
                "max_deblur_PSNR": round(max(psnrs), 2),
            }
            if imprs:
                results["summary"]["avg_improvement"] = round(np.mean(imprs), 2)
                results["summary"]["std_improvement"] = round(np.std(imprs), 2)
    
    # 保存结果
    summary_path = os.path.join(output_dir, "summary.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print("Summary saved to: %s" % summary_path)
    
    if "summary" in results:
        s = results["summary"]
        print()
        print("=== Final Summary ===")
        for k, v in s.items():
            print("  %s: %s" % (k, v))
    
    return results


# ============================================================
# 第五部分：命令行入口
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Kinematics-Guided Motion Deblurring for Robot Video")
    
    parser.add_argument("--video", type=str, required=True,
                        help="Input video path (or 'camera' for webcam)")
    parser.add_argument("--joints", type=str, required=True,
                        help="Joint angle CSV: timestamp,q1..q7,qd1..qd7")
    parser.add_argument("--output", type=str, default="deblur_output",
                        help="Output directory (default: deblur_output)")
    parser.add_argument("--gt", type=str, default=None,
                        help="Ground truth video path (optional)")
    
    # 相机参数
    parser.add_argument("--fx", type=float, default=500, help="Focal length x")
    parser.add_argument("--fy", type=float, default=500, help="Focal length y")
    parser.add_argument("--depth", type=float, default=0.5, help="Depth (m)")
    parser.add_argument("--exposure", type=float, default=0.03,
                        help="Exposure time (s)")
    
    # 去模糊参数
    parser.add_argument("--method", choices=["wiener", "rl"], default="wiener",
                        help="Deblurring method")
    parser.add_argument("--K", type=float, default=0.01,
                        help="Wiener K parameter (small=strong deblur)")
    parser.add_argument("--rl-iters", type=int, default=30,
                        help="RL iterations")
    
    # 控制参数
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames to process")
    parser.add_argument("--no-skip", action="store_true",
                        help="Don't skip blank frames")
    
    args = parser.parse_args()
    
    run_deblur_pipeline(
        video_path=args.video,
        joint_csv_path=args.joints,
        output_dir=args.output,
        ground_truth_path=args.gt,
        camera_params={
            "fx": args.fx,
            "fy": args.fy,
            "depth": args.depth,
            "exposure_time": args.exposure,
        },
        method=args.method,
        K=args.K,
        rl_iters=args.rl_iters,
        max_frames=args.max_frames,
        skip_blank=not args.no_skip
    )


if __name__ == "__main__":
    main()
