"""
main.py — 主函数：逐帧去模糊 pipeline
=========================================
功能：
  1. 读取视频或图片帧序列 + 同步的关节角数据
  2. 逐帧计算 PSF → 去模糊
  3. 保存去模糊后的视频或图片
  4. 如果有清晰真值视频则做评估

输入数据格式：
  A. 视频 + CSV 关节角
  B. 图片目录（step_XXXX_CAMERA.jpg）+ actions.csv

用法示例：
  python main.py --video input.mp4 --joints joint_data.csv
  python main.py --frames-dir ../droid/output_frames/ --camera 17368348

  带真值评估：
  python main.py --video blurry.mp4 --joints joints.csv --ground_truth clean.mp4
"""

import numpy as np
import cv2
import os, sys, csv, time, argparse, json, re
from datetime import datetime
from pathlib import Path

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
    从 CSV 文件加载关节角数据（标准格式）。
    
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
        header = next(reader, None)
        
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


def load_droid_actions_csv(csv_path):
    """
    加载 DROID 数据集导出的 actions.csv（来自 extract_sync_frames.py）。
    
    格式：step_idx, timestamp_ms, video_frame,
          action_joint_0..6, obs_joint_0..6,
          cart_x..z, cart_rotx..z, gripper_pos, ...
    
    从中提取：
      - timestamps: 每帧时间戳（秒）
      - q: 关节角 (N, 7) —— 使用 action_joint
      - q_dot: 用有限差分计算的关节角速度 (N, 7)
    
    返回:
        timestamps, q, q_dot
    """
    timestamps = []
    q_list = []
    
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            q = np.array([float(row[f"action_joint_{i}"]) for i in range(7)])
            t = float(row["timestamp_ms"]) / 1000.0  # ms → s
            timestamps.append(t)
            q_list.append(q)
    
    q = np.array(q_list)
    timestamps = np.array(timestamps)
    
    # 用有限差分计算 q_dot（前向差分，最后一帧用后向）
    qd = np.zeros_like(q)
    dt = np.diff(timestamps)
    for i in range(7):
        qd[:-1, i] = np.diff(q[:, i]) / np.maximum(dt, 1e-6)
        qd[-1, i] = qd[-2, i]  # 最后一帧拷贝前一帧速度
    
    print("Loaded %d DROID action frames from %s" % (len(timestamps), csv_path))
    print("  q range: [%.3f, %.3f]" % (q.min(), q.max()))
    print("  qd range: [%.3f, %.3f]" % (qd.min(), qd.max()))
    
    # 平滑速度（可选，减少噪声）
    # qd = np.apply_along_axis(lambda x: np.convolve(x, np.ones(3)/3, mode='same'), 0, qd)
    
    return timestamps, q, qd


def load_frames_from_dir(frames_dir, camera_serial, max_frames=None):
    """
    从图片目录加载帧序列。
    
    匹配 pattern: step_{step_idx:04d}_{camera_serial}.jpg
    帧按 step_idx 排序。
    
    返回:
        frame_paths: 按顺序的图片路径列表
        step_indices: 对应 step_idx
    """
    frames_dir = Path(frames_dir)
    pattern = re.compile(r"step_(\d{4})_" + re.escape(camera_serial) + r"\.jpg$")
    
    entries = []
    for f in frames_dir.iterdir():
        m = pattern.match(f.name)
        if m:
            entries.append((int(m.group(1)), str(f)))
    
    entries.sort(key=lambda x: x[0])
    
    if max_frames:
        entries = entries[:max_frames]
    
    paths = [e[1] for e in entries]
    indices = [e[0] for e in entries]
    
    print("Loaded %d frame images from %s (camera=%s)" % (
        len(paths), frames_dir, camera_serial))
    if paths:
        print("  Range: step_%04d ~ step_%04d" % (indices[0], indices[-1]))
    
    return paths, indices


def find_nearest_joint(frame_timestamp, joint_timestamps, q_list, qd_list):
    """
    找到与视频帧时间戳最接近的关节角数据。
    
    对于每个视频帧，在关节角时间序列中找
    时间差最小的那一帧作为匹配。
    """
    idx = np.argmin(np.abs(joint_timestamps - frame_timestamp))
    return q_list[idx], qd_list[idx], joint_timestamps[idx]


# ============================================================
# 第二部分：摄像头 / 视频加载
# ============================================================

def load_camera_or_video(video_path=None):
    """打开视频文件或摄像头。"""
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
# 第三部分：核心去模糊处理
# ============================================================

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
        (du, dv): 像素位移
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
# 第四部分：主去模糊 Pipeline
# ============================================================

def run_deblur_pipeline(video_path, joint_csv_path, output_dir,
                         ground_truth_path=None, camera_params=None,
                         method="wiener", K=0.01, rl_iters=30,
                         max_frames=None, skip_blank=True,
                         # DROID frames mode
                         frames_dir=None, camera_serial=None):
    """
    完整 pipeline：逐帧去模糊 + 评估。
    
    支持两种输入模式：
      1. 视频文件模式：video_path + joint_csv 标准格式
      2. 图片帧模式：frames_dir + droid_actions.csv 格式
    
    参数:
        video_path: 输入视频路径（或 "camera"）
        joint_csv_path: 关节角 CSV 路径
        output_dir: 输出目录
        ground_truth_path: 清晰真值视频路径（可选）
        camera_params: 相机参数字典
        method: "wiener" 或 "rl"
        K: 维纳滤波参数
        rl_iters: RL 迭代次数
        max_frames: 最大处理帧数
        skip_blank: 是否跳过全黑或全白帧
        frames_dir: DROID 图片帧目录（替代 video_path）
        camera_serial: 相机序列号
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
    
    is_frames_mode = frames_dir is not None
    
    if is_frames_mode:
        # ====== DROID 图片帧模式 ======
        print("Mode: DROID frame sequence")
        print()
        
        # 加载 actions.csv（DROID 格式）
        if os.path.isdir(joint_csv_path):
            joint_csv_path = os.path.join(joint_csv_path, "actions.csv")
        joint_timestamps, q_all, qd_all = load_droid_actions_csv(joint_csv_path)
        
        # 加载图片帧
        frame_paths, step_indices = load_frames_from_dir(
            frames_dir, camera_serial, max_frames)
        
        if len(frame_paths) == 0:
            raise ValueError("No frames found matching camera %s in %s" %
                             (camera_serial, frames_dir))
        
        # 建立 step_idx → joint data 的映射
        step_to_joint = {}  # step_idx → (q, q_dot, timestamp)
        for i, si in enumerate(step_indices):
            # step_indices 直接对应 actions.csv 的行号
            if si < len(q_all):
                step_to_joint[si] = (q_all[si], qd_all[si], joint_timestamps[si])
        
        total = len(frame_paths)
        fps = 15.0  # 参考帧率，用于 writer
        
        # 对于图片模式，创建视频只是辅助功能
        # 先读第一帧获取尺寸
        first_img = cv2.imread(frame_paths[0], cv2.IMREAD_GRAYSCALE)
        h, w = first_img.shape
        create_video = True
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video_path = os.path.join(output_dir, "deblurred_video.mp4")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), isColor=False)
        
        # 逐帧处理
        frame_idx = 0
        start_time = time.time()
        results = {
            "params": {
                "mode": "frames",
                "frames_dir": str(frames_dir),
                "camera_serial": camera_serial,
                "method": method,
                "K": K,
                "rl_iters": rl_iters,
                "camera_params": {k: float(v) if isinstance(v, (int, float)) else v
                                 for k, v in camera_params.items()}
            },
            "frames": []
        }
        
        save_interval = max(1, total // 10)
        
        for fi, (frame_path, step_i) in enumerate(zip(frame_paths, step_indices)):
            # 读图
            frame_bgr = cv2.imread(frame_path)
            if frame_bgr is None:
                print("  WARN: cannot read %s, skipping" % frame_path)
                continue
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            # 获取对应关节角
            if step_i in step_to_joint:
                q, q_dot, matched_t = step_to_joint[step_i]
            else:
                # step_idx 超出范围，用最近的时间匹配
                frame_t = step_i / fps if not is_frames_mode else step_i * 0.072
                q, q_dot, matched_t = find_nearest_joint(
                    frame_t, joint_timestamps, q_all, qd_all)
            
            # 去模糊
            deblurred, psf, (du, dv) = process_frame(
                frame_gray, q, q_dot, camera_params)
            
            # 评估去模糊前后质量 (compare original vs deblurred)
            eval_metrics, _ = evaluate(frame_gray, deblurred)
            
            # 保存去模糊后的图片
            out_img_path = os.path.join(
                output_dir, "deblurred_step_%04d_%s.jpg" % (step_i, camera_serial))
            cv2.imwrite(out_img_path, deblurred)
            
            # 也保存原图 + 去模糊的对比图
            compare = np.hstack([
                cv2.cvtColor(frame_gray, cv2.COLOR_GRAY2BGR),
                cv2.cvtColor(deblurred, cv2.COLOR_GRAY2BGR)
            ])
            cv2.putText(compare, "Original", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(compare, "Deblurred", (w + 10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.imwrite(os.path.join(output_dir, "compare_step_%04d.jpg" % step_i), compare)
            
            # 写入视频
            writer.write(deblurred)
            
            # 记录结果 (含评估指标)
            frame_result = {
                "frame": fi,
                "step_idx": int(step_i),
                "image": os.path.basename(frame_path),
                "joint_timestamp_s": float(matched_t),
                "pixel_displacement_du": float(du),
                "pixel_displacement_dv": float(dv),
                "psf_size": psf.shape[0],
                "psf_norm": float(np.sum(psf ** 2)),
                # 去模糊评估指标 (original vs deblurred)
                "PSNR": round(float(eval_metrics["PSNR_raw"]), 2),
                "SSIM": round(float(eval_metrics["SSIM_raw"]), 4),
                "PSNR_matched": round(float(eval_metrics["PSNR_matched"]), 2),
                "SSIM_matched": round(float(eval_metrics["SSIM_matched"]), 4),
            }
            results["frames"].append(frame_result)
            
            # 进度 (含评估指标)
            if fi % save_interval == 0:
                elapsed = time.time() - start_time
                fps_proc = (fi + 1) / elapsed if elapsed > 0 else 0
                print("Frame %d/%d | step#%04d | du=%.2f dv=%.2f | "
                      "PSNR=%.2f SSIM=%.4f | %.1f fps" % (
                    fi, total, step_i, du, dv,
                    frame_result["PSNR"], frame_result["SSIM"], fps_proc))
            
            frame_idx += 1
        
        writer.release()
        
    else:
        # ====== 视频模式（原有逻辑） ======
        joint_timestamps, q_all, qd_all = load_joint_csv(joint_csv_path)
        cap, fps, w, h, total = load_camera_or_video(video_path)
        
        gt_cap = None
        if ground_truth_path:
            gt_cap = cv2.VideoCapture(ground_truth_path)
            print("Loaded ground truth video: %s" % ground_truth_path)
        
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_video_path = os.path.join(output_dir, "deblurred_video.mp4")
        writer = cv2.VideoWriter(out_video_path, fourcc, fps, (w, h), isColor=False)
        print("Writing deblurred video to: %s" % out_video_path)
        
        results = {
            "params": {
                "mode": "video",
                "video": str(video_path),
                "method": method,
                "K": K,
                "rl_iters": rl_iters,
                "camera_params": {k: float(v) if isinstance(v, (int, float)) else v
                                 for k, v in camera_params.items()}
            },
            "frames": []
        }
        
        save_interval = max(1, total // 10) if total > 0 else 10
        frame_idx = 0
        start_time = time.time()
        
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            
            frame_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
            
            if skip_blank and frame_gray.std() < 2:
                frame_idx += 1
                continue
            
            frame_timestamp = frame_idx / fps
            q, q_dot, matched_t = find_nearest_joint(
                frame_timestamp, joint_timestamps, q_all, qd_all)
            
            deblurred, psf, (du, dv) = process_frame(
                frame_gray, q, q_dot, camera_params)
            
            writer.write(deblurred)
            
            gt_gray = None
            if gt_cap:
                ret_gt, gt_bgr = gt_cap.read()
                if ret_gt:
                    gt_gray = cv2.cvtColor(gt_bgr, cv2.COLOR_BGR2GRAY)
            
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
                
                m_blur, _ = evaluate(gt_gray, frame_gray)
                frame_result["blur_PSNR_raw"] = round(m_blur["PSNR_raw"], 2)
                frame_result["improvement"] = round(
                    m["PSNR_raw"] - m_blur["PSNR_raw"], 2)
            
            results["frames"].append(frame_result)
            
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
        
        cap.release()
        if gt_cap:
            gt_cap.release()
    
    writer.release()
    
    total_time = time.time() - start_time
    processed = len(results["frames"])
    print()
    print("Done. Processed %d frames in %.1f seconds (%.1f fps)" % (
        processed, total_time, processed / total_time))
    
    # 汇总统计
    if results["frames"]:
        dus = [f["pixel_displacement_du"] for f in results["frames"]]
        dvs = [f["pixel_displacement_dv"] for f in results["frames"]]
        
        # 兼容 frames 模式和 video 模式的指标键名
        psnrs = [f.get("PSNR", f.get("PSNR_raw", 0))
                 for f in results["frames"] if "PSNR" in f or "PSNR_raw" in f]
        ssims = [f.get("SSIM", f.get("SSIM_raw", 0))
                 for f in results["frames"] if "SSIM" in f or "SSIM_raw" in f]
        psnrs_m = [f.get("PSNR_matched", 0)
                   for f in results["frames"] if "PSNR_matched" in f]
        ssims_m = [f.get("SSIM_matched", 0)
                   for f in results["frames"] if "SSIM_matched" in f]
        imprs = [f.get("improvement", 0)
                 for f in results["frames"] if "improvement" in f]
        
        results["summary"] = {
            "total_frames": processed,
            "avg|du|": float(np.mean(np.abs(dus))),
            "avg|dv|": float(np.mean(np.abs(dvs))),
            "max|du|": float(np.max(np.abs(dus))),
            "max|dv|": float(np.max(np.abs(dvs))),
        }
        if psnrs:
            results["summary"].update({
                "avg_deblur_PSNR": round(np.mean(psnrs), 2),
                "std_deblur_PSNR": round(np.std(psnrs), 2),
                "min_deblur_PSNR": round(min(psnrs), 2),
                "max_deblur_PSNR": round(max(psnrs), 2),
            })
        if ssims:
            results["summary"].update({
                "avg_deblur_SSIM": round(np.mean(ssims), 4),
                "std_deblur_SSIM": round(np.std(ssims), 4),
                "min_deblur_SSIM": round(min(ssims), 4),
                "max_deblur_SSIM": round(max(ssims), 4),
            })
        if psnrs_m:
            results["summary"].update({
                "avg_PSNR_matched": round(np.mean(psnrs_m), 2),
            })
        if ssims_m:
            results["summary"].update({
                "avg_SSIM_matched": round(np.mean(ssims_m), 4),
            })
        if imprs:
            results["summary"].update({
                "avg_improvement": round(np.mean(imprs), 2),
            })
    
    # 保存结果 JSON
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
    
    # 输入模式二选一
    parser.add_argument("--video", type=str, default=None,
                        help="Input video path (or 'camera' for webcam)")
    parser.add_argument("--frames-dir", type=str, default=None,
                        help="DROID frames directory (alternative to --video)")
    parser.add_argument("--camera", type=str, default="17368348",
                        help="Camera serial for frames mode (default: 17368348)")
    parser.add_argument("--joints", type=str, required=True,
                        help="Joint angle CSV (standard or DROID actions.csv)")
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
    
    # 参数校验
    if not args.frames_dir and not args.video:
        parser.error("Either --video or --frames-dir must be provided.")
    
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
        skip_blank=not args.no_skip,
        frames_dir=args.frames_dir,
        camera_serial=args.camera,
    )


if __name__ == "__main__":
    main()
