# Robot-Kinematics-Guided Spatially-Varying Motion Deblurring

利用 Franka Panda 机器人关节角数据，通过正向运动学推算相机速度，为图像中每个像素区域计算空间变化的运动模糊 PSF，再通过非盲反卷积（Wiener / Richardson-Lucy）去除运动模糊。

## 概述

传统去模糊方法假设全图模糊均匀（单一 PSF），但在机器人场景中，相机安装在机械臂末端，**旋转运动导致图像不同区域的模糊方向和长度不同**。

本项目使用机器人运动学精确建模这一过程：

```
关节角 q  ─→  正运动学  ─→  雅可比  ─→  末端速度
                                            │
                        手眼标定  ───────────┤
                                            ▼
                                       相机速度 v_cam
                                            │
                   交互矩阵（像素-速度雅可比）─┤
                                            ▼
                                    像素位移 (du, dv)
                                            │
                                      运动模糊 PSF
                                            │
                                   Wiener / RL 反卷积
                                            ▼
                                      去模糊图像
```

## 文件结构

```
├── main.py             主函数 — 逐帧去模糊 pipeline
│                       支持两种输入模式：
│                         A. 视频 + 关节角 CSV
│                         B. 图片帧目录 + actions.csv (DROID 格式)
├── joint_deblur.py     核心算法 — 关节角 → PSF → 反卷积
│                       包含：正运动学、雅可比、手眼标定、
│                       交互矩阵、PSF 生成、Wiener/RL 去模糊
├── apply_blur.py       模糊工具 — 从像素位移生成 PSF 核，
│                       提供均匀模糊和空间变化模糊
├── evaluate.py         评估工具 — PSNR, SSIM, 直方图匹配评估
```

## 环境要求

- Python 3.8+
- NumPy
- OpenCV (`cv2`)
- h5py（可选，用于 HDF5 数据）

```bash
pip install numpy opencv-python h5py
```

## 用法

### 1. 图片帧模式（DROID 数据集）

从 `extract_sync_frames.py` 导出的图片帧 + actions.csv 做去模糊：

```bash
python main.py \
  --frames-dir ../output_frames \
  --camera 17368348 \
  --joints ../output_frames/actions.csv \
  --output deblur_output \
  --fx 600 --fy 600 \
  --depth 0.6 \
  --exposure 0.03 \
  --method wiener --K 0.01
```

所有参数均可选，常用调参：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--depth` | 0.5 | 物距（米），越大 PSF 越小 |
| `--exposure` | 0.03 | 曝光时间（秒），越长模糊越强 |
| `--K` | 0.01 | Wiener 参数，越小去模糊越强（噪声放大也越大） |
| `--method` | wiener | `wiener` 或 `rl`（Richardson-Lucy） |
| `--rl-iters` | 30 | RL 迭代次数（仅 method=rl） |
| `--fx, --fy` | 500 | 相机焦距（像素） |
| `--max-frames` | 全部 | 限制处理帧数 |

### 2. 视频模式

```bash
python main.py \
  --video blurry.mp4 \
  --joints joint_data.csv \
  --output deblur_output
```

### 3. 带真值评估

```bash
python main.py \
  --video blurry.mp4 \
  --joints joints.csv \
  --gt clean.mp4 \
  --output deblur_output
```

## 输出

```
deblur_output/
├── deblurred_step_0000_17368348.jpg   去模糊后的单帧图片
├── deblurred_step_0001_17368348.jpg
├── ...
├── compare_step_0000.jpg              原图 vs 去模糊 左右对比
├── compare_step_0001.jpg
├── ...
├── deblurred_video.mp4                去模糊后的合成视频
└── summary.json                       逐帧评估结果汇总
```

`summary.json` 包含每帧的：

- `pixel_displacement_du/dv` — 计算的像素位移量
- `PSNR`, `SSIM` — 原图与去模糊结果的结构相似度
- `PSNR_matched`, `SSIM_matched` — 直方图匹配后比较（消除亮度偏差）

## 评估指标

使用 `evaluate.py` 提供以下指标：

- **PSNR**：峰值信噪比，衡量像素级差异（>40dB = 极好）
- **SSIM**：结构相似度，范围 [0, 1]（>0.95 = 极好）
- **直方图匹配 PSNR/SSIM**：先做直方图匹配再比较，消除整体亮度偏差的影响

## 核心算法

### 正向运动学

使用 Franka Panda 的 D-H 参数计算关节角 → 末端位姿的 4×4 齐次变换矩阵。

### 几何雅可比

关节角速度 → 末端空间速度的线性映射：$v_{ee} = J(q) \dot{q}$

### 手眼标定

相机相对于法兰末端的固定变换，将末端速度转换到相机坐标系。

### 交互矩阵

将相机速度映射到像素速度：

$$
\begin{bmatrix} \dot{u} \\ \dot{v} \end{bmatrix} =
\begin{bmatrix}
-\frac{f_x}{Z} & 0 & \frac{u_c}{Z} & \frac{u_c v_c}{f_x} & -\frac{(1+u_c^2)}{f_x} & v_c f_x \\
0 & -\frac{f_y}{Z} & \frac{v_c}{Z} & \frac{(1+v_c^2)}{f_y} & -\frac{u_c v_c}{f_y} & -u_c f_y
\end{bmatrix}
\cdot v_{cam}
$$

### 反卷积

- **Wiener 滤波**：频域最小均方误差估计，速度快、参数少
- **Richardson-Lucy**：迭代泊松 MAP 估计，更锐利但可能产生振铃

## 参考

- [DROID Dataset](https://droid-dataset.github.io/)
- [Rerun Visualization](https://www.rerun.io/)
- Franka Panda D-H parameters from [libfranka](https://github.com/frankaemika/libfranka)
