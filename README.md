# Robot-Kinematics-Guided Spatially-Varying Motion Deblurring

利用 Franka Panda 机器人关节角数据，通过正向运动学推算**手腕相机**速度，为图像中每个像素区域计算空间变化的运动模糊 PSF，再通过非盲反卷积（Wiener / Richardson-Lucy）去除运动模糊。

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
│                       输入：视频或图片目录 + 关节角 CSV
│                       输出：去模糊视频/图片 + 评估
├── joint_deblur.py     核心算法 — 关节角 → PSF → 反卷积
│                       可配置 RobotConfig（D-H 参数）和
│                       HandEyeCalib（手眼标定），预设：
│                         PANDA — Franka Panda D-H 参数
│                         PANDA_HAND_EYE_SIMPLE — 简化手眼
│                         DROID_HAND_EYE_LEFT — DROID 左相机实测
│                         DROID_HAND_EYE_RIGHT — DROID 右相机实测
├── apply_blur.py       模糊工具 — 从像素位移生成 PSF 核
├── evaluate.py         评估工具 — PSNR, SSIM
```

## 环境要求

```bash
pip install numpy opencv-python h5py
```

---

## 用法：DROID 数据集去模糊

用你的 DROID 轨迹下去模糊：

### 1. 图片帧 + actions.csv（推荐）

```bash
python main.py \
  --frames ./output_frames/ \
  --joints ./output_frames/actions.csv \
  --hand-eye droid-left \
  --output deblur_output
```

相机参数默认已设为 **DROID ZED 内参（fx=fy=733.37）**，无需额外指定。

指定右相机：

```bash
python main.py \
  --frames ./output_frames/ \
  --joints ./output_frames/actions.csv \
  --hand-eye droid-right \
  --output deblur_output
```

### 2. 不指定 --hand-eye，用简化手眼标定

```bash
python main.py \
  --frames ./output_frames/ \
  --joints ./output_frames/actions.csv \
  --hand-eye simple \
  --fx 500 --fy 500 \
  --output deblur_output
```

### 3. 常用调参

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--hand-eye` | `simple` | 手眼标定：`simple` / `droid-left` / `droid-right` |
| `--fx, --fy` | 733.37 | 焦距（DROID ZED 默认） |
| `--depth` | 0.5 | 物距（米），越大 PSF 越小 |
| `--exposure` | 0.03 | 曝光时间（秒），越长模糊越强 |
| `--K` | 0.01 | Wiener 参数，越小去模糊越强（噪声放大也越大） |
| `--method` | wiener | `wiener` 或 `rl`（Richardson-Lucy） |
| `--rl-iters` | 30 | RL 迭代次数（仅 method=rl） |
| `--max-frames` | 全部 | 限制处理帧数，快速测试用 |

---

## 用法：视频文件

```bash
# 基本用法
python main.py \
  --video blurry.mp4 \
  --joints joint_data.csv \
  --hand-eye simple \
  --output deblur_output

# 带真值评估
python main.py \
  --video blurry.mp4 \
  --joints joints.csv \
  --gt clean.mp4 \
  --hand-eye simple \
  --output deblur_output
```

---

## 支持的数据格式

**关节角 CSV 自动识别：**
- **标准格式**：`timestamp, q1..q7, qd1..qd7`
- **DROID format**：`action_joint_0..action_joint_6`（自动有限差分算速度）

---

## 输出

```
deblur_output/
├── deblurred_step_0000.jpg     去模糊后的单帧图片
├── deblurred_step_0001.jpg
├── ...
├── deblurred_video.mp4         去模糊后的合成视频
```

---

## 核心算法

### 正向运动学

使用 Franka Panda 的 D-H 参数计算关节角 → 末端位姿的 4×4 齐次变换矩阵。支持通过 `RobotConfig` 切换不同机器人的 D-H 参数。

### 几何雅可比

关节角速度 → 末端空间速度的线性映射：$v_{ee} = J(q) \dot{q}$

### 手眼标定

相机相对于法兰末端的固定变换 `HandEyeCalib(R, t)`，将末端速度转换到相机坐标系。预设了两个 DROID 实测标定值（左/右相机）。

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

---

## 参考

- [DROID Dataset](https://droid-dataset.github.io/)
- Franka Panda D-H parameters from [libfranka](https://github.com/frankaemika/libfranka)
