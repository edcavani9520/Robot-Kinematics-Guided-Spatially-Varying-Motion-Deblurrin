# Robot-Kinematics-Guided Spatially-Varying Motion Deblurring

利用 Franka Panda / Kinova Gen3 机器人关节角数据，通过正向运动学推算**腕载相机**速度，为图像中每个像素区域计算空间变化的运动模糊 PSF，再通过非盲反卷积（Wiener / Richardson-Lucy）去除运动模糊。

## 概述

传统去模糊方法假设全图模糊均匀（单一 PSF），但在机器人场景中，相机安装在机械臂末端，**旋转运动导致图像不同区域的模糊方向和长度不同**。

本项目使用机器人运动学精确建模这一过程，支持**全局单一 PSF** 与**空间变化 PSF 网格**两种模式。

`
关节角 q  ──→  正运动学  ──→  雅可比  ──→  末端速度
                          │
                   手眼标定 ────────→ 相机速度 v_cam
                          │
                   交互矩阵（像素-速度雅可比）──→ 像素位移 (du, dv)
                          │
                       运动模糊 PSF
                          │
                    Wiener / RL 反卷积
                          │
                       去模糊图像
`

## 文件结构

`
├── main.py              主函数 —— 逐帧去模糊 pipeline
│                        输入：--video / --frames / --h5 + --joints
│                        输出：去模糊帧 + 视频 + 对比图 + PSF 报告
├── joint_deblur.py      核心算法：关节角 → PSF → 反卷积
│                        含 compute_psf / compute_psf_map /
│                        wiener_deconvolution / richardson_lucy /
│                        spatial_wiener / spatial_rl
├── robot_configs.py     机器人 DH 参数与手眼标定注册中心
│                        支持：panda (Franka), kinova-gen3
│                        手眼：simple, droid-left, droid-right, kinova-gen3
├── csv_loader.py        CSV 关节角数据加载器
│                        自动识别标准格式与 DROID action 格式
├── h5_loader.py         DROID / Episode h5 数据加载器
│                        自动检测格式、帧同步、JPEG 内嵌解码
├── evaluate.py          评估工具：PSNR, SSIM, 直方图匹配, 配准
└── sync.sh              Git 同步脚本
`

## 环境要求

`ash
pip install numpy opencv-python h5py
`

---

## 用法

### 1. 图片帧 + joints CSV（推荐用于 DROID）

`ash
python main.py \
  --frames ./output_frames/ \
  --joints ./output_frames/actions.csv \
  --hand-eye droid-left \
  --output deblur_output
`

相机参数默认已设为 **DROID ZED 内参（fx=fy=733.37）**，无需额外指定。

指定右相机：

`ash
python main.py \
  --frames ./output_frames/ \
  --joints ./output_frames/actions.csv \
  --hand-eye droid-right \
  --output deblur_output
`

### 2. h5 格式输入（自动检测 DROID / Episode）

`ash
# DROID trajectory.h5
python main.py \
  --h5 episode_0002.h5 \
  --episode-dir ./recordings/ \
  --camera 17368348 \
  --output deblur_output

# Episode h5（JPEG 内嵌）
python main.py \
  --h5 episode_0002.h5 \
  --output deblur_output
`

### 3. 视频文件

`ash
python main.py \
  --video blurry.mp4 \
  --joints joints.csv \
  --hand-eye simple \
  --output deblur_output

# 带真值评估
python main.py \
  --video blurry.mp4 \
  --joints joints.csv \
  --gt clean.mp4 \
  --output deblur_output
`

### 4. 空间变化 PSF（--spatial）

对图像进行网格划分，每格计算独立的 PSF，边缘重叠分块反卷积：

`ash
python main.py \
  --video blurry.mp4 \
  --joints joints.csv \
  --spatial \
  --grid-rows 4 --grid-cols 4 --overlap 0.25 \
  --output deblur_output
`

### 5. 切换机器人（--robot）

`ash
# 使用 Kinova Gen3
python main.py \
  --video blurry.mp4 \
  --joints joints.csv \
  --robot kinova-gen3 \
  --hand-eye kinova-gen3 \
  --output deblur_output
`

---

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --video | — | 输入视频路径 |
| --frames | — | 输入图片目录 |
| --h5 | — | h5 文件路径（自动检测 DROID / Episode 格式） |
| --joints | — | 关节角 CSV（--video / --frames 模式必填） |
| --episode-dir | — | Episode 目录（DROID h5 模式，自动找 recordings/MP4） |
| --camera | — | 摄像头 serial（仅 DROID h5 格式） |
| --gt | — | Ground truth 视频（仅 --video 模式） |
| --output | deblur_output | 输出目录 |
| --robot | panda | 机器人配置（panda / kinova-gen3） |
| --hand-eye | simple | 手眼标定（simple / droid-left / droid-right / kinova-gen3） |
| --fx, --fy | 733.37 | 焦距（DROID ZED 默认） |
| --depth | 0.5 | 物距（米），越大 PSF 越小 |
| --exposure | 0.03 | 曝光时间（秒），越长模糊越强 |
| --method | wiener | 反卷积方法（wiener / rl） |
| --K | 0.01 | Wiener 参数，越小去模糊越强（噪声放大也越大） |
| --rl-iters | 30 | RL 迭代次数（仅 method=rl） |
| --max-frames | 全部 | 限制处理帧数，快速测试用 |
| --use-obs-joint | False | h5 模式使用 observation 关节角 |
| --spatial | False | 使用空间变化 PSF（默认：全局单一 PSF） |
| --grid-rows | 4 | PSF 地图网格行数 |
| --grid-cols | 4 | PSF 地图网格列数 |
| --overlap | 0.25 | 空间反卷积分块重叠比例 |

---

## 输出结构

`
deblur_output/
├── blurred/             原始模糊帧
├── deblurred/           去模糊帧
├── comparison/          左右对比图（blurred | deblurred）
├── deblurred_video.mp4  去模糊合成视频
├── comparison_video.mp4 对比合成视频
└── psf_report.csv       每帧 PSF 参数
`

---

## 核心算法

### 正向运动学

使用 Franka Panda / Kinova Gen3 的 D-H 参数计算关节角 → 末端位姿的 4x4 齐次变换矩阵。通过 RobotConfig 支持切换不同机器人。

### 几何雅可比

关节角速度 → 末端空间速度的线性映射：{ee} = J(q) \dot{q}$

### 手眼标定

相机相对于法兰末端的固定变换 HandEyeCalib(R, t)，将末端速度转换到相机坐标系。预置了 DROID 实测标定值和 Kinova Gen3 标定值。

### 交互矩阵

将相机速度映射到像素速度：


\begin{bmatrix} \dot{u} \\ \dot{v} \end{bmatrix} =
\begin{bmatrix}
-\frac{f_x}{Z} & 0 & \frac{u_c}{Z} & \frac{u_c v_c}{f_x} & -\frac{(1+u_c^2)}{f_x} & v_c f_x \\
0 & -\frac{f_y}{Z} & \frac{v_c}{Z} & \frac{(1+v_c^2)}{f_y} & -\frac{u_c v_c}{f_y} & -u_c f_y
\end{bmatrix}
\cdot v_{cam}


### 反卷积

- **Wiener 滤波**：频域最小均方误差估计，速度快、参数少
- **Richardson-Lucy**：迭代泊松 MAP 估计，更锐利但可能产生振铃

---

## 支持的机器人

| 名称 | D-H 参数来源 |
|------|-------------|
| Franka Panda | [libfranka](https://github.com/frankaemika/libfranka) |
| Kinova Gen3 | [Kortex API](https://github.com/Kinovarobotics/ros2_kortex) |

## 参考

- [DROID Dataset](https://droid-dataset.github.io/)
- [libfranka](https://github.com/frankaemika/libfranka)
