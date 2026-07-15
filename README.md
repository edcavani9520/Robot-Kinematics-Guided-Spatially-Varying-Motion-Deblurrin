# Robot-Kinematics-Guided Motion Deblurring

利用机器人末端速度（tool_twist）推算腕载相机运动，通过 Bresenham 光栅化 + Wiener / TV-L2 ADMM 反卷积去除运动模糊。

## 快速测试

`ash
# 单帧 Wiener
python process_one_frame.py --frame 77 --K 0.03

# 单帧 TV-L2（推荐，无振铃）
python process_one_frame.py --frame 77 --method tv --tv-lam 0.002

# 批量评估
python batch_analyze.py --h5 episode_0001.h5 --K 0.03

# 全流水线
python pipeline.py --h5 episode_0001.h5 --robot kinova-gen3 --hand-eye kinova-gen3
`

## 流水线

`
h5 tool_twist + tool_pose ──→ 手眼标定 ──→ 相机速度 v_cam
                                      │
                               交互矩阵 ──→ 像素位移 (du, dv)
                                      │
                          Bresenham 光栅化 → 运动模糊 PSF
                                      │
                          ┌─ Wiener (快速, DC补偿)
                          ├─ TV-L2 ADMM (无振铃, 推荐)
                          └─ RL (最锐, 慢)
`

## 文件结构

`
├── pipeline.py            主流水线 (仅 H5 模式)
├── process_one_frame.py   单帧去模糊 + 对比图 + 评估
├── batch_analyze.py       批量评估 (h5 遍历)
├── joint_deblur.py        核心算法：PSF → 反卷积
│                          Wiener / TV-L2 / RL + Bresenham
├── robot_configs.py       机器人 DH 参数 + 手眼标定
├── h5_loader.py           H5 数据加载器 (Kinova/DROID/Episode)
└── evaluate.py            评估：PSNR, SSIM, 锐度
`

## 环境要求

`ash
pip install numpy opencv-python h5py scipy
`

## 参数说明

### 基本参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| --h5 | 必填 | H5 文件路径 |
| --robot | panda | 机器人 (panda / kinova-gen3) |
| --hand-eye | simple | 手眼标定 (kinova-gen3 / droid-left 等) |
| --K | 0.01 | Wiener 参数 (越小去模糊越强) |
| --method | wiener | 反卷积方法 (wiener / tv / rl) |
| --tv-lam | 0.002 | TV 正则化强度 |
| --rl-iters | 30 | RL 迭代次数 |
| --max-frames | 全部 | 限制帧数 |
| --psf-sigma | 0.0 | PSF 高斯正则化 (0.3 可减少振铃) |
| --reverse-psf | off | PSF 方向取反 (验证用) |

## 用法

### 单帧测试

`ash
# Wiener
python process_one_frame.py --frame 77 --K 0.03

# TV-L2 (推荐, Laplacian +187)
python process_one_frame.py --frame 77 --method tv --tv-lam 0.002

# PSF 正则化减少振铃
python process_one_frame.py --frame 77 --K 0.03 --psf-sigma 0.3

# 方向验证
python process_one_frame.py --frame 77 --K 0.03 --reverse-psf
`

### 批量评估

`ash
python batch_analyze.py --h5 episode_0001.h5 --K 0.03
python batch_analyze.py --h5 episode_0001.h5 --method tv --tv-lam 0.002
`

### 流水线

`ash
python pipeline.py --h5 episode_0001.h5 --robot kinova-gen3 --hand-eye kinova-gen3
python pipeline.py --h5 trajectory.h5 --hand-eye droid-left
`

## 核心算法

### 1. 手眼标定 (Kinova Gen3)

- 安装：沿 EE Z 向上 0.11m → 绕 Y 向前倾斜 20° → 沿光轴向前 0.03m
- R_he: 相机 Z_cam = [sin20°, 0, -cos20°]
- 	_he: [0.01026, 0, 0.08181] 米

### 2. 深度计算 (光轴距离)

`python
# 沿相机光轴到桌面的物理距离，不受 EE roll 翻转影响
depth = max(abs((table_z - cam_pos[2]) / abs(opt_z)), 0.02)
`

### 3. PSF (Bresenham + DC补偿)

Bresenham 线算法替代 round()，权重完全均匀。Wiener 后自动 DC 补偿。

### 4. 性能 (Frame 77)

| 方法 | Laplacian | SSIM | 特点 |
|------|:---------:|:----:|------|
| Wiener K=0.03 | **+187** | 0.79 | 快速 |
| **TV-L2 lam=0.002** | **+310** | 0.43 | **无振铃** |
| RL 50 iters | ~+200 | ~0.60 | 最锐 |

深度修正前后对比：

| depth | Laplacian | 说明 |
|:-----:|:---------:|------|
| 0.3m (旧 fallback) | +125 | 常数值，不准 |
| **0.055m (光轴距离)** | **+187** | **正确物理距离** |

## 评估指标

| 指标 | 含义 |
|------|------|
| Laplacian Variance | 无参考锐度，越大越清晰 |
| Tenengrad | 梯度锐度 |
| PSNR | 与模糊图的像素差异 |
| SSIM | 结构相似度 |

## 数据集

| 数据 | 分辨率 | 同步 | 有效帧率 |
|------|:------:|:----:|:--------:|
| episode_0001 | 320×240 | ✅ 完美 | **100%** |
| episode_0002 | 640×480 | ⚠️ 有偏移 | 66% |
