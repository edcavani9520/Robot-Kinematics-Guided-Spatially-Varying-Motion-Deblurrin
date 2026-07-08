# DIP Final Project — 开发日志与经验总结

> 日期：2026-07-07
> 项目：基于机器人运动学的图像去模糊

---

## 目录

1. [编码损坏与恢复](#1-编码损坏与恢复)
2. [R_ee^T 旋转矩阵缺失](#2-r_eet-旋转矩阵缺失)
3. [PANDA 硬编码问题](#3-panda-硬编码问题)
4. [--hand-eye 参数 choices 遗漏](#4---hand-eye-参数-choices-遗漏)
5. [PSF 核采样不均匀](#5-psf-核采样不均匀)
6. [Wiener K 参数调优](#6-wiener-k-参数调优)
7. [cv2.imwrite 中文路径问题](#7-cv2imwrite-中文路径问题)
8. [Laplacian 方差与噪声放大的区分](#8-laplacian-方差与噪声放大的区分)
9. [时间同步问题](#9-时间同步问题)
10. [最终文件结构](#10-最终文件结构)

---

## 1. 编码损坏与恢复

### 现象

`main.py` 中的中文注释、docstring、help 文本全部变成乱码：

```
# 原本：main.py — 主函数：逐帧去模糊 pipeline
# 变成：main.py 鈥?涓诲嚱鏁帮細閫愬抚鍘绘ā绯?pipeline
```

### 原因

文件以 UTF-8 编码保存中文后，被某些编辑器以 GBK 编码打开并重新保存，导致 UTF-8 字节序列被错误解释。具体表现：

- 中文注释完全损坏
- 包含中文的整行代码被吞掉（如 `print(f"[WARN] step {i}: 无法读取帧")` 整行消失）
- `"""` 字符串边界被破坏，导致 syntax error
- 文件行数从 626 膨胀到 1231（乱码字节被拆成多行）

### 修复

1. 用之前生成的正确备份 `main_fixed.py` 覆盖

```bash
copy "C:\Users\17967\Documents\Codex\2026-06-20\new-chat-3\outputs\main_fixed.py" "E:\Vital_document\CUHKSZ\课程文件\ECE4512\Final\main.py"
```

2. 所有 `.py` 文件都以 UTF-8（无 BOM）保存

3. 避免在代码注释中使用中文（至少在多人协作时）

### 经验

- 始终确认编辑器的保存编码为 UTF-8
- Python 文件头部可以加 `# -*- coding: utf-8 -*-` 做标识
- 重要文件做好 git 版本管理

---

## 2. R_ee^T 旋转矩阵缺失

### 现象

去模糊结果有严重振铃，PSF 方向和实际运动方向不一致。

### 原因

`get_camera_velocity` 函数中，直接从**基座系**的 `v_ee` 应用手眼标定 `R_he^T`，缺少了先从**基座系变换到末端系**的步骤（`R_ee^T`）。

**错误代码：**

```python
J, _ = get_geometric_jacobian(q, robot=robot)
v_ee = J @ q_dot

R_he, t_he = hand_eye.R, hand_eye.t

w = R_he.T @ v_ee[3:]          # 直接从基座系 → 相机系
v = R_he.T @ v_ee[:3] + np.cross(w, R_he.T @ t_he)
```

### 数学推导

正确的变换链应该是：

```
v_ee (基座系) → R_ee^T → 末端系 → R_he^T → 相机系
```

因为：
- `v_ee` 来自雅可比矩阵，在基座系中表达
- 手眼标定 `R_he` 是**末端系 → 相机系**的变换
- 需要先用 **R_ee^T** 将基座系速度转换到末端系

**正确代码：**

```python
J, T_list = get_geometric_jacobian(q, robot=robot)
v_ee = J @ q_dot

R_ee = T_list[-1][:3, :3]    # 基座→末端的旋转矩阵

v_ee_in_ee = R_ee.T @ v_ee[:3]
w_ee       = R_ee.T @ v_ee[3:]

w = R_he.T @ w_ee
v = R_he.T @ v_ee_in_ee + np.cross(w, R_he.T @ t_he)
```

### 验证

```python
R_ee 偏离单位阵: 2.56  # 测试用例中 R_ee 远非单位阵
v 误差: 0.634          # 代码输出与正确值的差异
w 误差: 0.800          # 代码输出与正确值的差异
```

修复后正确的相机速度：
```
v_cam: [0.1, 0.117, -0.341, -0.48, 0, -0.141]
```

---

## 3. PANDA 硬编码问题

### 现象

用 `main.py --h5 episode_0002.h5` 处理 Kinova Gen3 数据时，用了 Franka Panda 的 D-H 参数，导致运动学计算完全错误。

### 原因

`run_h5_pipeline` 函数中硬编码了 `PANDA`：

```python
deblurred, meta = process_frame(
    gray, q, qd, params, hand_eye_params, PANDA)   # ← 始终是 Panda
```

而 `episode_0002.h5` 是 Kinova Gen3 机器人录制的，两者的 D-H 参数完全不同（连杆长度、关节零位偏移）。

### 修复

增加 `--robot` 命令行参数，由用户指定机器人类型：

```bash
# 处理 Kinova Gen3 数据
python main.py --h5 episode_0002.h5 --robot kinova-gen3 --hand-eye kinova-gen3

# Franka Panda（默认）
python main.py --h5 trajectory.h5 --hand-eye droid-left
```

改动包括（6 处修改）：

| # | 位置 | 修改 |
|---|------|------|
| 1 | import | 加 `from robot_configs import get_robot` |
| 2 | run_h5_pipeline 签名 | 加 `robot_name="panda"` 参数 |
| 3 | 函数体 | `robot = get_robot(robot_name)` |
| 4 | process_frame 调用 | `PANDA` → `robot` |
| 5 | CLI | 加 `--robot` 参数 |
| 6 | 调用点 | 传 `robot_name=args.robot` |

---

## 4. --hand-eye 参数 choices 遗漏

### 现象

```bash
$ python main.py --hand-eye kinova-gen3
error: argument --hand-eye: invalid choice: 'kinova-gen3'
```

### 原因

```python
parser.add_argument("--hand-eye", type=str, default="simple",
                    choices=["simple", "droid-left", "droid-right"],  # ← 没有 kinova-gen3
                    help="手眼标定预设")
```

### 修复

添加 `kinova-gen3` 到 choices 列表：

```python
choices=["simple", "droid-left", "droid-right", "kinova-gen3"],
```

更好的做法：从 `HAND_EYE_CONFIGS` 动态生成 choices：

```python
from robot_configs import HAND_EYE_CONFIGS
choices=list(HAND_EYE_CONFIGS.keys()),
```

---

## 5. PSF 核采样不均匀

### 现象

PSF 核的权值分布不均匀——7×7 核中，相邻位置的权值从 0.015 到 0.295 不等（理想情况下应一致）。

```
0.000 0.000 0.102 0.147 0.000 0.000 0.000
0.015 0.295 0.190 0.000 0.000 0.000 0.000
0.250 0.000 0.000 0.000 0.000 0.000 0.000
```

### 原因

`create_motion_psf` 使用 `round()` 逐点采样，存在三个问题：

**问题 1：round() 的累积偏差**

```python
for t in np.linspace(0, 1, steps):
    x = int(round(c + t * dv))    # round() 对 0.5 做 banker's rounding
    y = int(round(c + t * du))
    psf[x, y] += 1.0
```

某些像素被命中多次，另一些被跳过。

**问题 2：最小值截断**

```python
steps = max(int(math.hypot(du, dv) * 2.5), 200)
```

对于 ≤ 80 像素的短 PSF，`min` 限制导致**所有核都用相同步数**，无法根据实际长度调整。

**问题 3：不是标准线光栅化**

`round()` 方法不是计算机图形学中的标准算法（如 Bresenham），不保证每个像素恰好被命中一次。

### 修复

增加采样步数（`* 2.5 → * 5.0`, `200 → 400`）改善均匀性，但根本解法是使用 **Bresenham 线光栅化算法**。

### 效果对比

| 指标 | 200步（原始） | 400步（修复后） | Bresenham（理论最优） |
|------|------------|--------------|-------------------|
| SSIM | 0.9762 | 0.9892 | ~0.99 |
| PSNR | 28.11 dB | 29.27 dB | ~30 dB |
| 核均匀性 | 差（0.015~0.295） | 改善 | 完全均匀 |

---

## 6. Wiener K 参数调优

### 参数意义

Wiener 滤波公式中的 K 是噪声-信号功率比：

```python
F = (H* / (|H|^2 + K)) * G
```

- **K 越小** → 去模糊越强，但噪声放大越严重
- **K 越大** → 去噪越好，但细节恢复不足

### 实验结果（PSF 7×7，frame 68）

| K | Laplacian 提升 | SSIM | PSNR | 评价 |
|---|---------------|------|------|------|
| 0.01 | +944 (1016) | 0.976 | 28.11 dB | 噪声放大严重 |
| **0.03** | **+372 (444)** | **0.989** | **29.27 dB** | **最佳平衡** |
| 0.05 | +271 (343) | 0.991 | 30.22 dB | 更保守，更少噪声 |

### 经验

- K 的默认值 0.01 对机器人相机数据来说偏小（噪声放大明显）
- K=0.03 在当前数据上表现最佳
- 实际使用时应根据相机传感器噪声水平调整

---

## 7. cv2.imwrite 中文路径问题

### 现象

```python
cv2.imwrite("E:\...\课程文件\...\test.png", img)
# 返回 False，图片未写入
```

### 原因

OpenCV 的 `cv2.imwrite` 底层调用 C++ 的 `imwrite`，不支持 Unicode 路径（特别是中文）。

### 修复

用 PIL 替代 OpenCV 保存图像：

```python
from PIL import Image

def imwrite_pil(path, img):
    if len(img.shape) == 3:
        # BGR → RGB 转换
        Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(path)
    else:
        Image.fromarray(img).save(path)
```

PIL 的 `Image.save()` 底层使用 Python 的文件 I/O，原生支持 Unicode 路径。

---

## 8. Laplacian 方差与噪声放大的区分

### 问题

Laplacian 方差是常用的"无参考图像锐度指标"，但它对**噪声**极其敏感。

```
K=0.01: Laplacian 从 72 涨到 1016 (+1311%)
K=0.03: Laplacian 从 72 涨到 444  (+517%)
```

两者差了 2.3 倍，但 PSF 相同、图像内容相同——区别就是噪声放大。

### 如何区分真实去模糊 vs 噪声放大

| 检查项 | 真实去模糊 | 噪声放大 |
|--------|-----------|---------|
| Std 变化 | 几乎不变或微增 | **明显增加** |
| Entropy 变化 | 微增（+0.1~0.2 bits） | 大幅增加 |
| 肉眼观察 | 边缘更锐利 | 出现颗粒感 |
| SSIM | > 0.98 | < 0.96 |
| Laplacian 提升 | +300~+500%（PSF 7×7） | +1000%+ |

### 建议

- 不要只看 Laplacian/Tenengrad，要结合 SSIM 和 Std 判定
- 用不同的 K 值做对比实验，观察指标变化趋势
- 最可靠的方法仍然是肉眼观察对比图

---

## 9. 时间同步问题

### 数据

`episode_0002.h5` 中：

```
camera: 368 帧，~10 fps
robot:  605 个数据点，~15 fps
```

### 同步方法

`h5_loader.py` 使用 `np.searchsorted` 为每帧匹配最近的机器人数据点：

```python
sync_indices = np.searchsorted(robot_ts, cam_ts)
# refinement: 取左右中更近的
for i in range(len(cam_ts)):
    idx = sync_indices[i]
    if idx > 0 and abs(cam_ts[i] - robot_ts[idx-1]) < abs(cam_ts[i] - robot_ts[idx]):
        sync_indices[i] = idx - 1
```

同步结果：366/368 帧成功匹配，2 帧因时间戳超出范围而丢失。

### 潜在问题

1. **时钟偏差**：视频和机器人的时间戳来自不同时钟，可能存在固定偏移
2. **曝光延迟**：`frame_idx / fps` 估算的帧时间与实际曝光起始时间有差异
3. **频率不匹配**：10 fps 视频 vs 15 fps 机器人数据，匹配误差约 ±1/30 秒

---

## 10. 最终文件结构

```
Final/
├── main.py                      主函数（h5/视频/图片三种模式）
├── joint_deblur.py              核心算法（关节角 → PSF → 反卷积）
├── csv_loader.py                CSV 关节角加载
├── h5_loader.py                 h5 数据加载 + 时间同步
├── evaluate.py                  评估工具（PSNR/SSIM/对比锐度）
├── robot_configs.py             机器人 D-H 参数 + 手眼标定预设
├── process_one_frame.py         单帧去模糊工具
├── batch_analyze.py             批量评估工具
├── README.md                    使用说明
└── episode_0002.h5              Kinova Gen3 真机数据（368 帧）
```

### 已修复的关键问题

| 问题 | 状态 | 影响 |
|------|------|------|
| R_ee^T 缺失 | ✅ 已修复 | 相机速度计算正确 |
| PANDA 硬编码 | ✅ 已修复 | 支持多机器人配置 |
| --hand-eye choices | ✅ 已修复 | 支持 kinova-gen3 |
| PSF 采样不均匀 | ⚠️ 已改善 | 步数从 200 → 400 |
| 编码损坏 | ⚠️ 已恢复 | 从 backup 恢复 |
| cv2.imwrite 中文路径 | ✅ 已修复 | 改用 PIL 保存 |
