# Robot-Kinematics-Guided RGB Motion Deblurring

利用 Kinova Gen3 关节运动推算相机运动和运动模糊 PSF，并对 RGB 三个通道分别执行 Wiener、Richardson–Lucy 或 TV-L2 去卷积。

## 唯一支持的数据格式

程序只接受当前数据采集器生成的 HDF5 episode：

```text
obs/image    (N, H, W, 3) uint8    RGB 彩色图像
obs/proprio  (N, 8)       float64  7 个关节角（度）+ 夹爪值
action       (N, 7)       float64  6 维末端位姿增量 + 夹爪目标
timestamps   (N,)         float64  Unix 时间戳（秒）
```

`data_collector.py` 在写入前执行 `BGR → RGB`，因此 `obs/image` 的通道顺序固定为 RGB。程序不再检测或兼容灰度图、`camera/rgb` JPEG、DROID MP4 或其他旧格式。

加载器会检查数据集名称、数组形状、图像类型、帧数对齐和时间戳递增关系；格式不正确时会立即给出错误。

## 彩色去模糊方式

同一帧的 R、G、B 通道使用完全相同的 PSF 和算法参数分别去卷积，然后按 RGB 顺序合并。处理过程中不会对各通道分别归一化，避免额外色偏。

评估方式：

- PSNR、SSIM、直方图匹配和像素统计使用完整 RGB 图像；
- Laplacian、Tenengrad、TV 和 Edge Ratio 使用由 RGB 统一转换出的亮度图。

## 环境

```powershell
pip install numpy opencv-python h5py scipy pillow pytest
```

## 快速使用

单帧处理：

```powershell
python process_one_frame.py --h5 episode_0001.h5 --frame 50 wiener --K 0.01
python process_one_frame.py --h5 episode_0001.h5 --frame 50 tv --tv-lam 0.002
python process_one_frame.py --h5 episode_0001.h5 --frame 50 rl --rl-iters 10
```

批量评估：

```powershell
python batch_analyze.py --h5 episode_0001.h5 wiener --K 0.01
python batch_analyze.py --h5 episode_0002.h5 --max-frames 20 tv --tv-lam 0.002
```

完整流水线：

```powershell
python pipeline.py --h5 episode_0001.h5 --output deblur_output wiener --K 0.01
```

## 关键参数

| 参数 | 默认值 | 说明 |
|---|---:|---|
| `--h5` | 必填/脚本默认值 | 当前格式 HDF5 episode |
| 方法子命令 | 必填 | `wiener`、`rl` 或 `tv`；公共参数必须写在子命令前 |
| `wiener --K` | `0.01` | 仅 Wiener 可用，必须大于 0 |
| `wiener --adaptive-k` | 关闭 | 仅 Wiener 可用 |
| `rl --rl-iters` | `30` | 仅 RL 可用，必须为正整数 |
| `tv --tv-lam` | `0.002` | 仅 TV-L2 可用，必须大于 0 |
| `--depth` | `0.5` | 有限且大于 0，目标深度（米） |
| `--exposure` | H5，否则 `0.01` | V4L2 Microdia 手动曝光时间（秒）；显式参数优先 |
| `--fx`, `--fy` | H5，否则约 `260.65` | 标签标称对角 FOV 75°、320×240 方形像素估计值；可成对覆盖 |
| `--psf-sigma` | `0` | 有限且非负；0 表示关闭 |
| `--max-frames` | 全部 | 正整数；0 和负数会被拒绝 |

每次运行会在输出根目录下创建包含全部有效参数的独立子目录，并写入
`run_config.json`。目标运行目录非空时默认拒绝覆盖；确认重跑同一配置时使用
`--overwrite`。覆盖模式只清理已知生成物，发现未知文件会停止运行。

当前 episode 不保存绝对末端位姿，因此 PSF 使用 `obs/proprio` 的关节角和时间戳差分出的关节速度计算；深度使用明确的 `--depth` 参数。`action[:6]` 是位姿增量，不会被错误解释为直接测量的 `tool_twist`。

## 输出

完整流水线生成：

```text
deblur_output/
├── blurred/                 RGB 原始帧
├── deblurred/               RGB 去模糊帧
├── comparison/              RGB 左右对比图
├── deblurred_video.mp4      彩色视频
├── comparison_video.mp4     彩色对比视频
└── psf_report.csv
```

RGB 仅在调用 OpenCV 图片或视频写入接口时转换为 BGR；Pillow 输出直接使用 RGB。

## 测试

```powershell
python -m pytest tests -v
```

测试覆盖严格 HDF5 schema、RGB 通道顺序、关节速度、三种全局去卷积、空间去卷积、彩色评估，以及图片/视频输出边界。

## 实时 WS 推理与 RGB 去模糊

实时入口为 `ws_inference_realtime_deblur.py`。它复用 Gen3 控制仓库中的
`Pi05WebSocketControl`，只重写相机取图步骤：模型收到的
`observation/image` 会被替换成实时 RGB Wiener 去模糊后的图像，控制和 WS
推理流程保持不变。

公开仓库：

- 去模糊代码：<https://github.com/edcavani9520/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git>
- Gen3 控制代码：<https://github.com/edcavani9520/fnii-gen3-controller.git>

推荐把两个仓库克隆为同级目录：

```powershell
git clone https://github.com/edcavani9520/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git
git clone https://github.com/edcavani9520/fnii-gen3-controller.git
cd Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin
pip install numpy opencv-python scipy
python ws_inference_realtime_deblur.py --controller-root ../fnii-gen3-controller --ws-host localhost --ws-port 8000 --K 0.01 --depth 0.5 --exposure 0.01
```

运行环境还需要 Gen3 控制仓库原本使用的 Kinova Kortex、OpenPI/WS 及相机依赖。
脚本优先使用机械臂反馈的关节速度；反馈不可用时才使用带角度环绕处理的相邻帧差分。
低于 `--min-motion-px` 的运动会直接旁路，避免静止图像被无意义地去卷积。
