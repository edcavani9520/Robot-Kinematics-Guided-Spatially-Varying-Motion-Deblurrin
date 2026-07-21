# RGB Single-Format Deblurring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the project accept only the current RGB Kinova HDF5 schema and process all three RGB channels through every deconvolution path.

**Architecture:** A strict `load_episode_h5` loader owns schema validation and returns RGB frames plus kinematic arrays. Deconvolution functions dispatch RGB inputs to their existing two-dimensional numerical kernels, while evaluation separates color-sensitive metrics from luminance-based sharpness metrics. All command-line entry points share this loader, the joint-kinematics PSF path, and explicit RGB/BGR conversions only at OpenCV output boundaries.

**Tech Stack:** Python, NumPy, OpenCV, h5py, pytest, Pillow, SciPy

---

## File Map

- `h5_loader.py`: strict current-format HDF5 validation, RGB frame access, joint velocity derivation.
- `joint_deblur.py`: two-dimensional numerical kernels plus public RGB dispatch and RGB-safe spatial blending.
- `evaluate.py`: RGB PSNR/SSIM/histogram handling and luminance sharpness evaluation.
- `pipeline.py`: unified current-format RGB full pipeline and color output.
- `batch_analyze.py`: current-format RGB batch evaluation.
- `process_one_frame.py`: current-format RGB single-frame evaluation and output.
- `kinova_batch.py`: delete the duplicate batch entry point.
- `README.md`: document the only accepted schema and RGB behavior.
- `tests/test_h5_loader.py`: loader schema, RGB order, and velocity tests.
- `tests/test_rgb_deblur.py`: global and spatial RGB deconvolution tests.
- `tests/test_color_evaluate.py`: color and luminance metric tests.
- `tests/test_rgb_output.py`: RGB/BGR output-boundary tests.

### Task 1: Strict RGB Episode Loader

**Files:**
- Create: `tests/test_h5_loader.py`
- Modify: `h5_loader.py`

- [ ] **Step 1: Write failing valid-schema and RGB-order tests**

Create a temporary HDF5 file with `obs/image` shaped `(3, 2, 2, 3)` and a sentinel RGB pixel `[255, 0, 17]`. Assert `load_episode_h5` returns `images`, radians joint positions, equal-length joint velocities, `actions`, `timestamps`, and `EpisodeFrameReader.read_frame(0)` preserves the sentinel without BGR conversion.

- [ ] **Step 2: Write failing invalid-schema tests**

Parametrize missing datasets, grayscale `(N,H,W)`, wrong image dtype, mismatched leading lengths, and non-increasing timestamps. Assert each raises `ValueError` containing the invalid dataset or condition.

- [ ] **Step 3: Run loader tests and verify RED**

Run: `python -m pytest tests/test_h5_loader.py -v`

Expected: failures because the existing loader unpacks RGB arrays as three dimensions and the reader converts or accepts legacy layouts.

- [ ] **Step 4: Implement the minimal strict loader**

Keep one public loader with this returned interface:

```python
{
    "images": images,
    "joint_positions": np.deg2rad(proprio[:, :7]),
    "joint_velocities": joint_velocities,
    "actions": actions,
    "timestamps": timestamps,
    "num_frames": n,
    "H": h,
    "W": w,
    "camera_fps": float(file_attr_or_timestamp_rate),
}
```

Make `EpisodeFrameReader` accept only `(N,H,W,3)` `uint8` RGB data. Remove format detection, DROID readers, JPEG branches, `load_kinova_h5`, and `KinovaFrameReader`. Derive centered finite-difference joint velocities with degree wrap handling and positive timestamp intervals.

- [ ] **Step 5: Run loader tests and verify GREEN**

Run: `python -m pytest tests/test_h5_loader.py -v`

Expected: all loader tests pass.

### Task 2: RGB Global and Spatial Deconvolution

**Files:**
- Create: `tests/test_rgb_deblur.py`
- Modify: `joint_deblur.py`

- [ ] **Step 1: Write failing RGB global-deconvolution tests**

Build an RGB image whose three channels contain distinct constant/impulse patterns. For each of `wiener_deconvolution`, `richardson_lucy`, and `tv_deconv`, compare RGB output channel `c` with a direct call on input channel `c`, and assert `(H,W,3)` plus `uint8` are preserved.

- [ ] **Step 2: Run global RGB tests and verify RED**

Run: `python -m pytest tests/test_rgb_deblur.py -k global -v`

Expected: existing FFT and shape unpacking code fails or returns an invalid shape for RGB input.

- [ ] **Step 3: Add a shared RGB dispatch helper**

Add an internal helper that validates either `(H,W)` or `(H,W,3)`, calls the private two-dimensional kernel for each RGB channel, and stacks along axis 2. Refactor the public Wiener, RL, and TV functions around `_wiener_2d`, `_richardson_lucy_2d`, and `_tv_deconv_2d` without changing two-dimensional numerical behavior.

- [ ] **Step 4: Run global RGB tests and verify GREEN**

Run: `python -m pytest tests/test_rgb_deblur.py -k global -v`

Expected: all global RGB tests pass.

- [ ] **Step 5: Write failing spatial RGB tests**

Use a `2x2` PSF map of identity kernels with an RGB input. Assert spatial Wiener and spatial RL preserve RGB shape/type and match the input within their existing boundary behavior.

- [ ] **Step 6: Run spatial tests and verify RED**

Run: `python -m pytest tests/test_rgb_deblur.py -k spatial -v`

Expected: broadcasting fails because blending weights are two-dimensional.

- [ ] **Step 7: Make spatial blending channel-safe**

Allocate `result` and `weight` with the input shape; expand each two-dimensional blend window as `w[..., None]` for RGB, while preserving two-dimensional behavior.

- [ ] **Step 8: Run all deconvolution tests and verify GREEN**

Run: `python -m pytest tests/test_rgb_deblur.py -v`

Expected: all deconvolution tests pass.

### Task 3: Color-Aware Evaluation

**Files:**
- Create: `tests/test_color_evaluate.py`
- Modify: `evaluate.py`

- [ ] **Step 1: Write failing color metric tests**

Assert RGB PSNR uses all channels, SSIM equals the average of per-channel SSIM values, and histogram matching operates independently per channel. Create two RGB images with identical luminance but rearranged colors and assert sharpness helpers use the documented RGB-to-luminance conversion.

- [ ] **Step 2: Run evaluation tests and verify RED**

Run: `python -m pytest tests/test_color_evaluate.py -v`

Expected: global SSIM/histogram behavior and OpenCV multichannel Laplacian/Sobel behavior disagree with the desired definitions.

- [ ] **Step 3: Implement color/luminance separation**

Add `_to_luminance` using OpenCV `COLOR_RGB2GRAY`. Make SSIM and histogram matching channel-aware. Route Laplacian, Tenengrad, total variation, and edge ratio through `_to_luminance`. Preserve existing result keys; add per-channel basic statistics without removing aggregate statistics.

- [ ] **Step 4: Run evaluation tests and verify GREEN**

Run: `python -m pytest tests/test_color_evaluate.py -v`

Expected: all evaluation tests pass.

### Task 4: RGB Entry Points and Output Boundaries

**Files:**
- Create: `tests/test_rgb_output.py`
- Modify: `pipeline.py`
- Modify: `batch_analyze.py`
- Modify: `process_one_frame.py`
- Delete: `kinova_batch.py`

- [ ] **Step 1: Write failing output conversion tests**

Exercise comparison-image construction with a sentinel RGB image and assert the rendered left/right pixels have correct colors after encoding/decoding. Assert the deblurred video writer is created with `isColor=True` and receives BGR frames.

- [ ] **Step 2: Run output tests and verify RED**

Run: `python -m pytest tests/test_rgb_output.py -v`

Expected: existing functions expect two-dimensional `gray` arrays and configure a grayscale video writer.

- [ ] **Step 3: Unify the main pipeline**

Import only `load_episode_h5` and `EpisodeFrameReader`. Remove format detection, DROID/Kinova branches, synchronization branches, and missing tool-pose access. Compute camera velocity with `get_camera_velocity(q, qd, ...)`, then compute the PSF with fixed `--depth`. Pass RGB to deconvolution/evaluation and convert to BGR only for `cv2.imwrite` and video writers.

- [ ] **Step 4: Update batch and single-frame scripts**

Replace gray variables and conversions with RGB frames. Use the same joint-kinematics PSF path and current loader in all scripts. Make comparison canvases RGB internally when saved with Pillow, or BGR internally when saved with OpenCV, with one explicit conversion at the boundary.

- [ ] **Step 5: Remove the duplicate Kinova batch entry point**

Delete `kinova_batch.py`; `batch_analyze.py` is the only batch analyzer and uses the current-format RGB loader.

- [ ] **Step 6: Run output and full unit tests**

Run: `python -m pytest tests -v`

Expected: all tests pass with no warnings from the project code.

### Task 5: Documentation and Real-Data Verification

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Update documentation**

Document only the current HDF5 schema, RGB channel order, per-channel shared-PSF deconvolution, fixed `--depth`, and current commands. Remove DROID, JPEG episode, grayscale, and new/old format descriptions.

- [ ] **Step 2: Validate all five episode schemas**

Run a read-only Python check that loads `episode_0001.h5` through `episode_0005.h5`, reads the first and last RGB frame, and reports shape/type.

Expected: every file loads as `(N,240,320,3)` `uint8` RGB with aligned arrays.

- [ ] **Step 3: Run bounded real-data smoke tests**

Run Wiener, RL, and TV-L2 on representative RGB frames and run the main pipeline with `--max-frames 1` into a temporary output directory.

Expected: every result is `(240,320,3)` `uint8`; saved files decode as color; no legacy-format or grayscale branch is reached.

- [ ] **Step 4: Run syntax and complete regression checks**

Run:

```powershell
python -m compileall -q h5_loader.py joint_deblur.py evaluate.py pipeline.py batch_analyze.py process_one_frame.py
python -m pytest tests -v
git diff --check
```

Expected: all commands exit zero.

- [ ] **Step 5: Review repository status**

Confirm only source, tests, docs, and any explicitly intended generated smoke-test files changed. Keep the user's five modified HDF5 files unstaged.
