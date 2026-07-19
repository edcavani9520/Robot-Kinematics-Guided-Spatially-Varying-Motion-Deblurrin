# Real-Time WebSocket RGB Deblurring Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a hardware-independent real-time RGB Wiener processor and a lightweight launcher that subclasses the external `Pi05WebSocketControl`, replacing its policy camera observation with the deblurred frame.

**Architecture:** `realtime_deblur.py` owns state estimation, kinematics, PSF construction, static bypass, and RGB Wiener processing without importing WebSocket or Kortex dependencies. `ws_inference_realtime_deblur.py` dynamically loads the parent controller from a user-supplied clone, builds a small subclass that overrides only `get_camera_image()`, and exposes the parent control arguments plus live Wiener tuning. Tests use synthetic/fake controllers and never connect to hardware.

**Tech Stack:** Python 3, NumPy, SciPy, OpenCV, Kortex SDK (runtime only), websockets/openpi-client (runtime only), pytest

---

## File Map

- Create `realtime_deblur.py`: live stateful RGB Wiener processor and diagnostics.
- Create `ws_inference_realtime_deblur.py`: dynamic parent loading, wrapper subclass, CLI, public clone guidance.
- Create `tests/test_realtime_deblur.py`: velocity, bypass, PSF, RGB, and validation tests.
- Create `tests/test_ws_realtime_deblur.py`: dynamic loading errors and fake-parent wrapper tests.
- Modify `README.md`: append public repository checkout and real-time launch instructions without rewriting concurrent strict-CLI documentation.

### Task 1: Stateful Real-Time RGB Wiener Core

**Files:**
- Create: `tests/test_realtime_deblur.py`
- Create: `realtime_deblur.py`

- [ ] **Step 1: Write failing validation and initialization tests**

Create tests that instantiate `RealTimeRGBDeblurrer`, reject non-RGB/non-`uint8` frames and invalid seven-joint state, and assert the first call without measured velocity returns the original RGB object values with diagnostics `applied=False` and `velocity_source="initializing"`.

- [ ] **Step 2: Run tests to verify RED**

Run: `python -m pytest tests/test_realtime_deblur.py -v`

Expected: collection fails because `realtime_deblur` does not exist.

- [ ] **Step 3: Implement configuration, diagnostics, and initial bypass**

Add immutable `RealtimeDeblurConfig` fields `K`, `depth`, `exposure`, `fx`, `fy`, `psf_sigma`, `adaptive_k`, and `min_motion_px`, validating finite physical domains. Add immutable `DeblurDiagnostics` fields `applied`, `velocity_source`, `du`, `dv`, `motion_px`, `psf_size`, and `elapsed_ms`. Implement `RealTimeRGBDeblurrer.process(image_rgb, joint_positions_deg, joint_velocities_deg_s=None, timestamp=None)` validation and first-sample state initialization.

- [ ] **Step 4: Run focused tests to verify GREEN**

Run: `python -m pytest tests/test_realtime_deblur.py -v`

Expected: validation and initialization tests pass.

- [ ] **Step 5: Write failing wrapped-velocity and measured-precedence tests**

Call the processor at timestamps `1.0` and `1.1` with joint zero moving `359 -> 0` degrees and no measured velocity. Patch the camera-velocity function to record radians/second and assert it receives positive `10 deg/s`, not `-3590 deg/s`. In a separate test, provide `joint_velocities_deg_s=[20,0,...]` and assert the measured velocity is used with diagnostics `velocity_source="measured"`.

- [ ] **Step 6: Run velocity tests to verify RED**

Run: `python -m pytest tests/test_realtime_deblur.py -k velocity -v`

Expected: failures because finite-difference and measured-velocity paths are not implemented.

- [ ] **Step 7: Implement velocity resolution**

Store the previous position/timestamp on every valid call. Prefer a finite measured seven-vector. Otherwise compute `(delta + 180) % 360 - 180` divided by positive `dt`. Reject non-increasing timestamps. Convert both position and velocity to radians before calling `get_camera_velocity` with Kinova Gen3 and its hand-eye calibration.

- [ ] **Step 8: Run velocity tests to verify GREEN**

Run: `python -m pytest tests/test_realtime_deblur.py -k velocity -v`

Expected: velocity tests pass.

- [ ] **Step 9: Write failing motion/bypass/RGB Wiener tests**

Patch PSF creation to return known displacements. Assert motion below `min_motion_px` returns the original RGB values without calling Wiener. Assert nonzero motion calls Wiener once with the complete `(H,W,3)` RGB array, a shared 2-D PSF, and the configured/effective K; output remains `(H,W,3) uint8`. Verify optional Gaussian PSF smoothing remains normalized.

- [ ] **Step 10: Run processing tests to verify RED**

Run: `python -m pytest tests/test_realtime_deblur.py -k "motion or wiener or smoothing" -v`

Expected: failures because PSF and Wiener processing are absent.

- [ ] **Step 11: Implement PSF and RGB Wiener processing**

Use `compute_psf_from_pose(..., v_cam_6d=camera_velocity)` at the RGB frame center. Apply SciPy Gaussian smoothing only when configured. Compute Euclidean pixel motion. Skip below threshold; otherwise calculate adaptive K with the existing PSF-size formula and call `wiener_deconvolution` once on RGB. Measure elapsed wall time with `time.perf_counter` and return diagnostics.

- [ ] **Step 12: Run all core tests**

Run: `python -m pytest tests/test_realtime_deblur.py -v`

Expected: all real-time core tests pass.

### Task 2: Lightweight External WS Controller Wrapper

**Files:**
- Create: `tests/test_ws_realtime_deblur.py`
- Create: `ws_inference_realtime_deblur.py`

- [ ] **Step 1: Write failing controller-loader error test**

Call `load_parent_controller` with a directory lacking `pi05_ws_control.py`. Assert `FileNotFoundError` contains the expected file path, `https://github.com/edcavani9520/fnii-gen3-controller.git`, and an HTTPS `git clone` command.

- [ ] **Step 2: Run loader test to verify RED**

Run: `python -m pytest tests/test_ws_realtime_deblur.py -k loader -v`

Expected: collection fails because the launcher does not exist.

- [ ] **Step 3: Implement deferred parent loading**

Define repository URL constants and `load_parent_controller(controller_root)`. Resolve the root, validate `pi05_ws_control.py`, add the root to `sys.path`, load with `importlib.util.spec_from_file_location`, execute the module, and return `Pi05WebSocketControl`. Keep Kortex/openpi imports out of launcher module import time.

- [ ] **Step 4: Run loader test to verify GREEN**

Run: `python -m pytest tests/test_ws_realtime_deblur.py -k loader -v`

Expected: loader error test passes.

- [ ] **Step 5: Write failing fake-parent wrapper tests**

Create a fake parent whose `get_camera_image()` returns a sentinel RGB frame, `get_joint_positions()` returns seven degrees, and `full_status.actuators` exposes seven `.velocity` values. Create a fake deblurrer that records arguments and returns a different sentinel. Assert `build_deblur_controller_class(FakeParent)` preserves parent initialization, passes measured velocity/timestamp to the deblurrer, and returns the processed frame.

- [ ] **Step 6: Run wrapper tests to verify RED**

Run: `python -m pytest tests/test_ws_realtime_deblur.py -k wrapper -v`

Expected: failures because the wrapper factory is absent.

- [ ] **Step 7: Implement the override-only subclass factory**

`build_deblur_controller_class(parent_class)` returns `Pi05DeblurWebSocketControl`. Its constructor accepts `deblurrer` and `deblur_log_every`, then delegates every other argument to `super().__init__`. `get_camera_image()` calls the parent first, extracts measured actuator velocities only when exactly seven finite values exist, calls `deblurrer.process`, logs periodically, and returns the processed RGB frame. No other parent method is overridden.

- [ ] **Step 8: Run wrapper tests to verify GREEN**

Run: `python -m pytest tests/test_ws_realtime_deblur.py -v`

Expected: all loader and wrapper tests pass without Kortex/openpi installed.

### Task 3: CLI and Public Download Instructions

**Files:**
- Modify: `ws_inference_realtime_deblur.py`
- Modify: `tests/test_ws_realtime_deblur.py`
- Modify: `README.md`

- [ ] **Step 1: Write failing CLI help/default tests**

Call `_parse_args` with `--controller-root` and a minimal argument list. Assert defaults match the parent controller (10 Hz, Twist mode, 320x240 camera behavior through the parent) and live Wiener defaults. Assert parser help contains both public repository HTTPS URLs and the exact sibling-checkout command.

- [ ] **Step 2: Run CLI tests to verify RED**

Run: `python -m pytest tests/test_ws_realtime_deblur.py -k cli -v`

Expected: failures because CLI construction is absent.

- [ ] **Step 3: Implement parser and launcher**

Expose the parent arguments needed by the current `Pi05WebSocketControl` constructor: WS host/port, robot/camera, prompt, dry-run/observe-only, frequency/action steps, action/workspace limits, control mode, logging, start pose, and camera drain frames. Add live Wiener arguments and `--controller-root` defaulting to `../fnii-gen3-controller`. `main()` loads the parent, builds the subclass, constructs `RealtimeDeblurConfig`, constructs the controller, and calls `run()`.

- [ ] **Step 4: Append README real-time section**

Append a bounded section containing both HTTPS `git clone` commands, recommended sibling layout, installation dependencies, launch example, Wiener parameters, and the explicit statement that the deblurred RGB output replaces `observation/image`. Do not rewrite concurrent strict-CLI sections.

- [ ] **Step 5: Run CLI tests and help smoke test**

Run:

```powershell
python -m pytest tests/test_ws_realtime_deblur.py -v
python ws_inference_realtime_deblur.py --help
```

Expected: tests pass and help prints without importing Kortex/openpi.

### Task 4: Full and Episode-Based Verification

**Files:**
- Verify: `realtime_deblur.py`
- Verify: `ws_inference_realtime_deblur.py`
- Verify: `tests/test_realtime_deblur.py`
- Verify: `tests/test_ws_realtime_deblur.py`

- [ ] **Step 1: Run syntax and full test suite**

Run:

```powershell
python -m compileall -q realtime_deblur.py ws_inference_realtime_deblur.py
python -m pytest -q
```

Expected: all existing and new tests pass.

- [ ] **Step 2: Run current-episode motion smoke test**

Load `episode_0001.h5`, choose the frame with maximum stored joint-velocity norm, convert its radian state back to degrees, and call `RealTimeRGBDeblurrer.process` with measured degree/second velocity. Assert output is `(240,320,3) uint8`, diagnostics show a finite nonzero displacement, and elapsed time is reported.

- [ ] **Step 3: Validate the real external controller checkout without hardware**

Call `load_parent_controller` against the local `fnii-gen3-controller` checkout and assert it returns `Pi05WebSocketControl`. Do not instantiate or connect the robot, camera, or WebSocket.

- [ ] **Step 4: Review ownership-safe diff**

Run `git status --short` and a path-scoped diff for the two new modules, two new test files, README appended section, this plan, and its design. Confirm concurrent strict-CLI files and the five user HDF5 files remain untouched by this feature.
