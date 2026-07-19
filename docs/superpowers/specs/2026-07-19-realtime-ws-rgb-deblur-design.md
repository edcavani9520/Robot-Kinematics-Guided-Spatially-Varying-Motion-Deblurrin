# Real-Time WebSocket RGB Deblurring Design

## Goal

Add a lightweight real-time RGB Wiener deblurring wrapper in the Final repository. The wrapper reuses the existing `Pi05WebSocketControl` implementation and replaces the RGB image returned by `get_camera_image()` so the deblurred frame becomes the policy's `observation/image`.

Robot connection, WebSocket transport, action chunks, action safety, Twist/IK execution, gripper control, start-pose handling, and cleanup remain owned by the Gen3 controller repository.

## Public Repositories

The implementation and documentation identify both public HTTPS clone locations:

- Deblurring repository: `https://github.com/edcavani9520/Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin.git`
- Gen3 WebSocket controller: `https://github.com/edcavani9520/fnii-gen3-controller.git`

Recommended checkout layout:

```text
workspace/
├── Robot-Kinematics-Guided-Spatially-Varying-Motion-Deblurrin/
└── fnii-gen3-controller/
```

The launch command supplies the controller location explicitly:

```powershell
python ws_inference_realtime_deblur.py --controller-root ../fnii-gen3-controller
```

`--controller-root` also accepts an absolute path. Its help text contains the controller repository URL and clone command. No machine-specific absolute path is embedded as a required dependency.

## Components

### `realtime_deblur.py`

This module contains robot- and WebSocket-independent real-time processing:

- `RealTimeRGBDeblurrer` validates `(H,W,3)` `uint8` RGB frames.
- It accepts joint positions in degrees and optional joint velocities in degrees per second.
- When measured velocities are unavailable, it estimates velocity from consecutive timestamped joint positions with `0/360` angle wrapping.
- Joint positions and velocities are converted to radians and passed through the existing Kinova Gen3 kinematics and hand-eye calibration.
- The camera velocity produces `(du,dv)` and one PSF for the current exposure/depth/camera parameters.
- If pixel motion is below `min_motion_px`, the original RGB frame is returned without an FFT.
- Otherwise, the existing RGB `wiener_deconvolution` processes R, G, and B independently with the shared PSF.
- The method returns the RGB result and structured diagnostics containing `du`, `dv`, PSF size, whether processing was applied, and elapsed milliseconds.

The live path is fixed to Wiener. Supported tuning parameters are `K`, `depth`, `exposure`, `fx`, `fy`, `psf_sigma`, `adaptive_k`, and `min_motion_px`.

### `ws_inference_realtime_deblur.py`

This launch script loads `pi05_ws_control.py` from `--controller-root` using an explicit file path. It verifies the file exists and gives a clone instruction when it does not.

`Pi05DeblurWebSocketControl` subclasses the imported `Pi05WebSocketControl` and overrides only `get_camera_image()`:

1. Call `super().get_camera_image()` for the same 320x240 RGB camera behavior used during collection and inference.
2. Read the latest seven joint positions from the parent controller.
3. Prefer seven actuator velocities from `full_status.actuators` when finite and available.
4. Pass the frame and joint state to `RealTimeRGBDeblurrer`.
5. Return the deblurred RGB result.

Because the parent run loop uses the return value for both preview and `observation/image`, the model receives the deblurred frame without changing the policy schema.

The wrapper periodically logs deblurring latency, `(du,dv)`, PSF size, and whether the static-frame bypass was used. It does not save images in the control loop.

The wrapper CLI includes the parent controller options required to construct `Pi05WebSocketControl`, plus the real-time Wiener options and `--controller-root`. Defaults match the existing WS controller and offline deblurring code.

## Data Flow

```text
USB camera BGR
  -> parent BGR-to-RGB capture
  -> latest Kinova joint position/velocity
  -> camera velocity from kinematics and hand-eye calibration
  -> pixel displacement and PSF
  -> shared-PSF RGB Wiener (or static bypass)
  -> deblurred RGB observation/image
  -> OpenPI WebSocket inference
  -> unchanged action safety and robot execution
```

## Timing and Failure Behavior

Deblurring runs synchronously when the parent controller requests a camera frame. This guarantees that the frame sent to the policy is the frame described by the logged PSF and avoids a stale asynchronous result.

At 320x240, RGB Wiener is the only live algorithm. RL and TV-L2 are not exposed by this wrapper. Static or near-static frames skip Wiener to preserve the 10 Hz loop budget.

If robot feedback has not arrived yet, the first frame is returned unchanged while the velocity estimator is initialized. Invalid images or non-finite state produce a clear error before inference. A missing controller checkout stops at startup with both the expected path and the public clone command.

## Testing

Tests avoid importing the Kortex SDK or connecting to hardware:

1. RGB input validation and output shape/type/channel order.
2. First-frame and below-threshold bypass behavior.
3. Wrapped finite-difference velocity across `359 -> 0` degrees.
4. Nonzero motion producing a nontrivial PSF and RGB Wiener output.
5. Measured joint velocity taking precedence over finite differences.
6. Wrapper source loading failure containing the public GitHub clone command.
7. A fake parent controller proving that the overridden camera method returns the processed frame used as `observation/image`.
8. Offline timing and output smoke tests using a maximum-motion frame from the current RGB episodes.

## Success Criteria

- The new code lives in the Final/deblurring repository.
- The parent WS controller is reused rather than copied.
- The deblurred RGB frame directly replaces `observation/image`.
- Live processing uses shared-PSF per-channel Wiener only.
- Robot/WS/action behavior outside image acquisition is unchanged.
- Repository URLs and direct clone/run instructions are visible in README and CLI help.
- Unit tests and an episode-based offline smoke test pass without robot hardware.
