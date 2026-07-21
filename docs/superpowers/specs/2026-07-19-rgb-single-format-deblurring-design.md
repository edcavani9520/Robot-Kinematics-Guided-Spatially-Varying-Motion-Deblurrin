# RGB Single-Format Deblurring Design

## Goal

Convert the motion-deblurring project from grayscale and multi-format compatibility to a single RGB pipeline for the five newly collected Kinova Gen3 episodes.

The only accepted HDF5 layout is:

- `obs/image`: `(N, H, W, 3)` `uint8`, stored in RGB order
- `obs/proprio`: `(N, 8)` floating point, containing seven joint angles in degrees and the gripper value
- `action`: `(N, 7)` floating point, containing six end-effector pose deltas and the gripper target
- `timestamps`: `(N,)` floating point Unix timestamps

The channel order is established by the collector, which converts OpenCV BGR frames to RGB before writing `obs/image`.

## Scope

The change covers data loading, Wiener/Richardson-Lucy/TV-L2 deconvolution, spatial deconvolution helpers, evaluation, image/video output, batch processing, single-frame processing, the main pipeline, tests, and user documentation.

DROID files, `camera/rgb` JPEG episodes, two-dimensional grayscale `obs/image` datasets, format auto-detection, and new/old format branches are removed. Existing generated result directories and the five HDF5 input files are not modified.

## Data Loading

`h5_loader.py` exposes one episode loader and one frame reader for the accepted layout. The loader validates required dataset names, ranks, shapes, data type, equal leading dimensions, non-empty input, and strictly increasing timestamps. Invalid files fail with an actionable `ValueError`.

Images remain RGB throughout the Python processing pipeline. The frame reader returns an `(H, W, 3)` RGB array without channel conversion.

Joint positions are converted from degrees to radians. Joint velocities are derived from timestamp-aware finite differences with angular wrap handling, matching the prior kinematics-based PSF path. `action` is returned using its collector-defined pose-delta meaning; it is not exposed as `tool_twist`.

## PSF and Robot Motion

All entry points use the current episode's joint positions and derived joint velocities with the Kinova Gen3 kinematics and hand-eye calibration to calculate camera velocity and the PSF. Because the current format does not store absolute tool pose, depth remains the explicit `--depth` parameter instead of attempting automatic table depth from nonexistent pose data.

This removes branches that previously selected between stored tool pose/twist and forward kinematics. It also removes the incorrect interpretation of `action[:6]` as a directly measured tool velocity.

## RGB Deconvolution

The public Wiener, Richardson-Lucy, and TV-L2 functions accept either a two-dimensional channel or a three-channel RGB image. Their two-dimensional implementations remain the numerical primitive. For RGB input, the function applies the same PSF and algorithm parameters independently to R, G, and B, then stacks the results in original channel order.

No per-channel histogram normalization or independent gain adjustment is added. Wiener DC compensation remains numerically identical for every channel. Output shape matches input shape and output type remains `uint8`.

Spatial Wiener and spatial Richardson-Lucy helpers allocate and blend arrays with the full input shape. Their scalar blending windows are expanded across the channel dimension for RGB input.

Unsupported image ranks, non-RGB three-dimensional arrays, empty images, and malformed PSFs fail clearly.

## Evaluation

Color-sensitive metrics operate on full RGB arrays:

- PSNR measures mean squared error across every channel.
- SSIM is computed per RGB channel and averaged.
- Histogram matching is performed independently per RGB channel.
- Basic statistics describe all RGB samples and may include per-channel summaries.

Structure and sharpness metrics operate on a deterministic luminance conversion derived from RGB:

- Laplacian variance
- Tenengrad
- total variation
- edge-strength ratio

This prevents channel differences from being counted as spatial sharpness and keeps the interpretation of existing sharpness reports stable.

## Entry Points and Output

`pipeline.py`, `batch_analyze.py`, and `process_one_frame.py` pass RGB images directly into deconvolution and evaluation. Grayscale conversions, gray-named variables, and gray-to-BGR display reconstruction are removed. The duplicate `kinova_batch.py` entry point is deleted so `batch_analyze.py` is the only batch analyzer.

OpenCV I/O boundaries explicitly convert RGB to BGR:

- `cv2.imwrite`
- `cv2.VideoWriter.write`
- OpenCV-labelled comparison canvases

Pillow output receives RGB directly. Comparison images preserve the original and deblurred colors. The main deblurred video writer is configured as a color writer.

The scripts share the same loader and PSF construction path. Duplicate Kinova-versus-Episode data branches are removed. Command-line interfaces retain deconvolution parameters and `--depth`; DROID-only and format-selection parameters are removed.

## Error Handling

The pipeline stops before processing when the HDF5 schema is wrong, RGB data has an invalid shape/type, timestamp lengths differ, or a requested frame is outside the episode. A failed frame read is treated as an input error rather than silently converted or skipped as another format.

Output conversion happens only at explicit library boundaries, making accidental RGB/BGR swaps observable and testable.

## Testing and Verification

Implementation follows red-green-refactor cycles. Tests are added before production changes for:

1. Loading a valid current-format RGB episode while preserving RGB channel order.
2. Rejecting former grayscale and `camera/rgb` layouts.
3. Deriving finite joint velocities with timestamp and angle-wrap handling.
4. Preserving shape, type, channel order, and channel independence in Wiener, RL, and TV-L2.
5. Preserving RGB shape in spatial deconvolution.
6. Computing color PSNR/SSIM and luminance-based sharpness metrics.
7. Converting RGB correctly at saved-image, comparison, and video boundaries.

After unit tests pass, all five real episodes receive schema validation. Representative frames from the five episodes are used for loader and color-order smoke tests. At least one frame is processed with each deconvolution method; the main pipeline is exercised with a bounded frame count so verification remains practical.

## Success Criteria

- The five current episodes load without format detection or compatibility branches.
- Frames remain RGB from HDF5 loading through deconvolution and evaluation.
- Wiener, RL, and TV-L2 process each RGB channel using one shared PSF.
- Saved images, comparison images, and videos have correct colors.
- No production path converts input frames to grayscale for deconvolution.
- Former input formats are rejected with clear errors.
- Automated tests and bounded real-data smoke tests pass.
