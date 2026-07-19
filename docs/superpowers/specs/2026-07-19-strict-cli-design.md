# Strict Deblurring CLI Design

## Goal

Make every exposed CLI argument meaningful, reject invalid numerical domains before processing, prevent method-incompatible parameter combinations, and make every experiment reproducible from its output directory.

## Supported entry points

The supported commands remain:

- `pipeline.py` for complete episode output;
- `batch_analyze.py` for episode-wide evaluation and one selected comparison;
- `process_one_frame.py` for one-frame inspection;
- `kinova_batch.py` as a compatibility alias for `batch_analyze.py`.

All commands operate only on the validated current Kinova Gen3 RGB HDF5 schema. Robot and hand-eye selection are internal constants for this schema, so `--robot` and `--hand-eye` are removed.

## Command structure

Each entry point uses required method subcommands:

```text
<entry-point> wiener [common arguments] --K FLOAT [--adaptive-k]
<entry-point> rl      [common arguments] --rl-iters INTEGER
<entry-point> tv      [common arguments] --tv-lam FLOAT
```

The method-specific namespace prevents invalid combinations structurally:

- Wiener accepts only `--K` and `--adaptive-k`;
- Richardson-Lucy accepts only `--rl-iters`;
- TV-L2 accepts only `--tv-lam`.

Common PSF arguments are `--h5`, `--depth`, `--exposure`, `--fx`, `--fy`, and `--psf-sigma`.

Entry-point-specific arguments are:

- pipeline: `--max-frames`, `--output`, `--overwrite`;
- batch: `--max-frames`, `--show-frame`, `--output-root`, `--overwrite`;
- one frame: `--frame`, `--reverse-psf`, `--output`, `--overwrite`.

## Configuration model

Add a focused `cli_config.py` module containing:

- finite positive, finite non-negative, positive integer, and non-negative integer argparse converters;
- common CLI argument registration;
- method subparser registration;
- immutable `WienerConfig`, `RLConfig`, and `TVConfig` objects;
- complete configuration serialization and stable run-name generation;
- output-directory preparation and overwrite protection.

The deconvolution boundary accepts exactly one method configuration object instead of accepting Wiener, RL, and TV parameters simultaneously. This preserves method validity for both CLI and direct Python callers.

## Validation rules

CLI parsing enforces:

| Value | Valid domain |
|---|---|
| depth | finite and `> 0` |
| exposure | finite and `> 0` |
| fx, fy | finite and `> 0` |
| Wiener K | finite and `> 0` |
| TV lambda | finite and `> 0` |
| RL iterations | integer and `>= 1` |
| PSF sigma | finite and `>= 0` |
| maximum frames | integer and `>= 1` |
| frame and selected frame | integer and `>= 0` |

After loading an episode, runtime validation enforces:

- requested frame indices are within the effective processing range;
- camera velocity, `du`, `dv`, PSF values, and adaptive K are finite;
- depth and camera intrinsics cannot yield a non-finite interaction matrix;
- PSF dimensions do not exceed image dimensions;
- OpenCV video writers opened successfully.

Invalid values fail with a specific error instead of being interpreted silently. In particular, `--max-frames 0` is invalid and `--show-frame` is never clamped.

## Output safety and reproducibility

Every run has a configuration-derived directory name containing all parameters that can change output, including depth, exposure, focal lengths, PSF sigma, method parameters, adaptive-K state, selected frame, and maximum-frame limit when applicable.

Every output directory contains `run_config.json` with:

- source H5 path;
- entry-point name;
- method and complete method configuration;
- all PSF and camera parameters;
- frame-selection parameters;
- generated run name.

The pipeline CSV additionally records the effective per-frame Wiener K when applicable.

If a target run directory is non-empty, execution fails by default. `--overwrite` permits reuse and removes only the known generated files and directories inside that exact run directory. It never recursively deletes an arbitrary output root.

## Compatibility

This is an intentional breaking CLI change. Old `--method` commands fail with argparse usage text showing the new method subcommands. `kinova_batch.py` continues to work, but exposes the same strict batch CLI and no legacy-only options.

README examples and parameter documentation are updated to the new syntax. The historical experiment inventory remains historical, but receives a notice that its command syntax and dataset metadata describe the pre-unification version.

## Testing

Tests cover:

- every numeric converter at valid, zero, negative, NaN, and infinity boundaries;
- method-specific parser acceptance and rejection;
- complete CLI-to-config propagation for all entry points;
- frame and selected-frame range validation;
- PSF finite-value and image-size validation;
- output collision behavior with and without overwrite;
- complete `run_config.json` content and stable run names;
- method configuration dispatch and adaptive-K calculation;
- continued success of the existing RGB, HDF5, evaluation, and deconvolution tests.

Completion requires the complete test suite to pass and CLI help for all four command names to show only valid method-specific arguments.
