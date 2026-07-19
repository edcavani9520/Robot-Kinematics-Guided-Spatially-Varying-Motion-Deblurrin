# Kinematics-Guided Deblurring and Pi0.5 Experiment Design

Date: 2026-07-19

## 1. Scope

This document defines the experiments to execute. It does not contain paper prose or assumed conclusions.

The experiment has two parts:

1. Offline parameter ablation and cross-episode evaluation of kinematics-guided deblurring.
2. A 2 x 2 real-robot comparison of raw/deblurred LoRA training data and raw/deblurred inference images.

No new camera calibration, data collection, synthetic benchmark, spatially varying PSF experiment, or additional robot task is required for this study.

## 2. Fixed System Configuration

Unless an experiment explicitly varies one parameter, use:

| Parameter | Value |
|---|---:|
| Image format | RGB |
| Resolution | 320 x 240 |
| Depth | 0.5 m |
| Exposure duration | 0.01 s |
| fx, fy | 260.65 px |
| cx, cy | 159.5, 119.5 px |
| PSF sigma | 0 |
| Adaptive K | disabled |
| Final online method | Wiener |
| Candidate final Wiener K | 0.03 |

The intrinsics and depth are task-specific estimates, not calibration results. The final Wiener K may be changed once, before LoRA dataset generation, if the corrected 0.01 s ablation selects a clearly better value. It must then remain frozen for both training preprocessing and online inference.

Archived runs named with `e0.03` are preliminary results produced with an older assumed exposure. They must not be presented as results obtained with the confirmed 0.01 s exposure. The main ablation should reuse the existing parameter grid with the corrected configuration; no robot recollection is required.

## 3. Offline Data Allocation

The current local data contain 528 RGB frames:

| Episode | Frames | Role |
|---|---:|---|
| episode_0003 | 104 | Development episode for all parameter selection |
| episode_0001 | 107 | Held-out cross-episode evaluation |
| episode_0002 | 126 | Held-out cross-episode evaluation |
| episode_0004 | 69 | Held-out cross-episode evaluation |
| episode_0005 | 122 | Held-out cross-episode evaluation |

All episodes are 320 x 240 and use `obs/image`, `obs/proprio`, `action`, and `timestamps`. Do not describe episode_0002 as a 640 x 480 episode or as a different H5 schema.

All parameter decisions are made using episode_0003. Parameters may not be changed after inspecting the held-out episode results.

## 4. Offline Experiments

### E1: Method comparison

Compare the raw image and three deconvolution methods on episode_0003:

| Condition | Parameters |
|---|---|
| Raw | No deconvolution |
| Wiener | K = 0.03 |
| Richardson-Lucy | iterations = 5 |
| TV-L2 | lambda = 0.002 |

If E2-E4 select a different development-episode optimum, update the corresponding E1 parameter before producing the final E1 table. Do not tune E1 using held-out episodes.

### E2: Wiener regularization ablation

Run on episode_0003:

```text
K = 0.0001, 0.0005, 0.001, 0.003, 0.005, 0.007,
    0.01, 0.02, 0.03, 0.15, 0.2, 0.5, 1.0
```

Select the operating point using all of the following rather than sharpness alone:

- positive but controlled Laplacian and Tenengrad gain;
- limited TV growth;
- preservation of Edge Ratio;
- limited SSIM reduction;
- real-time processing feasibility.

### E3: Richardson-Lucy iteration ablation

Run on episode_0003:

```text
iterations = 5, 10, 15, 20, 30, 50
```

Record quality metrics and processing time. The selected setting is the lowest iteration count after which additional sharpness is outweighed by artifact growth or runtime.

### E4: TV-L2 regularization ablation

Run on episode_0003:

```text
lambda = 0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.003, 0.004
```

Choose the intermediate setting that improves edge response without excessive smoothing or TV growth.

### E5: Physical-parameter sensitivity

Use Wiener K = 0.03 on episode_0003 and change one variable at a time:

```text
depth    = 0.3, 0.5, 0.8 m
exposure = 0.005, 0.01, 0.02 s
```

This experiment measures sensitivity to assumed model parameters. It does not identify the true depth or exposure.

### E6: Optional-component ablation

Use Wiener K = 0.03 on episode_0003:

| Condition | Adaptive K | PSF sigma |
|---|---:|---:|
| Base | Off | 0 |
| Adaptive only | On | 0 |
| Smoothing only | Off | 1.0 |
| Adaptive and smoothing | On | 1.0 |

Keep negative results. An optional component is enabled only if it improves artifact-controlled metrics rather than Laplacian/Tenengrad alone.

### E7: Cross-episode evaluation

Freeze the E2-E4 selected parameters and run Wiener, RL, and TV-L2 on episodes 0001, 0002, 0004, and 0005. No per-episode tuning is permitted.

## 5. Offline Metrics and Aggregation

For every run record:

- number of processed frames;
- mean, standard deviation, and median of each metric across frames;
- Laplacian before, after, and relative/absolute change;
- Tenengrad before, after, and relative/absolute change;
- TV before, after, and ratio;
- Edge Ratio;
- input-to-output SSIM;
- input-to-output PSNR if retained;
- total runtime, mean time per frame, and effective processing FPS;
- representative raw/deblurred image and PSF.

SSIM and PSNR are content-preservation measures because no sharp ground truth exists. Laplacian and Tenengrad are sharpness responses but can reward noise and ringing. A configuration must therefore be judged by the combined sharpness, artifact, preservation, and runtime evidence.

`metrics.txt` currently describes only the representative frame. Main quantitative comparisons must use whole-episode summaries. Representative frames are qualitative evidence only. Use a predetermined frame (the temporal midpoint) rather than selecting the visually best result.

## 6. Paired Pi0.5 Training Datasets

Create two LeRobot datasets from exactly the same successful demonstration episodes.

### D-Raw

- Keep the original RGB observations.
- Keep proprioception, actions, timestamps, task text, and episode boundaries unchanged.

### D-Deblur

- Replace only RGB observations with offline Wiener-deblurred RGB images.
- Use exactly the same Wiener/physical parameters as online inference.
- Keep proprioception, actions, timestamps, task text, and episode boundaries byte-identical where applicable.

Use an episode-level train/validation split and identical episode IDs in both splits. Do not split adjacent frames from one episode across training and validation.

Before training, verify:

- equal episode and frame counts;
- equal state/action arrays and task instructions;
- equal image shapes and color order;
- no image processing failures or missing frames;
- only RGB pixel values differ.

## 7. LoRA Training Conditions

Train two policies:

| Policy | Dataset |
|---|---|
| pi-Raw | D-Raw |
| pi-Deblur | D-Deblur |

Hold constant:

- Pi0.5 base checkpoint;
- LoRA rank, alpha, dropout, and target modules;
- optimizer, learning rate, scheduler, and weight decay;
- batch size and gradient accumulation;
- training steps, warmup, and checkpoint rule;
- image augmentation and normalization;
- dataset order/shuffling and random seed;
- hardware and software environment.

Select both checkpoints using the same rule, preferably the same final training step. Do not select checkpoints using real-robot test success.

One training seed per visual condition is accepted for the deadline-limited experiment. Do not interpret robot-trial variability as training-seed variability.

## 8. Downstream 2 x 2 Design

The single task is `Put the block into the bowl.`

| Group | Policy | Training images | Inference images | Trials |
|---|---|---|---|---:|
| G1 | pi-Raw | Raw | Raw | 20 |
| G2 | pi-Raw | Raw | Real-time deblurred | 20 |
| G3 | pi-Deblur | Deblurred | Raw | 20 |
| G4 | pi-Deblur | Deblurred | Real-time deblurred | 20 |

Total: 80 real-robot trials.

Primary contrasts:

- G4 minus G1: total pipeline effect;
- G2 minus G1: inference-only preprocessing effect;
- G3 minus G1: training-only preprocessing/domain-mismatch effect;
- G4 minus G2: deblurred-training effect under deblurred inference;
- G4 minus G3: online-deblurring effect for the deblurred policy.

Do not alter deblurring parameters, policy checkpoints, prompt, controller settings, workspace limits, or stopping rules after the first valid trial.

## 9. Initial Layouts

Use five in-distribution layouts. The bowl remains fixed.

| Layout | Block position |
|---|---|
| P1 | Nominal center |
| P2 | Approximately 3 cm left |
| P3 | Approximately 3 cm right |
| P4 | Approximately 3 cm toward the robot |
| P5 | Approximately 3 cm away from the robot |

Mark positions on the table and photograph the setup. Each group receives four trials at each layout, for 20 trials per group.

## 10. Balanced Test Order

For each layout, execute four blocks using the following balanced orders. The order of these four rows may be randomized before starting a layout, but all four must be used once.

| Block within layout | Test order |
|---:|---|
| 1 | G1, G2, G3, G4 |
| 2 | G2, G3, G4, G1 |
| 3 | G3, G4, G1, G2 |
| 4 | G4, G1, G2, G3 |

This gives every group each ordinal test position once per layout and avoids running all trials of one condition consecutively.

Before every trial:

1. Return the robot to the fixed initial joint pose.
2. Reset the block and bowl using the layout markers.
3. Confirm camera connection, resolution, exposure mode, and image orientation.
4. Reset policy/controller history.
5. Select the scheduled policy and inference preprocessing condition.
6. Use the identical language prompt.
7. Start the trial video and timer.

## 11. Outcome Definition

A trial is successful only if all conditions hold:

- the robot grasps the block;
- transports it to the bowl;
- releases it completely inside the bowl;
- the block remains inside for at least 3 seconds;
- no human intervention occurs;
- completion occurs within 60 seconds.

Count as failures:

- perception/localization failure;
- grasp failure;
- transport/drop failure;
- release or placement outside/on the rim;
- timeout;
- policy-triggered safety abort;
- any required human assistance.

Assign one primary failure code:

| Code | Meaning |
|---|---|
| P | Perception/localization |
| G | Grasp |
| T | Transport/drop |
| R | Release/placement |
| O | Timeout |
| S | Safety abort |

## 12. Invalid Trials and Stopping Rules

A trial may be marked invalid and repeated only for an external technical failure that occurs before the policy has a meaningful opportunity to act, such as camera disconnection, logging failure, robot API disconnection, or an operator placing the wrong condition/layout.

Policy errors, poor trajectories, policy-caused safety stops, and deblurring latency are outcomes, not invalid trials. Record every invalid trial and its reason. Do not stop early because a condition appears better or worse; complete all 80 valid trials.

## 13. Trial Record

Record one row per attempted trial:

```text
trial_id
date_time
layout
block_id
order_position
group
policy_checkpoint
training_preprocessing
inference_preprocessing
deblur_parameters
success
failure_code
completion_time_s
invalid_trial
invalid_reason
safety_abort
video_filename
operator_notes
```

Also record mean and P95 deblurring latency and, if available, effective control-loop frequency for raw and deblurred inference.

## 14. Statistical Analysis

For each group report:

- successes out of 20;
- success percentage;
- 95% Wilson confidence interval;
- absolute percentage-point difference from G1;
- failure-mode counts.

Primary model:

```text
logit(success) = beta0
               + beta1 * training_deblur
               + beta2 * inference_deblur
               + beta3 * training_deblur * inference_deblur
               + layout
```

If the logistic model is unstable because of complete/quasi-complete separation, report Fisher exact tests for the prespecified comparisons G4 vs G1, G2 vs G1, and G4 vs G2, with Holm correction. Report effect sizes and confidence intervals whether or not p < 0.05.

## 15. Required Outputs

Offline experiments:

- full per-run metrics table;
- whole-episode summary table;
- Wiener K curve;
- RL-iteration and TV-lambda curves;
- depth and exposure sensitivity results;
- optional-component ablation;
- cross-episode comparison;
- representative RGB/PSF comparisons;
- runtime table.

Downstream experiment:

- the two frozen LoRA training configurations;
- raw 80-trial log including invalid attempts;
- four-group success counts and Wilson intervals;
- prespecified statistical comparisons;
- failure-mode counts;
- online latency summary;
- links between every trial and its video.

## 16. Completion Criteria

The experiment is complete when:

- corrected 0.01 s parameter ablations and whole-episode summaries exist;
- the final Wiener configuration is frozen;
- paired LoRA datasets pass equality checks except for RGB pixels;
- both LoRA runs use matching hyperparameters and checkpoint rules;
- all 80 valid robot trials are completed under the balanced protocol;
- exclusions, technical failures, and parameter deviations are documented;
- success-rate confidence intervals and prespecified comparisons are calculated.
