# Remapping-only Docker benchmark

Two self-contained local images were built and verified. The candidate changes only one import and one nearest-neighbor call in the original model, plus remapping.py and hybrid_nearest.py. It preserves the original averaging and sparse sequence-ordering code, model weights, decoder and merge logic. Those original operations remain nondeterministic; unchanged code does not guarantee identical realized model inputs or sequences across fresh runs. The common runtime and LAS/LAZ wrapper match the prior controlled benchmark. No model source/configuration/checkpoint bind mounts were used.

## Results

| Metric | Original | Remapping only |
|---|---:|---:|
| Mean processing runtime (s) | 303.77 | 181.05 |
| Median processing runtime (s) | 168.19 | 143.20 |
| Minimum processing runtime (s) | 141.85 | 94.60 |
| Maximum processing runtime (s) | 844.07 | 285.89 |
| Mean launch-to-exit wall time (s) | 529.33 | 329.62 |
| Median launch-to-exit wall time (s) | 255.84 | 170.17 |
| Mean framework test iteration (s) | 46.70 | 31.50 |
| Median framework test iteration (s) | 33.69 | 35.21 |
| Mean workflow-window GPU utilization (%) | 3.65 | 1.15 |
| Mean inference-process GPU utilization (%) | 10.80 | 2.44 |
| Peak sampled GPU memory (MiB) | 4032.00 | 4298.00 |

Observed mean processing-time reduction: 40.40%; observed median processing-time reduction: 14.86%.

| Mean stage time (s) | Original | Remapping only |
|---|---:|---:|
| input_conversion | 1.541 | 2.106 |
| model_preprocess | 186.489 | 87.468 |
| model_inference | 92.448 | 85.410 |
| merge_output | 4.371 | 4.319 |

The runtime differences are observational. Large startup/block-I/O variation, together with the lookup-only measurement in this run, prevents attributing the full end-to-end difference to remapping. No repeatable end-to-end speedup is established by this small shared-host cohort.

## Individual runs in execution order

| Run | Processing (s) | Inference process (s) | GPU utilization (%) |
|---|---:|---:|---:|
| tree68_10m_main_01 | 844.073 | 227.804 | 0.83 |
| tree68_10m_remap_01 | 261.060 | 128.546 | 0.56 |
| tree68_10m_remap_02 | 285.891 | 116.240 | 0.59 |
| tree68_10m_main_02 | 158.582 | 64.454 | 4.56 |
| tree68_10m_main_03 | 141.850 | 62.635 | 5.21 |
| tree68_10m_remap_03 | 143.196 | 74.068 | 1.21 |
| tree68_10m_remap_04 | 94.601 | 42.894 | 1.90 |
| tree68_10m_main_04 | 168.194 | 58.520 | 4.33 |
| tree68_10m_main_05 | 206.165 | 48.825 | 3.35 |
| tree68_10m_remap_05 | 120.483 | 65.301 | 1.51 |

## Result stability

not accepted: excess variation observed.

| Metric | Original-original maximum | Original-remap maximum | Cross pairs above original maximum |
|---|---:|---:|---:|
| semantic_disagreement | 0.00171090 | 0.00793834 | 7/25 |
| assigned_unassigned_disagreement | 0.01993647 | 0.12388481 | 11/25 |
| instance_disagreement | 0.02207260 | 0.12605062 | 11/25 |
| one_minus_ari | 0.06221750 | 0.34331380 | 11/25 |
| score_mae | 0.04026394 | 0.20967513 | 11/25 |

Baked-image exact replay lookup means: original 6.196 s; hybrid 6.750 s. These timings come from this run, separately from the earlier frozen-prediction experiment.

Instance disagreement uses optimal label matching and includes background; it does not mistake arbitrary instance renumbering for changed segmentation. All 45 within-/cross-image pairs were compared. A separate streamed comparator verified the in-memory metrics. All ten outputs preserved the input points, order, attributes and relevant metadata. No ground-truth accuracy claim is made. Pairwise comparisons share runs and are correlated; five repetitions do not establish statistical equivalence.

The baked candidate helper also passed the saved six-window exact replay, including original final merge: all indices, semantic labels, raw instance IDs and scores were identical. This isolates mapping equivalence from the nondeterministic model steps executed anew in the fresh cohort. Exact replay does not clear an observed full-inference stability failure. This cohort does not establish the causal mechanism behind the excess variation; it is not proof of a nearest-neighbor index error or of degraded ground-truth accuracy. No deterministic-mean, sequence-ordering or tie-breaking prototype was included.

## Conditions and provenance

Sequential GPU2 RTX A6000, 10 CPUs, 50 GB RAM, no swap, no container network; five fresh original and five fresh candidate containers with alternating pair order. v6_epoch_1500_fix.pth, normal v6 configuration, seed 1838, original nondeterministic execution policy. The crop was selected because it previously exposed excess variation.

Processing runtime is wrapper execution time; launch-to-exit wall time also includes Docker creation/startup and shutdown; stages cover input conversion, preprocessing, model process and output export. GPU utilization is the arithmetic mean of per-run device samples restricted to container start through the final successful workflow stage; startup and workflow I/O idle time are included, while daemon cleanup after completion is excluded. Framework test-iteration time includes the model windows, final merge and other work inside the test loop; it is separate from model-process startup and is not GPU-kernel-only timing. First-call kernel initialization and output/evaluation work may affect it. GPU memory is a 200 ms sampled maximum, not a guaranteed allocation peak. This is not an actual Tesla T4 test. Timing is sensitive to shared-host I/O; report ranges and medians alongside means.

| Image | Immutable local ID |
|---|---|
| `forestmamba:original-v6-20260917t102454z` | `sha256:00a1a1d93ccf2d94e4f2f770f3cfdf0f7c63e99047931df9b103e24170faadf7` |
| `forestmamba:remap-only-v6-20260917t102454z` | `sha256:f7a5c88ae14b7b841c69ffa132657d06fb7fb0568bc28c60af53b5ff1e946a2e` |

Base image: `sha256:8fdb4ff27f765922d667366c5545433391fec438fc068aeb1d1993c17287b169`.
Original model source: `24a68f9c2c0bc407508a7b6990817509674af74b`.
Checkpoint SHA-256: `a398d6e5e79af8ce9cf697fdd10b1887927dc838cdafcb5ae5c3352a2d56d96a`.

A containerd restart delayed acknowledgement of original run 05 after its model process had completed successfully. Only that finished benchmark container was cleared; no shared service was restarted by this benchmark. Its launch-to-exit time is an infrastructure outlier, and GPU utilization excludes the stale post-completion interval. Details are retained in provenance/runtime-interruption.json.

The images are local and were not pushed or deployed. Dockerfile and complete build inputs are under build/. Image file hashes, inspection records and the two-line source patch are under provenance/. Exact commands, logs, telemetry, output LAZ files and comparator details are retained. Read README.md for usage.
