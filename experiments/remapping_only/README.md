# Experimental remapping-only review

**Draft research change: the fresh-inference stability gate failed. Do not deploy this as an accuracy-preserving optimization.** The purpose of this PR is to let the ForestMamba developer inspect one isolated candidate and its evidence. It does not merge the broader `bgruening:gpu-optim` changes.

Related to 3DT-2029 and 3DT-2031 in [ForestMamba Optimization](https://linear.app/geosense-ufr/project/forestmamba-optimization-c36cc7322374). The complete measured report is [results/2026-09-17/REPORT.md](results/2026-09-17/REPORT.md); the machine-readable summary, all 45 pair comparisons, individual stage records, preservation checks and frozen replay are committed alongside it.

## Additional full-cloud and diagnostic evidence

See [OBSERVATIONS.md](OBSERVATIONS.md) for the separate original-cloud experiment: **26m21s to 11m19s mean processing time (57.07% lower)**, exact mapping audit, and controlled sequence/voxel diagnostics. This gain was observed on the 38.2-million-point cloud, not established by the small baked-image cohort below. Stability still fails. Provide the original dense cloud; retain the existing internal sampling.

## Model change and original workflow

The base is upstream `main` at `24a68f9c2c0bc407508a7b6990817509674af74b`, also the original source used in the Docker comparison. In `ForAINetV2OneFormer3D_XAwarequery.predict`, the original sliding-window test path does this:

1. Select the raw points `pc1` inside a cylinder and retain their original row indices.
2. Apply the existing 0.2 m grid sampling and its existing averaging. If needed, randomly cap the resulting set at 640,000 points to obtain `pc3`.
3. Collate `pc3` into sparse coordinates/features, run the backbone, and compute embeddings and binary semantic logits.
4. Find a nearest `pc3` reference for every raw `pc1` query with `torch_cluster.knn(pc3.float(), pc1.float(), k=1)`.
5. Compute original voxel mean coordinates, select model queries, run the decoder, and predict masks/scores.
6. Apply the original edge/mask filters, project masks and semantic predictions using the nearest-neighbor indices, and assign per-point predictions.
7. Merge the overlapping windows and export predictions onto original LAS/LAZ rows.

The executable change to the existing model file is **one import and the call at step 4**. Two comments explain that the new lookup crosses to CPU. The new `remapping_knn` adapter retains the same two-row output: row 0 is query indices, row 1 is reference indices. All other model calls, including other uses of `cluster_knn`, are untouched.

No change is made to grid size, local centering, random sampling, voxel averages, sparse ordering, Mamba sequence policy, pooling, weights, query selection, decoder, thresholds, score tie handling or final merge. Prepared 20 cm input detection and actual T4 support are separate work. This PR deliberately leaves existing numerical and tie behavior in place so those changes cannot be mistaken for remapping effects.

## Replacement algorithm

`oneformer3d/hybrid_nearest.py` implements a guarded CPU spatial-index lookup:

- Accept only CUDA float32 XYZ tensors on the same device, with at least one reference. Support empty queries. The adapter supports only `k=1` and uses ten CPU workers, matching the benchmark allocation.
- Copy reference/query coordinates to CPU without changing their float32 values or order; build a SciPy `cKDTree`. The tree's distance calculation differs from the legacy CUDA arithmetic, so a direct tree replacement is not assumed equivalent.
- Query the closest **two** references with exact tree search (`eps=0`) in chunks of up to one million queries. Preserve original reference indices.
- Compare the squared-distance gap. When it is at most `64 * eps(float32) * second_squared_distance + 64 * tiny(float32)`, recompute that query with the original CUDA `torch_cluster.knn` against **all references in their original order**. This covers close neighbors, duplicate coordinates, multiway distance ties and tiny distances where relative tolerance alone would miss underflow effects.
- For nonfinite coordinates or coordinates outside the tested absolute range of 10,000 units, delegate the entire lookup to the original extension. This preserves the original behavior; it does not promise to sanitize invalid geometry.
- Return CUDA int64 indices and diagnostics. The adapter reconstructs the original two-row indexing layout; production inference does not consume the diagnostic counters.

This avoids exhaustive GPU distance searches on unambiguous geometry, but adds GPU-to-CPU synchronization, transfers, tree construction, CPU work, index transfer and fallback overhead. It is a hypothesis to benchmark, not an unconditional speed improvement. Both coordinate arrays are copied before chunking, so chunking bounds search batches rather than total host input memory. The guard is a conservative tested heuristic, not a proof for every CUDA architecture/compiler/extension version.

## Why identical code can still produce different full runs

The remapping call sits **between the backbone and query-coordinate averaging/decoder**, not at the very end of inference. Its CPU transfers change synchronization and scheduling. Original GPU reductions and ordering remain nondeterministic. Keeping their source identical does not prove that each fresh run realizes identical means, voxel membership, sparse sequence order, query selection or predictions. Equal-score mask assignment also remains unchanged and is not covered by the recorded real-window replay: no winning-score ties occurred in those windows.

Controlled diagnostics now confirm sequence sensitivity and voxel-rounding effects in the original model; see [OBSERVATIONS.md](OBSERVATIONS.md). The lookup-specific causal explanation of the measured excess variation remains unresolved. We have not shown that the new lookup produces wrong neighbors in a divergent fresh run, nor measured ground-truth accuracy. A fixed seed does not make all original CUDA operations deterministic.

The frozen replay separates the two questions: hold raw coordinates, sampled points, per-window predictions and merge inputs fixed, then change only the lookup. Replaying the original window/global merge gave exactly equal outputs for the saved fixture. Fresh inference reruns the whole nondeterministic pipeline and currently fails the agreed observed-baseline gate.

## What was actually validated

Common settings: `v6_epoch_1500_fix.pth`, SHA-256 `a398d6e5e79af8ce9cf697fdd10b1887927dc838cdafcb5ae5c3352a2d56d96a`, normal v6 configuration (`expand=2`, not the expand-1 configuration), seed 1838, deterministic execution disabled, RTX A6000 GPU2, 10 CPUs, 50,000,000,000 bytes host RAM, no swap, no container network. Images contain model/config/checkpoint/wrapper; the fresh benchmark did not bind-mount model source.

### Frozen replay

Earlier two-crop evidence covered 10,385,733 unique points and 15 windows, with original/hybrid lookups replayed through the original final merge. Lookup means were 6.235 vs 4.174 s and 5.235 vs 3.233 s, but those were lookup-only observations.

The packaged-image check repeated the six-window `tree68_10m` fixture in order: cached mapping control, original A, hybrid A, hybrid B, original B. **Every replay matched captured nearest indices, semantic labels, raw instance IDs and point scores exactly after the original final merge.** Its lookup means were **6.196 s original vs 6.750 s hybrid**, so the earlier lookup improvement did not repeat in this run. See `frozen-replay.json` for every window and fallback count.

### Fresh full inference

Five original and five candidate runs processed `tree68_10m`, a 7,008,010-point crop selected because it had already exposed variation. Input SHA-256: `b23cbca3e2cd350df70223d6fddf5ad0db7dbca2ce00a93da32ec0f59f91319b`. Pair order alternated: original1, candidate1, candidate2, original2, original3, candidate3, candidate4, original4, original5, candidate5. No inference run was discarded or retried.

All ten outputs preserved input point rows/order, coordinates, original dimensions and relevant metadata. All 45 result pairs were compared (10 original-original, 10 candidate-candidate, 25 cross). A second streaming implementation verified the primary comparator on one original pair, including the exact contingency table and scalar metrics; it did not independently recompute all 45 pairs.

Instance IDs are arbitrary: the comparator maximizes overlap with Hungarian matching between positive IDs, while matching background only to background. It also reports semantic disagreement, assigned/background disagreement, adjusted Rand disagreement, score differences, and anchored local splitting/background-loss/mixing screens. Pairwise comparisons share runs; they are not 45 independent trials. Millions of points are not treated as independent statistical replicates.

| Matched instance disagreement | Maximum | Pairs exceeding original-original maximum |
|---|---:|---:|
| Original-original | 2.2073% | baseline |
| Candidate-candidate | 11.7522% | 7/10 |
| Original-candidate | 12.6051% | 11/25 |

The requirement was **no additional variation beyond measured baseline variation**. This descriptive screen fails. It is neither a statistical equivalence test nor a ground-truth accuracy measurement. Full-cloud equivalence remains unproven.

| Timing/utilization | Original | Candidate |
|---|---:|---:|
| Median tool processing | 168.19 s | 143.20 s |
| Mean tool processing | 303.77 s | 181.05 s |
| Processing range | 141.85–844.07 s | 94.60–285.89 s |
| Median framework test iteration | 33.69 s | 35.21 s |
| Mean workflow GPU utilization | 3.65% | 1.15% |
| Mean inference-process GPU utilization | 10.80% | 2.44% |
| Peak sampled GPU memory | 4,032 MiB | 4,298 MiB |

The apparent 14.86% median total reduction is **not a reliable remapping speedup**. Startup/block-I/O variation is large; preprocessing also changed substantially despite identical preprocessing code. A shared containerd restart delayed acknowledgement of original run 5 after successful model completion. The host launch-to-exit measurement retains that delay; GPU telemetry for all runs ends at the last successful workflow stage. No shared service was restarted by the experiment. Framework test-iteration timing includes windows, merging and other loop work, not just GPU kernels. GPU memory is a 200 ms sampled maximum, not an allocation guarantee, and A6000 results do not validate Tesla T4.

## Reviewable additions and provenance

- `Dockerfile`: the exact two-target build recipe used for the measured pair. `original` installs pinned original model/config plus the common wrapper and checkpoint. `remap` derives from it and adds only the changed model file and two helpers. Measured images shared all 19 baseline layers; candidate added one layer, with identical runtime configuration except descriptive labels.
- `prepare_build.py`: new packaging helper that reconstructs that context from the pinned Git tree, verified checkpoint and common wrapper extracted from an existing local image. It does not run that image or download dependencies. It verifies wrapper files against measured hashes. The wrapper is a shared benchmark fixture, not another model optimization; it is not duplicated as thousands of unrelated PR lines.
- `commands/`: the recorded orchestration, stage measurement, GPU telemetry, LAS preservation validator, all-pair and streaming comparators, and frozen replay with original merge excerpts. Packaging replaces machine-specific Python/fixture/GPU paths with environment/manifest values; timezone becomes a manifest parameter. Comparison/model/replay algorithms are unchanged from the recorded experiment.
- `prepare_run.py`: new initializer for repeating the exact measured-image/fixture experiment in a fresh directory. It verifies the crop hash, six-window fixture completion, local immutable images and baked image source hashes before launching. It intentionally targets the measured crop and GPU2; it is not a general dataset benchmark framework.
- `tests/`: small CUDA oracle tests for the current PR helper (including exact/near/multiway distance ties, tiny distances, domain fallback, chunk boundaries, noncontiguous inputs and invalid contracts), and CPU checks for label-invariant comparison/background handling and evidence integrity. Distance-tie tests do not validate nondeterministic **mask-score** ties.
- `results/2026-09-17/`: original numeric results and hashes, with local path/GPU identifiers replaced by placeholders. Raw point clouds, weights, Docker layers and multi-gigabyte captured tensors are not committed. `original-harness-sha256.json` identifies the unrelocated scripts; the archive is a review record, not a runnable manifest.

The current PR helper has the same executable AST as the measured candidate; a stale helper docstring and two model comments were corrected. Historical image IDs/hashes describe the actual measured files, not a newly rebuilt image from this PR. The final-patch CUDA tests mount the PR source read-only in the measured runtime; they are distinct from the ten historical fresh-inference runs.

## Reproduction prerequisites and commands

**This is not yet a public one-command build.** The measured runtime/original/candidate images are local and have not been pushed to a registry. A reviewer needs them transferred by the maintainer (for example Docker save/load), plus the checkpoint/crop/frozen fixtures. A runtime recreated from the repository's original Dockerfile is not assumed numerically identical: extension/compiler versions matter. Request the measured assets or explicitly treat a new runtime/hardware combination as a new experiment. This PR provides all changed model code and the full measurement/comparison logic without requiring those large assets to read the findings.

Measured identities:

| Item | Immutable ID |
|---|---|
| Base runtime | `sha256:8fdb4ff27f765922d667366c5545433391fec438fc068aeb1d1993c17287b169` |
| Original | `sha256:00a1a1d93ccf2d94e4f2f770f3cfdf0f7c63e99047931df9b103e24170faadf7` |
| Candidate | `sha256:f7a5c88ae14b7b841c69ffa132657d06fb7fb0568bc28c60af53b5ff1e946a2e` |

Rebuild the targets, after loading the measured base and original image and obtaining the checkpoint:

```bash
python experiments/remapping_only/prepare_build.py \
  --checkpoint /absolute/v6_epoch_1500_fix.pth --output /absolute/new-context
docker tag sha256:8fdb4ff27f765922d667366c5545433391fec438fc068aeb1d1993c17287b169 forestmamba:review-base
docker build --network none --build-arg BASE_IMAGE=forestmamba:review-base \
  --target original -t forestmamba:review-original /absolute/new-context
docker build --network none --build-arg BASE_IMAGE=forestmamba:review-base \
  --target remap -t forestmamba:review-remap /absolute/new-context
```

Repeat the **recorded immutable images**, with the exact input and saved `tree68_10m_capture/frozen` folder available. Linux host requirements: Docker Unix socket access, NVIDIA container runtime, `nvidia-smi`, Python >=3.9, and a CPU analysis environment containing NumPy, SciPy, laspy and lazrs. GPU2 must be free; the harness refuses an occupied GPU and retains output/failure evidence rather than overwriting it.

```bash
export FM_CPU_PYTHON=/absolute/analysis-environment/bin/python
python experiments/remapping_only/prepare_run.py \
  --input /absolute/tree68_10m.laz \
  --frozen-outputs /absolute/frozen-experiment/outputs \
  --gpu-timezone Europe/Berlin --output /absolute/new-run
python /absolute/new-run/commands/benchmark.py
python /absolute/new-run/commands/report.py
```

The repeat initializer intentionally uses the recorded image IDs; rebuilding creates a separate artifact, which must be independently verified before substitution. Do not edit historical IDs or call a new image's results the recorded cohort. For other inputs/hardware, adapt a fresh experiment and compare new original repeats under the same conditions.

CPU tests:

```bash
python -m unittest discover -s experiments/remapping_only/tests -p test_evidence.py -v
```

CUDA oracle tests in the loaded measured runtime, with a read-only mount of this checkout:

```bash
docker run --rm --pull=never --network none --gpus device=2 \
  --cpus 10 --memory 50000000000 --memory-swap 50000000000 \
  -e PYTHONDONTWRITEBYTECODE=1 -e OMP_NUM_THREADS=10 -e OPENBLAS_NUM_THREADS=10 \
  -v "$PWD":/review:ro --workdir /tmp --entrypoint python \
  forestmamba:remap-only-v6-20260917t102454z \
  /review/experiments/remapping_only/tests/test_remapping.py -v
```

## Questions for the model developer before adoption

1. Is raw-to-sampled remapping the right replacement boundary? Would moving/caching it after the model predictions avoid perturbing execution, while retaining identical points and projection?
2. What reproducibility contract is intended for voxel reductions, sparse ordering and equal-score mask ownership? Which operations are expected to vary with identical input/seed?
3. Is the numerical ambiguity guard adequate for the supported torch_cluster build and target GPUs, or should more cases use the original extension?
4. Can a divergent fresh run be captured at the first differing stage (sampled means, sparse coordinates/order, embeddings, queries, masks, neighbor indices, merge), with same-schedule original controls to distinguish a mapping error from nondeterminism/scheduling?
5. After that cause is isolated, can independent full-cloud repeats meet the baseline envelope and produce a repeatable end-to-end improvement on an uncontended host, then on an actual Tesla T4 below its memory limit?

Until those gates pass, keep this PR in draft and retain the original implementation for production.
