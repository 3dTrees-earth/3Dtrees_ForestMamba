# Remapping-only: full-cloud performance and controlled diagnostics

The inference change remains one nearest-neighbor call and its helper modules.
Use the **original dense cloud** and retain ForestMamba's internal 20 cm sampling.
External subsampling/detection, new means, sequence sorting, overflow fixes and
checkpoint changes are outside this PR. This update changes evidence/documentation
and restores the documented tests omitted by the repository's broad `tests` ignore
rule; it changes no model/helper executable code.

## Full original cloud: measured performance gain

The 16 September run used the original 1838 cloud (38,206,792 points), three fresh
runs per variant, GPU2 RTX A6000, ten CPUs and 50 GB RAM. The checkpoint was
`v6_epoch_1500_fix.pth`, normal v6 configuration (expand=2), seed1838. The original
source was `24a68f9c2c0bc407508a7b6990817509674af74b`. Executable ASTs of the model
and both remapping helpers match this PR after excluding module docstrings.

| Mean measurement | Original | Remapping-only |
|---|---:|---:|
| Total processing | 1,581.11 s (26m21s) | 678.71 s (11m19s) |
| Input conversion | 8.77 s | 8.82 s |
| Preprocessing | 76.23 s | 74.03 s |
| Inference stage | 1,471.54 s | 571.20 s |
| Output merge/write | 24.09 s | 24.19 s |
| Remapping stream interval | 1,151.62 s | 263.56 s |
| Mean sampled GPU utilization | 74.92% | 16.92% |
| Peak sampled device memory | 12,316 MiB | 12,192 MiB |

**Observed saving: 902.40 seconds (~15 minutes), 57.07% less total time, or 2.33x
throughput for this dataset/runtime.** The mapping interval fell 77.11%. It is a
nested CUDA-event interval that includes CPU gaps, not a pure GPU-kernel metric;
do not add it to stage times. This experiment used instrumented source snapshots
mounted into a common runtime, unlike the later baked-image cohort already in
this PR. The separate mapping audit is excluded from performance samples.

The audit checked 64 windows and **1,318,704,290 point-window mappings with zero
index differences**. Despite that, maximum matched instance disagreement was
5.0786% within the three fresh originals and 10.1813% across versions. The speed
observation is therefore **not an accuracy-preserving performance acceptance**.
See [full-cloud.json](results/2026-09-17/additional-evidence/full-cloud.json) for
individual runs, stage times, mapping audit and stability screens.

Small-input overhead can dominate: the separate two-crop diagnostic had these
median processing times, and does not establish a general speedup:

| Requested 10 m crop | Original | Remapping-only | Original maximum instance disagreement | Cross-version maximum |
|---|---:|---:|---:|---:|
| Tree59, 3,377,723 points | 68.47 s | 66.69 s | 1.6436% | 1.5695% |
| Tree68, 7,008,010 points | 106.86 s | 145.09 s | 6.8412% | 11.4468% |

Each crop used five originals and five candidates, alternated in fresh processes,
with 45 comparisons per crop. Nine of the 25 tree68 cross-version pairs exceeded
the observed original instance-disagreement maximum. Tree59 had no cross-version
excess, but one within-candidate pair exceeded the original background/ARI ranges.
The later baked-image cohort remains a **third, separate measurement**, documented
in [README.md](README.md): its 14.86% median processing reduction was confounded
by startup/I/O, and its frozen lookup test did not show a speedup. Do not pool
these cohorts or present the full-cloud gain as universal. A6000 measurements do
not validate Tesla T4.

## Confirmed influences and unresolved attribution

Separate traces first differed at **window 0 voxel sampling**, before the first
lookup, in both original/original and original/candidate comparisons. Raw inputs
and recorded RNG states matched. A seed change is not the explanation.

1. **Internal sequence order is a confirmed major influence.** Identical sparse
   inputs/features produced 976,421 differing window assignments in an unchanged
   original repeat. The first observed internal sequence difference was
   `unet.u.mamba_block`: original slab sorting leaves some coordinates tied and
   dependent on incoming sparse row order. Freezing the exact sequence from the
   first original replay at every Mamba block reproduced that replay reference
   and all three comparisons bit-for-bit. This was independently reproduced on
   tree59 (65,494 original-control differences; zero with its sequence frozen).
2. **Voxel averaging/quantization also matters.** GPU averaging differed by up to
   0.1154 mm. With ordering controlled diagnostically, two changed-mean inputs
   created an extra sparse voxel and changed 46,843/46,845 assignments; the
   unchanged-coordinate control changed none. A third changed-mean input changed
   nine assignments. This is a different-input effect, not disagreement between
   lookup implementations on identical coordinates.
3. **Feature reductions have smaller observed effects.** With original sequences
   frozen, reduction differences up to 9.54e-7 changed logits, scores and 2–3 raw
   mask entries, but did not change final matched assignments in this window.
4. **Lookup-specific causation remains unresolved.** All checked indices agree.
   With a frozen backbone, original/hybrid/cached/synchronized lookup trials give
   exact decoder predictions and assignments. Before the backbone, both lookup
   modes show internal variation; freezing the original sequence removes all
   assignment variation in all 12 trials. Synchronization alone does not fix it.
   Changed scheduling/allocations are plausible influences, not proven causes of
   the larger complete-run range. Five original runs may not capture its full
   distribution of outcomes.
5. **Winning-score ties were not implicated in these windows.** Neither captured
   window had winning-score ties; 20 repeats per window gave identical winners.
   Other windows and datasets are not ruled out.

Canonical tie ordering and frozen-sequence interventions were diagnostic-only.
No production sort, averaging rule, model interface or other inference behavior
was changed. Window assignment counts are not whole-cloud accuracy measurements.
The recorded pointwise/crop comparisons remove arbitrary instance renumbering;
background is matched only to background. These shared-run pair comparisons are
observed ranges, not independent statistical trials or an equivalence guarantee.

The small experiment completed 20 clean runs, two calibrations, six traces, 90
pair comparisons and 28 passing output-preservation checks. The standalone
controlled suites took 67–95 seconds. Detailed numeric evidence is in
[additional-evidence](results/2026-09-17/additional-evidence/), including original
unchanged controls, frozen-sequence comparisons and lookup-side-effect probes.
Large input/tensor fixtures remain local; this evidence update does not claim
that those diagnostic fixtures are publicly downloadable.

## Decision

Keep the PR in draft. The dense original input is the intended path: retain fast
internal sampling and map predictions to the original rows. External 20 cm
preparation is unnecessary just to avoid that sampling step and can change voxel
alignment/representatives. Dense input still costs more I/O, RAM and preprocessing.

The remapping replacement has promising full-cloud performance and strong mapping
checks, but **has not met the no-additional-variation requirement**. Preserving
one captured original sequence validates an exact fixture-specific computation;
it is not a general deployable determinism fix. Further optimization should use
that control while preserving the original model logic. No ground-truth accuracy
or merge-readiness claim is made.
