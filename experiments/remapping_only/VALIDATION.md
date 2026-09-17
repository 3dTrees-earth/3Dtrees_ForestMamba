# PR packaging validation, 17 September 2026

These checks validate the review package. The ten historical full-inference runs are documented separately under `results/2026-09-17/`.

| Check | Result |
|---|---|
| CUDA oracle unit tests using current PR source, measured runtime and GPU2 RTX A6000 | 9 passed; none skipped |
| CPU comparison/evidence unit tests | 4 passed |
| Model/helper executable AST comparison against measured candidate | All 3 files identical after excluding module docstrings |
| Experiment Python syntax | All 21 Python files parsed |
| Build-context generator smoke test | Passed; checkpoint and common wrapper hashes verified |
| Build-context comparison against consumed historical inputs | All 36 original input files byte-identical |
| Repeat initializer smoke test | Passed; exact crop hash, six-window completion marker, original/candidate immutable image IDs and 35/37 baked file inventories verified |
| Complete staged patch whitespace check | Passed |

No new full-inference cohort was launched during PR packaging. The initialized repeat directory was left unexecuted. No image was pushed or deployed. The historical stability verdict remains **failed**, despite the focused tests passing.

Compared with the recorded helper source, the PR changes only documentation/comments. Harness packaging replaces host-specific locations with environment/manifest parameters and removes two extra end-of-file blank lines; original unrelocated script hashes are retained. The new build/repeat initializers are packaging additions, not new model logic.

## Additional evidence update

- Restored the documented CPU/CUDA test files, which the repository's broad `tests` ignore rule had omitted from the original PR commit; added a narrow ignore exception.
- Re-ran all 4 CPU comparison/evidence tests and all 9 CUDA oracle tests on free GPU2 in the measured runtime with 10 CPUs/50 GB RAM: all passed, none skipped.
- Verified the model and both helpers have identical executable ASTs to both the earlier full-cloud and small-diagnostic candidate snapshots, excluding module docstrings. This update changes no inference code.
- Added full-cloud stage/runtime evidence and controlled sequence/reduction/lookup observations in OBSERVATIONS.md and additional-evidence JSON. All JSON parses without machine-specific absolute paths; Python syntax and diff whitespace checks pass.
- No new full-cloud or fresh-inference cohort was run for this evidence update. The stability verdict remains failed; no production ordering or averaging fix is included.
