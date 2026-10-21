#!/usr/bin/env python3
"""Galaxy-compatible ForestMamba entrypoint.

The deployed Galaxy tool calls /workspace/src/run.py. Keep that stable and
delegate to the richer batch runner used by local validation.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


DEFAULT_REPO_DIR = Path("/workspace")
DEFAULT_CHECKPOINT = (
    DEFAULT_REPO_DIR
    / "work_dirs"
    / "forestmamba_chm_radius16_qp300_2many_v6"
    / "epoch_3000_fix.pth"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ForestMamba on a LAS/LAZ file.")
    parser.add_argument("--dataset-path", required=True, help="Input LAS/LAZ path.")
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument("--work-dir", required=True, help="Scratch work directory.")
    parser.add_argument(
        "--repo-dir",
        default=str(DEFAULT_REPO_DIR),
        help="ForestMamba repository root inside the container.",
    )
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument(
        "--config-base",
        default=str(
            DEFAULT_REPO_DIR
            / "configs"
            / "ForAINetv2"
            / "forestmamba_chm_radius16_qp300_2many_v6.py"
        ),
    )
    parser.add_argument("--preprocess-workers", default="1")
    parser.add_argument("--chunk-size", default="1000000")
    parser.add_argument("--semantic-dim", default="PredSemantic_FM")
    parser.add_argument("--instance-dim", default="PredInstance_FM")
    parser.add_argument("--score-dim", default="PredScore_FM")
    parser.add_argument("--bluepoint-iterations", default=None)
    parser.add_argument("--bluepoint-second-pass-threshold", default="0.01")
    parser.add_argument("--spatial-match-tolerance", default="0.01")
    return parser.parse_args()


def normalized_checkpoint(path: str) -> str:
    checkpoint = Path(path)
    if checkpoint.exists():
        return str(checkpoint)
    if checkpoint.name == "v6_epoch_1500_fix.pth" and DEFAULT_CHECKPOINT.exists():
        print(
            f"[compat] Checkpoint {checkpoint} not found; using {DEFAULT_CHECKPOINT}",
            flush=True,
        )
        return str(DEFAULT_CHECKPOINT)
    return str(checkpoint)


def main() -> int:
    args = parse_args()
    repo_dir = Path(args.repo_dir)
    runner = repo_dir / "scripts" / "forest_mamba_laz_batch.py"
    command = [
        sys.executable,
        str(runner),
        "--input",
        args.dataset_path,
        "--output-dir",
        args.output_dir,
        "--work-dir",
        args.work_dir,
        "--repo-dir",
        str(repo_dir),
        "--checkpoint",
        normalized_checkpoint(args.checkpoint),
        "--config-base",
        args.config_base,
        "--preprocess-workers",
        str(args.preprocess_workers),
        "--chunk-size",
        str(args.chunk_size),
        "--semantic-dim",
        args.semantic_dim,
        "--instance-dim",
        args.instance_dim,
        "--score-dim",
        args.score_dim,
        "--bluepoint-second-pass-threshold",
        str(args.bluepoint_second_pass_threshold),
        "--spatial-match-tolerance",
        str(args.spatial_match_tolerance),
    ]
    if args.bluepoint_iterations is not None:
        command.extend(["--bluepoint-iterations", str(args.bluepoint_iterations)])
    return subprocess.run(command, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
