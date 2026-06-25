#!/usr/bin/env python3
"""Galaxy entrypoint for ForestMamba LAS/LAZ inference."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


_SRC_DIR = Path(__file__).parent.resolve()
_REPO_DIR = _SRC_DIR.parent
_SCRIPTS_DIR = _REPO_DIR / "scripts"
for path in (_SRC_DIR, _SCRIPTS_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

try:
    from parameters import Parameters, print_params
except ImportError as exc:
    print(f"Error: Could not import parameters.py: {exc}")
    sys.exit(1)


def build_batch_args(params: Parameters) -> argparse.Namespace:
    inputs = [str(path) for path in params.all_inputs()]
    if not inputs and params.input_list is None:
        raise ValueError(
            "No input LAS/LAZ file was provided. Use --dataset-path, --input, "
            "--input-paths, or --input-list."
        )

    return argparse.Namespace(
        inputs=inputs,
        input_list=str(params.input_list) if params.input_list else None,
        output_dir=str(params.output_dir),
        work_dir=str(params.work_dir),
        repo_dir=str(params.repo_dir),
        checkpoint=str(params.checkpoint),
        config_base=str(params.config_base),
        gpu_device=params.gpu_device,
        preprocess_workers=params.preprocess_workers,
        chunk_size=params.chunk_size,
        tile_size=params.tile_size,
        tile_overlap=params.tile_overlap,
        tile_max_points=params.tile_max_points,
        keep_tile_work=params.keep_tile_work,
        crop_center_x=params.crop_center_x,
        crop_center_y=params.crop_center_y,
        crop_radius=params.crop_radius,
        prediction_dir=str(params.prediction_dir) if params.prediction_dir else None,
        skip_inference=params.skip_inference,
        skip_preprocess=params.skip_preprocess,
        semantic_dim=params.semantic_dim,
        instance_dim=params.instance_dim,
        score_dim=params.score_dim,
        benchmark_stage_log=(
            str(params.benchmark_stage_log) if params.benchmark_stage_log else None
        ),
        bluepoint_iterations=params.bluepoint_iterations,
        bluepoint_score_threshold=params.bluepoint_score_threshold,
        bluepoint_second_pass_threshold=params.bluepoint_second_pass_threshold,
        spatial_match_tolerance=params.spatial_match_tolerance,
    )


def load_batch_module():
    try:
        import forest_mamba_laz_batch
    except ImportError as exc:
        print(f"Error: Could not import ForestMamba batch wrapper: {exc}")
        sys.exit(1)
    return forest_mamba_laz_batch


def run(params: Parameters) -> None:
    batch_args = build_batch_args(params)
    load_batch_module().run_batch(batch_args)


def main(argv: list[str] | None = None) -> None:
    try:
        params = Parameters.from_cli(argv)
        if params.show_params:
            print_params(params)
            return
        run(params)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
