#!/usr/bin/env python3
"""Galaxy-facing parameter parsing for the ForestMamba wrapper."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


DEFAULT_CHECKPOINT = (
    "/workspace/work_dirs/forestmamba_chm_radius16_qp300_2many_v6/"
    "v6_epoch_1500_fix.pth"
)
DEFAULT_CONFIG_BASE = (
    "/workspace/configs/ForAINetv2/"
    "forestmamba_chm_radius16_qp300_2many_v6.py"
)
DEFAULT_REPO_DIR = Path(__file__).resolve().parents[1]


def parse_bool(value: str | bool) -> bool:
    """Accept bare flags plus true/false strings from Galaxy XML commands."""
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if normalized in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value!r}")


def add_bool(
    parser: argparse.ArgumentParser,
    *flags: str,
    dest: str,
    default: bool = False,
    help: str,
) -> None:
    parser.add_argument(
        *flags,
        dest=dest,
        nargs="?",
        const=True,
        default=default,
        type=parse_bool,
        help=help,
    )


def parse_path_list(value: str) -> list[Path]:
    return [
        Path(item.strip())
        for item in value.replace("\n", ",").split(",")
        if item.strip()
    ]


@dataclass
class Parameters:
    """Runtime parameters for ForestMamba LAZ/LAS inference."""

    dataset_path: Path | None = None
    input_path: Path | None = None
    inputs: list[Path] = field(default_factory=list)
    input_paths: list[Path] = field(default_factory=list)
    input_list: Path | None = None
    output_dir: Path = Path("/output")
    work_dir: Path = Path("forestmamba_batch_work")
    repo_dir: Path = DEFAULT_REPO_DIR
    checkpoint: Path = Path(DEFAULT_CHECKPOINT)
    config_base: Path = Path(DEFAULT_CONFIG_BASE)
    gpu_device: str = ""
    preprocess_workers: int = 1
    chunk_size: int = 1_000_000
    tile_size: float = 0.0
    tile_overlap: float = 16.0
    tile_max_points: int = 0
    keep_tile_work: bool = False
    crop_center_x: float | None = None
    crop_center_y: float | None = None
    crop_radius: float = 0.0
    prediction_dir: Path | None = None
    skip_inference: bool = False
    skip_preprocess: bool = False
    semantic_dim: str = "PredSemantic_FM"
    instance_dim: str = "PredInstance_FM"
    score_dim: str = "PredScore_FM"
    benchmark_stage_log: Path | None = None
    bluepoint_iterations: int = 0
    bluepoint_score_threshold: float | None = None
    bluepoint_second_pass_threshold: float = 0.01
    spatial_match_tolerance: float = 0.01
    show_params: bool = False

    @classmethod
    def from_cli(cls, argv: list[str] | None = None) -> "Parameters":
        namespace = build_parser().parse_args(argv)
        params = cls(
            dataset_path=namespace.dataset_path,
            input_path=namespace.input_path,
            inputs=namespace.inputs,
            input_paths=parse_path_list(namespace.input_paths),
            input_list=namespace.input_list,
            output_dir=namespace.output_dir,
            work_dir=namespace.work_dir,
            repo_dir=namespace.repo_dir,
            checkpoint=namespace.checkpoint,
            config_base=namespace.config_base,
            gpu_device=namespace.gpu_device,
            preprocess_workers=namespace.preprocess_workers,
            chunk_size=namespace.chunk_size,
            tile_size=namespace.tile_size,
            tile_overlap=namespace.tile_overlap,
            tile_max_points=namespace.tile_max_points,
            keep_tile_work=namespace.keep_tile_work,
            crop_center_x=namespace.crop_center_x,
            crop_center_y=namespace.crop_center_y,
            crop_radius=namespace.crop_radius,
            prediction_dir=namespace.prediction_dir,
            skip_inference=namespace.skip_inference,
            skip_preprocess=namespace.skip_preprocess,
            semantic_dim=namespace.semantic_dim,
            instance_dim=namespace.instance_dim,
            score_dim=namespace.score_dim,
            benchmark_stage_log=namespace.benchmark_stage_log,
            bluepoint_iterations=namespace.bluepoint_iterations,
            bluepoint_score_threshold=namespace.bluepoint_score_threshold,
            bluepoint_second_pass_threshold=namespace.bluepoint_second_pass_threshold,
            spatial_match_tolerance=namespace.spatial_match_tolerance,
            show_params=namespace.show_params,
        )
        params.validate()
        return params

    def all_inputs(self) -> list[Path]:
        paths: list[Path] = []
        for candidate in (self.dataset_path, self.input_path):
            if candidate is not None:
                paths.append(candidate)
        paths.extend(self.inputs)
        paths.extend(self.input_paths)
        return unique_paths(paths)

    def validate(self) -> None:
        positive_ints = {
            "preprocess_workers": self.preprocess_workers,
            "chunk_size": self.chunk_size,
        }
        for name, value in positive_ints.items():
            if value <= 0:
                raise ValueError(f"{name} must be positive")

        non_negative_numbers = {
            "tile_size": self.tile_size,
            "tile_overlap": self.tile_overlap,
            "tile_max_points": self.tile_max_points,
            "crop_radius": self.crop_radius,
            "bluepoint_iterations": self.bluepoint_iterations,
            "spatial_match_tolerance": self.spatial_match_tolerance,
        }
        for name, value in non_negative_numbers.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        if not 0 <= self.bluepoint_second_pass_threshold <= 1:
            raise ValueError("bluepoint_second_pass_threshold must be between 0 and 1")
        if (
            self.bluepoint_score_threshold is not None
            and not 0 <= self.bluepoint_score_threshold <= 1
        ):
            raise ValueError("bluepoint_score_threshold must be between 0 and 1")


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        key = path.expanduser()
        if key in seen:
            continue
        seen.add(key)
        result.append(path)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Galaxy-compatible ForestMamba LAS/LAZ wrapper."
    )
    parser.add_argument(
        "--dataset-path",
        "--dataset_path",
        dest="dataset_path",
        type=Path,
        help="Galaxy-style single input LAS/LAZ path.",
    )
    parser.add_argument(
        "--input-path",
        "--input_path",
        dest="input_path",
        type=Path,
        help="Alias for a single input LAS/LAZ path.",
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        type=Path,
        default=[],
        help="Input LAS/LAZ file. May be provided multiple times.",
    )
    parser.add_argument(
        "--input-paths",
        "--input_paths",
        dest="input_paths",
        default="",
        help="Comma- or newline-separated input LAS/LAZ paths.",
    )
    parser.add_argument(
        "--input-list",
        "--input_list",
        dest="input_list",
        type=Path,
        help="Text or JSON file containing LAS/LAZ paths to process.",
    )
    parser.add_argument(
        "--output-dir",
        "--output_dir",
        dest="output_dir",
        type=Path,
        default=Path("/output"),
        help="Output directory for enriched LAZ files.",
    )
    parser.add_argument(
        "--work-dir",
        "--work_dir",
        dest="work_dir",
        type=Path,
        default=Path("forestmamba_batch_work"),
        help="Scratch directory for staged datasets and model outputs.",
    )
    parser.add_argument(
        "--repo-dir",
        "--repo_dir",
        dest="repo_dir",
        type=Path,
        default=DEFAULT_REPO_DIR,
        help="ForestMamba repository root inside the container.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(DEFAULT_CHECKPOINT),
        help="ForestMamba checkpoint path.",
    )
    parser.add_argument(
        "--config-base",
        "--config_base",
        dest="config_base",
        type=Path,
        default=Path(DEFAULT_CONFIG_BASE),
        help="Base ForestMamba config path.",
    )
    parser.add_argument(
        "--gpu-device",
        "--gpu_device",
        dest="gpu_device",
        default="",
        help="CUDA device id for inference. Leave empty to let the runtime assign the GPU.",
    )
    parser.add_argument(
        "--preprocess-workers",
        "--preprocess_workers",
        dest="preprocess_workers",
        type=int,
        default=1,
        help="Workers for ForestMamba preprocessing.",
    )
    parser.add_argument(
        "--chunk-size",
        "--chunk_size",
        dest="chunk_size",
        type=int,
        default=1_000_000,
        help="Point chunk size for staged reads and streamed LAZ writes.",
    )
    parser.add_argument(
        "--tile-size",
        "--tile_size",
        dest="tile_size",
        type=float,
        default=0.0,
        help="Enable XY tiled inference with this core tile size in input units.",
    )
    parser.add_argument(
        "--tile-overlap",
        "--tile_overlap",
        dest="tile_overlap",
        type=float,
        default=16.0,
        help="XY overlap added around each tile for model context.",
    )
    parser.add_argument(
        "--tile-max-points",
        "--tile_max_points",
        dest="tile_max_points",
        type=int,
        default=0,
        help="Fail a tile above this point count. Use 0 to disable.",
    )
    add_bool(
        parser,
        "--keep-tile-work",
        "--keep_tile_work",
        dest="keep_tile_work",
        help="Keep per-tile scratch directories after tiled inference.",
    )
    parser.add_argument(
        "--crop-center-x",
        "--crop_center_x",
        dest="crop_center_x",
        type=float,
        help="Optional XY crop center X for interactive correction inference.",
    )
    parser.add_argument(
        "--crop-center-y",
        "--crop_center_y",
        dest="crop_center_y",
        type=float,
        help="Optional XY crop center Y for interactive correction inference.",
    )
    parser.add_argument(
        "--crop-radius",
        "--crop_radius",
        dest="crop_radius",
        type=float,
        default=0.0,
        help="Optional XY cylinder crop radius before inference.",
    )
    parser.add_argument(
        "--prediction-dir",
        "--prediction_dir",
        dest="prediction_dir",
        type=Path,
        help="Use existing ForestMamba prediction PLY files from this directory.",
    )
    add_bool(
        parser,
        "--skip-inference",
        "--skip_inference",
        dest="skip_inference",
        help="Skip model execution and require existing predictions.",
    )
    add_bool(
        parser,
        "--skip-preprocess",
        "--skip_preprocess",
        dest="skip_preprocess",
        help="Skip ForestMamba data conversion when predictions already exist.",
    )
    parser.add_argument(
        "--semantic-dim",
        "--semantic_dim",
        dest="semantic_dim",
        default="PredSemantic_FM",
        help="Extra dimension name for predicted semantic labels.",
    )
    parser.add_argument(
        "--instance-dim",
        "--instance_dim",
        dest="instance_dim",
        default="PredInstance_FM",
        help="Extra dimension name for predicted instance IDs.",
    )
    parser.add_argument(
        "--score-dim",
        "--score_dim",
        dest="score_dim",
        default="PredScore_FM",
        help="Extra dimension name for ForestMamba confidence scores.",
    )
    parser.add_argument(
        "--benchmark-stage-log",
        "--benchmark_stage_log",
        dest="benchmark_stage_log",
        type=Path,
        help="Optional JSONL path for benchmark-only stage timing.",
    )
    parser.add_argument(
        "--bluepoint-iterations",
        "--bluepoint_iterations",
        dest="bluepoint_iterations",
        type=int,
        default=0,
        help="Run iterative bluepoint inference this many rounds.",
    )
    parser.add_argument(
        "--bluepoint-score-threshold",
        "--bluepoint_score_threshold",
        dest="bluepoint_score_threshold",
        type=float,
        default=None,
        help="Optional model score_th override for bluepoint rounds.",
    )
    parser.add_argument(
        "--bluepoint-second-pass-threshold",
        "--bluepoint_second_pass_threshold",
        dest="bluepoint_second_pass_threshold",
        type=float,
        default=0.01,
        help="Second pass threshold as a fraction. Default 0.01 = 1 percent.",
    )
    parser.add_argument(
        "--spatial-match-tolerance",
        "--spatial_match_tolerance",
        dest="spatial_match_tolerance",
        type=float,
        default=0.01,
        help="Maximum coordinate distance for bluepoint prediction row matching.",
    )
    add_bool(
        parser,
        "--show-params",
        "--show_params",
        dest="show_params",
        help="Print parsed parameters and exit.",
    )
    return parser


def print_params(params: Parameters) -> None:
    """Print current parameter configuration."""
    for key, value in sorted(params.__dict__.items()):
        print(f"{key}: {value}")
