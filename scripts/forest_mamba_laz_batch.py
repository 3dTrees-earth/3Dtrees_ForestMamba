#!/usr/bin/env python3
"""Run ForestMamba on one or more LAS/LAZ files and enrich the originals."""

from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
from datetime import datetime, timezone
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import laspy
import numpy as np
from plyfile import PlyData, PlyElement
from laspy.vlrs.vlrlist import VLRList
from scipy.spatial import cKDTree


DEFAULT_CHECKPOINT = (
    "/workspace/work_dirs/forestmamba_chm_radius16_qp300_2many_v6/"
    "v6_epoch_1500_fix.pth"
)
DEFAULT_CONFIG_BASE = (
    "/workspace/configs/ForAINetv2/"
    "forestmamba_chm_radius16_qp300_2many_v6.py"
)
PLACEHOLDER_PLY_DTYPE = np.dtype(
    [
        ("x", "<f8"),
        ("y", "<f8"),
        ("z", "<f8"),
        ("semantic_seg", "<i4"),
        ("treeID", "<i4"),
    ]
)
LAZ_BACKEND = laspy.LazBackend.Lazrs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Batch ForestMamba inference for LAS/LAZ point clouds."
    )
    parser.add_argument(
        "--input",
        dest="inputs",
        action="append",
        default=[],
        help="Input LAS/LAZ file. May be provided multiple times.",
    )
    parser.add_argument(
        "--input-list",
        help="Text or JSON file containing LAS/LAZ paths to process.",
    )
    parser.add_argument("--output-dir", required=True, help="Output directory.")
    parser.add_argument(
        "--work-dir",
        default="forestmamba_batch_work",
        help="Scratch directory for staged datasets and model outputs.",
    )
    parser.add_argument(
        "--repo-dir",
        default=Path(__file__).resolve().parents[1],
        help="ForestMamba repository root inside the container.",
    )
    parser.add_argument(
        "--checkpoint",
        default=DEFAULT_CHECKPOINT,
        help="ForestMamba checkpoint path.",
    )
    parser.add_argument(
        "--config-base",
        default=DEFAULT_CONFIG_BASE,
        help="Base ForestMamba config path.",
    )
    parser.add_argument(
        "--gpu-device",
        default="",
        help="CUDA device id for inference. Leave empty to let the runtime assign the GPU.",
    )
    parser.add_argument(
        "--preprocess-workers",
        type=int,
        default=1,
        help="Workers for ForestMamba preprocessing.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1_000_000,
        help="Point chunk size for tiled reads, streamed staging, and LAZ writes.",
    )
    parser.add_argument(
        "--tile-size",
        type=float,
        default=0.0,
        help=(
            "Enable memory-bounded XY tiled inference with this core tile size "
            "in input units. The default 0 keeps whole-file inference."
        ),
    )
    parser.add_argument(
        "--tile-overlap",
        type=float,
        default=16.0,
        help="XY overlap added around each tile for model context.",
    )
    parser.add_argument(
        "--tile-max-points",
        type=int,
        default=0,
        help=(
            "Fail a tile above this point count. Use 0 to disable. This is a "
            "guardrail for dense clouds; reduce --tile-size if it trips."
        ),
    )
    parser.add_argument(
        "--keep-tile-work",
        action="store_true",
        help="Keep per-tile scratch directories after tiled inference.",
    )
    parser.add_argument(
        "--crop-center-x",
        type=float,
        help="Optional XY crop center X for interactive correction inference.",
    )
    parser.add_argument(
        "--crop-center-y",
        type=float,
        help="Optional XY crop center Y for interactive correction inference.",
    )
    parser.add_argument(
        "--crop-radius",
        type=float,
        default=0.0,
        help=(
            "Optional XY cylinder crop radius before inference. Z is not "
            "filtered; all points above/below the seed within the XY radius are kept."
        ),
    )
    parser.add_argument(
        "--prediction-dir",
        help="Use existing ForestMamba prediction PLY files from this directory.",
    )
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip model execution and require existing predictions.",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Debug option: skip ForestMamba data conversion when predictions already exist.",
    )
    parser.add_argument(
        "--semantic-dim",
        default="PredSemantic_FM",
        help="Extra dimension name for predicted semantic labels.",
    )
    parser.add_argument(
        "--instance-dim",
        default="PredInstance_FM",
        help="Extra dimension name for predicted instance IDs.",
    )
    parser.add_argument(
        "--score-dim",
        default="PredScore_FM",
        help="Extra dimension name for ForestMamba confidence scores.",
    )
    parser.add_argument(
        "--benchmark-stage-log",
        help=(
            "Optional JSONL path for benchmark-only stage timing. "
            "When omitted, no benchmark timing file is written."
        ),
    )
    parser.add_argument(
        "--bluepoint-iterations",
        type=int,
        default=0,
        help=(
            "Run iterative bluepoint inference this many rounds. "
            "0 keeps the standard one-pass inference path."
        ),
    )
    parser.add_argument(
        "--bluepoint-score-threshold",
        type=float,
        default=None,
        help="Optional model score_th override for bluepoint rounds.",
    )
    parser.add_argument(
        "--bluepoint-second-pass-threshold",
        type=float,
        default=0.01,
        help=(
            "Run the second bluepoint pass only when more than this fraction "
            "of first-pass points are non-ground with raw instance_pred == -1. "
            "Default 0.01 = 1%%."
        ),
    )
    parser.add_argument(
        "--spatial-match-tolerance",
        type=float,
        default=0.01,
        help=(
            "Maximum coordinate distance allowed when matching bluepoint "
            "PLY predictions back onto the original LAS/LAZ rows."
        ),
    )
    return parser.parse_args(argv)


def safe_stem(path: Path) -> str:
    stem = path.stem
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-")
    return stem or "pointcloud"


def crop_las_xy_radius(
    input_path: Path,
    output_path: Path,
    *,
    center_x: float,
    center_y: float,
    radius: float,
    chunk_size: int,
) -> int:
    if radius <= 0:
        raise ValueError("--crop-radius must be greater than 0 when crop center is set")

    print(
        f"[crop] {input_path}: center=({center_x:.3f}, {center_y:.3f}) "
        f"radius={radius:.3f}",
        flush=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    radius_sq = float(radius) * float(radius)
    kept = 0
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        header = copy.deepcopy(reader.header)
        strip_unsupported_output_vlrs(header)
        with laspy.open(
            str(output_path),
            mode="w",
            header=header,
            do_compress=True,
            laz_backend=LAZ_BACKEND,
        ) as writer:
            for points in reader.chunk_iterator(chunk_size):
                dx = np.asarray(points.x) - center_x
                dy = np.asarray(points.y) - center_y
                mask = (dx * dx + dy * dy) <= radius_sq
                if not np.any(mask):
                    continue
                cropped = points[mask]
                writer.write_points(cropped)
                kept += len(cropped)

    if kept == 0:
        raise ValueError(
            f"Crop around ({center_x}, {center_y}) with radius {radius} kept 0 points"
        )
    print(f"[crop] Wrote {kept:,} points to {output_path}", flush=True)
    return kept


def read_input_list(path: Path) -> list[Path]:
    text = path.read_text().strip()
    if not text:
        return []
    if text[0] in "[{":
        payload = json.loads(text)
        if isinstance(payload, dict):
            payload = payload.get("inputs", [])
        return [Path(str(item)) for item in payload]
    return [
        Path(line.strip())
        for line in text.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def collect_inputs(args: argparse.Namespace) -> list[Path]:
    inputs = [Path(item) for item in args.inputs]
    if args.input_list:
        inputs.extend(read_input_list(Path(args.input_list)))
    seen = set()
    unique: list[Path] = []
    for path in inputs:
        resolved = path.resolve()
        if resolved in seen:
            continue
        if path.suffix.lower() not in {".las", ".laz"}:
            raise ValueError(f"Unsupported point cloud extension: {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        seen.add(resolved)
        unique.append(path)
    if not unique:
        raise ValueError("No input LAS/LAZ files were provided")
    return unique


def validate_dim_name(name: str) -> None:
    if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]{0,31}", name):
        raise ValueError(
            f"Invalid LAS extra dimension name {name!r}; use 1-32 letters, "
            "numbers, and underscores, starting with a letter or underscore."
        )


def run(cmd: list[str], *, cwd: Path, env: dict[str, str]) -> None:
    print("+ " + " ".join(str(part) for part in cmd), flush=True)
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def append_benchmark_stage(log_path: Path | None, payload: dict[str, object]) -> None:
    if log_path is None:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")



@contextmanager
def benchmark_stage(
    log_path: Path | None,
    *,
    model: str,
    input_path: Path,
    sample_id: str,
    stage: str,
    metadata: dict[str, object] | None = None,
):
    if log_path is None:
        yield
        return

    started_at = utc_now()
    started = time.monotonic()
    status = "success"
    error = ""
    try:
        yield
    except Exception as exc:
        status = "failed"
        error = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        ended_at = utc_now()
        payload: dict[str, object] = {
            "model": model,
            "input_path": str(input_path),
            "sample_id": sample_id,
            "stage": stage,
            "started_at": started_at,
            "ended_at": ended_at,
            "duration_seconds": round(time.monotonic() - started, 6),
            "status": status,
        }
        if error:
            payload["error"] = error
        if metadata:
            payload["metadata"] = metadata
        append_benchmark_stage(log_path, payload)


def write_placeholder_ply(input_path: Path, ply_path: Path, chunk_size: int) -> int:
    print(f"[stage] Streaming {input_path}", flush=True)
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        n_points = int(reader.header.point_count)
        with ply_path.open("wb") as ply_file:
            write_binary_placeholder_ply_header(ply_file, n_points)
            written = 0
            for points in reader.chunk_iterator(chunk_size):
                chunk_len = len(points)
                vertex = np.empty(chunk_len, dtype=PLACEHOLDER_PLY_DTYPE)
                vertex["x"] = np.asarray(points.x)
                vertex["y"] = np.asarray(points.y)
                vertex["z"] = np.asarray(points.z)
                vertex["semantic_seg"] = 2
                vertex["treeID"] = 1
                vertex.tofile(ply_file)
                written += chunk_len
    if written != n_points:
        raise RuntimeError(
            f"Streamed {written:,} points from {input_path}, expected {n_points:,}"
        )
    print(f"[stage] Wrote {ply_path} with {n_points:,} points", flush=True)
    return n_points


def write_placeholder_ply_arrays(points_xyz: np.ndarray, ply_path: Path) -> int:
    n_points = len(points_xyz)
    vertex = np.empty(
        n_points,
        dtype=PLACEHOLDER_PLY_DTYPE,
    )
    vertex["x"] = points_xyz[:, 0]
    vertex["y"] = points_xyz[:, 1]
    vertex["z"] = points_xyz[:, 2]
    vertex["semantic_seg"] = 2
    vertex["treeID"] = 1
    ply_path.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(vertex, "vertex")], text=False).write(str(ply_path))
    return n_points


def stage_forainet_dataset(
    input_path: Path,
    dataset_root: Path,
    sample_id: str,
    chunk_size: int,
) -> int:
    test_data = dataset_root / "test_data"
    meta_data = dataset_root / "meta_data"
    (dataset_root / "train_val_data").mkdir(parents=True, exist_ok=True)
    test_data.mkdir(parents=True, exist_ok=True)
    meta_data.mkdir(parents=True, exist_ok=True)
    (meta_data / "train_list.txt").write_text("")
    (meta_data / "val_list.txt").write_text("")
    (meta_data / "test_list.txt").write_text(f"{sample_id}\n")
    return write_placeholder_ply(input_path, test_data / f"{sample_id}.ply", chunk_size)


def write_runtime_config(
    config_path: Path,
    config_base: Path,
    dataset_root: Path,
    *,
    score_threshold: float | None = None,
) -> None:
    lines = [
        f"_base_ = {str(config_base)!r}",
        "",
        f"data_root_forainetv2 = {str(dataset_root) + '/'!r}",
        "",
        "train_dataloader = dict(",
        "    num_workers=0,",
        "    persistent_workers=False,",
        "    dataset=dict(data_root=data_root_forainetv2))",
        "",
        "val_dataloader = dict(",
        "    num_workers=0,",
        "    persistent_workers=False,",
        "    dataset=dict(data_root=data_root_forainetv2))",
        "",
        "test_dataloader = dict(",
        "    num_workers=0,",
        "    persistent_workers=False,",
        "    sampler=dict(type='DefaultSampler', shuffle=False, round_up=False),",
        "    dataset=dict(data_root=data_root_forainetv2))",
        "",
    ]
    if score_threshold is not None:
        lines.extend(
            [
                f"score_th = {float(score_threshold)!r}",
                "model = dict(score_th=score_th)",
                "",
            ]
        )
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text("\n".join(lines))


def preprocess_dataset(
    repo_dir: Path,
    dataset_root: Path,
    workers: int,
    env: dict[str, str],
) -> None:
    instance_dir = dataset_root / "forainetv2_instance_data"
    run(
        [
            sys.executable,
            str(repo_dir / "data/ForAINetV2/batch_load_ForAINetV2_data.py"),
            "--output_folder",
            str(instance_dir),
            "--train_forainetv2_dir",
            str(dataset_root / "train_val_data"),
            "--test_forainetv2_dir",
            str(dataset_root / "test_data"),
            "--train_scan_names_file",
            str(dataset_root / "meta_data/train_list.txt"),
            "--val_scan_names_file",
            str(dataset_root / "meta_data/val_list.txt"),
            "--test_scan_names_file",
            str(dataset_root / "meta_data/test_list.txt"),
            "--num_workers",
            str(workers),
        ],
        cwd=repo_dir,
        env=env,
    )
    run(
        [
            sys.executable,
            str(repo_dir / "tools/create_data_forainetv2.py"),
            "forainetv2",
            "--root-path",
            str(dataset_root),
            "--out-dir",
            str(dataset_root),
            "--workers",
            str(workers),
        ],
        cwd=repo_dir,
        env=env,
    )


def checkpoint_epoch(checkpoint: Path) -> str:
    match = re.search(r"epoch[_-](\d+)", checkpoint.name)
    return f"_ep{match.group(1)}" if match else ""


def run_inference(
    repo_dir: Path,
    run_dir: Path,
    config_path: Path,
    checkpoint: Path,
    env: dict[str, str],
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)
    run(
        [
            sys.executable,
            str(repo_dir / "tools/test.py"),
            str(config_path),
            str(checkpoint),
        ],
        cwd=run_dir,
        env=env,
    )
    return run_dir / "work_dirs/inference" / f"{config_path.stem}{checkpoint_epoch(checkpoint)}"


def find_prediction(prediction_dir: Path, sample_id: str, input_path: Path) -> Path:
    candidates = [
        prediction_dir / f"{sample_id}.ply",
        prediction_dir / f"{input_path.stem}.ply",
        prediction_dir / f"test_{input_path.stem}.ply",
        prediction_dir / f"test_{input_path.stem}_3dtrees.ply",
        prediction_dir / f"{sample_id}_final_results.ply",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    all_ply = sorted(prediction_dir.glob("*.ply"))
    if len(all_ply) == 1:
        return all_ply[0]
    raise FileNotFoundError(
        f"No prediction PLY for sample {sample_id!r} in {prediction_dir}"
    )


def add_or_overwrite_dim(las: laspy.LasData, name: str, dtype: str, description: str) -> None:
    if name in las.point_format.dimension_names:
        return
    las.add_extra_dim(
        laspy.ExtraBytesParams(name=name, type=np.dtype(dtype), description=description)
    )


def strip_unsupported_output_vlrs(header: laspy.LasHeader) -> None:
    """Remove VLRs that laspy can read but cannot write to a plain LAZ."""
    unsupported_vlrs = (laspy.copc.CopcInfoVlr, laspy.copc.CopcHierarchyVlr)
    header.vlrs = VLRList(
        [vlr for vlr in header.vlrs if not isinstance(vlr, unsupported_vlrs)]
    )
    if header.evlrs is not None:
        header.evlrs = VLRList(
            [
                vlr
                for vlr in header.evlrs
                if not isinstance(vlr, unsupported_vlrs)
            ]
        )


def normalize_instance_ids_for_laz(instance_pred: np.ndarray) -> np.ndarray:
    """Use 0 for unassigned points while preserving valid ForestMamba IDs."""
    instance_pred = np.asarray(instance_pred, dtype=np.int32)
    return np.where(instance_pred < 0, 0, instance_pred).astype(np.int32)


def enrich_laz(
    input_path: Path,
    prediction_ply: Path,
    output_path: Path,
    semantic_dim: str,
    instance_dim: str,
    score_dim: str,
    chunk_size: int,
) -> dict[str, object]:
    print(f"[post] Reading predictions {prediction_ply}", flush=True)
    semantic_pred, instance_pred, score_pred = read_prediction_fields(prediction_ply)

    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        n_points = int(reader.header.point_count)
    if n_points != len(semantic_pred):
        raise ValueError(
            f"Point count mismatch for {input_path}: LAZ has {n_points}, "
            f"prediction has {len(semantic_pred)}"
        )

    write_enriched_laz_stream(
        input_path,
        output_path,
        semantic_dim,
        instance_dim,
        score_dim,
        semantic_pred,
        instance_pred,
        score_pred,
        chunk_size,
    )
    normalized_instance = normalize_instance_ids_for_laz(instance_pred)
    positive_instances = np.unique(normalized_instance[normalized_instance > 0])
    return {
        "input": str(input_path),
        "prediction_ply": str(prediction_ply),
        "output": str(output_path),
        "point_count": int(n_points),
        "positive_instance_count": int(len(positive_instances)),
        "semantic_dim": semantic_dim,
        "instance_dim": instance_dim,
        "score_dim": score_dim,
        "instance_id_normalization": "raw_negative_to_0_valid_preserved",
        "write_mode": "streamed",
    }


def read_prediction_fields(prediction_ply: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    print(f"[post] Reading predictions {prediction_ply}", flush=True)
    vertex = PlyData.read(str(prediction_ply))["vertex"].data
    required = {"semantic_pred", "instance_pred", "score"}
    missing = required.difference(vertex.dtype.names or ())
    if missing:
        raise ValueError(f"Prediction PLY is missing fields: {sorted(missing)}")
    return (
        np.asarray(vertex["semantic_pred"], dtype=np.int32),
        np.asarray(vertex["instance_pred"], dtype=np.int32),
        np.asarray(vertex["score"], dtype=np.float32),
    )


def load_prediction_ply(
    prediction_ply: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    vertex = PlyData.read(str(prediction_ply))["vertex"].data
    names = set(vertex.dtype.names or ())
    points = np.column_stack(
        [
            np.asarray(vertex["x"], dtype=np.float64),
            np.asarray(vertex["y"], dtype=np.float64),
            np.asarray(vertex["z"], dtype=np.float64),
        ]
    )
    if {"semantic_pred", "instance_pred"}.issubset(names):
        semantic = np.asarray(vertex["semantic_pred"], dtype=np.int32)
        instance = np.asarray(vertex["instance_pred"], dtype=np.int32)
        score_field = "score" if "score" in names else "instance_score"
        score = (
            np.asarray(vertex[score_field], dtype=np.float32)
            if score_field in names
            else np.zeros(len(vertex), dtype=np.float32)
        )
        return points, semantic, instance, score
    if {"semantic_pred", "semantic_seg", "treeID"}.issubset(names):
        semantic = np.asarray(vertex["semantic_pred"], dtype=np.int32)
        instance = np.full(len(vertex), -1, dtype=np.int32)
        score = np.zeros(len(vertex), dtype=np.float32)
        return points, semantic, instance, score
    raise ValueError(f"Unsupported bluepoint PLY fields in {prediction_ply}: {sorted(names)}")


def first_pass_unassigned_ratio(prediction_dir: Path, sample_id: str) -> dict[str, object]:
    """Estimate first-pass non-ground points with raw instance_pred == -1."""
    total_points = 0
    unassigned_points = 0
    parts: list[dict[str, object]] = []

    base_path = prediction_dir / f"{sample_id}_1.ply"
    if base_path.exists():
        _, base_semantic, base_instance, _ = load_prediction_ply(base_path)
        base_unassigned = int(
            np.count_nonzero((base_semantic != 0) & (base_instance < 0))
        )
        total_points = int(len(base_instance))
        unassigned_points = base_unassigned
        parts.append({
            "path": str(base_path),
            "point_count": int(len(base_instance)),
            "unassigned_point_count": base_unassigned,
        })
    else:
        bluepoint_path = prediction_dir / f"{sample_id}_bluepoints_1.ply"
        if bluepoint_path.exists():
            _, _, blue_instance, _ = load_prediction_ply(bluepoint_path)
            blue_unassigned = int(len(blue_instance))
            total_points = int(len(blue_instance))
            unassigned_points = blue_unassigned
            parts.append({
                "path": str(bluepoint_path),
                "point_count": int(len(blue_instance)),
                "unassigned_point_count": blue_unassigned,
            })

    if total_points == 0:
        raise FileNotFoundError(
            f"No first-pass bluepoint outputs found for {sample_id!r} in {prediction_dir}"
        )

    return {
        "unassigned_ratio": float(unassigned_points / total_points),
        "unassigned_point_count": unassigned_points,
        "point_count": total_points,
        "parts": parts,
    }


def locate_prediction_rows(
    input_path: Path,
    points: np.ndarray,
    *,
    tolerance: float,
    chunk_size: int,
) -> np.ndarray:
    """Map prediction coordinates back to original LAS/LAZ row indices."""
    if len(points) == 0:
        return np.empty(0, dtype=np.int64)
    tree = cKDTree(points)
    row_indices = np.full(len(points), -1, dtype=np.int64)
    best_distances = np.full(len(points), np.inf, dtype=np.float64)
    start = 0
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        for chunk in reader.chunk_iterator(chunk_size):
            chunk_points = np.column_stack(
                [
                    np.asarray(chunk.x, dtype=np.float64),
                    np.asarray(chunk.y, dtype=np.float64),
                    np.asarray(chunk.z, dtype=np.float64),
                ]
            )
            if len(chunk_points):
                distances, indices = tree.query(chunk_points, k=1)
                matches = np.flatnonzero(distances <= tolerance)
                if len(matches):
                    matched_overlay = indices[matches].astype(np.int64, copy=False)
                    candidate_rows = start + matches
                    candidate_distances = distances[matches]
                    order = np.lexsort((candidate_distances, matched_overlay))
                    sorted_overlay = matched_overlay[order]
                    first_for_overlay = np.r_[
                        True,
                        sorted_overlay[1:] != sorted_overlay[:-1],
                    ]
                    best_overlay = sorted_overlay[first_for_overlay]
                    best_rows = candidate_rows[order][first_for_overlay]
                    best_chunk_distances = candidate_distances[order][first_for_overlay]
                    improves = best_chunk_distances < best_distances[best_overlay]
                    if np.any(improves):
                        improved_overlay = best_overlay[improves]
                        row_indices[improved_overlay] = best_rows[improves]
                        best_distances[improved_overlay] = best_chunk_distances[improves]
            start += len(chunk)
    missing = np.flatnonzero(row_indices < 0)
    if len(missing):
        raise ValueError(
            f"Could not match {len(missing):,} bluepoint prediction rows within "
            f"tolerance {tolerance}"
        )
    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("Bluepoint predictions matched duplicate original LAS/LAZ rows")
    return row_indices


def overlay_bluepoint_prediction(
    input_path: Path,
    base_prediction: Path,
    overlay_prediction: Path | None,
    output_path: Path,
    semantic_dim: str,
    instance_dim: str,
    score_dim: str,
    chunk_size: int,
    tolerance: float,
) -> dict[str, object]:
    base_points, semantic_pred, instance_pred, score_pred = load_prediction_ply(
        base_prediction
    )
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        n_points = int(reader.header.point_count)
    if len(base_points) != n_points:
        raise ValueError(
            f"Base bluepoint prediction {base_prediction} has {len(base_points):,} "
            f"points but original has {n_points:,}. Regenerate predictions with the "
            "updated bluepoint writer so the first-pass PLY contains all points."
        )

    overlay_summary: dict[str, object] | None = None
    if overlay_prediction is not None:
        overlay_points, overlay_semantic, overlay_instance, overlay_score = load_prediction_ply(
            overlay_prediction
        )
        row_indices = locate_prediction_rows(
            input_path,
            overlay_points,
            tolerance=tolerance,
            chunk_size=chunk_size,
        )
        if len(row_indices) != len(overlay_points):
            raise ValueError(
                f"Overlay prediction {overlay_prediction} row mapping returned "
                f"{len(row_indices):,} rows for {len(overlay_points):,} points"
            )
        semantic_pred = semantic_pred.copy()
        instance_pred = instance_pred.copy()
        score_pred = score_pred.copy()
        semantic_pred[row_indices] = overlay_semantic
        instance_pred[row_indices] = overlay_instance
        score_pred[row_indices] = overlay_score
        overlay_summary = {
            "prediction_ply": str(overlay_prediction),
            "point_count": int(len(overlay_points)),
        }

    write_enriched_laz_stream(
        input_path,
        output_path,
        semantic_dim,
        instance_dim,
        score_dim,
        semantic_pred,
        instance_pred,
        score_pred,
        chunk_size,
    )
    normalized_instance = normalize_instance_ids_for_laz(instance_pred)
    positive_instances = np.unique(normalized_instance[normalized_instance > 0])
    return {
        "input": str(input_path),
        "prediction_ply": str(base_prediction),
        "overlay_prediction": overlay_summary,
        "output": str(output_path),
        "point_count": int(n_points),
        "positive_instance_count": int(len(positive_instances)),
        "semantic_dim": semantic_dim,
        "instance_dim": instance_dim,
        "score_dim": score_dim,
        "instance_id_normalization": "raw_negative_to_0_valid_preserved",
        "write_mode": "streamed",
    }

def add_or_overwrite_header_dim(
    header: laspy.LasHeader, name: str, dtype: str, description: str
) -> None:
    if name in header.point_format.dimension_names:
        return
    header.add_extra_dim(
        laspy.ExtraBytesParams(name=name, type=np.dtype(dtype), description=description)
    )


def write_enriched_laz_stream(
    input_path: Path,
    output_path: Path,
    semantic_dim: str,
    instance_dim: str,
    score_dim: str,
    semantic_pred: np.ndarray,
    instance_pred: np.ndarray,
    score_pred: np.ndarray,
    chunk_size: int,
) -> None:
    print(f"[post] Streaming enriched LAS/LAZ to {output_path}", flush=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        header = copy.deepcopy(reader.header)
        strip_unsupported_output_vlrs(header)
        add_or_overwrite_header_dim(
            header, semantic_dim, "int32", "ForestMamba semantic label"
        )
        add_or_overwrite_header_dim(
            header, instance_dim, "int32", "ForestMamba instance id"
        )
        add_or_overwrite_header_dim(
            header, score_dim, "float32", "ForestMamba confidence score"
        )
        with laspy.open(
            str(output_path),
            mode="w",
            header=header,
            do_compress=True,
            laz_backend=LAZ_BACKEND,
        ) as writer:
            start = 0
            for points in reader.chunk_iterator(chunk_size):
                end = start + len(points)
                out_points = laspy.ScaleAwarePointRecord.zeros(
                    len(points), header=header
                )
                out_points.copy_fields_from(points)
                out_points[semantic_dim] = semantic_pred[start:end]
                out_points[instance_dim] = normalize_instance_ids_for_laz(
                    instance_pred[start:end]
                )
                out_points[score_dim] = score_pred[start:end]
                writer.write_points(out_points)
                start = end
    print(f"[post] Wrote {output_path}", flush=True)


def tile_windows(
    input_path: Path, tile_size: float
) -> tuple[list[tuple[int, int, float, float, float, float, bool, bool]], int]:
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        x_min, y_min = float(reader.header.mins[0]), float(reader.header.mins[1])
        x_max, y_max = float(reader.header.maxs[0]), float(reader.header.maxs[1])
        point_count = int(reader.header.point_count)

    if tile_size <= 0:
        raise ValueError("--tile-size must be greater than 0 for tiled inference")

    nx = max(1, int(np.ceil((x_max - x_min) / tile_size)))
    ny = max(1, int(np.ceil((y_max - y_min) / tile_size)))
    windows = []
    for ix in range(nx):
        core_x0 = x_min + ix * tile_size
        core_x1 = x_max if ix == nx - 1 else core_x0 + tile_size
        for iy in range(ny):
            core_y0 = y_min + iy * tile_size
            core_y1 = y_max if iy == ny - 1 else core_y0 + tile_size
            windows.append((ix, iy, core_x0, core_x1, core_y0, core_y1, ix == nx - 1, iy == ny - 1))
    return windows, point_count


def tile_include_and_core_masks(
    x: np.ndarray,
    y: np.ndarray,
    window: tuple[int, int, float, float, float, float, bool, bool],
    overlap: float,
) -> tuple[np.ndarray, np.ndarray]:
    _, _, core_x0, core_x1, core_y0, core_y1, last_x, last_y = window
    tile_x0 = core_x0 - overlap
    tile_x1 = core_x1 + overlap
    tile_y0 = core_y0 - overlap
    tile_y1 = core_y1 + overlap
    include = (x >= tile_x0) & (x <= tile_x1) & (y >= tile_y0) & (y <= tile_y1)
    core_x = (x >= core_x0) & ((x <= core_x1) if last_x else (x < core_x1))
    core_y = (y >= core_y0) & ((y <= core_y1) if last_y else (y < core_y1))
    return include, core_x & core_y


def write_binary_placeholder_ply_header(handle, point_count: int) -> None:
    header = "\n".join(
        [
            "ply",
            "format binary_little_endian 1.0",
            f"element vertex {point_count}",
            "property double x",
            "property double y",
            "property double z",
            "property int semantic_seg",
            "property int treeID",
            "end_header",
            "",
        ]
    )
    handle.write(header.encode("ascii"))


def count_tile_points(
    input_path: Path,
    window: tuple[int, int, float, float, float, float, bool, bool],
    overlap: float,
    chunk_size: int,
) -> tuple[int, int]:
    point_count = 0
    core_count = 0
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        for points in reader.chunk_iterator(chunk_size):
            x = np.asarray(points.x)
            y = np.asarray(points.y)
            include, core = tile_include_and_core_masks(x, y, window, overlap)
            if np.any(include):
                point_count += int(np.count_nonzero(include))
                core_count += int(np.count_nonzero(core[include]))
    return point_count, core_count


def stage_tile_from_las(
    input_path: Path,
    window: tuple[int, int, float, float, float, float, bool, bool],
    *,
    overlap: float,
    chunk_size: int,
    ply_path: Path,
    original_indices_path: Path,
    core_mask_path: Path,
    max_points: int,
) -> tuple[int, int, np.memmap, np.memmap]:
    tile_point_count, core_point_count = count_tile_points(
        input_path, window, overlap, chunk_size
    )
    if tile_point_count == 0:
        return (
            0,
            0,
            np.empty((0,), dtype=np.int64),
            np.empty((0,), dtype=bool),
        )
    if max_points > 0 and tile_point_count > max_points:
        ix, iy, *_ = window
        raise MemoryError(
            f"Tile ({ix}, {iy}) has {tile_point_count:,} points, above "
            f"--tile-max-points={max_points:,}. Reduce --tile-size."
        )

    ply_path.parent.mkdir(parents=True, exist_ok=True)
    original_indices_path.parent.mkdir(parents=True, exist_ok=True)
    core_mask_path.parent.mkdir(parents=True, exist_ok=True)
    original_indices = np.memmap(
        original_indices_path, dtype=np.int64, mode="w+", shape=(tile_point_count,)
    )
    core_mask = np.memmap(
        core_mask_path, dtype=bool, mode="w+", shape=(tile_point_count,)
    )

    written = 0
    start = 0
    with laspy.open(str(input_path), laz_backend=LAZ_BACKEND) as reader:
        with ply_path.open("wb") as ply_file:
            write_binary_placeholder_ply_header(ply_file, tile_point_count)
            for points in reader.chunk_iterator(chunk_size):
                x = np.asarray(points.x)
                y = np.asarray(points.y)
                include, core = tile_include_and_core_masks(x, y, window, overlap)
                if not np.any(include):
                    start += len(points)
                    continue

                idx = np.nonzero(include)[0]
                end = written + len(idx)
                vertex = np.empty(len(idx), dtype=PLACEHOLDER_PLY_DTYPE)
                vertex["x"] = x[idx]
                vertex["y"] = y[idx]
                vertex["z"] = np.asarray(points.z)[idx]
                vertex["semantic_seg"] = 2
                vertex["treeID"] = 1
                vertex.tofile(ply_file)
                original_indices[written:end] = start + idx
                core_mask[written:end] = core[idx]
                written = end
                start += len(points)

    if written != tile_point_count:
        raise RuntimeError(
            f"Streamed {written:,} tile points but counted {tile_point_count:,}"
        )
    original_indices.flush()
    core_mask.flush()
    return tile_point_count, core_point_count, original_indices, core_mask


def run_tile_inference(
    sample_id: str,
    staged_ply: Path,
    *,
    repo_dir: Path,
    sample_work: Path,
    checkpoint: Path,
    config_base: Path,
    env: dict[str, str],
    preprocess_workers: int,
    skip_inference: bool,
    skip_preprocess: bool,
    prediction_dir: Path | None,
    benchmark_stage_log: Path | None,
) -> Path:
    dataset_root = sample_work / "ForAINetV2"
    test_data = dataset_root / "test_data"
    meta_data = dataset_root / "meta_data"
    (dataset_root / "train_val_data").mkdir(parents=True, exist_ok=True)
    test_data.mkdir(parents=True, exist_ok=True)
    meta_data.mkdir(parents=True, exist_ok=True)
    (meta_data / "train_list.txt").write_text("")
    (meta_data / "val_list.txt").write_text("")
    (meta_data / "test_list.txt").write_text(f"{sample_id}\n")
    staged_target = test_data / f"{sample_id}.ply"
    if staged_ply.resolve() != staged_target.resolve():
        shutil.copyfile(staged_ply, staged_target)

    runtime_config = sample_work / "configs" / f"forestmamba_{sample_id}.py"
    write_runtime_config(runtime_config, config_base, dataset_root)
    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba",
        input_path=staged_ply,
        sample_id=sample_id,
        stage="model_preprocess",
        metadata={"skipped": bool(skip_preprocess)},
    ):
        if skip_preprocess:
            print("[preprocess] Skipping ForestMamba data conversion", flush=True)
        else:
            preprocess_dataset(repo_dir, dataset_root, preprocess_workers, env)

    if skip_inference:
        if prediction_dir is None:
            raise ValueError("--skip-inference requires --prediction-dir")
        pred_dir = prediction_dir
    elif prediction_dir is not None:
        pred_dir = prediction_dir
    else:
        with benchmark_stage(
            benchmark_stage_log,
            model="forestmamba",
            input_path=staged_ply,
            sample_id=sample_id,
            stage="model_inference",
        ):
            pred_dir = run_inference(
                repo_dir,
                sample_work / "run",
                runtime_config,
                checkpoint,
                env,
            )
    return find_prediction(pred_dir, sample_id, test_data / f"{sample_id}.ply")


def build_env(repo_dir: Path, gpu_device: str | None) -> dict[str, str]:
    env = os.environ.copy()
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(repo_dir)
        if not existing_pythonpath
        else f"{repo_dir}{os.pathsep}{existing_pythonpath}"
    )
    env.setdefault("WANDB_MODE", "disabled")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/.cache")
    env.setdefault("HF_HOME", "/tmp/huggingface")
    env.setdefault("TRANSFORMERS_CACHE", "/tmp/huggingface/transformers")
    if gpu_device is not None and str(gpu_device).strip() != "":
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_device).strip()
    return env


def process_one(
    input_path: Path,
    *,
    repo_dir: Path,
    output_dir: Path,
    work_dir: Path,
    checkpoint: Path,
    config_base: Path,
    gpu_device: str,
    preprocess_workers: int,
    prediction_dir: Path | None,
    skip_inference: bool,
    skip_preprocess: bool,
    semantic_dim: str,
    instance_dim: str,
    score_dim: str,
    chunk_size: int,
    benchmark_stage_log: Path | None,
) -> dict[str, object]:
    stem = safe_stem(input_path)
    sample_id = f"test_{stem}"
    sample_work = work_dir / sample_id
    dataset_root = sample_work / "ForAINetV2"
    env = build_env(repo_dir, gpu_device)

    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba",
        input_path=input_path,
        sample_id=sample_id,
        stage="input_conversion",
    ):
        point_count = stage_forainet_dataset(
            input_path, dataset_root, sample_id, chunk_size
        )
    runtime_config = sample_work / "configs" / f"forestmamba_{sample_id}.py"
    write_runtime_config(runtime_config, config_base, dataset_root)
    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba",
        input_path=input_path,
        sample_id=sample_id,
        stage="model_preprocess",
        metadata={"skipped": bool(skip_preprocess)},
    ):
        if skip_preprocess:
            print("[preprocess] Skipping ForestMamba data conversion", flush=True)
        else:
            preprocess_dataset(repo_dir, dataset_root, preprocess_workers, env)

    if skip_inference:
        if prediction_dir is None:
            raise ValueError("--skip-inference requires --prediction-dir")
        pred_dir = prediction_dir
    elif prediction_dir is not None:
        pred_dir = prediction_dir
    else:
        with benchmark_stage(
            benchmark_stage_log,
            model="forestmamba",
            input_path=input_path,
            sample_id=sample_id,
            stage="model_inference",
        ):
            pred_dir = run_inference(
                repo_dir,
                sample_work / "run",
                runtime_config,
                checkpoint,
                env,
            )

    pred_ply = find_prediction(pred_dir, sample_id, input_path)
    output_path = output_dir / f"{stem}_forestmamba.laz"
    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba",
        input_path=input_path,
        sample_id=sample_id,
        stage="merge_output",
    ):
        summary = enrich_laz(
            input_path,
            pred_ply,
            output_path,
            semantic_dim,
            instance_dim,
            score_dim,
            chunk_size,
        )
    summary["sample_id"] = sample_id
    summary["staged_point_count"] = int(point_count)
    return summary


def process_one_bluepoint(
    input_path: Path,
    *,
    repo_dir: Path,
    output_dir: Path,
    work_dir: Path,
    checkpoint: Path,
    config_base: Path,
    gpu_device: str,
    preprocess_workers: int,
    prediction_dir: Path | None,
    skip_inference: bool,
    skip_preprocess: bool,
    semantic_dim: str,
    instance_dim: str,
    score_dim: str,
    chunk_size: int,
    bluepoint_iterations: int,
    bluepoint_score_threshold: float | None,
    bluepoint_second_pass_threshold: float,
    spatial_match_tolerance: float,
    benchmark_stage_log: Path | None,
) -> dict[str, object]:
    if bluepoint_iterations <= 0:
        raise ValueError("--bluepoint-iterations must be greater than 0")
    if not 0 <= bluepoint_second_pass_threshold <= 1:
        raise ValueError("--bluepoint-second-pass-threshold must be between 0 and 1")

    stem = safe_stem(input_path)
    sample_id = f"test_{stem}"
    sample_work = work_dir / sample_id
    dataset_root = sample_work / "ForAINetV2"
    test_data = dataset_root / "test_data"
    meta_data = dataset_root / "meta_data"
    env = build_env(repo_dir, gpu_device)
    env["FORESTMAMBA_OUTPUT_MODE"] = "bluepoints"

    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba-bluepoint",
        input_path=input_path,
        sample_id=sample_id,
        stage="input_conversion",
    ):
        point_count = stage_forainet_dataset(
            input_path, dataset_root, sample_id, chunk_size
        )

    runtime_config = sample_work / "configs" / f"forestmamba_bluepoint_{sample_id}.py"
    write_runtime_config(
        runtime_config,
        config_base,
        dataset_root,
        score_threshold=bluepoint_score_threshold,
    )

    second_pass_decision: dict[str, object] | None = None
    overlay_prediction: Path | None = None

    if prediction_dir is not None:
        pred_dir = prediction_dir
    elif skip_inference:
        raise ValueError("--skip-inference with bluepoints requires --prediction-dir")
    else:
        pred_dir = None
        current_sample_id = sample_id
        run_dir = sample_work / "bluepoint_run"
        for iteration in range(1, bluepoint_iterations + 1):
            (meta_data / "test_list.txt").write_text(f"{current_sample_id}\n")
            stale_instance_dir = dataset_root / "forainetv2_instance_data"
            if stale_instance_dir.exists():
                for stale_path in stale_instance_dir.glob("*bluepoints*"):
                    if stale_path.is_dir():
                        shutil.rmtree(stale_path)
                    else:
                        stale_path.unlink()
            with benchmark_stage(
                benchmark_stage_log,
                model="forestmamba-bluepoint",
                input_path=input_path,
                sample_id=current_sample_id,
                stage="model_preprocess",
                metadata={
                    "iteration": iteration,
                    "skipped": bool(skip_preprocess),
                },
            ):
                if skip_preprocess:
                    print("[preprocess] Skipping ForestMamba data conversion", flush=True)
                else:
                    preprocess_dataset(repo_dir, dataset_root, preprocess_workers, env)

            with benchmark_stage(
                benchmark_stage_log,
                model="forestmamba-bluepoint",
                input_path=input_path,
                sample_id=current_sample_id,
                stage="model_inference",
                metadata={"iteration": iteration},
            ):
                pred_dir = run_inference(
                    repo_dir,
                    run_dir,
                    runtime_config,
                    checkpoint,
                    env,
                )

            bluepoint_path = pred_dir / f"{sample_id}_bluepoints_{iteration}.ply"
            if iteration == 1 and bluepoint_iterations > 1:
                first_pass_stats = first_pass_unassigned_ratio(pred_dir, sample_id)
                should_run_second_pass = (
                    first_pass_stats["unassigned_ratio"]
                    > bluepoint_second_pass_threshold
                )
                second_pass_decision = {
                    **first_pass_stats,
                    "threshold": float(bluepoint_second_pass_threshold),
                    "run_second_pass": bool(should_run_second_pass),
                }
                print(
                    "[bluepoint] first-pass non-ground unassigned ratio "
                    f"{first_pass_stats['unassigned_ratio']:.4%} "
                    f"(threshold {bluepoint_second_pass_threshold:.4%}); "
                    f"run_second_pass={should_run_second_pass}",
                    flush=True,
                )
                if not should_run_second_pass:
                    break

            if iteration > 1:
                candidate_overlay = pred_dir / f"{sample_id}_{iteration}.ply"
                if candidate_overlay.exists():
                    overlay_prediction = candidate_overlay
            if not bluepoint_path.exists():
                print(
                    f"[bluepoint] No bluepoints for round {iteration}; stopping.",
                    flush=True,
                )
                break
            if iteration < bluepoint_iterations:
                next_sample_id = f"{sample_id}_bluepoints_{iteration}"
                shutil.copyfile(bluepoint_path, test_data / f"{next_sample_id}.ply")
                current_sample_id = next_sample_id

        if pred_dir is None:
            raise RuntimeError("Bluepoint inference did not produce a prediction directory")

    base_prediction = pred_dir / f"{sample_id}_1.ply"
    if not base_prediction.exists():
        raise FileNotFoundError(f"Bluepoint base prediction missing: {base_prediction}")
    if prediction_dir is not None:
        for candidate in sorted(pred_dir.glob(f"{sample_id}_*.ply")):
            match = re.fullmatch(rf"{re.escape(sample_id)}_(\d+)\.ply", candidate.name)
            if match and int(match.group(1)) > 1:
                overlay_prediction = candidate

    output_path = output_dir / f"{stem}_forestmamba_bluepoints.laz"
    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba-bluepoint",
        input_path=input_path,
        sample_id=sample_id,
        stage="apply_bluepoint_predictions",
    ):
        summary = overlay_bluepoint_prediction(
            input_path,
            base_prediction,
            overlay_prediction,
            output_path,
            semantic_dim,
            instance_dim,
            score_dim,
            chunk_size,
            spatial_match_tolerance,
        )
    summary["prediction_mode"] = "bluepoints"
    summary["sample_id"] = sample_id
    summary["staged_point_count"] = int(point_count)
    if second_pass_decision is not None:
        summary["second_pass_decision"] = second_pass_decision
    return summary


def process_one_tiled(
    input_path: Path,
    *,
    repo_dir: Path,
    output_dir: Path,
    work_dir: Path,
    checkpoint: Path,
    config_base: Path,
    gpu_device: str,
    preprocess_workers: int,
    prediction_dir: Path | None,
    skip_inference: bool,
    skip_preprocess: bool,
    semantic_dim: str,
    instance_dim: str,
    score_dim: str,
    chunk_size: int,
    tile_size: float,
    tile_overlap: float,
    tile_max_points: int,
    keep_tile_work: bool,
    benchmark_stage_log: Path | None,
) -> dict[str, object]:
    stem = safe_stem(input_path)
    env = build_env(repo_dir, gpu_device)
    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba",
        input_path=input_path,
        sample_id=stem,
        stage="tiling",
        metadata={"tile_size": float(tile_size), "tile_overlap": float(tile_overlap)},
    ):
        windows, point_count = tile_windows(input_path, tile_size)
    print(
        f"[tile] {input_path}: {point_count:,} points across {len(windows)} tiles "
        f"(tile_size={tile_size}, overlap={tile_overlap})",
        flush=True,
    )

    pred_dir = work_dir / f"{stem}_tiled_predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    semantic_pred = np.memmap(
        pred_dir / "semantic_pred.int32", dtype=np.int32, mode="w+", shape=(point_count,)
    )
    instance_pred = np.memmap(
        pred_dir / "instance_pred.int32", dtype=np.int32, mode="w+", shape=(point_count,)
    )
    score_pred = np.memmap(
        pred_dir / "score_pred.float32", dtype=np.float32, mode="w+", shape=(point_count,)
    )
    assigned = np.memmap(
        pred_dir / "assigned.bool", dtype=bool, mode="w+", shape=(point_count,)
    )
    semantic_pred[:] = 0
    instance_pred[:] = 0
    score_pred[:] = 0.0
    assigned[:] = False

    next_instance_offset = 0
    processed_tiles = 0
    skipped_tiles = 0

    for tile_index, window in enumerate(windows, start=1):
        ix, iy, *_ = window
        sample_id = f"test_{stem}_tile_{ix}_{iy}"
        sample_work = work_dir / sample_id
        staged_ply = sample_work / "streamed_stage" / f"{sample_id}.ply"
        original_indices_path = sample_work / "streamed_stage" / "original_indices.int64"
        core_mask_path = sample_work / "streamed_stage" / "core_mask.bool"
        with benchmark_stage(
            benchmark_stage_log,
            model="forestmamba",
            input_path=input_path,
            sample_id=sample_id,
            stage="input_conversion",
            metadata={"tile_index": tile_index, "tile_x": ix, "tile_y": iy},
        ):
            tile_points, core_points, original_indices, core_mask = stage_tile_from_las(
                input_path,
                window,
                overlap=tile_overlap,
                chunk_size=chunk_size,
                ply_path=staged_ply,
                original_indices_path=original_indices_path,
                core_mask_path=core_mask_path,
                max_points=tile_max_points,
            )
        if tile_points == 0 or core_points == 0:
            skipped_tiles += 1
            shutil.rmtree(sample_work, ignore_errors=True)
            continue

        print(
            f"[tile] {tile_index}/{len(windows)} ({ix}, {iy}) "
            f"points={tile_points:,} core={core_points:,}",
            flush=True,
        )
        tile_prediction = run_tile_inference(
            sample_id,
            staged_ply,
            repo_dir=repo_dir,
            sample_work=sample_work,
            checkpoint=checkpoint,
            config_base=config_base,
            env=env,
            preprocess_workers=preprocess_workers,
            skip_inference=skip_inference,
            skip_preprocess=skip_preprocess,
            prediction_dir=prediction_dir,
            benchmark_stage_log=benchmark_stage_log,
        )
        tile_sem, tile_inst, tile_score = read_prediction_fields(tile_prediction)
        if len(tile_sem) != tile_points:
            raise ValueError(
                f"Point count mismatch for tile ({ix}, {iy}): staged "
                f"{tile_points}, prediction has {len(tile_sem)}"
            )

        core_indices = original_indices[core_mask]
        core_sem = tile_sem[core_mask]
        core_inst = tile_inst[core_mask].copy()
        core_score = tile_score[core_mask]

        positive = core_inst > 0
        if np.any(positive):
            core_inst[positive] += next_instance_offset
            next_instance_offset = int(core_inst[positive].max())

        semantic_pred[core_indices] = core_sem
        instance_pred[core_indices] = core_inst
        score_pred[core_indices] = core_score
        assigned[core_indices] = True
        processed_tiles += 1

        semantic_pred.flush()
        instance_pred.flush()
        score_pred.flush()
        assigned.flush()
        if not keep_tile_work:
            shutil.rmtree(sample_work, ignore_errors=True)

    assigned_count = int(np.count_nonzero(assigned))
    if assigned_count != point_count:
        print(
            f"[tile] Warning: assigned {assigned_count:,}/{point_count:,} points; "
            "unassigned points keep zero predictions.",
            flush=True,
        )

    output_path = output_dir / f"{stem}_forestmamba.laz"
    with benchmark_stage(
        benchmark_stage_log,
        model="forestmamba",
        input_path=input_path,
        sample_id=stem,
        stage="merge_output",
        metadata={"processed_tile_count": processed_tiles, "skipped_tile_count": skipped_tiles},
    ):
        write_enriched_laz_stream(
            input_path,
            output_path,
            semantic_dim,
            instance_dim,
            score_dim,
            semantic_pred,
            instance_pred,
            score_pred,
            chunk_size,
        )
    positive_instances = np.unique(instance_pred[instance_pred > 0])
    return {
        "input": str(input_path),
        "prediction_mode": "tiled",
        "output": str(output_path),
        "point_count": int(point_count),
        "assigned_point_count": assigned_count,
        "processed_tile_count": processed_tiles,
        "skipped_tile_count": skipped_tiles,
        "tile_size": float(tile_size),
        "tile_overlap": float(tile_overlap),
        "positive_instance_count": int(len(positive_instances)),
        "semantic_dim": semantic_dim,
        "instance_dim": instance_dim,
        "score_dim": score_dim,
    }


def run_batch(args: argparse.Namespace) -> None:
    for name in (args.semantic_dim, args.instance_dim, args.score_dim):
        validate_dim_name(name)

    repo_dir = Path(args.repo_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    work_dir = Path(args.work_dir).resolve()
    checkpoint = Path(args.checkpoint).resolve()
    config_base = Path(args.config_base).resolve()
    prediction_dir = Path(args.prediction_dir).resolve() if args.prediction_dir else None
    benchmark_stage_log = (
        Path(args.benchmark_stage_log).resolve() if args.benchmark_stage_log else None
    )

    if not repo_dir.exists():
        raise FileNotFoundError(f"ForestMamba repository not found: {repo_dir}")
    if not args.skip_inference and not checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint}")
    if not config_base.exists():
        raise FileNotFoundError(f"Config base not found: {config_base}")
    if args.bluepoint_iterations > 0 and args.tile_size > 0:
        raise ValueError("Bluepoint inference is not supported with --tile-size yet")

    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)
    inputs = collect_inputs(args)

    summaries = []
    for index, input_path in enumerate(inputs, start=1):
        print(f"[batch] {index}/{len(inputs)} {input_path}", flush=True)
        model_input_path = input_path
        crop_point_count = None
        if args.crop_radius > 0:
            if args.crop_center_x is None or args.crop_center_y is None:
                raise ValueError(
                    "--crop-center-x and --crop-center-y are required with --crop-radius"
                )
            crop_dir = work_dir / "input_crops"
            crop_path = crop_dir / f"{safe_stem(input_path)}_crop_r{args.crop_radius:.2f}.laz"
            crop_point_count = crop_las_xy_radius(
                input_path,
                crop_path,
                center_x=args.crop_center_x,
                center_y=args.crop_center_y,
                radius=args.crop_radius,
                chunk_size=args.chunk_size,
            )
            model_input_path = crop_path
        if args.bluepoint_iterations > 0:
            summary = process_one_bluepoint(
                model_input_path,
                repo_dir=repo_dir,
                output_dir=output_dir,
                work_dir=work_dir,
                checkpoint=checkpoint,
                config_base=config_base,
                gpu_device=args.gpu_device,
                preprocess_workers=args.preprocess_workers,
                prediction_dir=prediction_dir,
                skip_inference=args.skip_inference,
                skip_preprocess=args.skip_preprocess,
                semantic_dim=args.semantic_dim,
                instance_dim=args.instance_dim,
                score_dim=args.score_dim,
                chunk_size=args.chunk_size,
                bluepoint_iterations=args.bluepoint_iterations,
                bluepoint_score_threshold=args.bluepoint_score_threshold,
                bluepoint_second_pass_threshold=args.bluepoint_second_pass_threshold,
                spatial_match_tolerance=args.spatial_match_tolerance,
                benchmark_stage_log=benchmark_stage_log,
            )
        elif args.tile_size > 0:
            summary = process_one_tiled(
                model_input_path,
                repo_dir=repo_dir,
                output_dir=output_dir,
                work_dir=work_dir,
                checkpoint=checkpoint,
                config_base=config_base,
                gpu_device=args.gpu_device,
                preprocess_workers=args.preprocess_workers,
                prediction_dir=prediction_dir,
                skip_inference=args.skip_inference,
                skip_preprocess=args.skip_preprocess,
                semantic_dim=args.semantic_dim,
                instance_dim=args.instance_dim,
                score_dim=args.score_dim,
                chunk_size=args.chunk_size,
                tile_size=args.tile_size,
                tile_overlap=args.tile_overlap,
                tile_max_points=args.tile_max_points,
                keep_tile_work=args.keep_tile_work,
                benchmark_stage_log=benchmark_stage_log,
            )
        else:
            summary = process_one(
                model_input_path,
                repo_dir=repo_dir,
                output_dir=output_dir,
                work_dir=work_dir,
                checkpoint=checkpoint,
                config_base=config_base,
                gpu_device=args.gpu_device,
                preprocess_workers=args.preprocess_workers,
                prediction_dir=prediction_dir,
                skip_inference=args.skip_inference,
                skip_preprocess=args.skip_preprocess,
                semantic_dim=args.semantic_dim,
                instance_dim=args.instance_dim,
                score_dim=args.score_dim,
                chunk_size=args.chunk_size,
                benchmark_stage_log=benchmark_stage_log,
            )
        if crop_point_count is not None:
            summary["original_input"] = str(input_path)
            summary["crop_input"] = str(model_input_path)
            summary["crop_center_xy"] = [float(args.crop_center_x), float(args.crop_center_y)]
            summary["crop_radius"] = float(args.crop_radius)
            summary["crop_point_count"] = int(crop_point_count)
        summaries.append(summary)

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps({"files": summaries}, indent=2) + "\n")
    print(f"[batch] Wrote {summary_path}", flush=True)


def main(argv: list[str] | None = None) -> None:
    run_batch(parse_args(argv))


if __name__ == "__main__":
    main()
