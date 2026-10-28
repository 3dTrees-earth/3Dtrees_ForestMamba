"""Bluepoint rounds keep tree ids unique and write the standard output name (CPU only)."""

from pathlib import Path
import sys
import tempfile
import unittest

import laspy
import numpy as np
from plyfile import PlyData, PlyElement

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import forest_mamba_laz_batch as batch  # noqa: E402

XYZ = np.array(
    [[572440.0 + i, 5709015.0 + 2 * i, 420.0 + i] for i in range(6)], dtype=np.float64
)


def write_input(path):
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = [0.001, 0.001, 0.001]
    header.offsets = [572000.0, 5709000.0, 0.0]
    las = laspy.LasData(header)
    las.x, las.y, las.z = XYZ[:, 0], XYZ[:, 1], XYZ[:, 2]
    las.write(path)


def write_round(path, rows, instance):
    vertex = np.zeros(
        len(rows),
        dtype=[("x", "f8"), ("y", "f8"), ("z", "f8"), ("semantic_pred", "i4"),
               ("instance_pred", "i4"), ("score", "f4")],
    )
    vertex["x"], vertex["y"], vertex["z"] = XYZ[rows, 0], XYZ[rows, 1], XYZ[rows, 2]
    vertex["semantic_pred"] = 1
    vertex["instance_pred"] = instance
    vertex["score"] = 0.9
    PlyData([PlyElement.describe(vertex, "vertex")]).write(str(path))


class BluepointOverlayTest(unittest.TestCase):
    def run_rounds(self, rounds):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name)
        source = root / "input.laz"
        write_input(source)
        predictions = root / "predictions"
        predictions.mkdir()
        for number, (rows, instance) in enumerate(rounds, start=1):
            write_round(predictions / f"test_input_{number}.ply", rows, instance)
        config = root / "config.py"
        config.write_text("")
        summary = batch.process_one_bluepoint(
            source,
            repo_dir=root,
            output_dir=root / "out",
            work_dir=root / "work",
            checkpoint=root / "model.pth",
            config_base=config,
            gpu_device="0",
            preprocess_workers=1,
            prediction_dir=predictions,
            skip_inference=True,
            skip_preprocess=True,
            semantic_dim="PredSemantic_FM",
            instance_dim="PredInstance_FM",
            score_dim="PredScore_FM",
            chunk_size=1000,
            bluepoint_iterations=len(rounds),
            bluepoint_score_threshold=None,
            bluepoint_second_pass_threshold=0.01,
            bluepoint_threshold_enabled=True,
            spatial_match_tolerance=0.01,
            benchmark_stage_log=None,
        )
        output = Path(summary["output"])
        return summary, output, np.asarray(laspy.read(output)["PredInstance_FM"])

    def test_second_round_ids_follow_the_first_rounds(self):
        # Both rounds number their trees from 1.
        summary, output, ids = self.run_rounds([
            (list(range(6)), [1, 1, 2, 2, -1, -1]),
            ([4, 5], [1, 2]),
        ])
        np.testing.assert_array_equal(ids, [1, 1, 2, 2, 3, 4])
        self.assertEqual(output.name, "input_forestmamba.laz")
        self.assertEqual(summary["positive_instance_count"], 4)
        self.assertEqual([r["instance_id_offset"] for r in summary["overlay_predictions"]], [2])

    def test_every_later_round_is_kept(self):
        summary, _, ids = self.run_rounds([
            (list(range(6)), [1, 1, 2, -1, -1, -1]),
            ([3, 4], [1, -1]),
            ([4, 5], [1, 1]),
        ])
        np.testing.assert_array_equal(ids, [1, 1, 2, 3, 4, 4])
        self.assertEqual([r["instance_id_offset"] for r in summary["overlay_predictions"]], [2, 3])

    def test_rounds_are_applied_in_numeric_order(self):
        rounds = [(list(range(6)), [1, -1, -1, -1, -1, -1])] + [([i], [1]) for i in range(1, 6)]
        _, _, ids = self.run_rounds(rounds)
        np.testing.assert_array_equal(ids, [1, 2, 3, 4, 5, 6])

    def test_round_list_is_numeric(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for number in (1, 2, 10, 3):
                (root / f"test_input_{number}.ply").touch()
            names = [p.name for p in batch.bluepoint_round_predictions(root, "test_input")]
            self.assertEqual(names, ["test_input_2.ply", "test_input_3.ply", "test_input_10.ply"])


if __name__ == "__main__":
    unittest.main()
