"""CUDA compatibility rule and the job-start GPU line (CPU only, no nvidia-smi needed)."""

from pathlib import Path
import json
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import cuda_arch_check as check  # noqa: E402
import forest_mamba_laz_batch as batch  # noqa: E402


class RunsOnTest(unittest.TestCase):
    def test_sass_runs_on_same_major_with_higher_minor(self):
        self.assertTrue(check.runs_on({(7, 0)}, set(), "7.5"))
        self.assertTrue(check.runs_on({(8, 0)}, set(), "8.6"))

    def test_sm80_only_does_not_run_on_t4(self):
        # MinkowskiEngine in the 1.0.2 image of 19 Nov 2026.
        self.assertFalse(check.runs_on({(8, 0)}, set(), "7.5"))

    def test_ptx_is_jit_compiled_for_newer_gpus(self):
        self.assertTrue(check.runs_on(set(), {(8, 6)}, "9.0"))
        self.assertFalse(check.runs_on(set(), {(8, 6)}, "7.5"))


class ReportGpuTest(unittest.TestCase):
    def run_report(self, smi_row, supported):
        with tempfile.TemporaryDirectory() as tmp:
            info = Path(tmp) / "cuda_archs.json"
            info.write_text(json.dumps({"supported_compute": supported}))
            done = subprocess.CompletedProcess([], 0, stdout=smi_row + "\n")
            with mock.patch.object(batch.subprocess, "run", return_value=done):
                batch.report_gpu({"CUDA_VISIBLE_DEVICES": "0"}, info)

    def test_supported_gpu_passes(self):
        self.run_report("0, Tesla T4, 7.5, 15360 MiB", ["7.0", "7.5", "8.0", "8.6"])

    def test_unsupported_gpu_stops_with_clear_message(self):
        with self.assertRaisesRegex(RuntimeError, "Tesla T4 \\(compute 7.5\\)"):
            self.run_report("0, Tesla T4, 7.5, 15360 MiB", ["8.0", "8.6"])


if __name__ == "__main__":
    unittest.main()
