"""The sliding-window speed-ups return exactly what the code they replace returned.

Each test compares a new implementation with a copy of the 1.0.2 code. Run inside
the ForestMamba image (it needs torch, torch_cluster and the model package):

    python tests/test_exact_speedups.py

GPU tests are skipped without CUDA or without the compiled fm_kernels_cuda
extension. End to end, FM_VERIFY=1 runs the same comparisons on every window of
a real run.
"""
from pathlib import Path
import sys
import unittest

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

try:
    from oneformer3d import fm_kernels
    from oneformer3d import oneformer3d as model
    from oneformer3d.remapping import remapping_knn
    from oneformer3d.query_decoder import MambaAggregation
    from torch_cluster import fps as cluster_fps
except ImportError as err:  # pragma: no cover - outside the image
    raise unittest.SkipTest(f"needs the ForestMamba image: {err}")

CUDA = torch.cuda.is_available() and fm_kernels.available()


# ---------------------------------------------------------------- 1.0.2 references
def z_filter_loop(scores, mask_pred, coordinates, ground_z_max, z_margin):
    scores = scores.clone()
    for i in range(mask_pred.size(0)):
        mask = mask_pred[i]
        if mask.sum().item() == 0:
            scores[i] = 0
            continue
        z_values = coordinates[mask, 2]
        if z_values.numel() > 0 and z_values.min().item() > ground_z_max + z_margin:
            scores[i] = 0
    return scores


def peaks_loop(chm, occupied, res, a, b, min_height):
    rows, cols = np.where(occupied & (chm >= min_height))
    out_r, out_c = [], []
    for r, c in zip(rows, cols):
        h = float(chm[r, c])
        w_cells = max(1, int(np.ceil(a * (h ** b) / res)))
        r0 = max(0, r - w_cells); r1 = min(chm.shape[0], r + w_cells + 1)
        c0 = max(0, c - w_cells); c1 = min(chm.shape[1], c + w_cells + 1)
        if chm[r, c] >= chm[r0:r1, c0:c1].max():
            out_r.append(r); out_c.append(c)
    return np.array(out_r, dtype=np.int64), np.array(out_c, dtype=np.int64)


def nms_loop(peak_xy, d_sq_thresh):
    kept = np.ones(len(peak_xy), dtype=bool)
    for i in range(len(peak_xy) - 1):
        if not kept[i]:
            continue
        d_sq = ((peak_xy[i + 1:] - peak_xy[i]) ** 2).sum(axis=1)
        kept[i + 1:][d_sq < d_sq_thresh] = False
    return kept


class _Peaks:
    """The attributes _detect_chm_peaks reads."""
    chm_min_height = 1.5
    chm_allometric_a = torch.tensor(0.25)
    chm_allometric_b = torch.tensor(0.5)


# ---------------------------------------------------------------- CPU tests
class CpuEquivalence(unittest.TestCase):
    def test_z_filter(self):
        g = torch.Generator().manual_seed(0)
        for case in range(50):
            k, n = int(torch.randint(1, 60, (1,), generator=g)), int(torch.randint(1, 3000, (1,), generator=g))
            masks = torch.rand(k, n, generator=g) < float(torch.rand(1, generator=g)) * 0.3
            masks[0] = False  # an empty mask
            coords = torch.rand(n, 3, generator=g) * 30
            coords[:, 2] = torch.round(coords[:, 2] * 10) / 10  # ties at the threshold
            scores = torch.rand(k, generator=g)
            ground = float(torch.round(torch.rand(1, generator=g) * 100) / 10)
            ref = z_filter_loop(scores, masks, coords, ground, 5.0)
            new = scores.clone()
            lowest = model._masked_row_min(masks, coords[:, 2]).double()
            new[(masks.sum(1) == 0) | (lowest > ground + 5.0)] = 0
            self.assertTrue(torch.equal(ref, new), f"case {case}")

    def test_chm_peaks(self):
        rng = np.random.default_rng(1)
        detect = model.ForAINetV2OneFormer3D_CHMquery._detect_chm_peaks
        for case in range(40):
            shape = rng.integers(1, 120, size=2)
            chm = (rng.random(shape) * 35).astype(np.float32)
            if case % 2:
                chm = np.round(chm).astype(np.float32)  # many equal heights
            occupied = rng.random(shape) > 0.2
            chm[~occupied] = 0.0
            for res in (0.3, 0.7):
                ref = peaks_loop(chm, occupied, res, 0.25, 0.5, 1.5)
                new = detect(_Peaks(), chm, occupied, res)
                self.assertTrue(np.array_equal(ref[0], new[0]) and np.array_equal(ref[1], new[1]),
                                f"case {case} res {res}")

    def test_peak_nms(self):
        rng = np.random.default_rng(2)
        for case in range(200):
            m = int(rng.integers(1, 2500))
            xy = (rng.random((m, 2)) * rng.choice([5, 20, 32])).astype(np.float32)
            if case % 3 == 0:
                xy = (np.round(xy * 10) / 10).astype(np.float32)  # exact 1.0 m distances
            self.assertTrue(np.array_equal(nms_loop(xy, 1.0), model._greedy_peak_nms(xy, 1.0)),
                            f"case {case}")


class DecoderIterPredGate(unittest.TestCase):
    def test_only_the_ssm_decoder_skips_iter_pred(self):
        """Inference turns iter_pred off only where layers ignore the per-layer
        masks; the transformer decoders use them as attention masks."""
        from oneformer3d import query_decoder as qd
        self.assertTrue(qd.ForAINetv2SSMQueryDecoder_XAwarequery
                        .final_prediction_independent_of_iter_pred)
        for name in ("QueryDecoder", "ForAINetv2QueryDecoder",
                     "ForAINetv2QueryDecoder_XAwarequery", "ScanNetQueryDecoder",
                     "OneDataQueryDecoder"):
            self.assertFalse(getattr(getattr(qd, name),
                                     "final_prediction_independent_of_iter_pred", False), name)


# ---------------------------------------------------------------- GPU tests
@unittest.skipUnless(CUDA, "needs CUDA and the fm_kernels_cuda extension")
class GpuEquivalence(unittest.TestCase):
    def test_fps_matches_torch_cluster(self):
        for seed in range(40):
            g = torch.Generator().manual_seed(seed)
            n = int(torch.randint(2, 120_000, (1,), generator=g))
            x = torch.randn(n, 5, generator=g)
            if seed % 3 == 0:
                x = (x * 4).round() / 4  # exact distance ties
            if seed % 5 == 0:
                x = torch.cat([x, x[: n // 3]])  # duplicate rows
            x = x.cuda()
            batch = torch.zeros(len(x), dtype=torch.long, device="cuda")
            ratio = min(400 / len(x), 1.0)
            torch.manual_seed(seed)
            ref = cluster_fps(x, batch, ratio=ratio)
            torch.manual_seed(seed)
            new = fm_kernels.fps(x, batch, ratio)
            self.assertTrue(torch.equal(ref, new), f"seed {seed}")

    def test_nearest_sampled_point_matches_remapping(self):
        grid_sample = model.ForAINetV2OneFormer3D_XAwarequery.grid_sample
        for seed in range(12):
            g = torch.Generator().manual_seed(seed)
            n = int(torch.randint(1_000, 400_000, (1,), generator=g))
            pts = torch.rand(n, 3, generator=g) * torch.tensor([32.0, 32.0, 30.0])
            pts = torch.round(pts * 1000) / 1000  # millimetre grid like LAS: means on voxel faces
            if seed % 2:
                pts[: n // 10, 2] = torch.round(pts[: n // 10, 2] / 0.2) * 0.2
            pts = pts.cuda()
            idx = torch.arange(n, device="cuda")
            pc3, _ = grid_sample(pts, idx, 0.2)
            new = fm_kernels.nearest_sampled_point(pc3.float(), pts.float(), 0.2)
            self.assertIsNotNone(new)
            ref = remapping_knn(pc3.float(), pts.float(), k=1)[1]
            self.assertTrue(torch.equal(ref, new), f"seed {seed}")

    def test_fps_multi_cloud_batch_falls_back(self):
        x = torch.randn(3000, 3, device="cuda")
        batch = torch.repeat_interleave(torch.arange(3, device="cuda"), 1000)
        torch.manual_seed(0)
        ref = cluster_fps(x, batch, ratio=0.1)
        torch.manual_seed(0)
        self.assertTrue(torch.equal(ref, fm_kernels.fps(x, batch, 0.1)))

    def test_nearest_sampled_point_hard_inputs(self):
        """Far from the origin, dense voxels, huge spans, tiny cells, no queries:
        the result is either None (caller falls back) or exactly the reference."""
        grid_sample = model.ForAINetV2OneFormer3D_XAwarequery.grid_sample
        g = torch.Generator().manual_seed(7)
        cases = []
        for offset, n in ((32.0, 200_000), (100.0, 100_000), (500.0, 50_000), (5000.0, 5_000)):
            pts = torch.rand(n, 3, generator=g) * 0.6 + offset  # ~27 voxels, thousands of points each
            cases.append((pts, 0.2))
        spread = torch.rand(101, 3, generator=g) * 450_000.0  # voxel key would overflow
        cases.append((spread, 0.2))
        cases.append((torch.rand(7, 3, generator=g) * 1e-4, 1e-5))
        for i, (pts, cell) in enumerate(cases):
            pts = pts.cuda()
            pc3, _ = grid_sample(pts, torch.arange(len(pts), device="cuda"), cell)
            new = fm_kernels.nearest_sampled_point(pc3.float(), pts.float(), cell)
            if new is not None:
                ref = remapping_knn(pc3.float(), pts.float(), k=1)[1]
                self.assertTrue(torch.equal(ref, new), f"case {i}")
        # Found in review: many identical points just below a voxel edge make
        # the float32 voxel mean drift 0.1 m into the neighbouring voxel.
        for offset, k in ((32.0, 200_000), (100.0, 100_000), (1000.0, 10_000)):
            edge = float(np.nextafter(np.float32(offset - 0.2), np.float32(-np.inf)))
            pts = torch.cat([
                torch.tensor([[offset + 0.005, 0, 0], [offset + 0.195, 0.195, 0.195],
                              [offset + 0.201, 0, 0]]),
                torch.tensor([[edge, 0.0, 0.0]]).repeat(k, 1)]).float().cuda()
            pc3, _ = grid_sample(pts, torch.arange(len(pts), device="cuda"), 0.2)
            new = fm_kernels.nearest_sampled_point(pc3.float(), pts.float(), 0.2)
            if new is not None:
                ref = remapping_knn(pc3.float(), pts.float(), k=1)[1]
                self.assertTrue(torch.equal(ref, new), f"drift at {offset} m")
        refs = torch.rand(10, 3, device="cuda")
        self.assertIsNone(fm_kernels.nearest_sampled_point(refs, refs[:0], 0.2))

    @unittest.skipUnless(torch.cuda.device_count() > 1, "needs two GPUs")
    def test_kernels_on_second_gpu(self):
        torch.cuda.set_device(0)
        x = torch.randn(5000, 3, device="cuda:1")
        batch = torch.zeros(5000, dtype=torch.long, device="cuda:1")
        torch.manual_seed(3)
        ref = cluster_fps(x, batch, ratio=0.05)
        torch.manual_seed(3)
        self.assertTrue(torch.equal(ref, fm_kernels.fps(x, batch, 0.05)))
        pts = (torch.round(torch.rand(20_000, 3, device="cuda:1") * 8000) / 1000)
        grid_sample = model.ForAINetV2OneFormer3D_XAwarequery.grid_sample
        pc3, _ = grid_sample(pts, torch.arange(len(pts), device="cuda:1"), 0.2)
        new = fm_kernels.nearest_sampled_point(pc3.float(), pts.float(), 0.2)
        self.assertTrue(torch.equal(remapping_knn(pc3.float(), pts.float(), k=1)[1], new))

    def test_decoder_knn_cache(self):
        torch.manual_seed(0)
        agg = MambaAggregation(64, k=8).cuda().eval()
        q, src = torch.randn(403, 64, device="cuda"), torch.randn(5000, 64, device="cuda")
        q_pos, s_pos = torch.rand(403, 3, device="cuda") * 32, torch.rand(5000, 3, device="cuda") * 32
        with torch.no_grad():
            ref = [agg(q, src, q_pos, s_pos) for _ in range(3)]
            cache = {}
            new = [agg(q, src, q_pos.clone(), s_pos, knn_cache=cache) for _ in range(3)]
        for a, b in zip(ref, new):
            self.assertTrue(torch.equal(a, b))


if __name__ == "__main__":
    unittest.main()
