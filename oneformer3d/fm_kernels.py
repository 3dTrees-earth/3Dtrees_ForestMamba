"""Exact, faster replacements for the per-window hot spots of sliding-window inference.

Every function here returns what the code it replaces returns (same indices, same
tie rules); tests/test_exact_speedups.py checks that against the originals.

* ``fps`` - farthest point sampling with torch_cluster.fps's picks, on all SMs.
* ``nearest_sampled_point`` - for every raw window point, its nearest grid-sampled
  point, as ``remapping_knn`` returns it, by searching neighbouring voxels.

Set ``FM_LEGACY=1`` to run the original implementations instead (comparison or
rollback). If the compiled extension is missing, the originals are used and a
warning is printed once.
"""
import collections
import os
import sys

import numpy as np
import torch
from torch_cluster import fps as cluster_fps, knn as cluster_knn

LEGACY = os.environ.get("FM_LEGACY", "0") == "1"
# FM_VERIFY=1: replaced implementations run alongside and disagreements are
# counted here (printed at the end of each tile).
VERIFY = os.environ.get("FM_VERIFY", "0") == "1"
verify_counts = collections.Counter()

try:  # built by oneformer3d/csrc/setup.py in Dockerfile.3dtrees
    import fm_kernels_cuda as _ext
except ImportError:  # pragma: no cover - depends on the image
    _ext = None

_warned = False


def available():
    """True when the compiled kernels can be used."""
    global _warned
    if LEGACY:
        return False
    if _ext is None and not _warned:
        print("[fm_kernels] extension not built; using the original torch_cluster paths",
              file=sys.stderr, flush=True)
        _warned = True
    return _ext is not None


def fps(src, batch, ratio):
    """``torch_cluster.fps(src, batch, ratio=ratio)`` for a single cloud (batch all zeros).

    Draws the random start exactly like torch_cluster (one ``torch.rand`` on the
    source device), so seeded runs pick the same points.
    """
    if not (available() and src.is_cuda and src.dtype == torch.float32):
        return cluster_fps(src, batch, ratio=ratio)
    n = src.size(0)
    r = torch.tensor(float(ratio), dtype=src.dtype, device=src.device)
    deg = torch.tensor([n], dtype=torch.long, device=src.device)
    k = int((deg.to(r.dtype) * r).ceil().long().item())
    start = torch.rand(1, dtype=src.dtype, device=src.device)
    start = int((start * deg.to(r.dtype)).long().item())
    return _ext.fm_fps(src.reshape(n, -1), start, k)


# Same ambiguity guard as hybrid_nearest: near ties go to torch_cluster's rule.
_RELATIVE_GUARD = 64 * np.finfo(np.float32).eps
_ABSOLUTE_GUARD = 64 * np.finfo(np.float32).tiny


def nearest_sampled_point(refs, queries, cell, chunk=131072):
    """Index of the nearest ``refs`` row for every ``queries`` row, or None.

    ``refs`` must be the voxel means of ``queries`` on the grid ``floor(p / cell)``
    in ``torch.unique(dim=0)`` order, as ``grid_sample`` builds them, so every
    query's own voxel holds a ref within sqrt(3) * cell. Candidates are the refs
    of the 27 neighbouring voxels; rows whose first or second candidate is not
    provably closer than every ref outside them are searched over 125 voxels.
    Distances are float64 like cKDTree; near ties use torch_cluster.knn's rule
    over those voxels, as ``hybrid_nearest`` does over all refs. Returns None when
    the precondition does not hold; the caller then uses ``remapping_knn``.
    """
    if not available() or not refs.is_cuda or len(refs) == 0:
        return None
    dev = refs.device
    qc = torch.floor(queries / cell).to(torch.int32).long()
    lo = qc.min(0).values - 3
    span = qc.max(0).values - lo + 4
    sy, sz = int(span[1]), int(span[2])

    def key(c):
        c = c - lo
        return (c[..., 0] * sy + c[..., 1]) * sz + c[..., 2]

    # A key monotone in (x, y, z) sorts like torch.unique(dim=0): unique key i is ref i.
    skey = torch.unique(key(qc))
    if len(skey) != len(refs):
        return None

    def offsets(r):
        rng = torch.arange(-r, r + 1, device=dev)
        return torch.stack(torch.meshgrid(rng, rng, rng, indexing="ij"), -1).reshape(-1, 3)

    offs27, offs125 = offsets(1), offsets(2)
    refs64 = refs.double()

    def search(q, qcell, offs, want_candidates=False):
        k = key(qcell.unsqueeze(1) + offs.unsqueeze(0))
        pos = torch.searchsorted(skey, k).clamp(max=len(skey) - 1)
        hit = skey[pos] == k
        if want_candidates:
            return torch.where(hit, pos, torch.full_like(pos, -1))
        d = ((refs64[pos] - q.double().unsqueeze(1)) ** 2).sum(-1)
        d = torch.where(hit, d, torch.full_like(d, float("inf")))
        two = torch.topk(d, 2, dim=1, largest=False)
        return pos.gather(1, two.indices[:, :1]).squeeze(1), two.values[:, 0], two.values[:, 1]

    out = torch.empty(len(queries), dtype=torch.long, device=dev)
    ambiguous = []
    for s in range(0, len(queries), chunk):
        q, qcell = queries[s:s + chunk], qc[s:s + chunk]
        idx, first, second = search(q, qcell, offs27)
        # A ref outside the 27 voxels is at least cell + (distance from the query
        # to its nearest voxel face) away, minus slack for the float32 voxel mean.
        u = q.double() - qcell.double() * cell
        margin = torch.minimum(u, cell - u).min(1).values.clamp(min=0)
        bound_sq = (cell + margin - 1e-4) ** 2
        redo = torch.nonzero((first >= bound_sq) | (second >= bound_sq)).squeeze(1)
        if redo.numel():
            i2, f2, s2 = search(q[redo], qcell[redo], offs125)
            idx[redo], first[redo], second[redo] = i2, f2, s2
        if not bool(torch.isfinite(first).all()):
            return None
        out[s:s + chunk] = idx
        finite = torch.isfinite(second)
        amb = torch.zeros_like(finite)
        amb[finite] = (second[finite] - first[finite]) <= (
            _RELATIVE_GUARD * second[finite] + _ABSOLUTE_GUARD)
        rows = torch.nonzero(amb).squeeze(1)
        if rows.numel():
            ambiguous.append(rows + s)
    if ambiguous:
        rows = torch.cat(ambiguous)
        candidates = search(queries[rows], qc[rows], offs125, want_candidates=True)
        out[rows] = _ext.fm_knn1(refs.float(), queries[rows].float(), candidates)
    return out


def reference_knn(refs, queries):
    """torch_cluster.knn(refs, queries, k=1) column, for verification."""
    return cluster_knn(refs.contiguous(), queries.contiguous(), k=1)[1]
