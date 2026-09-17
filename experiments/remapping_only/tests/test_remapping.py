"""Small CUDA oracle tests, loaded without importing the full model package."""
import importlib.util
from pathlib import Path
import sys
import types
import unittest

import torch
from torch_cluster import knn

ROOT = Path(__file__).resolve().parents[3]
PACKAGE = '_remapping_test_package'
module = types.ModuleType(PACKAGE)
module.__path__ = [str(ROOT/'oneformer3d')]
sys.modules[PACKAGE] = module


def load(name):
    spec = importlib.util.spec_from_file_location(PACKAGE+'.'+name, ROOT/'oneformer3d'/f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


hybrid = load('hybrid_nearest').hybrid_nearest
adapter = load('remapping').remapping_knn


@unittest.skipUnless(torch.cuda.is_available(), 'CUDA and the pinned torch_cluster extension are required')
class RemappingOracleTests(unittest.TestCase):
    def check_mapping(self, refs, queries, **kwargs):
        refs = torch.as_tensor(refs, dtype=torch.float32, device='cuda')
        queries = torch.as_tensor(queries, dtype=torch.float32, device='cuda')
        actual, stats = hybrid(refs, queries, workers=1, **kwargs)
        expected = knn(refs.contiguous(), queries.contiguous(), k=1)
        self.assertTrue(torch.equal(actual, expected[1]))
        self.assertTrue(torch.equal(adapter(refs, queries), expected))
        # Index equivalence must also preserve arbitrary per-reference predictions.
        predictions = torch.arange(len(refs), device='cuda') * 17
        self.assertTrue(torch.equal(predictions[actual], predictions[expected[1]]))
        return stats

    def test_random_multiple_chunks(self):
        g = torch.Generator().manual_seed(1838)
        refs = torch.randn(173, 3, generator=g)
        queries = torch.randn(901, 3, generator=g)
        stats = self.check_mapping(refs, queries, chunk_size=101)
        self.assertEqual(stats['chunks'], 9)

    def test_duplicates_and_multiway_ties_fall_back(self):
        stats = self.check_mapping([[0,0,0],[0,0,0],[2,0,0],[-2,0,0]],[[0,0,0],[1,0,0],[-1,0,0]])
        self.assertEqual(stats['fallback_queries'],3)

    def test_near_tie_float32_guard(self):
        stats = self.check_mapping([[0,0,0],[2,0,0]],[[1+1e-7,0,0]])
        self.assertEqual(stats['fallback_queries'],1)

    def test_subnormal_distance_guard(self):
        stats = self.check_mapping([[0,0,0],[1e-22,0,0]],[[3e-23,0,0]])
        self.assertEqual(stats['fallback_queries'],1)

    def test_single_reference(self):
        stats = self.check_mapping([[3,2,1]],[[0,0,0],[10,20,30]])
        self.assertEqual(stats['fallback_queries'],0)

    def test_outside_tested_domain_falls_back(self):
        stats = self.check_mapping([[10001,0,0],[10002,0,0]],[[10001,1,0]])
        self.assertEqual(stats['fallback_queries'],1)

    def test_noncontiguous(self):
        refs=torch.arange(60,device='cuda',dtype=torch.float32).reshape(10,6)[:,::2]
        queries=refs[::2]
        self.assertFalse(refs.is_contiguous())
        self.check_mapping(refs,queries,chunk_size=2)

    def test_empty_queries(self):
        result,stats=hybrid(torch.zeros((1,3),device='cuda'),torch.zeros((0,3),device='cuda'))
        self.assertEqual(result.numel(),0)
        self.assertEqual(stats['queries'],0)

    def test_unsupported_contracts_rejected(self):
        xyz=torch.zeros((2,3),device='cuda')
        for refs,queries,kw in [(xyz.cpu(),xyz,{}),(xyz.double(),xyz,{}),
                                (xyz[:0],xyz,{}),(xyz[:,:2],xyz,{}),
                                (xyz,xyz,{'chunk_size':0}),(xyz,xyz,{'workers':0})]:
            with self.subTest(shape=refs.shape,kwargs=kw),self.assertRaises(ValueError):
                hybrid(refs,queries,**kw)
        with self.assertRaises(ValueError):
            adapter(xyz,xyz,k=2)


if __name__=='__main__':
    unittest.main()
