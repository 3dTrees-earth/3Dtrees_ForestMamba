"""CPU checks for review evidence and label-invariant comparison semantics."""
import importlib.util
import json
from pathlib import Path
import sys
import unittest

import numpy as np

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(HERE/'commands'))
from compare_cohort import compare


def cloud(ids):
    ids=np.asarray(ids,dtype=np.int64)
    return {'instance':ids,'semantic':np.ones(len(ids),dtype=np.uint8),
            'score':np.ones(len(ids),dtype=np.float32)}


class ComparisonTests(unittest.TestCase):
    def test_instance_renumbering_is_not_segmentation_change(self):
        result=compare(cloud([0,1,1,2,2]),cloud([0,9,9,7,7]))
        self.assertEqual(result['instance_agreement_matched_including_background'],1)
        self.assertEqual(result['ari_including_background'],1)

    def test_background_cannot_be_matched_to_a_tree(self):
        result=compare(cloud([0,1,1,2,2]),cloud([0,0,0,7,7]))
        self.assertEqual(result['instance_agreement_matched_including_background'],3/5)
        self.assertEqual(result['assigned_unassigned_agreement'],3/5)

    def test_split_is_counted_even_with_optimal_matching(self):
        result=compare(cloud([1,1,1,1]),cloud([2,2,3,3]))
        self.assertEqual(result['instance_agreement_matched_including_background'],.5)

    def test_original_and_candidate_evidence_fail_gate(self):
        summary=json.loads((HERE/'results/2026-09-17/summary.json').read_text())
        self.assertEqual(summary['pairs'],45)
        self.assertEqual(summary['input_preservation_passes'],10)
        self.assertTrue(summary['excess_variation_observed'])
        self.assertTrue(summary['packaged_helper_exact_replay_passed'])
        self.assertEqual(len(list((HERE/'results/2026-09-17/pairs').glob('*.json'))),45)
        d=summary['global_screens']['original_vs_remap']['instance_disagreement']
        self.assertEqual(len(d['candidate_exceedances']),11)
        self.assertGreater(d['candidate_max'],d['baseline_max'])


if __name__=='__main__':
    unittest.main()
