"""Validate LAS/LAZ output against original rows using bounded-memory chunks."""
import argparse
import collections
import itertools
import json
from pathlib import Path

import laspy
import numpy as np

PREDICTIONS = ['PredSemantic_FM', 'PredInstance_FM', 'PredScore_FM']


def equal_metadata(left, right):
    if left is None or right is None:
        return left is right
    left, right = np.asarray(left), np.asarray(right)
    if left.dtype.kind in 'fc' and right.dtype.kind in 'fc':
        return np.array_equal(left, right, equal_nan=True)
    return np.array_equal(left, right)


def validate(original, output, semantic_classes=3):
    report = dict(original=str(original), output=str(output), point_count=0,
                  mismatched_dimensions={}, instances={}, finite_scores=True,
                  header_errors=[], label_errors=[])
    histogram = collections.Counter()
    with laspy.open(original) as src, laspy.open(output) as out:
        sh, oh = src.header, out.header
        for key in ['point_count', 'version', 'point_format', 'scales', 'offsets',
                    'mins', 'maxs', 'uuid', 'system_identifier', 'generating_software',
                    'creation_date', 'global_encoding']:
            left, right = getattr(sh, key), getattr(oh, key)
            if key == 'point_format':
                left, right = left.id, right.id
            if key == 'global_encoding':
                left, right = left.value, right.value
            if not equal_metadata(left, right):
                report['header_errors'].append(key)
        dims = list(sh.point_format.dimension_names)
        output_dims = set(oh.point_format.dimension_names)
        missing = set(dims + PREDICTIONS) - output_dims
        report['header_errors'].extend('missing dimension ' + name for name in sorted(missing))
        # The Extra Bytes VLR changes when adding predictions. Original dimension
        # definitions still must survive, including dtype, scale and no-data metadata.
        for name in dims:
            if name in PREDICTIONS or name not in output_dims:
                continue
            before = sh.point_format.dimension_by_name(name)
            after = oh.point_format.dimension_by_name(name)
            for field in ['kind', 'num_bits', 'num_elements', 'is_standard',
                          'description', 'offsets', 'scales', 'no_data']:
                if not equal_metadata(getattr(before, field), getattr(after, field)):
                    report['header_errors'].append(f'dimension {name}: {field}')

        def vlrs(header):
            return [(v.user_id, v.record_id, v.record_data_bytes().hex())
                    for v in list(header.vlrs) + list(header.evlrs or [])
                    if v.user_id != 'laszip encoded'
                    and not (v.user_id == 'LASF_Spec' and v.record_id == 4)]

        if vlrs(sh) != vlrs(oh):
            report['header_errors'].append('non-compression VLR/EVLR content')
        for name in PREDICTIONS[:2]:
            if name in output_dims and oh.point_format.dimension_by_name(name).dtype.kind not in 'iu':
                report['label_errors'].append(name + ' must use an integer dtype')
        if not missing:
            for s, o in itertools.zip_longest(src.chunk_iterator(500000), out.chunk_iterator(500000)):
                if s is None or o is None or len(s) != len(o):
                    report['header_errors'].append('row count')
                    break
                report['point_count'] += len(s)
                for name in dims:
                    if name in PREDICTIONS:
                        continue
                    left, right = np.asarray(s[name]), np.asarray(o[name])
                    equal = left == right
                    if np.issubdtype(left.dtype, np.floating):
                        equal |= np.isnan(left) & np.isnan(right)
                    bad = int(np.count_nonzero(~equal))
                    if bad:
                        report['mismatched_dimensions'][name] = report['mismatched_dimensions'].get(name, 0) + bad
                report['finite_scores'] &= bool(np.isfinite(o[PREDICTIONS[2]]).all())
                semantic, instances = np.asarray(o[PREDICTIONS[0]]), np.asarray(o[PREDICTIONS[1]])
                for invalid, message in [
                    (not np.isfinite(semantic).all() or bool(((semantic < 0) | (semantic >= semantic_classes)).any()) or not np.equal(semantic, np.floor(semantic)).all(), 'semantic labels must be integer class IDs in configured range'),
                    (not np.isfinite(instances).all() or bool((instances < 0).any()) or not np.equal(instances, np.floor(instances)).all(), 'instance labels must be finite nonnegative integers'),
                ]:
                    if invalid and message not in report['label_errors']:
                        report['label_errors'].append(message)
                if np.isfinite(instances).all() and np.equal(instances, np.floor(instances)).all():
                    ids, counts = np.unique(instances, return_counts=True)
                    histogram.update(dict(zip(map(int, ids), map(int, counts))))
    report['instances'] = {str(k): v for k, v in sorted(histogram.items())}
    report['passed'] = not report['header_errors'] and not report['label_errors'] and not report['mismatched_dimensions'] and report['finite_scores']
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('original')
    parser.add_argument('output')
    parser.add_argument('report')
    parser.add_argument('--semantic-classes', type=int, default=3)
    args = parser.parse_args()
    if args.semantic_classes <= 0:
        parser.error('--semantic-classes must be positive')
    result = validate(args.original, args.output, args.semantic_classes)
    Path(args.report).write_text(json.dumps(result, indent=2))
    print(json.dumps(result))
    raise SystemExit(0 if result['passed'] else 1)
