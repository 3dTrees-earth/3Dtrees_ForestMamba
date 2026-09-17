import numpy as np
def merge_overlapping_instances_by_score_speedup(all_pre_ins, best_masks, overlap_threshold=0.3):
    N = all_pre_ins.shape[0]
    merged_instance_labels = np.full(N, -1, dtype=int)
    merged_instance_scores = np.full(N, -1.0, dtype=float)
    if not best_masks:
        return (merged_instance_labels, [], merged_instance_scores)
    best_masks = sorted(best_masks, key=lambda x: -x[2])
    taken_flag = np.zeros(N, dtype=np.bool_)
    kept_masks = []
    for pts_idx, inst_id, score in best_masks:
        pts_idx = np.asarray(pts_idx, dtype=int)
        overlap = taken_flag[pts_idx].mean()
        if overlap > overlap_threshold:
            continue
        kept_masks.append((pts_idx, inst_id, score))
        merged_instance_labels[pts_idx] = inst_id
        merged_instance_scores[pts_idx] = score
        taken_flag[pts_idx] = True
    return (merged_instance_labels, kept_masks, merged_instance_scores)
