final_semantic_labels = votes_counter.argmax(1)
final_semantic_labels[votes_counter.sum(1) == 0] = -1

# (B) Wipe instance labels on ground points.
all_pre_ins[final_semantic_labels == 0] = -1

# (C) Drop tiny instances (<10 points).
uniq, cnt = np.unique(all_pre_ins, return_counts=True)
to_kill = np.isin(all_pre_ins, uniq[(cnt < 10) & (uniq != -1)])
all_pre_ins[to_kill] = -1

# (D) Drop mask records whose instance id is no longer present.
unique_best_masks = [
    (pts, iid, sc) for (pts, iid, sc) in best_masks
    if np.any(all_pre_ins[pts] == iid)
]

# (E) Merge overlapping instances coming from different windows.
clean_all_pre_ins, merged_masks, merged_instance_scores = \
    self.merge_overlapping_instances_by_score_speedup(
        all_pre_ins, unique_best_masks, overlap_threshold=score_th2)

# (F) Relabel instance ids to a dense 1..M range (bg stays -1).
# IDs start from 1 to match the GT convention (0 = background) so that
# the evaluator's `ins_arr > 0` filter correctly captures all instances.
# Array-LUT lookup is much faster than np.vectorize(dict.get).
# Shift by +1 so bg (-1) maps to lut[0] = -1.
max_id = int(clean_all_pre_ins.max()) if clean_all_pre_ins.size else -1
lut = np.full(max_id + 2, -1, dtype=np.int64)
pos_ids = np.unique(clean_all_pre_ins[clean_all_pre_ins >= 0])
lut[pos_ids + 1] = np.arange(1, pos_ids.size + 1)
clean_all_pre_ins = lut[clean_all_pre_ins + 1]
