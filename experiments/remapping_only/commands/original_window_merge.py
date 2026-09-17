pc3_dist_sq = ((pc3[:, 0] - region[0]) ** 2 +
               (pc3[:, 1] - region[1]) ** 2)
edge_pts = pc3_dist_sq > edge_thresh_sq                  # (N_pc3,)
touches_edge = (masks_t & edge_pts.unsqueeze(0)).any(1)  # (K,)
keep = torch.where(
    (scores_t > score_th1) & ~touches_edge)[0]

if keep.numel():
    masks_kept  = masks_t[keep]                   # (K',N_pc3)
    scores_kept = scores_t[keep]                  # (K',)

    # (8) Project kept masks onto pc1 via the NN index: each pc1
    # point inherits the masks of its nearest pc3 voxel.
    mk_bool = masks_kept[:, nn_idx_pc1]           # (K',N_pc1)
    rows, cols = mk_bool.nonzero(as_tuple=True)
    score_per_hit = scores_kept[rows]             # (nnz,)

    # (9) For each pc1 point pick the highest-scoring kept mask.
    N1 = pc1.shape[0]
    best_score = torch.full((N1,), -1.0, device=device)
    best_mid   = torch.full(
        (N1,), -1, dtype=torch.long, device=device)
    best_score.index_reduce_(
        0, cols, score_per_hit, reduce='amax')
    improved = score_per_hit == best_score[cols]
    best_mid.index_put_(
        (cols[improved],), rows[improved], accumulate=False)

    # (10) Merge into globals: only overwrite a point if this
    # window's score beats whatever a previous (overlapping)
    # window wrote for it.
    pts_glob   = pc1_indices.cpu().numpy()
    new_scores = best_score.cpu().numpy()
    better_pts = new_scores > global_instance_scores[pts_glob]

    if better_pts.any():
        best_mid_np     = best_mid.cpu().numpy()
        scores_kept_np  = scores_kept.cpu().numpy()
        global_instance_scores[pts_glob[better_pts]] = \
            new_scores[better_pts]
        all_pre_ins[pts_glob[better_pts]] = (
            max_instance + best_mid_np[better_pts])

        # Record the FULL pc1 point list for each mid that
        # contributed at least one "better" point.
        # Using mk_bool[mid] directly avoids re-scanning rows/cols.
        for mid in np.unique(best_mid_np[better_pts]).tolist():
            sel_pts = mk_bool[mid].nonzero(
                as_tuple=True)[0].cpu().numpy()
            best_masks.append((
                pts_glob[sel_pts],
                max_instance + int(mid),
                float(scores_kept_np[int(mid)])))

    max_instance += int(masks_kept.size(0))

# (11) Semantic voting using the decoder's semantic head.
sem_pred_pc3 = results_list[0].pts_semantic_mask[0]
if isinstance(sem_pred_pc3, np.ndarray):
    sem_pred_pc3 = torch.from_numpy(sem_pred_pc3).to(device)
cyl_sem_pc1 = sem_pred_pc3[nn_idx_pc1]

ids_np = pc1_indices.cpu().numpy()
sem_np = cyl_sem_pc1.cpu().numpy().astype(int)
np.add.at(votes_counter, (ids_np, sem_np), 1)
