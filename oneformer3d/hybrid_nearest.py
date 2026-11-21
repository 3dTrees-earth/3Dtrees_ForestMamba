"""Experimental 3D float32 CUDA mapping with indexed search and legacy fallback.

Used only for raw-window remapping. The ambiguity guard is validated against
pinned torch_cluster; broad numerical/hardware acceptance is still required.
"""
import numpy as np
import torch
from scipy.spatial import cKDTree
from torch_cluster import knn

# Broad guard over the handful of FP32 operations in a 3D squared distance.
# Also guard underflow/flush-to-zero rather than relying on a relative gap alone.
RELATIVE_GUARD=64*np.finfo(np.float32).eps
ABSOLUTE_GUARD=64*np.finfo(np.float32).tiny

def hybrid_nearest(refs,queries,workers=10,chunk_size=1000000):
    if workers<1 or chunk_size<1:raise ValueError('workers and chunk_size must be positive')
    for a in (refs,queries):
        if a.dtype!=torch.float32 or a.ndim!=2 or a.shape[1]!=3 or not a.is_cuda:
            raise ValueError('Expected CUDA float32 XYZ tensors')
    if refs.device!=queries.device:raise ValueError('Devices must match')
    if len(refs)==0:raise ValueError('At least one reference is required')
    out=torch.empty(len(queries),device=queries.device,dtype=torch.long)
    stats={'queries':len(queries),'fallback_queries':0,'relative_guard':float(RELATIVE_GUARD),
           'absolute_guard':float(ABSOLUTE_GUARD),'chunks':0}
    if not len(queries):return out,stats
    x=refs.detach().cpu().numpy();y=queries.detach().cpu().numpy()
    # Keep the prototype's indexed path within the local-coordinate regime tested.
    # Unsupported geometry takes exactly the existing whole-input operation.
    if (not np.isfinite(x).all() or not np.isfinite(y).all() or
            np.max(np.abs(x))>10000 or np.max(np.abs(y))>10000):
        stats['fallback_queries']=len(queries);stats['reason']='outside tested coordinate domain'
        return knn(refs.contiguous(),queries.contiguous(),k=1)[1],stats
    tree=cKDTree(x)
    for start in range(0,len(queries),chunk_size):
        stop=min(start+chunk_size,len(queries));stats['chunks']+=1
        distance,index=tree.query(y[start:stop],k=2,eps=0,p=2,workers=workers)
        first=distance[:,0]**2;second=distance[:,1]**2
        # Infinite second distance represents the single-reference case.
        ambiguous=np.zeros(len(first),dtype=bool)
        finite=np.isfinite(second)
        ambiguous[finite]=(second[finite]-first[finite]) <= RELATIVE_GUARD*second[finite]+ABSOLUTE_GUARD
        out[start:stop]=torch.from_numpy(np.ascontiguousarray(index[:,0])).to(refs.device)
        rows=np.flatnonzero(ambiguous)
        if len(rows):
            locations=torch.from_numpy(rows+start).to(refs.device)
            # Search ALL original references, in unchanged order, with exactly the
            # same extension. This handles duplicates and multiway ties as well.
            out[locations]=knn(refs.contiguous(),queries[locations].contiguous(),k=1)[1]
            stats['fallback_queries']+=len(rows)
    return out,stats
