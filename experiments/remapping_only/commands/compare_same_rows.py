"""Streaming label-invariant comparison on identical original point rows."""
import argparse,json,pathlib,time
import laspy,numpy as np
from scipy.optimize import linear_sum_assignment
p=argparse.ArgumentParser();p.add_argument('reference');p.add_argument('candidate');p.add_argument('report',type=pathlib.Path);a=p.parse_args()
start=time.perf_counter();counter={};bounds={};n=sem_eq=assigned_eq=0;score_sum=0.;score_max=0.
with laspy.open(a.reference) as ref,laspy.open(a.candidate) as cand:
    if ref.header.point_count!=cand.header.point_count:raise ValueError('Point counts differ')
    for x,y in zip(ref.chunk_iterator(500000),cand.chunk_iterator(500000)):
        if len(x)!=len(y) or any(not np.array_equal(np.asarray(x[d]),np.asarray(y[d])) for d in ["x","y","z"]):raise ValueError('Coordinates/order differ')
        i=np.asarray(x['PredInstance_FM'],dtype=np.int64);j=np.asarray(y['PredInstance_FM'],dtype=np.int64)
        xyz=np.column_stack([np.asarray(x[d]) for d in ['x','y','z']])
        ids,inverse=np.unique(i,return_inverse=True)
        lo=np.full((len(ids),3),np.inf);hi=np.full((len(ids),3),-np.inf)
        np.minimum.at(lo,inverse,xyz);np.maximum.at(hi,inverse,xyz)
        for k,ident in enumerate(ids):
            if ident<=0:continue
            key=int(ident)
            if key in bounds:
                bounds[key]=(np.minimum(bounds[key][0],lo[k]),np.maximum(bounds[key][1],hi[k]))
            else:bounds[key]=(lo[k],hi[k])
        pairs,counts=np.unique(np.stack([i,j],axis=1),axis=0,return_counts=True)
        for pair,c in zip(pairs,counts):key=tuple(map(int,pair));counter[key]=counter.get(key,0)+int(c)
        n+=len(x);sem_eq+=int(np.count_nonzero(x['PredSemantic_FM']==y['PredSemantic_FM']));assigned_eq+=int(np.count_nonzero((i>0)==(j>0)))
        d=np.abs(np.asarray(x['PredScore_FM'],dtype='f8')-np.asarray(y['PredScore_FM'],dtype='f8'))
        if not np.isfinite(d).all():raise ValueError('Non-finite scores')
        score_sum+=float(d.sum());score_max=max(score_max,float(d.max()))
rows=sorted({i for i,j in counter});cols=sorted({j for i,j in counter});ri={v:i for i,v in enumerate(rows)};ci={v:i for i,v in enumerate(cols)}
m=np.zeros((len(rows),len(cols)),dtype=np.int64)
for (i,j),count in counter.items():m[ri[i],ci[j]]=count
rp=np.array([i for i,v in enumerate(rows) if v>0],dtype=int);cp=np.array([i for i,v in enumerate(cols) if v>0],dtype=int)
sub=m[np.ix_(rp,cp)];lr,lc=linear_sum_assignment(-sub);matched=int(sub[lr,lc].sum());background=int(m[ri[0],ci[0]]) if 0 in ri and 0 in ci else 0
comb=lambda arr:sum(int(v)*(int(v)-1)//2 for v in np.asarray(arr).ravel())
total=n*(n-1)//2;observed=comb(m);r_pairs=comb(m.sum(1));c_pairs=comb(m.sum(0));expected=r_pairs*c_pairs/total if total else 0;den=.5*(r_pairs+c_pairs)-expected
ari=(observed-expected)/den if den else 1.
trees=[]
for k,v in enumerate(rows):
    if v<=0:continue
    amount=int(m[k].sum());targets=[{'candidate_id':cols[c],'overlap_points':int(m[k,c]),'reference_fraction':float(m[k,c]/amount)} for c in np.argsort(-m[k]) if m[k,c]>0]
    trees.append({'reference_id':v,'points':amount,'bounds_min':bounds[v][0].tolist(),'bounds_max':bounds[v][1].tolist(),'overlaps':targets})
r={'reference':a.reference,'candidate':a.candidate,'points':n,'semantic_agreement':sem_eq/n,'assigned_unassigned_agreement':assigned_eq/n,'instance_agreement_matched_including_background':(matched+background)/n,'foreground_matched_points':matched,'ari_including_background':ari,'score_mae':score_sum/n,'score_max_difference':score_max,'reference_trees':len(rp),'candidate_trees':len(cp),'per_reference_tree':trees,'contingency_rows':rows,'contingency_columns':cols,'contingency_counts':m.tolist(),'analysis_seconds':time.perf_counter()-start,'interpretation':'Same-row output comparison. Interpret differences against independently repeated unchanged-baseline runs on this exact input. Tree IDs are matched by overlap; local splits/merges and unmatched trees remain visible. A single comparison cannot establish stability.'}
a.report.write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k not in ['per_reference_tree','contingency_counts']},indent=2))
