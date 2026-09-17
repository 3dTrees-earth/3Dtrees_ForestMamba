"""Load small outputs once, compute exact contingencies, verify one streamed pair."""
import argparse,csv,hashlib,itertools,json,statistics,subprocess,sys
from pathlib import Path
import laspy,numpy as np
from scipy.optimize import linear_sum_assignment
from cohort_report import METRICS,local_changes,envelope,active_gpu_samples

def compare(a,b):
 i,j=a['instance'],b['instance'];n=len(i);assert n==len(j)
 rows,ri=np.unique(i,return_inverse=True);cols,ci=np.unique(j,return_inverse=True)
 table=np.bincount(ri*len(cols)+ci,minlength=len(rows)*len(cols)).reshape(len(rows),len(cols))
 rp=np.flatnonzero(rows>0);cp=np.flatnonzero(cols>0);sub=table[np.ix_(rp,cp)];u,v=linear_sum_assignment(-sub);matched=int(sub[u,v].sum())
 bg=int(table[np.flatnonzero(rows==0)[0],np.flatnonzero(cols==0)[0]]) if 0 in rows and 0 in cols else 0
 def comb(x):return sum(int(k)*(int(k)-1)//2 for k in x.ravel())
 total=n*(n-1)//2;obs=comb(table);rr=comb(table.sum(1));cc=comb(table.sum(0));exp=rr*cc/total if total else 0;den=.5*(rr+cc)-exp
 delta=np.abs(a['score'].astype('f8')-b['score']);assert np.isfinite(delta).all()
 return dict(points=n,semantic_agreement=float(np.mean(a['semantic']==b['semantic'])),assigned_unassigned_agreement=float(np.mean((i>0)==(j>0))),instance_agreement_matched_including_background=(matched+bg)/n,ari_including_background=(obs-exp)/den if den else 1.,score_mae=float(delta.mean()),score_max_difference=float(delta.max()),contingency_rows=rows.tolist(),contingency_columns=cols.tolist(),contingency_counts=table.tolist())
def main():
 p=argparse.ArgumentParser();p.add_argument('root',type=Path);p.add_argument('group');args=p.parse_args();r=args.root;group=args.group
 m=json.loads((r/'manifest.json').read_text());cohort=next(x for x in m['cohorts'] if x['group']==group);labels=cohort['labels'];assert len(labels)==10
 arrays={};coordhash=None
 for label in labels:
  d=laspy.read(r/'outputs'/label/'input_forestmamba.laz');h=hashlib.sha256()
  for dim in ['X','Y','Z']:h.update(np.asarray(d[dim]).tobytes())
  h.update(d.header.scales.tobytes());h.update(d.header.offsets.tobytes())
  if coordhash is None:coordhash=h.hexdigest()
  assert h.hexdigest()==coordhash
  arrays[label]=dict(instance=np.array(d['PredInstance_FM'],dtype=np.int64),semantic=np.array(d['PredSemantic_FM']),score=np.array(d['PredScore_FM']));del d
 pairs={}
 for a,b in itertools.combinations(labels,2):
  result=compare(arrays[a],arrays[b]);result.update(reference=a,candidate=b);pairs[a+'__'+b]=result
  (r/'logs'/f'{a}_vs_{b}.json').write_text(json.dumps(result,indent=2))
 mains=[x for x in labels if '_main_' in x];remaps=[x for x in labels if '_remap_' in x]
 def pair(a,b):return pairs.get(a+'__'+b,pairs.get(b+'__'+a))
 # Scalar metrics are symmetric; anchored local screens explicitly retain main_01 first.
 original=[pair(a,b) for a,b in itertools.combinations(mains,2)];candidate=[pair(a,b) for a,b in itertools.combinations(remaps,2)];cross=[pair(a,b) for a,b in itertools.product(mains,remaps)]
 screens={}
 for name,values in [('remap_within',candidate),('original_vs_remap',cross)]:
  screens[name]={}
  for metric,(field,invert) in METRICS.items():
   val=lambda d:1-d[field] if invert else d[field]
   screens[name][metric]=envelope([val(d) for d in original],[val(d) for d in values])
 bl=[local_changes(pair(mains[0],x)) for x in mains[1:]];cl=[local_changes(pair(mains[0],x)) for x in remaps]
 local={t:{k:envelope([d[t][k] for d in bl],[d[t][k] for d in cl]) for k in bl[0][t]} for t in bl[0]}
 oracle=r/'logs'/f'{group}.streamed-verification.json'
 subprocess.run([sys.executable,str(r/'commands/compare_same_rows.py'),str(r/'outputs'/mains[0]/'input_forestmamba.laz'),str(r/'outputs'/mains[1]/'input_forestmamba.laz'),str(oracle)],check=True,stdout=subprocess.DEVNULL)
 expected=json.loads(oracle.read_text());actual=pair(mains[0],mains[1])
 for field,_ in METRICS.values():assert abs(expected[field]-actual[field])<1e-12,(field,expected[field],actual[field])
 assert expected['contingency_counts']==actual['contingency_counts']
 performance={};runs=[]
 for label in labels:
  resource=json.loads((r/'logs'/f'{label}.resources.json').read_text());stages=[json.loads(l) for l in (r/'logs'/f'{label}.stages.jsonl').read_text().splitlines()];state=json.loads((r/'logs'/f'{label}.container.json').read_text())['State']
  with (r/'logs'/f'{label}.gpu.csv').open() as f:samples=list(csv.DictReader(f))
  # Stop telemetry at the final successful workflow stage, not stale daemon cleanup.
  gpu_window={'StartedAt':state['StartedAt'],'FinishedAt':stages[-1]['ended_at']}
  active,_=active_gpu_samples(samples,gpu_window,m.get('gpu_timezone','Europe/Berlin'))
  runs.append(dict(label=label,gpu_sample_window=gpu_window,seconds=resource['elapsed_seconds'],stages={d['stage']:d['duration_seconds'] for d in stages if d['status']=='success'},mean_gpu_utilization=statistics.mean(float(d[' utilization.gpu [%]'].split()[0]) for d in active),peak_device_mib=max(float(d[' memory.used [MiB]'].split()[0]) for d in active)))
 for name,ls in [('main',mains),('remap',remaps)]:
  times=[x['seconds'] for x in runs if x['label'] in ls];performance[name]=dict(mean_seconds=statistics.mean(times),median_seconds=statistics.median(times),min_seconds=min(times),max_seconds=max(times))
 excess=any(d['candidate_exceedances'] for groupmetrics in screens.values() for d in groupmetrics.values()) or any(d['persistent_exceedance'] for metrics in local.values() for d in metrics.values())
 cross_excess=any(v['candidate_exceedances'] for v in screens['original_vs_remap'].values()) or any(v['persistent_exceedance'] for metrics in local.values() for v in metrics.values())
 result=dict(cross_variant_excess_observed=bool(cross_excess),group=group,points=len(arrays[labels[0]]['instance']),pairs=45,performance=performance,runs=runs,global_screens=screens,local_screens=local,streamed_comparator_verified=True,excess_variation_observed=bool(excess),interpretation='Observed envelope, not a significance test. Ten original pairs share five runs; cross pairs are correlated. Cropping changes spatial context; diagnosis must be confirmed with controlled replays.')
 (r/'outputs'/f'{group}.comparison.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:result[k] for k in ['group','points','performance','excess_variation_observed','global_screens']},indent=2))
if __name__=='__main__':main()
