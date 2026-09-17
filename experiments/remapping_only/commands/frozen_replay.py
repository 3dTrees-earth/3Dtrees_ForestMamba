"""Replay only original projection/merging with identical saved predictions."""
import importlib.util,json,os,time,types
from pathlib import Path
import numpy as np
import torch
from torch_cluster import knn
from original_overlap_merge import merge_overlapping_instances_by_score_speedup
root=Path('/run');request=json.loads(Path(os.environ['FM_REPLAY_REQUEST']).read_text());label=request['capture'];folder=Path('/fixtures')/label/'frozen';meta=json.loads((folder/'metadata.json').read_text());done=json.loads((folder/'complete.json').read_text())
assert done['completed'] and done['windows_captured']==meta['windows_expected']
out=root/'outputs'/os.environ['FM_REPLAY_LABEL'];out.mkdir()
raw=torch.load(folder/'original_points.pt',map_location='cpu').cuda();expected=np.load(folder/'original_final.npz')
spec=importlib.util.spec_from_file_location('hybrid',Path('/workspace/oneformer3d/hybrid_nearest.py'));hybrid=importlib.util.module_from_spec(spec);spec.loader.exec_module(hybrid)
window_code=compile((root/'commands/original_window_merge.py').read_text(),'original_window_merge.py','exec');global_code=compile((root/'commands/original_global_merge.py').read_text(),'original_global_merge.py','exec')
report={'capture':label,'points':len(raw),'windows':done['windows_captured'],'runs':[],'completed':False}
def save(): (out/'results.json').write_text(json.dumps(report,indent=2))
with torch.no_grad():
 for mode in ['cached','original_a','hybrid_a','hybrid_b','original_b']:
  env=dict(np=np,torch=torch,device=raw.device,score_th2=meta['score_th2'],score_th1=meta['score_th1'],edge_thresh_sq=meta['edge_thresh_sq'],votes_counter=np.zeros((len(raw),meta['num_cls']),dtype=np.int16),all_pre_ins=np.full(len(raw),-1,dtype=np.int64),global_instance_scores=np.zeros(len(raw),dtype=float),best_masks=[],max_instance=0,self=types.SimpleNamespace(merge_overlapping_instances_by_score_speedup=merge_overlapping_instances_by_score_speedup))
  run={'mode':mode,'windows':[]};start=time.perf_counter()
  for p in sorted(folder.glob('window_*.pt')):
   d=torch.load(p,map_location='cpu');refs=d['pc3'].cuda();ids=d['pc1_indices'].cuda();queries=raw[ids];saved=d['nn_idx_pc1'].cuda()
   torch.cuda.synchronize();t=time.perf_counter();stats={}
   if mode=='cached':mapped=saved
   elif mode.startswith('original'):mapped=knn(refs.float(),queries.float(),k=1)[1]
   else:mapped,stats=hybrid.hybrid_nearest(refs.float(),queries.float(),workers=10)
   torch.cuda.synchronize();seconds=time.perf_counter()-t
   ndiff=int((mapped!=saved).sum());assert ndiff==0,(mode,d['window'],ndiff)
   env.update(pc1=queries,pc1_indices=ids,pc3=refs,region=d['region'],nn_idx_pc1=mapped,masks_t=d['masks'].cuda(),scores_t=d['scores'].cuda(),results_list=[types.SimpleNamespace(pts_semantic_mask=[d['sem_pred_pc3'].cuda()])])
   exec(window_code,env)
   ties=0
   if env['keep'].numel():
    ties=int((torch.bincount(env['cols'][env['improved']],minlength=len(queries))>1).sum())
   run['windows'].append(dict(window=d['window'],queries=len(queries),mapping_seconds=seconds,mapping_differences=ndiff,winning_score_tie_points=ties,hybrid_stats=stats))
   # Drop transient GPU tensors; retain only the original accumulated CPU buffers.
   for k in ['pc1','pc1_indices','pc3','nn_idx_pc1','masks_t','scores_t','results_list','masks_kept','scores_kept','mk_bool','rows','cols','score_per_hit','best_score','best_mid','improved','sem_pred_pc3','cyl_sem_pc1','pc3_dist_sq','edge_pts','touches_edge','keep']:
    env.pop(k,None)
   del refs,ids,queries,saved,mapped,d
  exec(global_code,env)
  arrays=dict(semantic=env['final_semantic_labels'],instances=env['clean_all_pre_ins'],scores=env['merged_instance_scores'])
  np.savez(out/(mode+'.npz'),**arrays)
  run['seconds_including_fixture_io']=time.perf_counter()-start;run['mapping_seconds']=sum(w['mapping_seconds'] for w in run['windows'])
  run['final_comparison']={k:dict(exact=bool(np.array_equal(a,expected[k])),different_points=int(np.count_nonzero(a!=expected[k])),max_absolute=float(np.max(np.abs(a-expected[k])))) for k,a in arrays.items()}
  report['runs'].append(run);save();print('REPLAY',mode,'mapping_s',run['mapping_seconds'],'final',run['final_comparison'],flush=True)
  assert all(v['exact'] for v in run['final_comparison'].values()),'Original merge replay differs from captured output; inspect tie counts and retained arrays'
 report['completed']=True;save();print('FROZEN_MERGE_REPLAY_PASSED',flush=True)
