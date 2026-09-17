import csv,json,re,statistics
from datetime import datetime,timezone
from pathlib import Path
from cohort_report import active_gpu_samples

r=Path(__file__).resolve().parents[1]
m=json.loads((r/'manifest.json').read_text());a=json.loads((r/'outputs/tree68_10m.comparison.json').read_text())
assert len(m['stages'])==10 and a['pairs']==45 and a['streamed_comparator_verified']
replay=json.loads((r/'outputs/image_replay/results.json').read_text());assert replay['completed']
assert all(json.loads((r/'logs'/(s['name']+'.validation.json')).read_text())['passed'] for s in m['stages'])
images={}
for variant in ['main','remap']:
 check=json.loads((r/'logs'/('verify_'+variant+'.log')).read_text());assert check['passed'];images[variant]=check
out={'image_verification':images,'images':m['images'],'tags':m['tags'],'runs':a['runs'],'performance':{},'global_screens':a['global_screens'],'local_screens':a['local_screens'],'cross_variant_excess_observed':a['cross_variant_excess_observed'],'excess_variation_observed':a['excess_variation_observed'],'packaged_helper_exact_replay_passed':True,'input_preservation_passes':10,'pairs':45,'scope':'Five fresh runs per image on tree68_10m (7,008,010 points); descriptive repeatability screen, not proof of equivalence or ground-truth accuracy.'}
for variant in ['main','remap']:
 runs=[v for v in a['runs'] if '_'+variant+'_' in v['label']]
 v=a['performance'][variant].copy()
 host_times=[json.loads((r/'logs'/(x['label']+'.stage.json')).read_text())['host_elapsed_seconds'] for x in runs]
 v['mean_launch_to_exit_seconds']=statistics.mean(host_times)
 v['median_launch_to_exit_seconds']=statistics.median(host_times)
 loop_times=[]
 for x in runs:
  log=(r/'logs'/(x['label']+'.log')).read_text()
  hits=re.findall(r'Epoch\(test\).*?\btime: ([0-9.]+)',log)
  if hits:loop_times.append(float(hits[-1]))
 v['inference_loop_seconds']=loop_times
 v['mean_inference_loop_seconds']=statistics.mean(loop_times) if loop_times else None
 v['median_inference_loop_seconds']=statistics.median(loop_times) if loop_times else None
 for x in runs:
  stages=[json.loads(line) for line in (r/'logs'/(x['label']+'.stages.jsonl')).read_text().splitlines()]
  stage=next(t for t in stages if t['stage']=='model_inference' and t['status']=='success')
  with (r/'logs'/(x['label']+'.gpu.csv')).open() as f:samples=list(csv.DictReader(f))
  active,_=active_gpu_samples(samples,{'StartedAt':stage['started_at'],'FinishedAt':stage['ended_at']},m.get('gpu_timezone','Europe/Berlin'))
  x['inference_process_gpu_utilization']=statistics.mean(float(row[' utilization.gpu [%]'].split()[0]) for row in active)
 v['mean_inference_process_gpu_utilization']=statistics.mean(x['inference_process_gpu_utilization'] for x in runs)
 v['mean_gpu_utilization']=statistics.mean(x['mean_gpu_utilization'] for x in runs)
 v['peak_sampled_device_mib']=max(x['peak_device_mib'] for x in runs)
 v['mean_stage_seconds']={stage:statistics.mean(x['stages'][stage] for x in runs) for stage in runs[0]['stages']}
 v['peak_host_bytes']=max(int(json.loads((r/'logs'/(x['label']+'.resources.json')).read_text())['memory.peak']) for x in runs)
 out['performance'][variant]=v
out['runtime_interruption']=json.loads((r/'provenance/runtime-interruption.json').read_text()) if (r/'provenance/runtime-interruption.json').exists() else None
out['packaged_lookup_mean_seconds']={v:statistics.mean(x['mapping_seconds'] for x in replay['runs'] if x['mode'].startswith(v)) for v in ['original','hybrid']}
out['mean_time_saved_percent']=100*(1-out['performance']['remap']['mean_seconds']/out['performance']['main']['mean_seconds'])
out['median_time_saved_percent']=100*(1-out['performance']['remap']['median_seconds']/out['performance']['main']['median_seconds'])
out['adoption_status']='not accepted: excess variation observed' if out['excess_variation_observed'] else 'no excess observed in this small cohort; full-cloud stability remains unproven'
(r/'benchmark-summary.json').write_text(json.dumps(out,indent=2))
lines=['# Remapping-only Docker benchmark','','Two self-contained local images were built and verified. The candidate changes only one import and one nearest-neighbor call in the original model, plus remapping.py and hybrid_nearest.py. It preserves the original averaging and sparse sequence-ordering code, model weights, decoder and merge logic. Those original operations remain nondeterministic; unchanged code does not guarantee identical realized model inputs or sequences across fresh runs. The common runtime and LAS/LAZ wrapper match the prior controlled benchmark. No model source/configuration/checkpoint bind mounts were used.','','## Results','','| Metric | Original | Remapping only |','|---|---:|---:|']
for title,key in [('Mean processing runtime (s)','mean_seconds'),('Median processing runtime (s)','median_seconds'),('Minimum processing runtime (s)','min_seconds'),('Maximum processing runtime (s)','max_seconds'),('Mean launch-to-exit wall time (s)','mean_launch_to_exit_seconds'),('Median launch-to-exit wall time (s)','median_launch_to_exit_seconds'),('Mean framework test iteration (s)','mean_inference_loop_seconds'),('Median framework test iteration (s)','median_inference_loop_seconds'),('Mean workflow-window GPU utilization (%)','mean_gpu_utilization'),('Mean inference-process GPU utilization (%)','mean_inference_process_gpu_utilization'),('Peak sampled GPU memory (MiB)','peak_sampled_device_mib')]:
 lines.append(f"| {title} | {out['performance']['main'][key]:.2f} | {out['performance']['remap'][key]:.2f} |")
lines+=['',f"Observed mean processing-time reduction: {out['mean_time_saved_percent']:.2f}%; observed median processing-time reduction: {out['median_time_saved_percent']:.2f}%.",'','| Mean stage time (s) | Original | Remapping only |','|---|---:|---:|']
for stage in out['performance']['main']['mean_stage_seconds']:
 lines.append(f"| {stage} | {out['performance']['main']['mean_stage_seconds'][stage]:.3f} | {out['performance']['remap']['mean_stage_seconds'][stage]:.3f} |")
lines+=['','The runtime differences are observational. Large startup/block-I/O variation, together with the lookup-only measurement in this run, prevents attributing the full end-to-end difference to remapping. No repeatable end-to-end speedup is established by this small shared-host cohort.','','## Individual runs in execution order','','| Run | Processing (s) | Inference process (s) | GPU utilization (%) |','|---|---:|---:|---:|']
for run in out['runs']:
 lines.append(f"| {run['label']} | {run['seconds']:.3f} | {run['stages']['model_inference']:.3f} | {run['mean_gpu_utilization']:.2f} |")
lines+=['','## Result stability','',out['adoption_status']+'.','','| Metric | Original-original maximum | Original-remap maximum | Cross pairs above original maximum |','|---|---:|---:|---:|']
for metric,d in out['global_screens']['original_vs_remap'].items():
 lines.append(f"| {metric} | {d['baseline_max']:.8f} | {d['candidate_max']:.8f} | {len(d['candidate_exceedances'])}/25 |")
lines+=['',f"Baked-image exact replay lookup means: original {out['packaged_lookup_mean_seconds']['original']:.3f} s; hybrid {out['packaged_lookup_mean_seconds']['hybrid']:.3f} s. These timings come from this run, separately from the earlier frozen-prediction experiment.",'','Instance disagreement uses optimal label matching and includes background; it does not mistake arbitrary instance renumbering for changed segmentation. All 45 within-/cross-image pairs were compared. A separate streamed comparator verified the in-memory metrics. All ten outputs preserved the input points, order, attributes and relevant metadata. No ground-truth accuracy claim is made. Pairwise comparisons share runs and are correlated; five repetitions do not establish statistical equivalence.','','The baked candidate helper also passed the saved six-window exact replay, including original final merge: all indices, semantic labels, raw instance IDs and scores were identical. This isolates mapping equivalence from the nondeterministic model steps executed anew in the fresh cohort. Exact replay does not clear an observed full-inference stability failure. This cohort does not establish the causal mechanism behind the excess variation; it is not proof of a nearest-neighbor index error or of degraded ground-truth accuracy. No deterministic-mean, sequence-ordering or tie-breaking prototype was included.','','## Conditions and provenance','','Sequential GPU2 RTX A6000, 10 CPUs, 50 GB RAM, no swap, no container network; five fresh original and five fresh candidate containers with alternating pair order. v6_epoch_1500_fix.pth, normal v6 configuration, seed 1838, original nondeterministic execution policy. The crop was selected because it previously exposed excess variation.','','Processing runtime is wrapper execution time; launch-to-exit wall time also includes Docker creation/startup and shutdown; stages cover input conversion, preprocessing, model process and output export. GPU utilization is the arithmetic mean of per-run device samples restricted to container start through the final successful workflow stage; startup and workflow I/O idle time are included, while daemon cleanup after completion is excluded. Framework test-iteration time includes the model windows, final merge and other work inside the test loop; it is separate from model-process startup and is not GPU-kernel-only timing. First-call kernel initialization and output/evaluation work may affect it. GPU memory is a 200 ms sampled maximum, not a guaranteed allocation peak. This is not an actual Tesla T4 test. Timing is sensitive to shared-host I/O; report ranges and medians alongside means.','','| Image | Immutable local ID |','|---|---|']
for variant in ['main','remap']:lines.append(f"| `{m['tags'][variant]}` | `{m['images'][variant]}` |")
lines+=['',f"Base image: `{m['base_image']}`.",f"Original model source: `{m['source_commit']}`.",f"Checkpoint SHA-256: `{m['checkpoint_sha256']}`.",'',('A containerd restart delayed acknowledgement of original run 05 after its model process had completed successfully. Only that finished benchmark container was cleared; no shared service was restarted by this benchmark. Its launch-to-exit time is an infrastructure outlier, and GPU utilization excludes the stale post-completion interval. Details are retained in provenance/runtime-interruption.json.' if out['runtime_interruption'] else 'No runtime interruption was recorded for this run.'),'', 'The images are local and were not pushed or deployed. Dockerfile and complete build inputs are under build/. Image file hashes, inspection records and the two-line source patch are under provenance/. Exact commands, logs, telemetry, output LAZ files and comparator details are retained. Read README.md for usage.']
(r/'REPORT.md').write_text('\n'.join(lines)+'\n')
m.update(status='completed',completed_at=datetime.now(timezone.utc).isoformat(),benchmark_summary='benchmark-summary.json',report='REPORT.md',adoption_status=out['adoption_status']);(r/'manifest.json').write_text(json.dumps(m,indent=2))
print(json.dumps({k:out[k] for k in ['performance','mean_time_saved_percent','median_time_saved_percent','global_screens','adoption_status']},indent=2))
