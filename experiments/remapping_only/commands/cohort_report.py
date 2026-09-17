"""Summarize controlled repetitions without treating points as independent trials."""
import argparse,csv,json,pathlib,statistics
from datetime import datetime
from zoneinfo import ZoneInfo

METRICS={
 'semantic_disagreement':('semantic_agreement',True),
 'assigned_unassigned_disagreement':('assigned_unassigned_agreement',True),
 'instance_disagreement':('instance_agreement_matched_including_background',True),
 'one_minus_ari':('ari_including_background',True),
 'score_mae':('score_mae',False),
}

def local_changes(comparison):
    rows=comparison['contingency_rows'];cols=comparison['contingency_columns']
    matrix=comparison['contingency_counts']
    positive_rows=[i for i,v in enumerate(rows) if v>0]
    positive_cols=[j for j,v in enumerate(cols) if v>0]
    zero_col=cols.index(0) if 0 in cols else None
    foreground_totals={j:sum(matrix[i][j] for i in positive_rows) for j in positive_cols}
    result={}
    for i in positive_rows:
        total=sum(matrix[i])
        foreground=[matrix[i][j] for j in positive_cols]
        mixing=sum(matrix[i][j]*(foreground_totals[j]-matrix[i][j])/max(1,foreground_totals[j])
                   for j in positive_cols)/max(1,total)
        result[str(rows[i])]=dict(
            fragmentation=(sum(foreground)-max(foreground,default=0))/max(1,total),
            lost_to_background=(matrix[i][zero_col] if zero_col is not None else 0)/max(1,total),
            mixing_with_other_reference_trees=mixing)
    return result

def envelope(reference,candidate):
    if not reference or not candidate:return {'status':'insufficient comparisons'}
    high=max(reference)
    exceeds=[i+1 for i,v in enumerate(candidate) if v>high+1e-12]
    return dict(baseline_min=min(reference),baseline_max=high,
                candidate_min=min(candidate),candidate_max=max(candidate),
                candidate_exceedances=exceeds,
                persistent_exceedance=len(exceeds)==len(candidate))

def active_gpu_samples(samples, state, timezone):
    start=datetime.fromisoformat(state['StartedAt'].replace('Z','+00:00')).timestamp()
    end=datetime.fromisoformat(state['FinishedAt'].replace('Z','+00:00')).timestamp()
    zone=ZoneInfo(timezone)
    active=[row for row in samples if start <= datetime.strptime(
        row['timestamp'], '%Y/%m/%d %H:%M:%S.%f').replace(tzinfo=zone).timestamp() <= end]
    if not active:raise ValueError('No GPU samples within container runtime')
    return active, end-start

def summarize(root, candidate_variant='corrected'):
    root=pathlib.Path(root);manifest=json.loads((root/'manifest.json').read_text())
    runs=[]
    for stage in manifest['stages']:
        if stage['status']!='completed':continue
        name=stage['name'];resource=root/'logs'/f'{name}.resources.json'
        validation=root/'logs'/f'{name}.validation.json'
        if not resource.exists() or not validation.exists():continue
        r=json.loads(resource.read_text());v=json.loads(validation.read_text())
        profile=json.loads((root/'logs'/f'{name}.model_profile.json').read_text())
        summary=json.loads((root/'outputs'/name/'summary.json').read_text())
        with (root/'logs'/f'{name}.gpu.csv').open() as f:gpu=list(csv.DictReader(f))
        state=json.loads((root/'logs'/f'{name}.container.json').read_text())['State']
        raw_samples=len(gpu)
        gpu,container_seconds=active_gpu_samples(gpu,state,json.loads((root/'provenance/telemetry-window.json').read_text())['timezone'])
        utilization=[float(row[' utilization.gpu [%]'].split()[0]) for row in gpu]
        memory=[float(row[' memory.used [MiB]'].split()[0]) for row in gpu]
        runs.append(dict(label=name,variant=stage['variant'],passed_preservation=v['passed'],
                         seconds=r['elapsed_seconds'],host_seconds=stage['host_elapsed_seconds'],
                         container_seconds=container_seconds,raw_gpu_samples=raw_samples,active_gpu_samples=len(gpu),
                         peak_host_bytes=int(r['memory.peak']),
                         peak_allocated_bytes=profile['peak_allocated_bytes'],
                         peak_reserved_bytes=profile['peak_reserved_bytes'],
                         sampled_peak_device_mib=max(memory),
                         mean_gpu_utilization=statistics.mean(utilization),
                         trees=summary['files'][0]['positive_instance_count']))
    baseline=[];candidate=[]
    for label,dest in [('main',baseline),(candidate_variant,candidate)]:
        for p in sorted((root/'logs').glob(f'main01_vs_{label}[0-9][0-9].json')):
            dest.append(json.loads(p.read_text()))
    global_screen={}
    for metric,(field,invert) in METRICS.items():
        value=lambda d:1-d[field] if invert else d[field]
        global_screen[metric]=envelope([value(d) for d in baseline],[value(d) for d in candidate])
    baseline_local=[local_changes(d) for d in baseline]
    candidate_local=[local_changes(d) for d in candidate]
    local_screen={}
    if baseline_local and candidate_local:
        for tree in baseline_local[0]:
            local_screen[tree]={metric:envelope([d[tree][metric] for d in baseline_local],
                                              [d[tree][metric] for d in candidate_local])
                                for metric in baseline_local[0][tree]}
    performance={}
    for variant in ['main',candidate_variant]:
        items=[r for r in runs if r['variant']==variant]
        if items:
            times=[r['seconds'] for r in items]
            performance[variant]=dict(runs=len(items),mean_seconds=statistics.mean(times),
                                     median_seconds=statistics.median(times),
                                     min_seconds=min(times),max_seconds=max(times),
                                     mean_gpu_utilization=statistics.mean(r['mean_gpu_utilization'] for r in items),
                                     mean_container_seconds=statistics.mean(r['container_seconds'] for r in items),
                                     mean_host_seconds=statistics.mean(r['host_seconds'] for r in items),
                                     max_host_bytes=max(r['peak_host_bytes'] for r in items),
                                     max_sampled_device_mib=max(r['sampled_peak_device_mib'] for r in items),
                                     tree_counts=[r['trees'] for r in items])
    complete=all(performance.get(v,{}).get('runs',0)>=5 for v in ['main',candidate_variant])
    result=dict(input=manifest['input'],crop=manifest['crop'],runs=runs,performance=performance,
                global_observed_envelopes=global_screen,local_observed_envelopes=local_screen,
                cohort_complete=complete and len(baseline)>=4 and len(candidate)>=5 and all(r['passed_preservation'] for r in runs if r['variant'] in ['main',candidate_variant]),
                candidate_variant=candidate_variant,
                interpretation='Descriptive observed-envelope screen against the first unchanged run. Four baseline comparisons share that anchor; they are not four independent pairwise experiments. Five runs do not establish statistical equivalence. Report isolated exceedances as unresolved and persistent candidate-only local changes as requiring investigation. Raw per-tree contingencies and spatial bounds remain in comparison files.')
    (root/'cohort-summary.json').write_text(json.dumps(result,indent=2))
    return result

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('root');p.add_argument('--candidate',default='corrected');a=p.parse_args()
    result=summarize(a.root,a.candidate)
    print(json.dumps({k:result[k] for k in ['cohort_complete','performance','global_observed_envelopes']},indent=2))
