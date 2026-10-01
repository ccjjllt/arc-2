import json,collections,itertools,hashlib
from pathlib import Path
exec(Path(__file__).with_name('arc009-semantics-audit.py').read_text().split("result={'status'")[0])
out=Path('experiments/009/episode-semantics.json');result=json.loads(out.read_text())
extra=['37d3e8b2','ed74f2f2','ac605cbb','6ea4a07e','17cae0c1']
def categories(fid,g,y):
 if fid=='00dbd492':return set('thick:'+str(v) for v in oracle(fid,g)[1]['thicknesses'])
 if fid=='ed74f2f2':return {'marker:'+''.join(str(g[r][c]) for r in [1,2,3] for c in [1,2,3])}
 if fid=='6ea4a07e':return {'input_color:'+str(v) for row in g for v in row if v}
 if fid=='17cae0c1':return {'shape:'+''.join(str(g[r][c+offset]) for r in range(3) for c in range(3)) for offset in [0,3,6]}
 if fid=='ac605cbb':return ({'input_color:'+str(v) for row in g for v in row if v}|({'join_branch'} if any(4 in row for row in y) else set()))
 if fid=='37d3e8b2':
  cs=set()
  for co in components(g,8):
   r0=min(r for r,c in co);r1=max(r for r,c in co);c0=min(c for r,c in co);c1=max(c for r,c in co)
   crop=[row[c0:c1+1] for row in g[r0:r1+1]]
   cs.add('hole_count:'+str(len(components(crop,0))))
  return cs
 raise ValueError(fid)
flags=[];repair=[]
for line,b in enumerate(rows,1):
 fid=b['family_id']
 demo=set(v for d in b['demos'] for wh in ['input','output'] for r in d[wh] for v in r)
 q=set(v for r in b['query_input'] for v in r);t=set(v for r in b['target'] for v in r)
 missing=t-demo-q
 if missing:
  qcats=categories(fid,b['query_input'],b['target']); dcats=set().union(*(categories(fid,d['input'],d['output']) for d in b['demos']))
  flag={'line':line,'family':fid,'instance':b['instance'],'output_colors_absent_from_visible_grids':sorted(missing),'query_categories_absent_demos':sorted(qcats-dcats),'classification':'unexposed_mapping_or_interaction; labels source-consistent, not proven contradictory','proposed_action':'quarantine original episode from clean in-context SFT slice; recompose demos from same allowed training family or keep explicit separate extrapolation slice'}
  flags.append(flag)
  reservoir=[a for a in allpairs if a['family']==fid and a['input']!=b['query_input']]
  unique={key(a['input']):a for a in reservoir}
  candidates=list(unique.values())
  chosen=None
  for k in range(1,4):
   for removed in itertools.combinations(range(3),k):
    kept=[b['demos'][i] for i in range(3) if i not in removed]
    keptcat=set().union(*(categories(fid,d['input'],d['output']) for d in kept)) if kept else set()
    for adds in itertools.combinations(candidates,k):
     if len(set(key(d['input']) for d in kept+list(adds)))!=3:continue
     cover=keptcat|set().union(*(categories(fid,d['input'],d['output']) for d in adds))
     if qcats<=cover:
      chosen={'base_line':line,'family':fid,'instance':b['instance'],'remove_demo_slots':list(removed),'replacement_pairs':[selected(a) for a in adds],'changes':k,'remaining_query_missing_categories':[],'status':'proposal only; materialization not performed; category coverage is not full identifiability proof'};break
    if chosen:break
   if chosen:break
  repair.append(chosen or {'base_line':line,'status':'no 3-demo repair from 20 existing pairs'})
result['color_mapping_or_interaction_flags']=flags
result['same_family_demo_repair_proposals']=repair
result['definitions']={
 'contradiction':'Same family and exactly same full input assigned different output. Zero observed is finite-sample evidence only; it does not validate 448 whole generators.',
 'underexposed_mapping':'A query activates a category/interaction whose label/output behavior is absent from demos; a source-consistent pair can still be unsuitable as a self-contained in-context episode.',
 'distribution_shift':'Different visible parameters, counts, dimensions, or copied colors are not themselves semantic inconsistency.',
 'new_color_probe':'Only a triage signal. Source review of flagged family determines whether arbitrary output mapping or new branch is unexposed. Do not reject colors that occur in query input and are copied to output.',
 'rule_check':'Six hand-written input-to-output reconstructions used only for one-off CPU audit of already-generated training pairs; exact matches are not learned-model or competition scores.',
 'repair':'Preserve query input and label; replace the minimum demos using same-family allowed generated training pairs with different query input. This is a proposal, not a claim the data file changed.'}
root='https://github.com/google/ARC-GEN/blob/a15cbdb44c776610aeeb9f487a06af875d3d0878/'
result['source_references']={'read_date':'2026-10-01','repository':root,'validate_semantics':{'path':'arc_gen.py','lines':'25-37','url':root+'arc_gen.py#L25-L37','finding':'CLI validator calls task-specific validate(), then compares entire fixed example set with original benchmark JSON; task-specific validate() is not a function that validates arbitrary new draws.'},'families':{}}
info={
 '0b17323b':('29-38','Input blue diagonal points reveal start/spacing. Extend the arithmetic progression with color 2. No cross-slot shared spacing/start needed.','usable; avoid exact query-input copies; vary visible progression parameters'),
 '2072aba6':('27-40','Random branch has only five binary positions with >=3 active; output is local 2x2 motif. Positions 5..8 are never active in raw draws. validate() lines45-51 uses nine positions.','usable as narrow-distribution data; a separately recorded correction should sample all nine positions (e.g. 3..6 actives), without altering rule'),
 '00dbd492':('30-54','Thickness/color lookup is arbitrary and must be demonstrated for query thickness. num_boxes=size//4-1 yields zero boxes at size7.','quarantine line897 until demo recomposition; avoid all-empty demos where informative slots are scarce; query-thickness coverage'),
 '3b4c2228':('33-43,45-71','Count green(color3) 2x2 components; size/color observable. Internal blues variable names the OUTPUT blue count, not blue input. All 20 sampled counts are1/2, 15 are1.','usable but low count diversity; choose demos with >=2 green-large counts plus distractor variations; do not delete repeated outputs'),
 '8597cfd7':('29-53','Compare bottom-minus-top red/yellow counts, output winner color. All five current bases show both winner colors in demos.','usable; sample both ordering branches; include cases distinguishing increment comparison from bottom-height or total-count shortcuts'),
 'c8b7cc0f':('35-63','Count foreground pixels strictly inside blue frame and fill row-major output with same query foreground color. Three of five query colors absent demos are harmless input-visible copies.','usable; improve count diversity (17/20 count3,3/20 count4,none5/6); need multiple counts and outside distractors, not all output counts nor all colors'),
 '37d3e8b2':('29-61','Hole count maps to fixed colors {1:1,2:2,3:3,4:7}; missing query hole-count mapping is underexposed.','quarantine/recompose lines106,1002'),
 'ed74f2f2':('28-44','Left marker shape encodes fixed output color1/2/3; query marker category missing demos.','quarantine/recompose lines422,1318,1766'),
 'ac605cbb':('30-71,74-85','Fixed color-specific rays; when gray rays cross, novel yellow diagonal branch. Base102 demos contain no join.','quarantine/recompose line1199 to demonstrate join; preserve a no-join contrast'),
 '6ea4a07e':('28-39','Complemented mask recolored by fixed lookup{8:2,3:1,5:4}; unseen source color needs its mapping.','quarantine/recompose lines1540,1988'),
 '17cae0c1':('27-47','Five fixed 3x3 shapes map to fixed colors. Query shape leading to3 absent demos.','quarantine/recompose line1825')}
for fid,(lines,finding,action) in info.items():
 result['source_references']['families'][fid]={'path':'tasks/task_'+fid+'.py','lines':lines,'url':root+'tasks/task_'+fid+'.py','finding':finding,'decision':action}
result['validate_correction']='008 correctly warned rectangular validate() output is not oracle, but its 60/60 wording should be read as constructibility checks. Never claim independent slots invalid merely because parameters differ.'
result['recommended_training_slices']={'unchanged_original':'retain raw2240 for reproducibility; do not silently delete','clean_provisional':'2223 bases after separately setting aside 10 source-reviewed underexposed mappings/interactions and 7 exact query-input repeats; this is an eligibility proposal, not all-family semantic certification; 10 can be repaired from existing raw pairs','repairable_count':sum('changes' in x for x in repair),'quarantined_source_error_count':0,'needs_independent_audit':'Other 437 families not source-reviewed in this audit; unseen-color probe misses underexposed behavior using already-visible output colors.'}
result['small_sample_biases']={'0b17323b':{'unique_inputs':5,'pairs':20,'exact_query_demo_duplicates':3,'source_input_space':6},'2072aba6':{'random_input_space':16,'occupied_positions':'first5 of9'},'3b4c2228':{'count_histogram':{'1':15,'2':5},'pairs':20},'c8b7cc0f':{'count_histogram':{'3':17,'4':3},'pairs':20},'8597cfd7':{'bases_with_both_branches_demonstrated':5,'bases':5}}
result['reproducibility']={'manual_analysis_scripts':['/tmp/arc009-semantics-audit.py','/tmp/arc009-semantics-enrich.py'],'no_framework_or_test_files_added':True,'input_unchanged':hashlib.sha256(p.read_bytes()).hexdigest()==result['source']['sha256'],'generated_input_only':True,'scope':'Only generated training pairs at known committed ARC-GEN revision; no official evaluation JSON or split holdout query labels opened.'}
out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print('flags',len(flags),'repairable',sum('changes'in x for x in repair))
print(json.dumps(repair,ensure_ascii=False,indent=2))
