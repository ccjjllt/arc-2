import json,copy,hashlib,time
from pathlib import Path
exec(Path(__file__).with_name('arc009-semantics-audit.py').read_text().split("result={'status'")[0])
def shape_features(g):
 thicks=[]
 for co in components(g,2):
  if len(co)==1:continue
  rs=[r for r,c in co];cs=[c for r,c in co]
  side=max(rs)-min(rs)+1
  if side==max(cs)-min(cs)+1 and side in [5,7,9]:thicks.append((side-3)//2)
 return {'thicknesses':sorted(thicks),'cells':len(g)*len(g[0])}
original=copy.deepcopy(next(b for b in rows if b['family_id']=='00dbd492' and b['instance']==102))
qfeatures=shape_features(original['query_input']);dfeatures=[shape_features(d['input']) for d in original['demos']]
missing=set(qfeatures['thicknesses'])-set(t for f in dfeatures for t in f['thicknesses'])
blank=[i for i,d in enumerate(original['demos']) if all(v==0 for row in d['input'] for v in row)]
if missing!={2} or blank!=[2]:raise ValueError('Expected witness not reproduced')
candidates=[]
for a in allpairs:
 if a['family']!='00dbd492' or a['line']==897 or a['input']==original['query_input']:continue
 if any(a['input']==d['input'] for d in original['demos']):continue
 feat=shape_features(a['input'])
 if not missing<=set(feat['thicknesses']):continue
 candidates.append((feat['cells'],a['line'],a['slot'],a))
candidates.sort(key=lambda a:a[:3]);chosen=candidates[0][3]
# Source-rule verification is separate from query-input-only selection.
pred,info=oracle('00dbd492',chosen['input'])
if pred!=chosen['output']:raise ValueError('Chosen training pair does not follow source rule')
repaired=copy.deepcopy(original)
repaired['demos'][2]={'input':copy.deepcopy(chosen['input']),'output':copy.deepcopy(chosen['output'])}
repaired['base_instance_id']='gen_instance:102|repair009_thickness_demo'
repaired['repair_status']='CPU episode repair pilot; model generalization and score untested'
repaired['parent_base_instance_id']=original['base_instance_id']
record={'schema':'original_repaired_episode_pair_v1','date':'2026-10-01','source_file':'experiments/008/data/gen_newbases.jsonl','source_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'source_line':897,'family_id':'00dbd492','instance':102,'reason':'query input has thickness2; original demos show thickness1/3 plus empty; arbitrary thickness2->color4 mapping not demonstrated','selection':{'uses_query_target_for_selection':False,'query_input_features':qfeatures,'original_demo_input_features':dfeatures,'missing_input_category':[2],'replacement_slot':2,'candidate_count':len(candidates),'ranking':'fewest input cells, then source line, then slot; same allowed training family; candidate input different from query and existing demos; candidate category covers thickness missing in demos','chosen_source_line':chosen['line'],'chosen_instance':chosen['instance'],'chosen_slot':chosen['slot'],'chosen_input_features':shape_features(chosen['input']),'chosen_pair_source_rule_matches':True,'source_rule_reference':'https://github.com/google/ARC-GEN/blob/a15cbdb44c776610aeeb9f487a06af875d3d0878/tasks/task_00dbd492.py#L30-L54'},'lineage':{'original_demos':[{'line':897,'slot':i} for i in range(3)],'repaired_demos':[{'line':897,'slot':0},{'line':897,'slot':1},{'line':chosen['line'],'slot':chosen['slot']}],'query':{'line':897,'slot':3}},'verification':{'query_input_unchanged':repaired['query_input']==original['query_input'],'target_unchanged':repaired['target']==original['target'],'no_repaired_demo_input_equals_query':all(d['input']!=repaired['query_input'] for d in repaired['demos']),'query_thickness_covered':set(qfeatures['thicknesses'])<=set(t for d in repaired['demos'] for t in shape_features(d['input'])['thicknesses']),'all_4_pairs_source_rule_match':all(oracle('00dbd492',z['input'])[0]==z['output'] for z in repaired['demos']+[{'input':repaired['query_input'],'output':repaired['target']}]),'original_pool_modified':False},'limitations':['Only one explicitly witnessed generated training episode repaired.','Coverage of thickness-to-color mapping is not a proof of unique rule identification or generalization to unseen instances.','No GPU training, model inference, competition submission, or score increase measured.','Replacement reuses an existing training pair and adds no independent generated pair.'],'original':original,'repaired':repaired}
Path('experiments/009/episode-repair-pilot.jsonl').write_text(json.dumps(record,ensure_ascii=False,separators=(',',':'))+'\n')
x=json.loads(Path('experiments/009/episode-semantics.json').read_text())
x['repair_pilot']={'file':'experiments/009/episode-repair-pilot.jsonl','records':1,'original_line':897,'replaced_demo_slot':2,'replacement_source_line':chosen['line'],'replacement_source_slot':chosen['slot'],'selection_query_input_only':True,'source_pair_matches':True,'query_and_target_unchanged':True,'note':'The materialized pilot uses blank demo slot2, preferable to the generic minimum-change proposal which had slot0. Other nine proposals unmaterialized.'}
x['reproducibility']['manual_analysis_scripts'].append('/tmp/arc009-episode-repair-pilot.py')
Path('experiments/009/episode-semantics.json').write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:record[k] for k in ['selection','lineage','verification']},ensure_ascii=False,indent=2))
