import collections,copy,hashlib,json,platform,time
from pathlib import Path
start=time.monotonic()
p=Path('experiments/008/data/gen_newbases.jsonl')
rows=[json.loads(l) for l in p.read_text().splitlines()]
ids=['0b17323b','2072aba6','00dbd492','3b4c2228','8597cfd7','c8b7cc0f']
key=lambda g:json.dumps(g,separators=(',',':'))
allpairs=[]
for line,b in enumerate(rows,1):
 for slot,pair in enumerate(b['demos']+[{'input':b['query_input'],'output':b['target']}]):
  allpairs.append({'family':b['family_id'],'line':line,'instance':b['instance'],'slot':slot,'input':pair['input'],'output':pair['output']})
seen={}; conflicts=[]
for a in allpairs:
 k=(a['family'],key(a['input'])); y=key(a['output'])
 if k in seen and seen[k][0]!=y:
  conflicts.append({'family':a['family'],'first':seen[k][1],'second':{j:a[j] for j in ['line','instance','slot']}})
 else: seen[k]=(y,{j:a[j] for j in ['line','instance','slot']})
def components(g,color):
 remaining={(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==color}; result=[]
 while remaining:
  todo=[remaining.pop()]; comp=[]
  while todo:
   r,c=todo.pop(); comp.append((r,c))
   for q in [(r-1,c),(r+1,c),(r,c-1),(r,c+1)]:
    if q in remaining: remaining.remove(q); todo.append(q)
  result.append(comp)
 return result
def oracle(fid,g):
 if fid=='0b17323b':
  pos=[r for r,row in enumerate(g) if row[r]==1]; gap=pos[1]-pos[0]
  y=copy.deepcopy(g)
  for r in range(pos[-1]+gap,len(g),gap): y[r][r]=2
  return y,{'spacing':gap,'start':pos[0]}
 if fid=='2072aba6':
  y=[[0]*6 for _ in range(6)]
  for r,row in enumerate(g):
   for c,v in enumerate(row):
    if v==5:
     for dr,dc,col in [(0,0,1),(1,0,2),(1,1,1),(0,1,2)]: y[2*r+dr][2*c+dc]=col
  return y,{'nonzero_positions':[3*r+c for r,row in enumerate(g) for c,v in enumerate(row) if v],'active_count':sum(v!=0 for row in g for v in row)}
 if fid=='00dbd492':
  y=copy.deepcopy(g); thicks=[]
  for co in components(g,2):
   if len(co)==1: continue
   r0=min(r for r,c in co); r1=max(r for r,c in co); c0=min(c for r,c in co); c1=max(c for r,c in co)
   size=r1-r0+1; assert size==c1-c0+1 and size in [5,7,9]
   thick=(size-3)//2; thicks.append(thick)
   for r in range(r0+1,r1):
    for c in range(c0+1,c1):
     if y[r][c]!=2:y[r][c]=[8,4,3][thick-1]
  return y,{'size':len(g),'thicknesses':sorted(thicks),'box_count':len(thicks)}
 if fid=='3b4c2228':
  counts={str(col):collections.Counter(len(c) for c in components(g,col)) for col in [2,3]}
  n=counts['3'][4]; y=[[int(r==c and r<n) for c in range(3)] for r in range(3)]
  return y,{'green_large':n,'green_all':sum(counts['3'].values()),'red_large':counts['2'][4],'red_all':sum(counts['2'].values()),'all_large':counts['3'][4]+counts['2'][4]}
 if fid=='8597cfd7':
  mid=len(g)//2
  counts={str(col):[sum(v==col for row in half for v in row) for half in [g[:mid],g[mid+1:]]] for col in [2,4]}
  dr=counts['2'][1]-counts['2'][0]; dy=counts['4'][1]-counts['4'][0]; color=2 if dr>dy else 4
  return [[color]*2 for _ in range(2)],{'reds':counts['2'],'yellows':counts['4'],'delta_red':dr,'delta_yellow':dy,'winner':color}
 if fid=='c8b7cc0f':
  blues=[(r,c) for r,row in enumerate(g) for c,v in enumerate(row) if v==1]
  r0=min(r for r,c in blues);r1=max(r for r,c in blues);c0=min(c for r,c in blues);c1=max(c for r,c in blues)
  colors=set(v for row in g for v in row)-{0,1}; assert len(colors)==1; color=colors.pop()
  n=sum(g[r][c]==color for r in range(r0+1,r1) for c in range(c0+1,c1)); total=sum(v==color for row in g for v in row)
  return [[color if 3*r+c<n else 0 for c in range(3)] for r in range(3)],{'color':color,'inside_count':n,'all_count':total,'frame':[r0,c0,r1,c1]}
 raise ValueError(fid)
def selected(a):return {j:a[j] for j in ['line','instance','slot']}
six={}
for fid in ids:
 pairs=[a for a in allpairs if a['family']==fid]; rec=[]; mismatches=[]
 for a in pairs:
  pred,info=oracle(fid,a['input'])
  if pred!=a['output']:mismatches.append(selected(a))
  rec.append({**selected(a),'features':info,'source_rule_exact':pred==a['output']})
 bases=[]
 for b in [b for b in rows if b['family_id']==fid]:
  slot=[x for x in rec if x['instance']==b['instance']]; ds=[x['features'] for x in slot[:3]]; q=slot[3]['features']
  z={'instance':b['instance'],'line':slot[0]['line'],'demo_features':ds,'query_features':q,'query_input_equals_a_demo':any(b['query_input']==d['input'] for d in b['demos']),'query_output_equals_a_demo':any(b['target']==d['output'] for d in b['demos'])}
  if fid=='00dbd492':z['query_thickness_not_shown_in_demos']=sorted(set(q['thicknesses'])-set(t for d in ds for t in d['thicknesses']))
  if fid=='8597cfd7':
   def alt(f,mode):
    if mode=='constant2':return 2
    if mode=='constant4':return 4
    r,y=f['reds'],f['yellows']; vals={'lower':(r[1],y[1]),'upper':(r[0],y[0]),'total':(sum(r),sum(y))}[mode]
    return 2 if vals[0]>vals[1] else (4 if vals[1]>vals[0] else None)
   z['simple_rival_rules_fitting_demos_but_disagreeing_query']=[m for m in ['constant2','constant4','lower','upper','total'] if all(alt(d,m)==d['winner'] for d in ds) and alt(q,m) is not None and alt(q,m)!=q['winner']]
   z['both_winners_shown_in_demos']=len(set(d['winner'] for d in ds))==2
  if fid=='3b4c2228':
   def alt(f,m):return int(m[-1]) if m.startswith('constant') else f[m]
   z['simple_rival_rules_fitting_demos_but_disagreeing_query']=[m for m in ['constant1','constant2','constant3','green_all','red_large','red_all','all_large'] if all(alt(d,m)==d['green_large'] for d in ds) and alt(q,m)!=q['green_large']]
  if fid=='c8b7cc0f':
   z['query_color_absent_demos']=q['color'] not in [d['color'] for d in ds]
   z['query_count_absent_demos']=q['inside_count'] not in [d['inside_count'] for d in ds]
   z['constant_count_fits_demos_but_disagrees_query']=len(set(d['inside_count'] for d in ds))==1 and q['inside_count']!=ds[0]['inside_count']
   z['total_count_fits_demos_but_disagrees_query']=all(d['all_count']==d['inside_count'] for d in ds) and q['all_count']!=q['inside_count']
  bases.append(z)
 six[fid]={'pairs':len(pairs),'unique_inputs':len(set(key(a['input']) for a in pairs)),'source_rule_matches':len(pairs)-len(mismatches),'source_rule_mismatches':mismatches,'bases':bases}
agg=collections.Counter(); repeat=[]
for line,b in enumerate(rows,1):
 same=any(b['query_input']==d['input'] for d in b['demos']);eq=any(b['target']==d['output'] for d in b['demos'])
 palette_demo=set(v for d in b['demos'] for which in ['input','output'] for r in d[which] for v in r)
 palette_query=set(v for r in b['query_input'] for v in r)
 palette_target=set(v for r in b['target'] for v in r)
 introduced=palette_target-palette_demo-palette_query
 agg['query_input_equals_demo']+=same;agg['query_output_equals_demo']+=eq;agg['output_has_color_absent_from_all_visible_grids']+=bool(introduced)
 if same:repeat.append({'line':line,'family':b['family_id'],'instance':b['instance']})
result={'status':'CPU manual data/source audit; no model, GPU, official evaluation solutions, or heldout query outputs read','date':'2026-10-01','source':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'arcgen_commit':'a15cbdb44c776610aeeb9f487a06af875d3d0878'},'hardware':platform.platform()+' '+platform.machine(),'base_count':len(rows),'family_count':len(set(b['family_id'] for b in rows)),'pair_count':len(allpairs),'contradictions_same_family_identical_input_different_output':conflicts,'unique_family_input_count':len(seen),'aggregate_flags':dict(agg),'query_input_duplicates':repeat,'six_families':six,'elapsed_seconds':round(time.monotonic()-start,4)}
Path('experiments/009/episode-semantics.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k not in ['six_families','query_input_duplicates']},ensure_ascii=False,indent=2))
for f,d in six.items():print(f,json.dumps(d,ensure_ascii=False))
