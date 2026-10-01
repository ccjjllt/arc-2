import json,hashlib,collections
from pathlib import Path
exec(Path(__file__).with_name('arc009-semantics-audit.py').read_text().split("result={'status'")[0])
def extra_oracle(fid,g):
 if fid=='37d3e8b2':
  y=[[0]*len(g[0]) for _ in g]
  for co in components(g,8):
   r0=min(r for r,c in co);r1=max(r for r,c in co);c0=min(c for r,c in co);c1=max(c for r,c in co)
   n=len(components([row[c0:c1+1] for row in g[r0:r1+1]],0));color={1:1,2:2,3:3,4:7}[n]
   for r,c in co:y[r][c]=color
  return y
 if fid=='6ea4a07e':
  color=(set(v for row in g for v in row)-{0}).pop();color={8:2,3:1,5:4}[color]
  return [[color if v==0 else 0 for v in row] for row in g]
 if fid=='ed74f2f2':
  color=1 if g[1][1]==5 and g[1][3]==5 else (2 if g[1][1]==5 else 3)
  return [[color if g[r+1][c+5] else 0 for c in range(3)] for r in range(3)]
 if fid=='17cae0c1':
  shapes={1:{(2,0),(2,1),(2,2)},3:{(0,0),(0,1),(0,2),(1,0),(1,2),(2,0),(2,1),(2,2)},4:{(1,1)},6:{(0,0),(0,1),(0,2)},9:{(2,0),(1,1),(0,2)}}
  y=[[0]*9 for _ in range(3)]
  for offset in [0,3,6]:
   occupied={(r,c) for r in range(3) for c in range(3) if g[r][c+offset]}
   color=next(k for k,v in shapes.items() if v==occupied)
   for r in range(3):
    for c in range(3):y[r][c+offset]=color
  return y
 if fid=='ac605cbb':
  y=[row[:] for row in g];gray=collections.Counter()
  for r,row in enumerate(g):
   for c,v in enumerate(row):
    if not v:continue
    off,end={1:([(0,1),(0,2)],(-1,2)),2:([(0,-1),(0,-2),(0,-3)],(0,-4)),3:([(1,0),(2,0)],(3,0)),6:([(-1,0),(-2,0),(-3,0),(-4,0),(-5,0)],(-6,0))}[v]
    for dr,dc in off:gray[r+dr,c+dc]+=1
    y[r+end[0]][c+end[1]]=v
  for (r,c),n in gray.items():y[r][c]=5
  for (r,c),n in gray.items():
   if n<2:continue
   while r<len(g) and c>=0:y[r][c]=4;r+=1;c-=1
  return y
out=Path('experiments/009/episode-semantics.json');x=json.loads(out.read_text());stats={}
for fid in ['37d3e8b2','ed74f2f2','ac605cbb','6ea4a07e','17cae0c1']:
 pairs=[a for a in allpairs if a['family']==fid]
 mismatches=[selected(a) for a in pairs if extra_oracle(fid,a['input'])!=a['output']]
 stats[fid]={'pairs':len(pairs),'source_rule_matches':len(pairs)-len(mismatches),'source_rule_mismatches':mismatches}
x['expanded_source_rule_reconstruction']=stats
for flag in x['color_mapping_or_interaction_flags']:
 b=rows[flag['line']-1]
 vis=set(v for d in b['demos'] for wh in ['input','output'] for row in d[wh] for v in row)|set(v for row in b['query_input'] for v in row)
 src=flag['output_colors_absent_from_visible_grids'][0]
 dst=next(c for c in range(1,10) if c not in vis and c not in set(v for row in b['target'] for v in row))
 alt=[[dst if v==src else v for v in row] for row in b['target']]
 flag['counterexample_witness']={'source_label':src,'alternate_label':dst,'both_labels_absent_visible_input_and_demos':True,'hypothesis':'same generator transformation followed by swapping the two absent output labels; all current demos unchanged; query output differs','changed_query_cells':sum(v==src for row in b['target'] for v in row),'alternate_target_sha256':hashlib.sha256(key(alt).encode()).hexdigest(),'qualification':'Demonstrates ambiguity within this explicit two-rule hypothesis class, not impossibility for a pretrained model with prior task memory; not a contradictory labeling claim.'}
x['reproducibility']['manual_analysis_scripts'].append('/tmp/arc009-semantics-extra.py')
out.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(stats,indent=2))
