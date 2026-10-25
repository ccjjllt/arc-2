import json,hashlib
from collections import Counter,deque,defaultdict
from pathlib import Path
ROOT=Path('/Users/zhaohuaxiang/ccjjllt/arc-2/experiments/026')
D4=[(-1,0),(1,0),(0,-1),(0,1)]; D8=D4+[(-1,-1),(-1,1),(1,-1),(1,1)]
def bg(g): return Counter(x for row in g for x in row).most_common(1)[0][0]
def cells(g,c): return {(i,j) for i,row in enumerate(g) for j,v in enumerate(row) if v==c}
def bbox(s):
 return min(i for i,j in s),min(j for i,j in s),max(i for i,j in s),max(j for i,j in s)
def crop(g,s):
 a,b,c,d=bbox(s);return [r[b:d+1] for r in g[a:c+1]]
def comps(g,mono=True,diag=False,background=None):
 if background is None: background=bg(g)
 remain={(i,j) for i,row in enumerate(g) for j,x in enumerate(row) if x!=background}; out=[]
 while remain:
  first=min(remain); todo=[first];remain.remove(first); comp={first}
  while todo:
   i,j=todo.pop()
   for di,dj in (D8 if diag else D4):
    n=i+di,j+dj
    if n in remain and (not mono or g[n[0]][n[1]]==g[first[0]][first[1]]):remain.remove(n);todo.append(n);comp.add(n)
  out.append(comp)
 return out
def norm(s):
 a,b,_,_=bbox(s);return frozenset((i-a,j-b) for i,j in s)
def reconstruct(t,g):
 h,w=len(g),len(g[0]);o=[r[:] for r in g];back=bg(g)
 if t=='14754a24':
  mark=set()
  for i in range(h):
   for j in range(w):
    p={(i+di,j+dj) for di,dj in D4+[(0,0)] if 0<=i+di<h and 0<=j+dj<w}
    y={x for x in p if g[x[0]][x[1]]==4}
    if len(p)<4 or not (2<=len(y)<len(p)) or any(g[a][b] not in (4,5) for a,b in p):continue
    if any(0<=a+di<h and 0<=b+dj<w and g[a+di][b+dj]==4 and (a+di,b+dj) not in p for a,b in y for di,dj in D8[4:]):continue
    mark|={x for x in p if g[x[0]][x[1]]==5}
  for i,j in mark:o[i][j]=2
  return o
 if t=='18419cfa':
  for s in [s for s in comps(g) if g[next(iter(s))[0]][next(iter(s))[1]]==8]:
   a,b,c,d=bbox(s)
   for i in range(a+1,c):
    for j in range(b+1,d):
     if any(g[x][y]==2 for x,y in [(i,j),(a+c-i,j),(i,b+d-j),(a+c-i,b+d-j)]):o[i][j]=2
  return o
 if t=='2037f2c7':
  a,b=map(norm,comps(g,False)); s=a^b; x0,y0,x1,y1=bbox(s);return [[8 if (i,j) in s else 0 for j in range(y0,y1+1)] for i in range(x0,x1+1)]
 if t=='2de01db2':
  return [[(next(c for c in row if c in (4,6,8)) if x!=next(c for c in row if c in (4,6,8)) else 0) for x in row] for row in g]
 if t=='3194b014':
  s=max(comps(g),key=len);i,j=next(iter(s));return [[g[i][j]]*3 for _ in range(3)]
 if t=='396d80d7':
  parts=[cells(g,c) for c in set(x for r in g for x in r)-{back}]
  s=min(parts,key=lambda p:(bbox(p)[2]-bbox(p)[0]+1)*(bbox(p)[3]-bbox(p)[1]+1));v=g[next(iter(s))[0]][next(iter(s))[1]];other=next(x for x in parts if x!=s)
  diagonal={(i+di,j+dj) for i,j in other for di,dj in D8[4:]};orth={(i+di,j+dj) for i,j in other for di,dj in D4}
  for i,j in diagonal-orth-other:
   if 0<=i<h and 0<=j<w and g[i][j]==back:o[i][j]=v
  return o
 if t=='5587a8d0':
  counts=Counter(x for r in g for x in r if x!=back);colors=sorted(counts,key=lambda c:-counts[c]);side=2*len(colors)-1
  return [[colors[min(i,j,side-1-i,side-1-j)] for j in range(side)] for i in range(side)]
 if t=='6df30ad6':
  s=cells(g,5);colors=set(x for r in g for x in r)-{0,5}
  v=min(colors,key=lambda c:(min(abs(i-a)+abs(j-b) for i,j in s for a,b in cells(g,c)),len(cells(g,c)),c))
  return [[v if (i,j) in s else 0 for j in range(w)] for i in range(h)]
 if t=='7bb29440':
  s=min(comps(g,False),key=lambda s:(sum(g[i][j] in (4,6) for i,j in s),*bbox(s)));return crop(g,s)
 if t=='aa62e3f4':
  s={(i,j) for i,row in enumerate(g) for j,x in enumerate(row) if x!=back};v=Counter(g[i][j] for i,j in s).most_common()[-1][0]
  border={(i+di,j+dj) for i,j in s for di,dj in D4}-s
  return [[v if (i,j) in border else back for j in range(w)] for i in range(h)]
 if t=='aaecdb9a':
  counts=Counter(g[next(iter(s))[0]][next(iter(s))[1]] for s in comps(g,diag=True));height=max(counts.values());colors=[5,2,8,9,6]
  return [[c if i>=height-counts[c] else 7 for c in colors] for i in range(height)]
 if t=='b15fca0b':
  a,b=sorted(cells(g,2))
  def dist(start):
   d={start:0};q=deque([start])
   while q:
    i,j=q.popleft()
    for di,dj in D4:
     n=i+di,j+dj
     if 0<=n[0]<h and 0<=n[1]<w and g[n[0]][n[1]]!=1 and n not in d:d[n]=d[i,j]+1;q.append(n)
   return d
  da,db=dist(a),dist(b)
  return [[4 if g[i][j]==0 and da.get((i,j),999)+db.get((i,j),999)==da[b] else g[i][j] for j in range(w)] for i in range(h)]
 if t=='cd3c21df':
  parts=comps(g,False);keys=[]
  for s in parts:
   a,b,_,_=bbox(s);keys.append(frozenset((g[i][j],i-a,j-b) for i,j in s))
  rare=min(Counter(keys),key=Counter(keys).get);return crop(g,parts[keys.index(rare)])
 if t=='d56f2372':
  for c in set(x for r in g for x in r)-{back}:
   s=cells(g,c);a,b,c0,d=bbox(s)
   if s=={(i,b+d-j) for i,j in s}:return crop(g,s)
  raise ValueError('no unique symmetric color partition')
 if t=='e729b7be':return [[g[h-1-i][w-1-j] if g[i][j]==back else g[i][j] for j in range(w)] for i in range(h)]
 if t=='f5c89df1':
  a,b=next(iter(cells(g,3)));offsets={(i-a,j-b) for i,j in cells(g,8)};out={(i+di,j+dj) for i,j in cells(g,2) for di,dj in offsets}
  return [[8 if (i,j) in out else 0 for j in range(w)] for i in range(h)]
 raise ValueError(t)
rows=[]
for f in ['train.jsonl','family-holdout.jsonl']:
 rows += [x for line in (ROOT/'data-pilot-v1'/f).read_text().splitlines() if (x:=json.loads(line))['source']=='water-vapor/synth-rearc']
reports=[]
for t in sorted(set(x['family_id'] for x in rows)):
 matches=[]
 for r in rows:
  if r['family_id']!=t:continue
  for p in r['train']+[{'input':r['query_input'],'output':r['query_output']}]:
   try:matches.append(reconstruct(t,p['input'])==p['output'])
   except Exception as e:matches.append(str(e))
 reports.append({'family_id':t,'independent_reconstruction_matches':sum(x is True for x in matches),'pairs':len(matches),'failures':[x for x in matches if x is not True]})
print(json.dumps(reports,indent=2))
report={'method':'manual source interpretation followed by a separate stdlib reconstruction; no upstream DSL/generator/verifier imports in audit','script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'counts':reports,'status':'passed' if all(x['independent_reconstruction_matches']==x['pairs'] for x in reports) else 'needs_review','scope':'16 purpose-selected pilot families; source interpretations and generated labels only, not a proof of episode uniqueness or ARC2 score improvement'}
(ROOT/'data-pilot-v1'/'manual-rule-audit.json').write_text(json.dumps(report,indent=2)+'\n')
