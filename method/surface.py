"""Six deterministic surface features; unchanged from the frozen model."""
import re
import numpy as np

def style(bank):
    rows=[]
    for q in bank:
        cs=[]
        for c in q['candidates']:
            t=c['text'];n=max(len(t),1)
            cs.append([np.log1p(len(t)),np.log1p(len(t.split())),t.count('\n')/n,sum(t.count(ch) for ch in '*#$\\')/n,sum(ch.isdigit() for ch in t)/n,len(re.findall(r'\b(?:clearly|obviously|certainly|definitely|confident)\b',t,re.I))/max(len(t.split()),1)])
        rows.append(cs)
    return np.array(rows,dtype=float)
