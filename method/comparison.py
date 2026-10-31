"""Small learned candidate comparison; NumPy-only frozen inference."""
from collections import defaultdict
import numpy as np

FEATURES=['own_score','other_same_answer_votes','supporter_mean_minus_own','best_competitor_max_minus_own','supporter_mean_minus_best_competitor_mean','votes_times_own_minus_best_competitor_max']


def features(scores,answers):
    out=np.zeros((*scores.shape,6),dtype=np.float64)
    for i,aa in enumerate(answers):
        groups=defaultdict(list)
        for j,a in enumerate(aa):
            if a is not None:groups[a].append(j)
        for j,a in enumerate(aa):
            own=float(scores[i,j]);out[i,j,0]=own
            if a is None:continue
            idx=groups[a];votes=len(idx)-1;mean=float(scores[i,idx].mean());others=[v for k,v in groups.items() if k!=a]
            compmax=max((float(scores[i,v].max()) for v in others),default=own)
            compmean=max((float(scores[i,v].mean()) for v in others),default=mean)
            out[i,j]=[own,votes,mean-own,compmax-own,mean-compmean,votes*(own-compmax)]
    return out


def choose(scores,answers):
    scores=scores.copy()
    for i,aa in enumerate(answers):
        valid=np.array([a is not None for a in aa])
        if valid.any():scores[i,~valid]=-np.inf
    return scores.argmax(axis=1)

