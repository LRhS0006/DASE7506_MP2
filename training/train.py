"""Reproduce the final verifier from supplied TRAIN data only.

Optional nested cross-validation evaluates the prespecified baseline and final
model. Export always uses five-fold question-level OOF base scores, exactly as
in the frozen experiment. Validation and test references are never opened.
"""
import os
os.environ.update(OMP_NUM_THREADS='4',OPENBLAS_NUM_THREADS='4',MKL_NUM_THREADS='4',VECLIB_MAXIMUM_THREADS='4')
import argparse,json,sys,time,hashlib,warnings
from pathlib import Path
import numpy as np
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--code-dir',type=Path,required=True,help='Unmodified official code/ directory')
    parser.add_argument('--output-dir',type=Path,required=True,help='New output directory, separate from release checkpoint')
    parser.add_argument('--cross-validate',action='store_true',help='Also reproduce 5 outer / 3 inner fold comparison')
    args=parser.parse_args()
    sys.path.insert(0,str(args.code_dir.resolve()))
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'method'))
    from common import load_inputs,read_jsonl,parse_answer
    from submission import build_features,FEATURE_NAMES
    from comparison import choose
    if args.output_dir.exists():
        raise FileExistsError('Use a new output directory; never overwrite released checkpoint')
    args.output_dir.mkdir(parents=True)
    start=time.perf_counter()
    bank,arrays=load_inputs('train',args.code_dir/'data')
    gold=read_jsonl(args.code_dir/'data/gsm8k_train.jsonl')
    assert len(bank)==len(gold)
    answers=[[parse_answer(c['text']) for c in q['candidates']] for q in bank]
    assert all(q['id']==g['id'] and q['question']==g['question'] for q,g in zip(bank,gold))
    labels=np.array([[a is not None and a==parse_answer(g['answer']) for a in aa] for aa,g in zip(answers,gold)])
    valid=np.array([[a is not None for a in aa] for aa in answers])
    hard=np.array([len({a for a in aa if a is not None})>=2 for aa in answers])
    x=arrays['embeddings']; all_indices=np.arange(len(bank))
    def fit_pair(inputs,targets):
        scaler=StandardScaler().fit(inputs)
        model=LogisticRegression(C=1,max_iter=1000,random_state=42)
        model.fit(scaler.transform(inputs),targets)
        return scaler,model
    def fit_base(indices):
        use=indices[hard[indices]]
        return fit_pair(x[use].reshape(-1,1024),labels[use].reshape(-1))
    def score(pair,inputs):
        return pair[1].decision_function(pair[0].transform(inputs)).reshape(-1,4)
    def score_base(pair,indices):
        return score(pair,x[indices].reshape(-1,1024))
    def features(indices,scores):
        return build_features([bank[i] for i in indices],[answers[i] for i in indices],scores)
    def fit_meta(indices,values,dimension):
        use=(valid[indices]&hard[indices,None]).reshape(-1)
        return fit_pair(values[...,:dimension].reshape(-1,dimension)[use],labels[indices].reshape(-1)[use])
    folds=[]
    with threadpool_limits(limits=4),warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter('always')
        oof=np.full((len(bank),4),np.nan)
        for k,(fit,held) in enumerate(KFold(5,shuffle=True,random_state=42).split(all_indices)):
            base=fit_base(fit);oof[held]=score_base(base,held)
            if args.cross_validate:
                inner=np.full((len(fit),4),np.nan)
                for tr,dev in KFold(3,shuffle=True,random_state=42).split(fit):
                    inner[dev]=score_base(fit_base(fit[tr]),fit[dev])
                train_features=features(fit,inner);held_features=features(held,oof[held])
                row={'fold':k,'hard_n':int(hard[held].sum()),'correct':{}}
                pred=choose(oof[held],[answers[i] for i in held])
                row['correct']['regression']=int(labels[held][np.arange(len(held)),pred][hard[held]].sum())
                for name,dimension in [('comparison_baseline',8),('comparison_plus_style',14)]:
                    pair=fit_meta(fit,train_features,dimension)
                    pred=choose(score(pair,held_features[...,:dimension].reshape(-1,dimension)),[answers[i] for i in held])
                    row['correct'][name]=int(labels[held][np.arange(len(held)),pred][hard[held]].sum())
                folds.append(row);print(json.dumps(row),flush=True)
        assert np.isfinite(oof).all()
        base=fit_base(all_indices)
        meta=fit_meta(all_indices,features(all_indices,oof),14)
        weights={}
        for prefix,(scaler,model) in [('base',base),('meta',meta)]:
            for name,value in [('mean',scaler.mean_),('scale',scaler.scale_),('coef',model.coef_),('intercept',model.intercept_)]:
                weights[prefix+'_'+name]=value
        np.savez(args.output_dir/'model.npz',**weights)
    log={'train_n':len(bank),'hard_train_n':int(hard.sum()),'training_only':True,'seed':42,'final_OOF_folds':5,'nested_inner_folds':3,'C':1,'max_iter':1000,'features':FEATURE_NAMES,'outer_folds':folds,'seconds':time.perf_counter()-start,'warnings':[str(w.message) for w in captured],'model_sha256':hashlib.sha256((args.output_dir/'model.npz').read_bytes()).hexdigest()}
    if folds:log['internal_hard_totals']={name:sum(f['correct'][name] for f in folds) for name in folds[0]['correct']}
    (args.output_dir/'training.json').write_text(json.dumps(log,indent=2)+'\n')
    print(json.dumps(log,indent=2))


if __name__=='__main__':main()
