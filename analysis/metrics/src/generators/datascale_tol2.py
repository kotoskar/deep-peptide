"""Re-eval tolerance-vs-data WITH per-protein storage so we can bootstrap CIs.
For each (tag,frac, fold,protein): all-type corrected tp/fn/fp at tol 3 and tol 0.
Writes analysis/metrics/datascale_tol_perprotein.csv. GPU inference.
"""
import os,sys,csv,warnings; warnings.filterwarnings("ignore"); sys.path.insert(0,os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import numpy as np, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders, PEPTIDE_START_STATE,
    PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein
dev="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True; torch.set_float32_matmul_precision("high")
MODELS={
 "baseline":{40:("runs/scale_baseline_40",2102),55:("runs/scale_baseline_55",2914),70:("runs/scale_baseline_70",3679),85:("runs/scale_baseline_85",4527),100:("runs/2026_baseline_esm2",5314)},
 "proj":{40:("runs/scale_proj_40",2131),55:("runs/scale_proj_55",2952),70:("runs/scale_proj_70",3726),85:("runs/scale_proj_85",4586),100:("runs/2026_esmc6b_3di_zeroctrl",5384)},
 "3di":{40:("runs/scale_3di_40",2088),55:("runs/scale_3di_55",2897),70:("runs/scale_3di_70",3660),85:("runs/scale_3di_85",4504),100:("runs/2026_esmc6b_3di_gated_boundary",5287)},
}
rows=[]
for tag,fr in MODELS.items():
    for frac in sorted(fr):
        rd,ntr=fr[frac]
        args=load_run_args(Path(rd),SimpleNamespace(batch_size=None,device=None,seed=42))
        m=get_model(args).to(dev); bil=getattr(getattr(m,"feature_extractor",None),"biLSTM",None)
        if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
        m.load_state_dict(load_state_dict(os.path.join(rd,"model.pt"),dev)); m.eval()
        for fold in (2,5):
            _,_,t=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],test_partitions=[fold],device=dev)
            _,_,preds,_,_=run_dataloader(t,m,optimizer=None,do_train=False,device=dev,collect_outputs=True,desc=f"{tag}{frac} f{fold}")
            data=t.dataset.data; names=t.dataset.names
            for i,pr in enumerate(preds):
                row=data.loc[names[i]]; c={0:[0,0,0],3:[0,0,0]}
                for task,(ss,es) in [("peptides",(PEPTIDE_START_STATE,PEPTIDE_END_STATE)),("propeptides",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE))]:
                    true=[(int(a),int(b)) for a,b in (row["true_peptides"] if task=="peptides" else row["true_propeptides"])]
                    pb=convert_path_to_peptide_borders(pr,start_state=ss,stop_state=es,offset=1)
                    for tol in (0,3):
                        g,pm=match_protein(true,pb,tol)
                        c[tol][0]+=sum(x["matched"] for x in g); c[tol][1]+=sum(not x["matched"] for x in g); c[tol][2]+=sum(not x for x in pm)
                rows.append(dict(tag=tag,frac=frac,n_train=ntr,fold=fold,protein=names[i],
                                 tp3=c[3][0],fn3=c[3][1],fp3=c[3][2],tp0=c[0][0],fn0=c[0][1],fp0=c[0][2]))
        print(f"[done] {tag} {frac}%",flush=True)
import pandas as pd
pd.DataFrame(rows).to_csv("analysis/metrics/datascale_tol_perprotein.csv",index=False)
print("wrote analysis/metrics/datascale_tol_perprotein.csv",len(rows))
