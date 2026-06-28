"""A2 per-SEGMENT data with lengths (for the by-length plot), corrected matcher tol=3,
folds {2,5}. Emits clean_tol-style rows: true (length,m3) + pred (plen,m3), task pep/propep."""
import os, sys; sys.path.insert(0, os.getcwd())
from pathlib import Path; from types import SimpleNamespace
import pandas as pd, torch
from infer import load_run_args, load_state_dict
from src.train_loop_crf import get_dataloaders, get_model, run_dataloader
from src.utils.manuscript_metrics import (convert_path_to_peptide_borders,
    PEPTIDE_START_STATE, PEPTIDE_END_STATE, PROPEPTIDE_START_STATE, PROPEPTIDE_END_STATE)
from analysis.errors.src.error_analysis import match_protein

device="cuda:0" if torch.cuda.is_available() else "cpu"
torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
torch.set_float32_matmul_precision("high")
run_dir="runs/2026_esmc6b_adapter256_seqonly"
args=load_run_args(Path(run_dir),SimpleNamespace(batch_size=None,device=None,seed=42))
TRUE=[]; PRED=[]
for fold in (2,5):
    _,_,test=get_dataloaders(args,train_partitions=[0,3,4,6],valid_partitions=[1],
                             test_partitions=[fold],device=device)
    model=get_model(args).to(device)
    bil=getattr(getattr(model,"feature_extractor",None),"biLSTM",None)
    if bil is not None and hasattr(bil,"flatten_parameters"): bil.flatten_parameters()
    model.load_state_dict(load_state_dict(os.path.join(run_dir,"model.pt"),device)); model.eval()
    _,_,preds,_,_=run_dataloader(test,model,optimizer=None,do_train=False,device=device,
                                 collect_outputs=True,desc=f"A2seg:f{fold}")
    data=test.dataset.data; names=test.dataset.names
    for i,pred in enumerate(preds):
        row=data.loc[names[i]]
        for task,(ss,es) in [("pep",(PEPTIDE_START_STATE,PEPTIDE_END_STATE)),
                             ("propep",(PROPEPTIDE_START_STATE,PROPEPTIDE_END_STATE))]:
            tkey="true_peptides" if task=="pep" else "true_propeptides"
            true=[(int(a),int(b)) for a,b in row[tkey]]
            pb=convert_path_to_peptide_borders(pred,start_state=ss,stop_state=es,offset=1)
            g,pm=match_protein(true,pb,3)
            for rec in g:
                TRUE.append({"model":"adapter256","fold":fold,"protein":names[i],"task":task,
                             "length":rec["length"],"m3":int(rec["matched"])})
            for (ps,pe),matched in zip(pb,pm):
                PRED.append({"model":"adapter256","fold":fold,"protein":names[i],"task":task,
                             "plen":pe-ps+1,"m3":int(matched)})
    del model; torch.cuda.empty_cache()
pd.DataFrame(TRUE).to_csv("analysis/metrics/adapter256_seg_true.csv",index=False)
pd.DataFrame(PRED).to_csv("analysis/metrics/adapter256_seg_pred.csv",index=False)
print(f"wrote adapter256_seg_true.csv ({len(TRUE)} true segs) + adapter256_seg_pred.csv ({len(PRED)} preds)")
