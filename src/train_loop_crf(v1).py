import json
import pickle
from typing import Dict, List, Tuple
import os
from torch.utils.data import DataLoader

from src.models import LSTMCNNCRF, LSTMCNNCRFProjector, SimpleLSTMCNNCRF, SelfAttentionCRF
from src.utils.dataset import PrecomputedCSVForOverlapCRFDataset
from src.utils.manuscript_metrics import compute_all_metrics
from torch.optim import Adam
import torch
import numpy as np
import argparse
from torch.amp import GradScaler
from typing import List, Tuple
from tqdm import tqdm

def log_metrics(metrics: dict, filepath: str, prefix: str = "", aim_run=None, epoch=None):
    """Log metrics from a dictionary to a file, one per line, sorted by key. Optionally add a prefix to each line."""
    with open(filepath, 'a') as f:
        f.write(f'Epoch: {epoch}\n')
        for key in sorted(metrics.keys()):
            sep = "" if prefix == "" else f"{prefix}/"
            f.write(f"{sep}{key}: {metrics[key]}\n")
    if aim_run:
        aim_run.track(metrics, epoch=epoch, context={'subset': prefix.lower()})
    

def get_dataloaders_loo(args: argparse.Namespace, partitions: List[int] = [0,1,2,3,4], restrict = None, leave_idxs = [], device = None)-> Tuple[DataLoader, DataLoader]:
    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=partitions, label_type=args.label_type, restrict=restrict, device=device)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=partitions, label_type=args.label_type, restrict=leave_idxs, device=device)

    print('Loading LOO data...')
    train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_set.collate_fn,
            num_workers=8,        
            pin_memory=True,
            persistent_workers=True,  
            prefetch_factor=2,
        )
    
    valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            collate_fn=valid_set.collate_fn,
            num_workers=1,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )

    print(f'Loaded data. {len(train_set)} train sequences, {len(valid_set)} validation sequences.')

    return train_loader, valid_loader

def get_dataloaders(args: argparse.Namespace, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], restrict = None, device = None) -> Tuple[DataLoader, DataLoader, DataLoader]:
    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=train_partitions, label_type=args.label_type, restrict=restrict, device=device)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=valid_partitions, label_type=args.label_type, restrict=restrict, device=device)
        test_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=test_partitions, label_type=args.label_type, restrict=restrict, device=device)

    if restrict is None:
        print('Loading data...')
    else:
        print('Loading restricted data...')
    train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            shuffle=True,
            collate_fn=train_set.collate_fn,
            num_workers=0,        
            pin_memory=False,
            persistent_workers=False,  
            prefetch_factor=None,
        )
    
    valid_loader = DataLoader(
            valid_set,
            batch_size=args.batch_size,
            collate_fn=valid_set.collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None
        )
    
    test_loader = DataLoader(test_set,
            batch_size=args.batch_size,
            collate_fn=test_set.collate_fn,
            num_workers=0,
            pin_memory=False,
            persistent_workers=False,
            prefetch_factor=None
        )
    print(f'Loaded data. {len(train_set)} train sequences (p.{train_partitions}), {len(valid_set)} validation sequences (p.{valid_partitions}), {len(test_set)} test sequences (p.{test_partitions}).')

    return train_loader, valid_loader, test_loader


def get_model(args: argparse.Namespace):
    if args.model == 'lstmcnncrf':
        model = LSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            dropout_input=args.dropout,
            num_states= 101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
            feature_extractor=args.feature_extractor
        )
    elif args.model == 'lstmcnncrf_simple':
        model = SimpleLSTMCNNCRF(
            input_size = args.embedding_dim,
            num_labels=3 if args.label_type == 'simple_with_propeptides' else 2,
            dropout_input=args.dropout,
            num_states= 3 if args.label_type == 'simple_with_propeptides' else 2,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_conv1=args.conv_dropout,
        )

    # NOTE just use already existing CLI args with names that don't really match. Works.
    elif args.model == 'selfattentioncrf':
        model = SelfAttentionCRF(
            input_size = args.embedding_dim,
            hidden_size= args.hidden_size,
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            dropout_input=args.dropout,
            num_states= 121 if 'with_propeptides' in args.label_type else 61,
            n_heads=args.num_filters,
            attn_dropout=args.conv_dropout,
        )
    elif args.model == 'lstmcnncrf_projector':
        model = LSTMCNNCRFProjector(
            input_size=args.embedding_dim,
            # projector:
            proj_size=256,                 # D_proj (можешь менять)
            dropout_projector=0.4, # или фикс 0.2
            # остальное как в LSTMCNNCRF:
            num_labels=3 if 'with_propeptides' in args.label_type else 2,
            num_states=101 if 'with_propeptides' in args.label_type else 51,
            n_filters=args.num_filters,
            hidden_size=args.hidden_size,
            filter_size=args.kernel_size,
            dropout_input=args.dropout,
            dropout_conv1=args.conv_dropout,
        )
    else:
        raise NotImplementedError(args.model)

    print('trainable params: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    return model


def train_homo_loo(args, run = None):
    ''' 
    DEPRECATED
    '''
    assert torch.cuda.is_available()
    device = f'cuda:{args.device}'
    
    with open('./data/protein_id_homo.txt', 'r') as f:
        homo_ids = [line.strip() for line in f if line.strip()]
        
    run['hparams'] = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'feature_extractor': args.feature_extractor,
        'homo_only': args.homo_only
    }
    
    for i in tqdm(range(0, len(homo_ids), args.K), desc='LOO partitions'):
        train_loader, valid_loader = get_dataloaders_loo(args, list(range(5)), restrict=homo_ids[:i] + homo_ids[min(i+args.K, len(homo_ids)):], leave_idxs=homo_ids[i:i+args.K], device=device)
        
        model = get_model(args).to(device)

        if args.feature_extractor == 'LSTMCNN':
            model.feature_extractor.biLSTM.flatten_parameters()

        optimizer = Adam(model.parameters(), lr = args.lr)
        use_amp = getattr(args, "amp", False)

        amp_dtype = torch.bfloat16  # default
        if getattr(args, "amp_dtype", "bf16") == "fp16":
            amp_dtype = torch.float16

        scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))
        
        for epoch in tqdm(range(args.epochs)):
            train_loss, train_probs, train_preds, train_peptides, train_labels = run_dataloader(train_loader, model, optimizer, do_train=True, device=device, scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype, collect_outputs=False)
            
            # Log valid metrics
            valid_loss, valid_probs, valid_preds, valid_peptides, valid_labels = run_dataloader(valid_loader, model, optimizer, do_train=False, device=device, scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype)
            valid_metrics = compute_all_metrics(valid_probs, valid_preds, valid_labels, valid_loader.dataset.names, valid_loader.dataset.data, windows = [3])[0]
            log_metrics(valid_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix=f'Part {i//args.K} of LLO with K={args.K} (Homo)', aim_run=run, epoch=epoch)     
        
    return None

def train(args, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], run = None):
    
    # FOR 4060TI 16GB
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")

    assert torch.cuda.is_available()
    device = f'cuda:{args.device}'

    with open('./data/protein_id_homo.txt', 'r') as f:
        homo_ids = [line.strip() for line in f if line.strip()]
        
    if args.homo_only:
        # return train_homo_loo(args, run)
        train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions, device=device, restrict=homo_ids)
    else:
        train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions, device=device)
        homo_train_loader, homo_valid_loader, homo_test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions, restrict=homo_ids, device=device)
    
    
    if not os.path.exists(args.checkpoints_dir):
        os.mkdir(args.checkpoints_dir)

    model = get_model(args).to(device)

    if args.feature_extractor == 'LSTMCNN':
        model.feature_extractor.biLSTM.flatten_parameters()

    optimizer = Adam(model.parameters(), lr = args.lr)
    use_amp = getattr(args, "amp", False)

    amp_dtype = torch.bfloat16  # default
    if getattr(args, "amp_dtype", "bf16") == "fp16":
        amp_dtype = torch.float16

    scaler = GradScaler("cuda", enabled=(use_amp and amp_dtype == torch.float16))

    previous_best = -100000000000

    run['hparams'] = {
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'feature_extractor': args.feature_extractor,
        'homo_only': args.homo_only
    }

    for epoch in tqdm(range(args.epochs)):
        train_loss, train_probs, train_preds, train_peptides, train_labels = run_dataloader(train_loader, model, optimizer, do_train=True, device=device, scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype, collect_outputs=False)
        
        # Log valid metrics
        valid_loss, valid_probs, valid_preds, valid_peptides, valid_labels = run_dataloader(valid_loader, model, optimizer, do_train=False, device=device, scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype)
        valid_metrics = compute_all_metrics(valid_probs, valid_preds, valid_labels, valid_loader.dataset.names, valid_loader.dataset.data, windows = [3])[0]
        log_metrics(valid_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Valid', aim_run=run, epoch=epoch)
        print(f'Epoch {epoch} completed')
        print(f'Metrics: {valid_metrics}')
        print(f'Validation loss {valid_loss:.2f}')
        if ((epoch + 1) % 10 == 0) or epoch == 0:
            torch.save(model.state_dict(), f'{args.checkpoints_dir}/model_{epoch}.pth')    

        # Best checkpoint selection
        stopping_metric = (valid_metrics['f1 peptides'] + valid_metrics['f1 propeptides'])/2
        if stopping_metric > previous_best:
            previous_best = stopping_metric
            best_val_metrics = valid_metrics
            pickle.dump((valid_probs, valid_preds, valid_labels, valid_loader.dataset.names), open(os.path.join(args.out_dir, 'valid_outputs.pickle'), 'wb'))
            valid_metrics['epoch'] = epoch # keep track of best early stopping.
            json.dump(valid_metrics, open(os.path.join(args.out_dir, 'valid_metrics.json'), 'w'), indent=2)
            torch.save(model.state_dict(), os.path.join(args.out_dir, 'model.pt'))
            
        # Log metrics on homo subset
        if not args.homo_only:
            homo_valid_loss, homo_valid_probs, homo_valid_preds, homo_valid_peptides, homo_valid_labels = run_dataloader(homo_valid_loader, model, optimizer, do_train=False, device=device, scaler=scaler, use_amp=use_amp, amp_dtype=amp_dtype)
            homo_valid_metrics = compute_all_metrics(homo_valid_probs, homo_valid_preds, homo_valid_labels, homo_valid_loader.dataset.names, homo_valid_loader.dataset.data, windows = [3])[0]
            log_metrics(homo_valid_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Valid (Homo)', aim_run=run, epoch=epoch)
            print(f'Metrics (Homo): {homo_valid_metrics}')
            print(f'Validation loss (Homo) {homo_valid_loss:.2f}')

    # Log final test_metrics on best checkpoint
    model.load_state_dict(torch.load(os.path.join(args.out_dir, 'model.pt')))
    
    test_loss, test_probs, test_preds, test_peptides, test_labels = run_dataloader(test_loader, model, optimizer, do_train=False, device=device)
    test_metrics = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [3])[0]
    log_metrics(test_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Test', aim_run=run, epoch=epoch)
    pickle.dump((test_probs, test_preds, test_labels, test_loader.dataset.names), open(os.path.join(args.out_dir, 'test_outputs.pickle'), 'wb'))
    json.dump(test_metrics, open(os.path.join(args.out_dir, 'test_metrics.json'), 'w'), indent=2)
    print('Test complete.')
    
    if not args.homo_only:
        homo_test_loss, homo_test_probs, homo_test_preds, homo_test_peptides, homo_test_labels = run_dataloader(homo_test_loader, model, optimizer, do_train=False, device=device)
        homo_test_metrics = compute_all_metrics(homo_test_probs, homo_test_preds, homo_test_labels, homo_test_loader.dataset.names, homo_test_loader.dataset.data, windows = [3])[0]
        log_metrics(homo_test_metrics, os.path.join(args.out_dir, 'all_metrics.txt'), prefix='Test (Homo)', aim_run=run, epoch=epoch)
        pickle.dump((homo_test_probs, homo_test_preds, homo_test_labels, homo_test_loader.dataset.names), open(os.path.join(args.out_dir, 'homo_test_outputs.pickle'), 'wb'))
        json.dump(homo_test_metrics, open(os.path.join(args.out_dir, 'homo_test_metrics.json'), 'w'), indent=2)
        print('Test (Homo) complete.')
    
    return best_val_metrics, test_metrics


def run_dataloader(
    loader: torch.utils.data.DataLoader,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    do_train: bool = True,
    device: str = "cpu",
    scaler=None,                 
    use_amp: bool = False,
    amp_dtype=torch.bfloat16,
    collect_outputs: bool = True,
) -> Tuple[float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    """
    Run a dataloader through the model. Collect predicted probabilities and
    true labels. Can be used both for training and prediction.
    """

    true = []
    labels = []
    probs = []
    preds = []
    epoch_loss = []

    model.train() if do_train else model.eval()

    n = 0
    # fp16 требует scaler, bf16 — нет
    use_scaler = bool(scaler is not None and getattr(scaler, "is_enabled", lambda: False)())
    if amp_dtype != torch.float16:
        use_scaler = False

    for idx, batch in enumerate(loader):
        embeddings, mask, label, peptides = batch

        embeddings = embeddings.to(device, non_blocking=True)
        mask = mask.to(device, non_blocking=True)
        label = label.to(device, non_blocking=True).long()   # <-- фикс dtype обязателен

        if do_train:
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                pos_probs, pos_preds, loss = model(embeddings, mask, label, skip_marginals=True, decode=False, return_probs=False)

            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

            if use_scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
                optimizer.step()

        else:
            with torch.no_grad():
                with torch.amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                    pos_probs, pos_preds, loss = model(embeddings, mask, label)

        if collect_outputs:
            true.extend(peptides)
            probs.append(pos_probs.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            preds.extend(pos_preds)
        epoch_loss.append(float(loss.item()))

        n += 1

    epoch_loss = sum(epoch_loss) / max(1, len(epoch_loss))
    return epoch_loss, probs, preds, true, labels



def parse_arguments():
    '''Parse arguments, prepare output directory and dump run configuration.'''
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_dir', type=str, help='Embeddings dir produced by `extract.py`', default = '/data3/fegt_data/embeddings/')
    p.add_argument('--checkpoints_dir', type=str, help='Dir to save checkpoints', default = './checkpoints')
    p.add_argument('--data_file', '-df', type=str, help='Sequences with Graph-Part headers', default = 'data/labeled_sequences.csv')
    p.add_argument('--partitioning_file', '-pf', type=str, help='Graph-Part output. Assume train-val-test split.', default = 'data/graphpart_assignments.csv')
    p.add_argument('--embedding', '-em', type=str, help='Sequence embedding strategy.', default='precomputed')
    p.add_argument('--embedding_dim', '-ed', type=int, help='Sequence embedding dimension.', default=1280)

    p.add_argument('--model', '-m', type=str, default='lstmcnncrf')

    p.add_argument('--out_dir', '-od', type=str, help='name that will be added to the runs folder output', default='runs/train_run')
    p.add_argument('--epochs', type=int, default=100, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', '-bs', type=int, default=48, help='samples that will be processed in parallel')

    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--conv_dropout', type=float, default=0.1)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--num_filters', type=int, default=32)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--port', type=int, default=12355)
    p.add_argument('--feature_extractor', type=str, default='LSTMCNN')
    p.add_argument('--homo_only', action='store_true')
    p.add_argument('--K', type=int, default=10)
    p.add_argument('--amp', action='store_true', help='Enable mixed precision (AMP)')
    p.add_argument('--amp_dtype', type=str, default='bf16', choices=['fp16','bf16'])

    p.add_argument('--label_type', type=str, default='multistate_with_propeptides')

    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'config.json'), 'w'), indent=3)

    return args
