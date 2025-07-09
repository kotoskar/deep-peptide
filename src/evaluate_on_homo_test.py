import json
import pickle
from typing import Dict, List, Tuple
import os
from torch.utils.data import DataLoader

from models import LSTMCNNCRF, SimpleLSTMCNNCRF, SelfAttentionCRF
from utils.dataset import PrecomputedCSVForOverlapCRFDataset
#from .utils.metrics_cleaned import compute_metrics, compute_metrics_with_propeptides
from utils.manuscript_metrics import compute_all_metrics
from torch.optim import Adam
import torch
import numpy as np
import argparse
# from torch.utils.tensorboard import SummaryWriter

from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from fairscale.nn.wrap import enable_wrap, wrap

from tqdm import tqdm



def parse_arguments():
    '''Parse arguments, prepare output directory and dump run configuration.'''
    p = argparse.ArgumentParser()

    p.add_argument('--embeddings_dir', type=str, help='Embeddings dir produced by `extract.py`', default = '/data3/fegt_data/embeddings/')
    p.add_argument('--checkpoints_dir', type=str, help='Dir to save checkpoints', default = './checkpoints')
    p.add_argument('--data_file', '-df', type=str, help='Sequences with Graph-Part headers', default = 'data/uniprot_12052022_cv_5_50/labeled_sequences.csv')
    p.add_argument('--partitioning_file', '-pf', type=str, help='Graph-Part output. Assume train-val-test split.', default = 'data/uniprot_12052022_cv_5_50/graphpart_assignments.csv')
    p.add_argument('--embedding', '-em', type=str, help='Sequence embedding strategy.', default='precomputed')
    p.add_argument('--embedding_dim', '-ed', type=int, help='Sequence embedding dimension.', default=1280)
    p.add_argument('--model', '-m', type=str, default='lstmcnncrf')
    p.add_argument('--out_dir', '-od', type=str, help='name that will be added to the runs folder output', default='train_run')
    p.add_argument('--epochs', type=int, default=100, help='number of times to iterate through all samples')
    p.add_argument('--batch_size', '-bs', type=int, default=100, help='samples that will be processed in parallel')
    p.add_argument('--lr', type=float, default=1e-4)
    p.add_argument('--dropout', type=float, default=0.1)
    p.add_argument('--conv_dropout', type=float, default=0.1)
    p.add_argument('--kernel_size', type=int, default=3)
    p.add_argument('--num_filters', type=int, default=32)
    p.add_argument('--hidden_size', type=int, default=64)
    p.add_argument('--device', type=int, default=0)
    p.add_argument('--port', type=int, default=12355)
    p.add_argument('--feature_extractor', type=str, default='LSTMCNN')
    p.add_argument('--label_type', type=str, default='multistate_with_propeptides')
    
    p.add_argument('--model_path', type=str)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out_dir, 'config.json'), 'w'), indent=3)

    return args


def log_metrics(metrics: dict, filepath: str, prefix: str = ""):
    """Log metrics from a dictionary to a file, one per line, sorted by key. Optionally add a prefix to each line."""
    with open(filepath, 'a') as f:
        for key in sorted(metrics.keys()):
            f.write(f"{prefix}{key}: {metrics[key]}\n")


def get_dataloaders(args: argparse.Namespace, train_partitions: List[int] = [0,1,2], valid_partitions: List[int] = [3], test_partitions: List[int] = [4], test_restrict = None) -> Tuple[DataLoader, DataLoader, DataLoader]:

    if args.embedding == 'precomputed':
        train_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=train_partitions, label_type=args.label_type)
        valid_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=valid_partitions, label_type=args.label_type)
        test_set = PrecomputedCSVForOverlapCRFDataset(args.embeddings_dir, args.data_file, args.partitioning_file, partitions=test_partitions, label_type=args.label_type, restrict=test_restrict)

    print(f'Loaded data. {len(train_set)} train sequences (p.{train_partitions}), {len(valid_set)} validation sequences (p.{valid_partitions}), {len(test_set)} test sequences (p.{test_partitions}).')
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, collate_fn=train_set.collate_fn, num_workers=2)
    valid_loader = DataLoader(valid_set, batch_size=args.batch_size, collate_fn=valid_set.collate_fn, num_workers=1)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, collate_fn=valid_set.collate_fn, num_workers=1)

    return train_loader, valid_loader, test_loader

def run_dataloader(loader: torch.utils.data.DataLoader,
                    model: torch.nn.Module,
                    optimizer: torch.optim.Optimizer,
                    do_train: bool = True,
                    device = 'cpu'
                ) -> Tuple[float, List[np.ndarray], List[List[int]], List[np.ndarray], List[np.ndarray]]:
    '''
    Run a dataloader through the model. Collect predicted probabilitities and
    true labels. Can be used both for training and prediction.
    '''
    global global_step

    true = [] # peptide coordinates
    labels = [] # labels made from coordinates
    probs = [] # per-position probabilities
    preds = [] # viterbi paths
    epoch_loss = []

    if do_train:
        model.train()
    else:
        model.eval()

    for idx, batch in enumerate(loader):

        model.zero_grad()

        embeddings, mask, label, peptides = batch
        embeddings = embeddings.to(device)
        mask = mask.to(device)
        label = label.to(device)

        if do_train:
            pos_probs, pos_preds, loss = model(embeddings, mask, label, skip_marginals=True)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
            loss.backward()
            optimizer.step()
            # writer.add_scalar('Train/loss', loss.item(), global_step=global_step)
            global_step += 1
        else:
            with torch.no_grad():
                pos_probs, pos_preds, loss = model(embeddings, mask, label)

        true.extend(peptides)
        probs.append(pos_probs.detach().cpu().numpy())
        labels.append(label.detach().cpu().numpy())
        preds.extend(pos_preds)
        epoch_loss.append(loss.item())


    epoch_loss = sum(epoch_loss)/len(epoch_loss)

    return epoch_loss, probs, preds, true, labels



if __name__ == "__main__":
    args = parse_arguments()
    device = f'cuda:{args.device}'
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

    with open('./data/protein_id_homo.txt', 'r') as f:
        homo_ids = [line.strip() for line in f if line.strip()]

    model.load_state_dict(torch.load(args.model_path))
    model.to(device)
    train_partitions = [0,1,2]
    valid_partitions = [3]
    test_partitions = [4]
    is_initiated = False

    optimizer = Adam(model.parameters(), lr = args.lr)

    train_loader, valid_loader, test_loader = get_dataloaders(args, train_partitions, valid_partitions, test_partitions, test_restrict=homo_ids)
    test_loss, test_probs, test_preds, test_peptides, test_labels = run_dataloader(test_loader, model, optimizer, do_train=False, device=device)
    test_metrics = compute_all_metrics(test_probs, test_preds, test_labels, test_loader.dataset.names, test_loader.dataset.data, windows = [3])[0]
    json.dump(test_metrics, open(os.path.join(args.out_dir, 'test_metrics_homo.json'), 'w'), indent=2)