# python3 src/utils/make_embeddings_saprot.py src/utils/Foldseek/foldseek data/structures data/protein_sequences.fasta data/SaProt_embeddings 1

import os
import argparse

from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer
from SaProt.model.saprot.base import SaprotBaseModel
from transformers import EsmTokenizer
from scipy.stats import spearmanr
import numpy as np
import pandas as pd

import torch
from torch.nn import CrossEntropyLoss
from Foldseek.foldseek_util import get_struc_seq
from tqdm import tqdm
import pathlib
from Bio import SeqIO

foldseek_struc_vocab = "pynwrqhgdlvtmfsaeikc#"

def calc_embedding(foldseek_bin, model, tokenizer, seq, pdb_file=None, device='cuda:0'):
    # Get 3Di sequence
    struc_seq = get_struc_seq(foldseek_bin, pdb_file, ["A"], plddt_mask=True, plddt_threshold=70)["A"][1].lower()

    seq = "".join([a + b for a, b in zip(seq, struc_seq)])
    
    #tokens = tokenizer.tokenize(seq)
    inputs = tokenizer(seq, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    embeddings = model.get_hidden_states(inputs, reduction="mean")
    # print(embeddings[0].shape)

    return embeddings[0]


def build_id_to_sequence_dict(fasta_path):
    """
    Читает FASTA-файл и создаёт словарь {id_белка: последовательность}.
    """
    id_to_seq = {}
    with open(fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            id_to_seq[record.id] = str(record.seq)
    return id_to_seq


def main():
    """
    Main script to score sets of mutated protein sequences (substitutions or indels) with SaProt.
    """
    parser = argparse.ArgumentParser(description='SaProt scoring')
    parser.add_argument('foldseek_bin',
                        default="",
                        type=str, help='Path to foldseek binary file')
    parser.add_argument('struc_dir',
                        type=pathlib.Path,
                        help='Path to folder with structures (.pdb)')
    parser.add_argument('seq_file',
                        type=pathlib.Path,
                        help='Path to fasta file with protein sequences')
    parser.add_argument('output_dir',
                        type=pathlib.Path,
                        help="output directory for extracted representations"
                        )
    parser.add_argument('cuda_no',
                        type=int,
                        help="index of cuda device"
                        )
    args = parser.parse_args()

    num_gpus = torch.cuda.device_count()
    if args.cuda_no > num_gpus:
      raise ValueError(f'No cuda device no. {args.cuda_no}. {num_gpus} devices available.')
    device = f'cuda:{args.cuda_no}'
    device_name = torch.cuda.get_device_name(args.cuda_no)
    print(f'Running on {device} {device_name}')

    config = {
    "task": "base",
    "config_path": "/home/user14/DeepPeptide/src/utils/SaProt_650M_AF2", # Note this is the directory path of SaProt, not the ".pt" file
    "load_pretrained": True,
    }
    model = SaprotBaseModel(**config)
    # model = AutoModelForMaskedLM.from_pretrained("westlake-repl/SaProt_650M_AF2")
    model.to(device)
    tokenizer = EsmTokenizer.from_pretrained(config["config_path"])
    # tokenizer = AutoTokenizer.from_pretrained("westlake-repl/SaProt_650M_AF2")

    pdb_files = os.listdir(path=args.struc_dir)
    pdb_paths = [str(args.struc_dir / pdb) for pdb in pdb_files]
    names = [pdb[:-4] for pdb in pdb_files]

    print('looking for seqs...')
    seqs = build_id_to_sequence_dict(args.seq_file)
    print('found them all')

    for prot_id, pdb_file in tqdm(zip(names, pdb_paths)):
        emb_filename = args.output_dir / (prot_id + '.csv')
        if os.path.exists(emb_filename):
            print("Scores already computed for: {}".format(prot_id))
            continue
        # pdb_range = [int(x) for x in pdb_ranges[pdb_index].split("-")]
        emb = calc_embedding(foldseek_bin=args.foldseek_bin, model=model,
                            tokenizer=tokenizer, seq=seqs[prot_id], pdb_file=pdb_file, device=device)
        
        output_file = open(f'{args.output_dir}/{prot_id}.pt', 'wb')
        torch.save(emb, output_file)
        output_file.close()
        


if __name__ == '__main__':
    main()