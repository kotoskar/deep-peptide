import torch
from pathlib import Path
import pathlib
from esm.models.esm3 import ESM3
from esm.sdk.api import ESMProtein, LogitsConfig
from huggingface_hub import login
import argparse
from Bio import SeqIO
from tqdm import tqdm
import os
from dotenv import load_dotenv

load_dotenv()

def build_id_to_sequence_dict(fasta_path):
    """
    Читает FASTA-файл и создаёт словарь {id_белка: последовательность}.
    """
    id_to_seq = {}
    with open(fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            id_to_seq[record.id] = str(record.seq)
    return id_to_seq


def run_esm3(pdb_file, model):
    protein = ESMProtein.from_pdb(pdb_file)
    tokens = model.encode(protein)

    out = model.logits(
        tokens,
        LogitsConfig(       # request both tracks so gradients are aligned
            sequence=True,
            structure=True,
            return_embeddings=True
        )
        )
    residue_emb = out.embeddings.squeeze(0)[1:-1, :]
    # protein_emb = residue_emb.mean(0)

    return residue_emb



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--struc_dir',
                        type=pathlib.Path,
                        help='Path to folder with structures (.pdb)',
                        default = 'data/structures'
                        )                   
    parser.add_argument('--cuda_no',
                        type=int,
                        help="index of cuda device",
                        default=2,
                        required=False
                        )
    parser.add_argument('--output_dir',
                        type=pathlib.Path,
                        help="output directory for embeddings",
                        default='./embeddings/esm3_struc',
                        required=False
                        )
    args = parser.parse_args()
    
    num_gpus = torch.cuda.device_count()
    if args.cuda_no > num_gpus:
      raise ValueError(f'No cuda device no. {args.cuda_no}. {num_gpus} devices available.')
    device = f"cuda:{args.cuda_no}" if torch.cuda.is_available() else "cpu"

    hf_token = os.getenv("HF_TOKEN")
    login(hf_token)
    model = ESM3.from_pretrained("esm3-sm-open-v1")
    model = model.to(device)

    os.makedirs(args.output_dir, exist_ok=True)

    print(f'Running on {device} {torch.cuda.get_device_name(args.cuda_no)}')
    print('________________________________________________________________')

    pdb_files = os.listdir(path=args.struc_dir)[::-1]
    pdb_paths = {pdb[:-4] : str(args.struc_dir / pdb) for pdb in pdb_files}

    names = [pdb[:-4] for pdb in pdb_files]
    # print('looking for seqs...')
    # seqs = build_id_to_sequence_dict(args.seq_file)
    # print('found them all')

    for name in names:
        filename = f'{args.output_dir}/{name}.pt'
        if os.path.isfile(filename):
            continue
        output_file = open(filename, 'wb')
        emb = run_esm3(pdb_paths[name], model)
        torch.save(emb, output_file)
        output_file.close()
