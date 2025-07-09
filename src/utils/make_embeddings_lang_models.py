import os

import ankh
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel, T5Tokenizer

from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_esm3_model_tokenizers

import argparse
import pathlib
from Bio import SeqIO


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def select_model_tokenizer(model_name: str):
    if model_name == "ankh":
        model, tokenizer = ankh.load_large_model()

    elif model_name == "esm":
        tokenizer = get_esm3_model_tokenizers()
        model = ESM3_sm_open_v0()

    elif model_name == "prot5":
        print('getting ProtT5')
        # tokenizer = T5Tokenizer.from_pretrained(
        #     "Rostlab/prot_t5_xl_uniref50", do_lower_case=False
        # )
        tokenizer = T5Tokenizer.from_pretrained(
            "Rostlab/ProstT5", do_lower_case=False
        )
        # model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
        model = T5EncoderModel.from_pretrained("Rostlab/ProstT5")

    return model, tokenizer


def calculate_embeds(
    tokenizer, model, seq: str, model_name: str, device: torch.device
) -> np.ndarray:
    if model_name == "ankh":
        inputs = tokenizer(
            [seq],
            add_special_tokens=False,
            padding=False,
            is_split_into_words=True,
            return_tensors="pt",
        )

        with torch.no_grad():
            inputs.to(device)
            output = model(**inputs)

    elif model_name == "esm":
        inputs = tokenizer.sequence.encode(seq)# , return_tensors="pt")
        inputs = torch.tensor(inputs).unsqueeze(0)
        with torch.no_grad():
            inputs = inputs.to(device)
            output = model(sequence_tokens=inputs)
            embedding = output.embeddings.cpu().squeeze(0)[1:-1, :]
            # output = model(**inputs)

    elif model_name == "prot5":
        item = []
        for i in range(len(seq)):
            if i != 0 and i != len(seq):
                item.append(" ")
            item.append(seq[i])

        item = ["".join(item)]

        ids = tokenizer.batch_encode_plus(item, add_special_tokens=False, padding=False)
        input_ids = torch.tensor(ids["input_ids"]).to(device)
        attention_mask = torch.tensor(ids["attention_mask"]).to(device)

        with torch.no_grad():
            output = model(input_ids=input_ids, attention_mask=attention_mask)
    if model_name != 'esm':
        embedding = output.last_hidden_state.cpu().numpy()  # mean(axis=1).view(-1)
        embedding = np.squeeze(embedding)
    return embedding


def get_embeds(
    seqs: dict, model_name: str,
        device: torch.device) -> dict[str, np.ndarray]:

    model, tokenizer = select_model_tokenizer(model_name)
    model.to(device)
    model.eval()
    print('models loaded.')

    outputs = {}
    for identifier, seq in tqdm(seqs.items()):
        if os.path.isfile(f'{args.output_dir}/{identifier}.pt'):
            continue
        embedding = calculate_embeds(tokenizer, model, seq, model_name, device)
        # outputs[identifier] = embedding
        output_file = open(f'{args.output_dir}/{identifier}.pt', 'wb')
        torch.save(embedding, output_file)
        output_file.close()
    # return outputs

def build_id_to_sequence_dict(fasta_path):
    """
    Читает FASTA-файл и создаёт словарь {id_белка: последовательность}.
    """
    id_to_seq = {}
    with open(fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            id_to_seq[record.id] = str(record.seq)
    return id_to_seq

def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    if args.cuda_no > num_gpus:
      raise ValueError(f'No cuda device no. {args.cuda_no}. {num_gpus} devices available.')
    device = f'cuda:{args.cuda_no}'
    device_name = torch.cuda.get_device_name(args.cuda_no)
    print(f'Running on {device} {device_name}')
    print('________________________________________________________________')

    print('looking for seqs...')
    seqs = build_id_to_sequence_dict(args.seq_file)
    print('found them all')

    get_embeds(seqs, args.model, device)
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('seq_file',
                        type=pathlib.Path,
                        help='Path to fasta file with protein sequences')
    parser.add_argument('output_dir',
                        type=pathlib.Path,
                        help="output directory for extracted representations"
                        )
    parser.add_argument('model',
                        type=str,
                        help="model name (ankh, esm, prot5)"
                        )
    parser.add_argument('cuda_no',
                        type=int,
                        help="index of cuda device"
                        )
    args = parser.parse_args()
    main(args)

# python3 src/utils/make_embeddings_esm3.py data/protein_sequences.fasta data/model_embeddings model 1