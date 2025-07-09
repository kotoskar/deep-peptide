from transformers import AutoModelForMaskedLM, AutoTokenizer
import argparse
import pathlib
import os
from tqdm import tqdm
from hashlib import md5
import torch
# from .ProSST.prosst.structure.quantizer import PdbQuantizer
from ProSST.prosst.structure.quantizer import PdbQuantizer
from Bio import SeqIO
import time


_MODEL = None
_TOKENIZER = None
_PROCESSOR = None


def _load_models(device):
  """Load the ProSST model, tokenizer, and processor if not already loaded.

  Returns
  -------
  tuple
    A tuple containing (model, tokenizer, processor)
  """
  global _MODEL, _TOKENIZER, _PROCESSOR
  try:
    if _MODEL is None:
      _MODEL = AutoModelForMaskedLM.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True, output_hidden_states=True)
      _MODEL.eval()  # Set model to evaluation mode
      _MODEL.to(device)
    if _TOKENIZER is None:
      _TOKENIZER = AutoTokenizer.from_pretrained("AI4Protein/ProSST-2048", trust_remote_code=True, output_hidden_states=True)
    if _PROCESSOR is None:
      _PROCESSOR = PdbQuantizer(device=device)
    return _MODEL, _TOKENIZER, _PROCESSOR
  except Exception as e:
    raise RuntimeError(f"Failed to load ProSST models: {str(e)}")


def build_id_to_sequence_dict(fasta_path):
    """
    Читает FASTA-файл и создаёт словарь {id_белка: последовательность}.
    """
    id_to_seq = {}
    with open(fasta_path, "r") as fasta_file:
        for record in SeqIO.parse(fasta_file, "fasta"):
            id_to_seq[record.id] = str(record.seq)
    return id_to_seq


def run_prosst(input_seq, pdb_fpath, device):
    """
    Parameters
    ----------
    input_seq : str
        The input protein sequence.
    pdb_fpath : str
        The path to the PDB file of the protein structure.

    Returns
    -------
    pred_scores : pandas.DataFrame
        The pivoted DataFrame containing the mutation score matrix.

    Raises
    ------
    FileNotFoundError
        If the PDB file does not exist.
    RuntimeError
        If there's an error during model inference or processing.
    """
    if not os.path.exists(pdb_fpath):
        raise FileNotFoundError(f"PDB file not found: {pdb_fpath}")

    # if not input_seq or not all(aa in "".join(SINGLE_LETTER_CODES) for aa in input_seq):
    #   raise ValueError(f"Invalid protein sequence: {input_seq}")

    # Load models only once
    print('Loading models...')
    model, tokenizer, processor = _load_models(device)
    # Process structure
    print('Processing structure...')
    structure_sequence = processor(pdb_fpath)
    structure_key = os.path.basename(pdb_fpath)
    # print(structure_sequence)
    structure_sequence_offset = [i + 3 for i in structure_sequence]#["2048"][structure_key]["struct"]]
    structure_sequence_length = len(structure_sequence)#["2048"][structure_key]["struct"])
    if len(input_seq) != structure_sequence_length:
        raise ValueError(
        f"Input sequence length ({len(input_seq)}) does not match PDB structure sequence length "
        f"({structure_sequence_length}). Please ensure wt_seq matches the sequence in {pdb_fpath}"
    )

    # Tokenize input sequence
    print('Tokenizing...')
    tokenized_res = tokenizer([input_seq], return_tensors="pt")
    input_ids = tokenized_res["input_ids"]
    attention_mask = tokenized_res["attention_mask"]
    structure_input_ids = torch.tensor([1, *structure_sequence_offset, 2], dtype=torch.long).unsqueeze(0)
    print('Running inference...')
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    structure_input_ids = structure_input_ids.to(device)
  # Run model inference
    with torch.no_grad():
        outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        ss_input_ids=structure_input_ids,
    )
    return outputs.hidden_states[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
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
    os.makedirs(args.output_dir, exist_ok=True)
    num_gpus = torch.cuda.device_count()
    if args.cuda_no > num_gpus:
      raise ValueError(f'No cuda device no. {args.cuda_no}. {num_gpus} devices available.')
    device = f'cuda:{args.cuda_no}'
    device_name = torch.cuda.get_device_name(args.cuda_no)
    print(f'Running on {device} {device_name}')
    print('________________________________________________________________')
    pdb_files = os.listdir(path=args.struc_dir)[::-1]
    pdb_paths = [str(args.struc_dir / pdb) for pdb in pdb_files]

    names = [pdb[:-4] for pdb in pdb_files]
    print('looking for seqs...')
    seqs = build_id_to_sequence_dict(args.seq_file)
    print('found them all')
    for i in tqdm(range(len(names))):
    # for i in tqdm(shortrange):
      if i % num_gpus != args.cuda_no:
        continue
      if os.path.isfile(f'{args.output_dir}/{names[i]}.pt'):
        continue
      # try:
      out = run_prosst(seqs[names[i]], pdb_paths[i], device)[0, 1:-1, :]
      out[out!=out] = 0.0
      output_file = open(f'{args.output_dir}/{names[i]}.pt', 'wb')
      torch.save(out, output_file)
      output_file.close()
      # except:
      #   print(f'Skipping {names[i]} due to internal error')
    # time_end = time.perf_counter()
    # print(f'Time taken: {(time_end - time_start):.6f} s.')
    # outs = {pred['name'][:-4]: torch.tensor(pred['2048_sst_seq']).cpu() for pred in preds}

    # for name, out in outs.items():
    #     out[out!=out] = 0.0
    #     output_file = open(f'{args.output_dir}/{name}.pt', 'wb')
    #     torch.save(out, output_file)
    #     output_file.close()
    # 

