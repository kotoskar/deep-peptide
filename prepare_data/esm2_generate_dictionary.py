# Aim of this script is to prepare a dictionary with keys being a string of 2 aminoacids
# and meanings being corresponding embeddings generated using ESM-2 model

import torch
import esm
from itertools import product
from typing import List, Dict
# from esm import Alphabet, FastaBatchedDataset, ProteinBertModel, pretrained, FastaBatchedDataset


# 1. Define amino acid alphabet (21 letters including "X")
def get_amino_acids() -> List[str]:
    return list("ACDEFGHIKLMNPQRSTVWYX")


# 2. Generate all 21x21 amino acid pairs
def generate_amino_acid_pairs(alphabet: List[str]) -> List[str]:
    return [a1 + a2 for a1, a2 in product(alphabet, repeat=2)]


# 3. Load pretrained ESM model
def load_esm_model(model_name: str = "esm2_t33_650M_UR50D"):
    model, alphabet = esm.pretrained.load_model_and_alphabet(model_name)
    model.eval()
    if torch.cuda.is_available():
        model = model.cuda()
    return model, alphabet


# 4. Compute 2D embedding (one per amino acid in pair)
def compute_2d_embedding(seq: str, model, alphabet) -> torch.Tensor:
    batch_converter = alphabet.get_batch_converter()
    batch = [("pair", seq)]
    _, _, tokens = batch_converter(batch)
    if torch.cuda.is_available():
        tokens = tokens.cuda()
    with torch.no_grad():
        results = model(tokens, repr_layers=[33], return_contacts=False)
    token_representations = results["representations"][33]
    # Slice off BOS (0) and EOS (-1), return [2, emb_dim]
    return token_representations[1:-1].cpu()


# 5. Compute 2D embeddings for all pairs
def compute_all_pair_embeddings(pairs: List[str], model, alphabet) -> Dict[str, torch.Tensor]:
    embedding_dict = {}
    for pair in pairs:
        emb_2d = compute_2d_embedding(pair, model, alphabet)
        embedding_dict[pair] = emb_2d  # shape: [2, emb_dim]
    return embedding_dict


# 6. Entry point
def main():
    aa_alphabet = get_amino_acids()
    aa_pairs = generate_amino_acid_pairs(aa_alphabet)
    model, alphabet = load_esm_model()
    embeddings = compute_all_pair_embeddings(aa_pairs, model, alphabet)

    # Save full 2D embeddings
    torch.save(embeddings, "/home/user14/data/train-test_dataset/embeddings_esm2/amino_acid_pair_2d_embeddings.pt")
    print(f"Saved 2D embeddings for {len(embeddings)} pairs.")


if __name__ == "__main__":
    main()
