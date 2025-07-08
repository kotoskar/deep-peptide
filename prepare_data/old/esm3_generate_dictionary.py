# Aim of this script is to prepare a dictionary with keys being a string of 2 aminoacids
# and meanings being corresponding embeddings generated using ESM-3 model

import torch
from itertools import product
from typing import List, Dict
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_esm3_model_tokenizers


# 1. Define amino acid alphabet (21 letters including "X")
def get_amino_acids() -> List[str]:
    return list("ACDEFGHIKLMNPQRSTVWYX")


# 2. Generate all 21x21 amino acid pairs
def generate_amino_acid_pairs(alphabet: List[str]) -> List[str]:
    return [a1 + a2 for a1, a2 in product(alphabet, repeat=2)]


# 3. Load pretrained ESM-3 model with specific device (cuda:1 if available)
def load_esm_model():
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    tokenizers = get_esm3_model_tokenizers()
    model = ESM3_sm_open_v0()
    model = model.eval().to(device)
    return model, tokenizers, device


# 4. Compute 2D embedding (one per amino acid in pair)
def compute_2d_embedding(seq: str, model, tokenizers, device) -> torch.Tensor:
    tokens = tokenizers.sequence.encode(seq)
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(sequence_tokens=tokens)
    
    # Get embeddings and remove BOS/EOS tokens
    token_representations = outputs.embeddings.cpu().squeeze(0)[1:-1]
    return token_representations  # shape: [2, emb_dim]


# 5. Compute 2D embeddings for all pairs
def compute_all_pair_embeddings(pairs: List[str], model, tokenizers, device) -> Dict[str, torch.Tensor]:
    embedding_dict = {}
    for pair in pairs:
        emb_2d = compute_2d_embedding(pair, model, tokenizers, device)
        embedding_dict[pair] = emb_2d  # shape: [2, emb_dim]
    return embedding_dict


# 6. Entry point
def main():
    aa_alphabet = get_amino_acids()
    aa_pairs = generate_amino_acid_pairs(aa_alphabet)
    model, tokenizers, device = load_esm_model()
    embeddings = compute_all_pair_embeddings(aa_pairs, model, tokenizers, device)

    # Save full 2D embeddings
    torch.save(embeddings, "/home/user14/data/train-test_dataset/embeddings_esm3/amino_acid_pair_2d_embeddings.pt")
    print(f"Saved 2D embeddings for {len(embeddings)} pairs.")


if __name__ == "__main__":
    main()