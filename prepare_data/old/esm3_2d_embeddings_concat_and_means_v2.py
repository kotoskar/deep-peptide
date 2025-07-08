# Aim of this script is to get the embeddings of flanking regions based on ESM-3
# Concatenate embeddings for start and end of each peptide/propeptide
# Get the average of concatenated embeddings for each protein
# Use pre-computed embeddings

import torch
import pandas as pd
import os
from tqdm import tqdm
from collections import defaultdict
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_esm3_model_tokenizers

class HybridEmbedder:
    def __init__(self, device: str = 'cpu'):
        self.device = torch.device(device)
        
        # Load pre-computed ESM-3 embeddings
        self.pair_embeddings = torch.load(
            "/home/user14/data/train-test_dataset/embeddings_esm3/amino_acid_pair_2d_embeddings.pt"
        )
        self.embed_dim = next(iter(self.pair_embeddings.values())).shape[-1]
        
        # Initialize ESM-3 model only for special cases
        self.model, self.tokenizers = None, None
        self.at_embedding = torch.randn(self.embed_dim, device=self.device)
    
    def _init_esm3(self):
        """Lazy initialization of ESM-3 for special cases"""
        if self.model is None:
            self.model = ESM3_sm_open_v0().eval().to(self.device)
            self.tokenizers = get_esm3_model_tokenizers()
    
    def embed_sequence(self, sequence: str) -> torch.Tensor:
        """Generate 2*embed_dim embedding using hybrid approach"""
        if pd.isna(sequence) or not sequence:
            return torch.zeros(self.embed_dim * 2, device=self.device)
        
        # Handle special @ cases with ESM-3
        if '@' in sequence:
            self._init_esm3()
            if sequence == '@':
                return torch.cat([self.at_embedding, self.at_embedding])
            elif sequence == '@@':
                return torch.cat([self.at_embedding, self.at_embedding])
            else:
                return self._process_mixed_sequence(sequence)
        
        # Use pre-computed embeddings for normal AA pairs
        if sequence in self.pair_embeddings:
            return self.pair_embeddings[sequence].mean(dim=0)  # Average the two AAs
        
        # Fallback for unknown pairs (shouldn't happen with proper input)
        print(f"Warning: No embedding found for '{sequence}', using zeros")
        return torch.zeros(self.embed_dim * 2, device=self.device)
    
    def _process_mixed_sequence(self, sequence: str) -> torch.Tensor:
        """Process sequences with @ symbols using ESM-3"""
        embeddings = []
        for char in sequence:
            if char == '@':
                embeddings.append(self.at_embedding)
            else:
                tokens = self.tokenizers.sequence.encode(char)
                with torch.no_grad():
                    outputs = self.model(sequence_tokens=torch.tensor(tokens).unsqueeze(0).to(self.device))
                emb = outputs.embeddings[0, 1:-1].mean(dim=0)  # Remove BOS/EOS
                embeddings.append(emb)
        return torch.cat(embeddings)

def process_hybrid(csv_path: str, output_dir: str, device: str = 'cpu'):
    embedder = HybridEmbedder(device)
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    protein_embeddings = defaultdict(list)
    expected_size = embedder.embed_dim * 4  # 4*embed_dim for pre+post

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):
        protein_id = row['protein_id']
        coords = row['coordinates'].replace('-', '_')
        
        # Get pre and post embeddings (each 2*embed_dim)
        pre_emb = embedder.embed_sequence(row['pre-region'])
        post_emb = embedder.embed_sequence(row['post-region'])
        
        # Combine to 4*embed_dim
        combined = torch.cat([pre_emb, post_emb])
        assert combined.shape[0] == expected_size, \
               f"Size mismatch: {combined.shape[0]} != {expected_size}"
        
        # Save individual
        torch.save(combined, os.path.join(output_dir, f"{protein_id}_{coords}.pt"))
        protein_embeddings[protein_id].append(combined)
    
    # Save mean embeddings
    mean_dir = os.path.join(output_dir, "mean_embeddings")
    os.makedirs(mean_dir, exist_ok=True)
    
    for prot_id, emb_list in tqdm(protein_embeddings.items(), desc="Averaging"):
        mean_emb = torch.stack(emb_list).mean(dim=0)
        torch.save(mean_emb, os.path.join(mean_dir, f"{prot_id}_mean.pt"))

if __name__ == "__main__":
    config = {
        "csv_path": "/home/user14/data/train-test_dataset/flanking_regions_no_missing.csv",
        "output_dir": "/home/user14/data/train-test_dataset/esm3_processed_embeddings_2d_v2/",
        "device": "cuda:1" if torch.cuda.is_available() else "cpu"
    }
    process_hybrid(**config)