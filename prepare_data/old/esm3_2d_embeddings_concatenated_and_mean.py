# Aim of this script is to generate embeddings based on ESM-3 model for flanking regions.
# Concatenate embeddings for each peptide or propeptide (start and end).
# Get average from all embeddings assigned to the same protein_id

import torch
import pandas as pd
import os
from tqdm import tqdm
from typing import Dict, DefaultDict
from collections import defaultdict
import numpy as np

class FlankingRegionProcessor:
    def __init__(self, pair_embeddings_path: str, device: str = 'cpu'):
        self.device = device
        self.pair_embeddings = torch.load(pair_embeddings_path)
        self._initialize_special_embeddings()
    
    def _initialize_special_embeddings(self):
        sample_embedding = next(iter(self.pair_embeddings.values()))
        self.embed_dim = sample_embedding.shape[-1]
        self.at_embedding = torch.randn(self.embed_dim, device=self.device)
    
    def _get_aa_embedding(self, aa: str) -> torch.Tensor:
        if aa == '@':
            return self.at_embedding
        dummy_pair = aa + aa
        return self.pair_embeddings.get(dummy_pair, 
                                      torch.zeros(self.embed_dim, device=self.device))[0]
    
    def get_region_embedding(self, region: str) -> torch.Tensor:
        if pd.isna(region):
            return torch.zeros(self.embed_dim * 2, device=self.device)
        
        region = str(region)
        if not region:
            return torch.zeros(self.embed_dim * 2, device=self.device)
        
        embeddings = [self._get_aa_embedding(char) for char in region]
        concatenated = torch.cat(embeddings * 2) if len(region) == 1 else torch.cat(embeddings)
        return concatenated

def process_flanking_regions(csv_path: str, pair_embeddings_path: str, output_dir: str, device: str = 'cpu'):
    processor = FlankingRegionProcessor(pair_embeddings_path, device)
    df = pd.read_csv(csv_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Storage for protein-wise embeddings
    protein_embeddings = defaultdict(list)
    embedding_size = None
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing regions"):
        protein_id = row['protein_id']
        coordinates = row['coordinates'].replace('-', '_')
        
        # Get individual embedding (4*embed_dim)
        pre_emb = processor.get_region_embedding(row['pre-region'])
        post_emb = processor.get_region_embedding(row['post-region'])
        combined_emb = torch.cat([pre_emb, post_emb])
        
        # Verify consistent size
        if embedding_size is None:
            embedding_size = combined_emb.shape[0]
        assert combined_emb.shape[0] == embedding_size, \
               f"Inconsistent embedding size: {combined_emb.shape[0]} vs {embedding_size}"
        
        # Save individual embedding
        torch.save(combined_emb, os.path.join(output_dir, f"{protein_id}_{coordinates}.pt"))
        
        # Store for mean calculation
        protein_embeddings[protein_id].append(combined_emb)
    
    # Calculate and save mean embeddings
    mean_dir = os.path.join(output_dir, "mean_embeddings")
    os.makedirs(mean_dir, exist_ok=True)
    
    for protein_id, embeddings in tqdm(protein_embeddings.items(), desc="Calculating means"):
        stacked = torch.stack(embeddings)
        mean_emb = torch.mean(stacked, dim=0)
        
        # Verify mean embedding size matches
        assert mean_emb.shape[0] == embedding_size, \
               f"Mean embedding size mismatch: {mean_emb.shape[0]} vs {embedding_size}"
        
        torch.save(mean_emb, os.path.join(mean_dir, f"{protein_id}_mean.pt"))
    
    print(f"\nProcessing complete. All embeddings have size {embedding_size}")
    print(f"Individual embeddings saved to: {output_dir}")
    print(f"Mean embeddings saved to: {mean_dir}")

def main():
    config = {
        "pair_embeddings_path": "/home/user14/data/train-test_dataset/embeddings_esm2/amino_acid_pair_2d_embeddings.pt",
        "flanking_regions_csv": "/home/user14/data/train-test_dataset/flanking_regions_no_missing.csv",
        "output_dir": "/home/user14/data/train-test_dataset/combined_flanking_embeddings",
        "device": "cuda:1" if torch.cuda.is_available() else "cpu"
    }
    
    process_flanking_regions(
        csv_path=config["flanking_regions_csv"],
        pair_embeddings_path=config["pair_embeddings_path"],
        output_dir=config["output_dir"],
        device=config["device"]
    )

if __name__ == "__main__":
    main()