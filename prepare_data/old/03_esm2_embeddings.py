# Aim of this script is to generate ESM2 embeddings using the flanking regions

# Aim of this script is to generate embeddings from
# flanking regions based on ESM-2 model

import os
import configparser
from typing import Callable, Dict
import pandas as pd
import torch
from tqdm import tqdm
from esm import pretrained


def generate_random_vector(hid_state, device='cpu', dtype=torch.float32):
    """
    Generates a random vector of size hid_state using a normal distribution.
    """
    return torch.randn(hid_state, device=device, dtype=dtype)


def model_init(device: torch.device):
    model, alphabet = pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    return model, batch_converter, alphabet


class RegionEmbedder:
    def __init__(self, model, batch_converter, alphabet, device: torch.device):
        self.model = model
        self.batch_converter = batch_converter
        self.alphabet = alphabet
        self.device = device
        self.special_embeddings: Dict[str, torch.Tensor] = {}
        
        # Get hidden state size by processing a single 'A' amino acid
        data = [("protein1", "A")]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], return_contacts=True)
        self.hid_state = results["representations"][33].shape[-1]
        self.special_embeddings['@'] = generate_random_vector(self.hid_state, device=device)
    
    @torch.inference_mode()
    def calculate_region_embeddings(self, sequence: str) -> torch.Tensor:
        if not sequence:
            return torch.zeros(self.hid_state, device=self.device)
        
        # Check if sequence is exactly '@'
        if sequence == '@':
            return self.special_embeddings['@']
        
        data = [("protein1", sequence)]
        _, _, tokens = self.batch_converter(data)
        tokens = tokens.to(self.device)
        
        with torch.no_grad():
            results = self.model(tokens, repr_layers=[33], return_contacts=True)
        embeddings = results["representations"][33]
        
        # Remove start and end tokens and average across sequence length
        return embeddings[0, 1:-1, :].mean(dim=0)


def process_dataset(df: pd.DataFrame, embedder: RegionEmbedder, output_dir: str):
    """
    Process a dataset to generate and save embeddings for each pre-region and post-region.
    Each embedding is saved as a separate file with a name containing protein_id and coordinates.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        protein_id = row['protein_id']
        coordinates = row['coordinates']
        base_filename = f"{protein_id}_{coordinates.replace('-', '_')}"
        
        # Process pre-region
        pre_region = row['pre-region']
        if pre_region != pre_region:  # Check for NaN
            pre_region = 'NA'
        pre_embedding = embedder.calculate_region_embeddings(pre_region)
        torch.save(pre_embedding, os.path.join(output_dir, f"{base_filename}_pre_0.pt"))
        
        # Process post-region
        post_region = row['post-region']
        if post_region != post_region:  # Check for NaN
            post_region = 'NA'
        post_embedding = embedder.calculate_region_embeddings(post_region)
        torch.save(post_embedding, os.path.join(output_dir, f"{base_filename}_post_1.pt"))


def main():
    # Load your CSV file
    df = pd.read_csv('/home/user14/data/train-test_dataset/flanking_regions_no_missing.csv')
    
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    model, batch_converter, alphabet = model_init(device)
    
    # Create embedder with special symbol handling
    embedder = RegionEmbedder(model, batch_converter, alphabet, device)
    
    # Output directory for embeddings
    output_dir = "/home/user14/data/train-test_dataset/embeddings_esm2/" 
    
    # Process dataset and save embeddings
    process_dataset(df, embedder, output_dir)
    
    print(f"All embeddings saved to {output_dir} directory")
    print(f"Special symbol '@' embedding: {embedder.special_embeddings['@'].shape}")


if __name__ == "__main__":
    main()