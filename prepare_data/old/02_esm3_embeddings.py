# Aim of this script is to generate embeddings from
# flanking regions based on ESM-3 model

import os
import configparser
from typing import Callable, Dict
import pandas as pd
import torch
from tqdm import tqdm
from esm.pretrained import ESM3_sm_open_v0
from esm.tokenization import get_esm3_model_tokenizers


def generate_random_vector(hid_state, device='cpu', dtype=torch.float32):
    """
    Generates a random vector of size hid_state using a normal distribution.
    """
    return torch.randn(hid_state, device=device, dtype=dtype)


def model_init(device: torch.device):
    tokenizers = get_esm3_model_tokenizers()
    model = ESM3_sm_open_v0()
    model = model.eval().to(device)
    return model, tokenizers


class RegionEmbedder:
    def __init__(self, model, tokenizers, device: torch.device):
        self.model = model
        self.tokenizers = tokenizers
        self.device = device
        self.special_embeddings: Dict[str, torch.Tensor] = {}
        

        tokens = self.tokenizers.sequence.encode('A')
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)
        # Pre-generate embedding for '@' symbol
        # hid_state = model.embed_dim
        self.hid_state = len(model(sequence_tokens=tokens).embeddings.cpu().squeeze(0)[1:-1, :].mean(dim=0))
        self.special_embeddings['@'] = generate_random_vector(self.hid_state, device=device)
    
    @torch.inference_mode()
    def calculate_region_embeddings(self, sequence: str) -> torch.Tensor:
        if not sequence:
            # return torch.zeros(self.model.embed_dim, device=self.device)
            return torch.zeros(self.hid_state, device=self.device)
        
        # Check if sequence is exactly '@'
        if sequence == '@':
            return self.special_embeddings['@']
        
        tokens = self.tokenizers.sequence.encode(sequence)
        tokens = torch.tensor(tokens).unsqueeze(0).to(self.device)

        outputs = self.model(sequence_tokens=tokens)
        embeddings = outputs.embeddings.cpu().squeeze(0)
        embeddings = embeddings[1:-1, :]  # remove start and end tokens

        # Average all token embeddings for the sequence
        return embeddings.mean(dim=0)


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
        # if os.path.isfile(f'{args.output_dir}/{names[i]}.pt'):
        #     continue
        
        # Process pre-region
        pre_region = row['pre-region']
        if pre_region != pre_region:
            pre_region = 'NA'
        pre_embedding = embedder.calculate_region_embeddings(pre_region)
        torch.save(pre_embedding, os.path.join(output_dir, f"{base_filename}_pre_0.pt"))
        
        # Process post-region
        post_region = row['post-region']
        if post_region != post_region:
            post_region = 'NA'
        post_embedding = embedder.calculate_region_embeddings(post_region)
        torch.save(post_embedding, os.path.join(output_dir, f"{base_filename}_post_1.pt"))


def main():
    # Load your CSV file
    df = pd.read_csv('/home/user14/data/train-test_dataset/flanking_regions_no_missing.csv')  # replace with your file path
    
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    model, tokenizers = model_init(device)
    
    # Create embedder with special symbol handling
    embedder = RegionEmbedder(model, tokenizers, device)
    
    # Output directory for embeddings
    output_dir = "/home/user14/data/train-test_dataset/embeddings_esm3/"
    
    # Process dataset and save embeddings
    process_dataset(df, embedder, output_dir)
    
    print(f"All embeddings saved to {output_dir} directory")
    print(f"Special symbol '@' embedding: {embedder.special_embeddings['@'].shape}")


if __name__ == "__main__":
    main()