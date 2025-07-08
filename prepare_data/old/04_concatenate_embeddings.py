# Aim of this script is to concatenate 4 embeddings for each peptide or propeptide
# and get an average of the concatenated embeddings for each sequence  


import torch
import glob
import os
from collections import defaultdict

# Define input and output directories
input_dir = "/home/user14/data/train-test_dataset/embeddings_esm3"
output_dir = "/home/user14/data/train-test_dataset/embeddings_esm3_processed"  # New directory for output

# Create output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Group files by their prefix (part before first "_")
file_groups = defaultdict(list)
for filepath in glob.glob(os.path.join(input_dir, "*.pt")):
    filename = os.path.basename(filepath)
    prefix = filename.split('_')[0]
    file_groups[prefix].append(filepath)

# Process each group
for prefix, files in file_groups.items():
    # Sort files in the required order: pre_0, pre_1, post_0, post_1
    ordered_files = sorted(files, key=lambda x: (
        os.path.basename(x).split('_')[-2],  # 'pre' or 'post'
        int(os.path.basename(x).split('_')[-1].split('.')[0])  # 0 or 1
    ))
    
    # Load embeddings in order
    embeddings = []
    for file in ordered_files:
        emb = torch.load(file)
        embeddings.append(emb)
    
    # Concatenate embeddings
    concatenated = torch.cat(embeddings, dim=0)
    
    # Compute mean embedding
    mean_embedding = torch.mean(concatenated, dim=0, keepdim=True)
    
    # Save mean embedding
    output_filename = os.path.join(output_dir, f"{prefix}_mean.pt")
    torch.save(mean_embedding, output_filename)
    print(f"Saved mean embedding to {output_filename}")

print("All done! Mean embeddings saved in:", output_dir)