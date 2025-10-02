import pandas as pd

ref_pth = 'data/labeled_sequences.csv'
df = pd.read_csv(ref_pth)
homo_ids = df.loc[df['organism'] == "Homo sapiens (Human)"]['protein_id']
with open('data/protein_id_homo.txt', 'w') as f:
    f.write('\n'.join(homo_ids))