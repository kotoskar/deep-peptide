# DeepPeptide Advanced
Predicting cleaved peptides in protein sequences.
Fork of original DeepPeptide project with major changes: different embedding models added.

[![DOI](https://zenodo.org/badge/593202385.svg)](https://zenodo.org/badge/latestdoi/593202385)

Made by ChemAI team on Summer with AIRI school. 


### Environment
`conda create --name DeepPeptide python=3.10`

`conda activate DeepPeptide`

`python -m pip install -r requirements.txt`


### Training the model

#### Download AF2 structures
```bash
python3 src/utils/download_af2_strucs.py data/protein_sequences.fasta data/structures/
```


#### Precompute embeddings
##### ESM embeddings
```bash 
python3 src/utils/make_embeddings.py data/protein_sequences.fasta data/embeddings
```
##### ProSST structure embeddings
```bash
python3 src/utils/make_embeddings_prosst.py data/structrures
```

#### Train
```
python3 run.py --embeddings_dir data/embeddings -df data/labeled_sequences.csv -pf data/graphpart_assignments.csv
```

Note that parameters `--lr`, `--batch_size`, `--dropout`, `--conv_dropout`, `--kernel_size`, `--num_filters`, `--hidden_size` were optimized in a nested CV hyperparameter search and not used at their defaults.

### Evaluation
- PeptideLocator was evaluated as a licensed executable and cannot be provided in this repo.
- We used 5-fold nested CV to select 20 model checkpoints trained using `src/train_loop_crf.py`. The selected checkpoints are hardcoded in `evaluation/measure_performance.py`, which computes the performance metrics from the checkpoints' saved predictions.

### Predicting

[See the predictor README](predictor/README.md)
