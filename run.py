import time
from src.train_loop_crf import parse_arguments, train
import wandb
import torch

import os
os.environ["WANDB_START_METHOD"] = "thread"

start_time = time.time()
model_name = "ESM3_homo"
run_name = f"DeepPeptide_{model_name}_homo"
with wandb.init(project="DeepPeptide_homo", name=run_name, reinit=True) as run:
    train(parse_arguments(), model_name=model_name, wandb_run=run)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
