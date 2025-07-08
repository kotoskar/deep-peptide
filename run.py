import time
from src.train_loop_crf import parse_arguments, train
import wandb
import torch

import os
os.environ["WANDB_START_METHOD"] = "thread"


start_time = time.time()
run_name = f"run{int(start_time)}"

run = wandb.init(project="DeepPeptide", name=run_name, reinit=True)
config = run.config
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# print(f'Running on device: {device}')
train(parse_arguments(), wandb_run=run)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
