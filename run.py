import time
from src.train_loop_crf import parse_arguments, train
from aim import Run

import os

run = Run()

start_time = time.time()
model_name = "ESM3_struc"
run_name = f"DeepPeptide_{model_name}"
train(parse_arguments(), model_name=model_name, run=run)
end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")
