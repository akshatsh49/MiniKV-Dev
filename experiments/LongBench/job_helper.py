import os, sys, json
import itertools
import numpy as np
from tqdm import tqdm

os.system("rm slurm_jobs/*")

with open("sample.slurm", "r") as f:
    original = f.read()

# job_list = [
#     #  MiniKV: H2O + quantization
#     "python pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy uniform --use_eviction_flash False --quant_bits 2 --group_size 16 --residual_length 128",
#     "python pred_minikv.py --model llama2-13b-chat-4k --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy uniform --use_eviction_flash False --quant_bits 2 --group_size 16 --residual_length 128",
#     "python pred_minikv.py --model mistral-7B-instruct-v0.2 --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy uniform --use_eviction_flash False --quant_bits 2 --group_size 16 --residual_length 128",
    
#     "python pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy pyramid --use_eviction_flash False --quant_bits 2 --group_size 16 --residual_length 128",
#     "python pred_minikv.py --model llama2-13b-chat-4k --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy pyramid --use_eviction_flash False --quant_bits 2 --group_size 16 --residual_length 128",
#     "python pred_minikv.py --model mistral-7B-instruct-v0.2 --e --full_model False --use_snap False --heavy_ratio 0.25 --recent_ratio 0.25 --eviction_strategy pyramid --use_eviction_flash False --quant_bits 2 --group_size 16 --residual_length 128",
    
#     # snapKV
#     "python pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.15 --quant_bits 16",
#     "python pred_minikv.py --model llama2-13b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.15 --quant_bits 16",
#     "python pred_minikv.py --model mistral-7B-instruct-v0.2  --e --full_model False --use_snap True --prompt_sparsity_ratio 0.15 --quant_bits 16",
    
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.15 --eviction_strategy pyramid --quant_bits 16",
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model llama2-13b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.15 --eviction_strategy pyramid --quant_bits 16",
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model mistral-7B-instruct-v0.2 --e --full_model False --use_snap True --prompt_sparsity_ratio 0.15 --eviction_strategy pyramid --quant_bits 16",
    
#     # snapKV + quantization
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.5 --eviction_strategy uniform --quant_bits 2 --group_size 16 --residual_length 128",
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model llama2-13b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.5 --eviction_strategy uniform --quant_bits 2 --group_size 16 --residual_length 128",
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model mistral-7B-instruct-v0.2 --e --full_model False --use_snap True --prompt_sparsity_ratio 0.5 --eviction_strategy uniform --quant_bits 2 --group_size 16 --residual_length 128",
    
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model llama2-7b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.5 --eviction_strategy pyramid --quant_bits 2 --group_size 16 --residual_length 128",
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model llama2-13b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio 0.5 --eviction_strategy pyramid --quant_bits 2 --group_size 16 --residual_length 128",
#     "TOKENIZERS_PARALLELISM=false python pred_minikv.py --model mistral-7B-instruct-v0.2 --e --full_model False --use_snap True --prompt_sparsity_ratio 0.5 --eviction_strategy pyramid --quant_bits 2 --group_size 16 --residual_length 128",
    
#     # add full models
#     "python pred_minikv.py --model llama2-7b-chat-4k --e --full_model True",
#     "python pred_minikv.py --model llama2-13b-chat-4k --e --full_model True",
#     "python pred_minikv.py --model mistral-7B-instruct-v0.2 --e --full_model True",
# ]

# for idx, job in enumerate(tqdm(job_list, desc="Creating SLURM jobs")):
#     job_content = original + job
#     with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
#         f.write(job_content)

# create variable budget jobs for mkv and mkv pyramid
total_budget = [0.2,0.4,0.5,0.7,0.8]
model = ['llama2-7b-chat-4k']
strategy = ['uniform']
use_snaps = [False]
# quant_bits = [2, 4, 8, 16]
# use_eviction_flashs = [False]
pooling_strategys = ['maxpool', 'avgpool', None]

for idx, (b, m, s, u, p) in enumerate(tqdm(itertools.product(total_budget, model, strategy, use_snaps, pooling_strategys), desc="Creating SLURM jobs")):
    job = f'TOKENIZERS_PARALLELISM=false python pred_minikv.py --model {m} --e --full_model False --use_snap {u} --heavy_ratio {b/2} --recent_ratio {b/2} --eviction_strategy {s} --use_eviction_flash False --k_bits 2 --k_dim channel --v_bits 2 --v_dim token --group_size 16 --residual_length 128 --pooling {p}'
    job_content = original + job
    with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
        f.write(job_content)
