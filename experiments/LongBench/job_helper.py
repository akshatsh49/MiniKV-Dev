import os, sys, json
import itertools
import numpy as np

with open("sample.slurm", "r") as f:
    original = f.read()

use_snaps = [True]
use_eviction_flashs = [False]

quant_bits = [16]
group_sizes = [16]
residual_lengths = [128]

# heavy_ratios = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]
# recent_ratios = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]
# heavy_ratios = [i/2 for i in heavy_ratios]
# recent_ratios = [i/2 for i in recent_ratios]

# prompt_sparsity_ratios = [0.1, 0.2, 0.5, 0.6, 0.8, 0.9]
prompt_sparsity_ratios = [0.15]

# total_budget = 0.5
# heavy_ratios = np.arange(0, total_budget, 0.05)

# model_names = ['llama2-7b-chat-4k',]
# model_names = ['llama2-13b-chat-4k',]
model_names = ['llama2-13b-chat-4k', 'mistral-7B-instruct-v0.2',]

os.makedirs("slurm_jobs", exist_ok=True)
# remove existing job*.slurm files in slurm_jobs
for f in os.listdir("slurm_jobs"):
    if f.startswith("job_"):
        os.remove(os.path.join("slurm_jobs", f))
        print(f"Removed {f}")

# for idx, (model_name, ratio) in enumerate(itertools.product(model_names, prompt_sparsity_ratio)):
# for idx, (model_name, ratio, quant_bit, g_size, length) in enumerate(itertools.product(model_names, prompt_sparsity_ratio, quant_bits, group_sizes, residual_lengths)):
# for idx, (model_name, use_snap, heavy_ratio, recent_ratio, use_eviction_flash, quant_bit, g_size, length) in enumerate(itertools.product(model_names, use_snaps, heavy_ratios, recent_ratios, use_eviction_flashs, quant_bits, group_sizes, residual_lengths)): # minikv call
# for idx, (model_name, use_snap, heavy_ratio, use_eviction_flash, quant_bit, g_size, length) in enumerate(itertools.product(model_names, use_snaps, heavy_ratios, use_eviction_flashs, quant_bits, group_sizes, residual_lengths)):
for idx, (model_name, use_snap, prompt_sparsity_ratio) in enumerate(itertools.product(model_names, use_snaps, prompt_sparsity_ratios)): # snapkv call
    # if heavy_ratio != recent_ratio:
    #     continue

    print(f"Creating job_{idx}.slurm")
    # job_content = original + f"CUDA_VISIBLE_DEVICES=0 python3 -u pred_long_bench.py --e --model_name_or_path {model_name} --full_model False --k_bits 2 --v_bits 2 --group_size {size} --recent_ratio {ratio} --heavy_ratio {ratio} --residual_length 128 --use_flash False"
    # job_content = original + f"CUDA_VISIBLE_DEVICES=0 python3 -u pred_long_bench.py --e --model_name_or_path mistralai/Mistral-7B-v0.3 --full_model False --k_bits 2 --v_bits 2 --group_size {size} --recent_ratio {ratio} --heavy_ratio {ratio} --residual_length 128"
    
    # job_content = original + f"CUDA_VISIBLE_DEVICES=0 python3 -u pred_snap.py --model {model_name} --e --full_model False --prompt_sparsity_ratios {ratio} --quant_bits {quant_bit} --group_size {g_size} --residual_length {length}"
    
    # recent_ratio = total_budget - heavy_ratio
    # job_content = original + f"TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python3 -u pred_snap.py --model {model_name} --e --full_model False --use_snap {use_snap} --heavy_ratio {heavy_ratio} --recent_ratio {recent_ratio} --use_eviction_flash {use_eviction_flash} --quant_bits {quant_bit} --group_size {g_size} --residual_length {length}"
    
    # mistral snapkv call
    job_content = original + f"TOKENIZERS_PARALLELISM=false CUDA_VISIBLE_DEVICES=0 python3 -u pred_snap.py --model {model_name} --e --full_model False --use_snap {use_snap} --prompt_sparsity_ratio {prompt_sparsity_ratio} --quant_bits 16"
    
    with open(f"slurm_jobs/job_{idx}.slurm", "w") as f:
        f.write(job_content)
