'''
For debugging
CUDA_LAUNCH_BLOCKING=1 python -m pdb mem_spd_test.py

python -u mem_spd_test.py
'''

import torch
import os, sys, json
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
from tqdm import tqdm
import argparse
from minikv.monkeypatch.monkeypatch import replace_llama, replace_mistral, replace_mixtral

def parse_args(args=None):
    parser = argparse.ArgumentParser()
    return parser.parse_args(args)

K_BITS = 2
V_BITS = 2
GROUP_SIZE = 32
RESIDUAL_LENGTH = 128

r_ratio = 0.01
h_ratio = 0.01
use_eviction_flash = False

BATCH_SIZE = 64
prompt_lenth = int(1e3)
output_length = 100
num_repeats = 5
PATH_TO_YOUR_SAVE_DIR = os.getenv('HF_HUB_CACHE', './')
CACHE_DIR = PATH_TO_YOUR_SAVE_DIR

model2path = json.load(open("config/model2path.json", "r"))
model_name_or_path = 'llama2-7b-chat-4k'
path = model2path[model_name_or_path]


from deepspeed.accelerator import get_accelerator
def memory_usage():
    alloc = "mem_allocated: {:.4f} GB".format(get_accelerator().memory_allocated() / (1024 * 1024 * 1024))
    max_alloc = "max_mem_allocated: {:.4f} GB".format(get_accelerator().max_memory_allocated() /
                                                        (1024 * 1024 * 1024))
    cache = "cache_allocated: {:.4f} GB".format(get_accelerator().memory_cached() / (1024 * 1024 * 1024))
    max_cache = "max_cache_allocated: {:.4f} GB".format(get_accelerator().max_memory_cached() /
                                                        (1024 * 1024 * 1024))
    return {
        "alloc": alloc,
        "max_alloc": max_alloc,
        "cache": cache,
        "max_cache": max_cache
    }

device = 'cuda'
if K_BITS < 16 and V_BITS < 16:
    print(f"[INFO] Loading MiniKV model.")
    if 'llama' in model_name_or_path.lower():
        model = AutoModelForCausalLM.from_pretrained(path, torch_dtype = torch.float16, _attn_implementation = 'flash_attention_2').to(device)   # cant use torch_dtype=torch.bfloat16 as kivi's quantization kernels dont support it
    elif 'mistral' in model_name_or_path.lower():
        raise NotImplementedError("Mistral model not supported yet.")
    
    args = parse_args()
    args.use_snap = False
    args.quant_bits = K_BITS
    args.group_size = GROUP_SIZE
    args.residual_length = RESIDUAL_LENGTH
    args.recent_ratio = r_ratio
    args.heavy_ratio = h_ratio
    args.use_eviction_flash = use_eviction_flash
    
    replace_llama(args)
    replace_mistral()
    replace_mixtral()
    
    layers = len(model.model.layers)
    for i in range(layers):
        model.model.layers[i].self_attn.config.use_snap = args.use_snap
        
        # these values are meaningless
        model.model.layers[i].self_attn.config.prompt_sparsity_ratio = 0.25
        model.model.layers[i].self_attn.config.window_size = 32
        model.model.layers[i].self_attn.config.kernel_size = 7
        model.model.layers[i].self_attn.config.pooling = 'maxpool'
        
        model.model.layers[i].self_attn.config.heavy_ratio = h_ratio
        model.model.layers[i].self_attn.config.recent_ratio = r_ratio
        model.model.layers[i].self_attn.config.use_eviction_flash = use_eviction_flash
        model.model.layers[i].self_attn.config.quant_bits = V_BITS
        model.model.layers[i].self_attn.config.group_size = GROUP_SIZE
        model.model.layers[i].self_attn.config.residual_length = RESIDUAL_LENGTH
                
else:
    from transformers import LlamaForCausalLM
    print(f"[INFO] Loading full model.")
    model = AutoModelForCausalLM.from_pretrained(
        pretrained_model_name_or_path=path,
        cache_dir=CACHE_DIR,
        torch_dtype=torch.float16,
        _attn_implementation = 'flash_attention_2',
    ).to(device)

model.eval()
tokenizer = AutoTokenizer.from_pretrained(
    path, 
    use_fast=False, 
    trust_remote_code=True, 
    tokenizer_type='llama')

print(f"After model loading {memory_usage()}")

context = []
batch_size = BATCH_SIZE
for _ in range(batch_size):
    string = 't,' * (prompt_lenth // 2)
    context.append(string[:-1])
inputs = tokenizer(context, return_tensors="pt").to('cuda')
input_ids = inputs['input_ids']
print(f"bs: {batch_size}, seqlen: {input_ids.shape[1]}+{output_length}\nmodel:{path}")
torch.cuda.reset_peak_memory_stats()
with torch.no_grad():
    torch.cuda.synchronize()
    st = time.time()
    for i in tqdm(range(num_repeats), desc = "repeats"):
        print(f"Before .generate {memory_usage()}")
        outputs = model.generate(**inputs,     
                                max_new_tokens=output_length, 
                                num_beams = 1, 
                                do_sample = False,
                                temperature = 1.0)
        
        print(f"After .generate {memory_usage()}")
        
        # torch cuda clear cache and defragment memory
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
            
    torch.cuda.synchronize()
    et = time.time()
    used_mem = torch.cuda.max_memory_allocated()
    
    with open(f"time_{K_BITS}.log", "a+") as f:
        print(f'used time: {(et - st) / num_repeats * 1000:0.4f} ms', file = f)
    with open(f"mem_{K_BITS}.log", "a+") as f:
        print(f'peak mem: {used_mem / 1024 ** 3:0.4f} GB', file = f)
