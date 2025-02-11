from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
import requests

from minikv.monkeypatch.monkeypatch import replace_mistral
import argparse
def parse_args(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default=None, choices=[
        "llama2-7b-chat-4k", "llama2-13b-chat-4k", "llama3-8b-instruct", "longchat-v1.5-7b-32k", "xgen-7b-8k", 
        "internlm-7b-8k", "chatglm2-6b", "chatglm2-6b-32k", "chatglm3-6b-32k", "vicuna-v1.5-7b-16k",
        "mistral-7B-instruct-v0.2", "mistral-7B-instruct-v0.1", "llama-2-7B-32k-instruct", "mixtral-8x7B-instruct-v0.1","lwm-text-chat-1m", "lwm-text-1m",
        "Yarn-llama-2-7b-128k"])
    parser.add_argument('--compress_args_path', type=str, default=None, help="Path to the compress args")
    parser.add_argument('--e', action='store_true', help="Evaluate on LongBench-E")
    parser.add_argument('--full_model', type=lambda x: x.lower() == 'true', help="Use uncompressed model", default=False)
    parser.add_argument('--use_snap', type=lambda x: x.lower() == 'true', help="Use snapKV for eviction", default=False)
    
    # snapKV args
    parser.add_argument('--prompt_sparsity_ratios', type=float, help="The sparsity ratio of the prompt", default=0.5)
    parser.add_argument('--window_sizes', type=int, help="The window size of the prompt", default=32)
    parser.add_argument('--kernel_sizes', type=int, help="The kernel size of the prompt", default=7)
    parser.add_argument('--pooling', type=str, help="The pooling method of the prompt", default="maxpool")
    
    # h2o args
    parser.add_argument('--heavy_ratio', type=float, help="The ratio of heavy hitter", default=0.25)
    parser.add_argument('--recent_ratio', type=float, help="The ratio of recent window", default=0.25)
    parser.add_argument('--eviction_strategy', type=str, help="The eviction strategy", default="uniform", choices=["uniform", "pyramid"])
    parser.add_argument('--use_eviction_flash', type=lambda x: x.lower() == 'true', help="Use custom flash_attn kernel which returns the cumulative attention map", default=False)
    
    # quantization args
    parser.add_argument('--quant_bits', type=int, help="The number of bits for key/value", default=2)
    parser.add_argument('--group_size', type=int, help="The group size", default=16)
    parser.add_argument('--residual_length', type=int, help="The residual length", default=128)
    
    args = parser.parse_args(args)
    if args.pooling == 'None':
        args.pooling = None
    return args

def setup_model_with_compression(model, args):
    
    if not args.full_model:
        layers = len(model.language_model.model.layers)
        window_sizes = [args.window_sizes] * layers if not isinstance(args.window_sizes, list) else args.window_sizes
        prompt_sparsity_ratios = [args.prompt_sparsity_ratios] * layers if not isinstance(args.prompt_sparsity_ratios, list) else args.prompt_sparsity_ratios
        kernel_sizes = [args.kernel_sizes] * layers if not isinstance(args.kernel_sizes, list) else args.kernel_sizes
        
        for i in range(layers):
            layer_config = model.language_model.model.layers[i].self_attn.config
            layer_config.use_snap = args.use_snap
            layer_config.prompt_sparsity_ratio = prompt_sparsity_ratios[i]
            layer_config.window_size = window_sizes[i]
            layer_config.kernel_size = kernel_sizes[i]
            layer_config.pooling = args.pooling
            layer_config.heavy_ratio = args.heavy_ratio
            layer_config.recent_ratio = args.recent_ratio
            layer_config.eviction_strategy = args.eviction_strategy
            layer_config.use_eviction_flash = args.use_eviction_flash
            layer_config.quant_bits = args.quant_bits
            layer_config.group_size = args.group_size
            layer_config.residual_length = args.residual_length

def main():
    args = parse_args()
    
    # disable legacy processing is correct
    replace_mistral(args)
    
    processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = LlavaNextForConditionalGeneration.from_pretrained(
        "llava-hf/llava-v1.6-mistral-7b-hf",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        attn_implementation="flash_attention_2",
    ).to("cuda:0")
    
    setup_model_with_compression(model, args)
    
    url = "https://github.com/haotian-liu/LLaVA/blob/1a91fc274d7c35a9b50b3cb29c4247ae5837ce39/images/llava_v1_5_radar.jpg?raw=true"
    image = Image.open(requests.get(url, stream=True).raw)
    
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": "What is shown in this image?"},
            ],
        },
    ]
    
    prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(image, prompt, return_tensors="pt").to("cuda:0")
    
    output = model.generate(**inputs, max_new_tokens=100)
    print(processor.decode(output[0], skip_special_tokens=True))

if __name__ == "__main__":
    main()
# python example/inference_llava.py --model llama2-7b-chat-4k --e --full_model False --use_snap True --prompt_sparsity_ratio .9 --quant_bits 16
