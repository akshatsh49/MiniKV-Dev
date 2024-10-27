from importlib.metadata import version
import warnings
import transformers
# from snapkv.monkeypatch.llama_hijack_4_37 import llama_flash_attn2_forward as llama_flash_attn2_forward_4_37, prepare_inputs_for_generation_llama as prepare_inputs_for_generation_llama_4_37
from snapkv.monkeypatch.snap_minikv_llama_hijack_4_37 import \
        sparsity_llama_flash_attn2_forward, sparsity_prepare_inputs_for_generation_llama, \
        snap_minikv_llama_flash_attn2_forward, snap_minikv_prepare_inputs_for_generation_llama

from snapkv.monkeypatch.minikv_llama_hijack_4_37 import \
        minikv_llama_flash_attn2_forward, minikv_prepare_inputs_for_generation_llama

# from snapkv.monkeypatch.mistral_hijack_4_37 import mistral_flash_attn2_forward as mistral_flash_attn2_forward_4_37, prepare_inputs_for_generation_mistral as prepare_inputs_for_generation_mistral_4_37
from snapkv.monkeypatch.snap_minikv_mistral_hijack_4_37 import sparsity_mistral_flash_attn2_forward, sparsity_prepare_inputs_for_generation_mistral
from snapkv.monkeypatch.minikv_mistral_hijack_4_37 import minikv_mistral_flash_attn2_forward, minikv_prepare_inputs_for_generation_mistral

from snapkv.monkeypatch.mixtral_hijack_4_37 import mixtral_flash_attn2_forward as mixtral_flash_attn2_forward_4_37, prepare_inputs_for_generation_mixtral as prepare_inputs_for_generation_mixtral_4_37

def check_version():
    try:
        transformers_version = version("transformers")
    except Exception as e:
        print(f"Transformers not installed: {e}")
    return transformers_version

def replace_llama(args = None):
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
        
    if args.use_snap and args.quant_bits == 16:
        # use sparsity-based attn_head
        print(f"[INFO] Loading Sparsity fwd pass")
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = sparsity_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = sparsity_llama_flash_attn2_forward
    
    elif args.use_snap and args.quant_bits == 2:
        # use quantization-based attn_head
        print(f"[INFO] Loading Snap+MKV fwd pass")
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = snap_minikv_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = snap_minikv_llama_flash_attn2_forward
        
    elif not args.use_snap and args.quant_bits == 2:
        transformers.models.llama.modeling_llama.LlamaForCausalLM.prepare_inputs_for_generation = minikv_prepare_inputs_for_generation_llama
        transformers.models.llama.modeling_llama.LlamaFlashAttention2.forward = minikv_llama_flash_attn2_forward

    else :
        raise NotImplementedError(f"This configuration is not supported: {args = }")

def replace_mistral(args = None):
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    # transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mistral_4_37
    # transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = mistral_flash_attn2_forward_4_37
    
    if args.use_snap and args.quant_bits == 16:
        # use sparsity-based attn_head
        print(f"[INFO] Loading Sparsity fwd pass")
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = sparsity_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = sparsity_mistral_flash_attn2_forward
    
    elif args.use_snap and args.quant_bits == 2:
        raise NotImplementedError(f"SnapKV does not support Mistral model with quant_bits = 2 yet")
        
    elif not args.use_snap and args.quant_bits == 2:
        transformers.models.mistral.modeling_mistral.MistralForCausalLM.prepare_inputs_for_generation = minikv_prepare_inputs_for_generation_mistral
        transformers.models.mistral.modeling_mistral.MistralFlashAttention2.forward = minikv_mistral_flash_attn2_forward

    else :
        raise NotImplementedError(f"This configuration is not supported: {args = }")

def replace_mixtral():
    transformers_version = check_version()
    version_list = ['4.37']
    warning_flag = True
    for version in version_list:
        if version in transformers_version:
            warning_flag = False
            break
    if warning_flag:
        warnings.warn(f"Transformers version {transformers_version} might not be compatible with SnapKV. SnapKV is tested with Transformers version {version_list}.")
    transformers.models.mixtral.modeling_mixtral.MixtralForCausalLM.prepare_inputs_for_generation = prepare_inputs_for_generation_mixtral_4_37
    transformers.models.mixtral.modeling_mixtral.MixtralFlashAttention2.forward = mixtral_flash_attn2_forward_4_37
