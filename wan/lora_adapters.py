import os
import torch
import logging
from typing import Optional, Dict, Union, List
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model, get_peft_model, PeftModel, PeftMixedModel

def get_named_lora_parameters(adapter_state_dict, adapter_name, load_all_dict=False):
    # Нормализация ключей (только там где есть lora)
    normalized_state_dict = {}
    num_i = 0
    
    for k, v in adapter_state_dict.items():
        if ('lora' in k and not load_all_dict) or load_all_dict :
            new_key = k
            
            if "diffusion_model." in new_key:
                if  "base_model.model." not in new_key:
                    new_key = new_key.replace("diffusion_model.", "base_model.model.")
                else:
                    new_key = new_key.replace("diffusion_model.", "")
            if (
                (".default.weight" not in new_key and f".{adapter_name}.weight" not in new_key)
                and  (new_key.split(".")[-2] == 'lora_A' or new_key.split(".")[-2] == 'lora_B')
            ):
                new_key = new_key.replace(".weight", f".{adapter_name}.weight")

            elif  f".{adapter_name}.weight" not in new_key and ".default.weight" in new_key:
                new_key = new_key.replace(".default.weight", f".{adapter_name}.weight")
                
            # if not rank and num_i < 20 : print(f'k={k} --> new_key={new_key}')
            normalized_state_dict[new_key] = v
            num_i += 1
    return normalized_state_dict

def get_lora_config(lora_dir_path, lora_weights_dict, default_r=128, default_lora_alpha=128):
    if any("time_projection." in k.lower() for k in lora_weights_dict.keys()):
        # target_modules=[
        #     "ffn.0", "ffn.2",
        #     "self_attn.q", "self_attn.k","self_attn.v", "self_attn.o",
        #     "time_embedding.0", "time_embedding.2", "time_projection.1",
        # ]
        target_modules= ["time_embedding.0", "time_embedding.2", "time_projection.1",]
        for i in range(40):  # For all 40 blocks
            target_modules.extend([
                f"blocks.{i}.ffn.0",
                f"blocks.{i}.ffn.2",
                f"blocks.{i}.self_attn.q",
                f"blocks.{i}.self_attn.k",
                f"blocks.{i}.self_attn.v",
                f"blocks.{i}.self_attn.o",
            ])
    else:
        # target_modules=[
        #     "ffn.0", "ffn.2",
        #     "self_attn.q", "self_attn.k","self_attn.v", "self_attn.o",
        #     "cross_attn.q", "cross_attn.k","cross_attn.v", "cross_attn.o",
        # ]
        target_modules= []
        for i in range(40):  # For all 40 blocks
            target_modules.extend([
                f"blocks.{i}.ffn.0",
                f"blocks.{i}.ffn.2",
                f"blocks.{i}.self_attn.q",
                f"blocks.{i}.self_attn.k",
                f"blocks.{i}.self_attn.v",
                f"blocks.{i}.self_attn.o",
                f"blocks.{i}.cross_attn.q",
                f"blocks.{i}.cross_attn.k",
                f"blocks.{i}.cross_attn.v",
                f"blocks.{i}.cross_attn.o",
            ])

    if os.path.exists(os.path.join(lora_dir_path, "adapter_config.json" )):
        logging.info(f"Load lora config from {lora_dir_path}")
        lora_config = LoraConfig.from_pretrained(lora_dir_path)
        # Сhange init target_modules on this ones for vace compatibility
        lora_config.target_modules = target_modules
    else:
        logging.info(f"Create config with lora alpha = {default_lora_alpha}, r={default_r}")
        lora_config = LoraConfig(
            r=default_r, target_modules=target_modules, lora_alpha=default_lora_alpha,
            inference_mode=True,
        )

    lora_config.exclude_modules = "vace_blocks"
    return lora_config

def load_peft_adapter(
    model: torch.nn.Module,
    path: str,
    adapter_config: Optional[LoraConfig],
    adapter_name: str = None,
    lora_weight: int = 1,
    load_all_dict: bool = False,
    r: int = 128,
    lora_alpha: int = 128,
) -> PeftModel:
    """
    Загружает PEFT адаптер в модель с обработкой различных форматов
    
    Args:
        model: Базовая модель
        path: Путь к адаптеру (.pt/.safetensors) или до директории с файла
        adapter_config: Конфиг LoRA (если None, будет загружен из директории или создан дефолтный)
        adapter_name: Имя адаптера, if None -- создастся по имени папки и весу
        lora_weight: Вес для лоры
        load_all_dict: загружать все веса или только те где есть lora
        
    """

    if os.path.isdir(path):
        if os.path.exists(os.path.join(path, "adapter_model.safetensors")):
            path = os.path.join(path, "adapter_model.safetensors")
        elif os.path.exists(os.path.join(path, "gathered_adapter.pt")):
            path = os.path.join(path, "gathered_adapter.pt")
        else:
            raise ValueError(f"There is no Lora in: {path}")
    # Load weights
    if path.endswith(".safetensors"):
        lora_weights_dict = load_file(path)
    elif path.endswith(".pt"):
        lora_weights_dict = torch.load(path, map_location="cpu")
    else:
        raise ValueError(f"Unsupported adapter format: {path}")
    lora_dir_path = '/'.join(path.split('/')[:-1])
    
    # Load or create lora config
    if adapter_config is None:
        logging.info(f"\n\ncreate lora config for adapter_name={adapter_name}, lora_alpha={lora_alpha} (lora_dir_path={lora_dir_path})")
        adapter_config = get_lora_config(lora_dir_path, lora_weights_dict, default_r=r, default_lora_alpha=lora_alpha)
    logging.info(f"\nmultiply lora_alpha on {lora_weight} -> lora_alpha = {adapter_config.lora_alpha * lora_weight}")
    adapter_config.lora_alpha = adapter_config.lora_alpha * lora_weight
    adapter_config.exclude_modules = "vace_blocks"
    if adapter_name is None:
        # name for adapter is like  dir name + epoch
        # TODO: pretty name
        adapter_name =  lora_dir_path.split('/')[-3] +  '_' + lora_dir_path.split('/')[-2] +  '_' + lora_dir_path.split('/')[-1] # f"lora{i}"

    logging.info(f"\nLoading custom '{adapter_name}' LoRa from {path}, with config={adapter_config}\n")
    # Init Peft model
    if not isinstance(model, PeftModel) and not isinstance(model, PeftMixedModel) :
        model = get_peft_model(
            model, adapter_config,
            adapter_name=adapter_name,
            mixed=True
        )
    # or add new adapter    
    else:
        
        while adapter_name in model.peft_config:
            print('adapter_name=', adapter_name)
            if adapter_name[-1].isdigit():
                adapter_name[-1] = int(adapter_name[-1]) + 1
            else:
                adapter_name += "_1"
        model.add_adapter(adapter_name, adapter_config)
    # rename layers for matching the name
    lora_state_dict = get_named_lora_parameters(lora_weights_dict, adapter_name, load_all_dict=False)
    
    if  load_all_dict:
        # Load all weights
        load_result = model.load_state_dict(lora_state_dict, strict=False)
        if len(load_result.missing_keys) > 0:
            logging.warning(f"Missing keys: {[k for i, k in enumerate(load_result.missing_keys) if ('lora' in k and i < 50)]}")
        if len(load_result.unexpected_keys) > 0:
            logging.warning(f"Unexpected keys: {[k for i, k in enumerate(load_result.unexpected_keys) if (i < 50)] }")
    else:
        # Load only LoRA weights
        # if not rank : print('model state dict', [k for i, k in enumerate(model.state_dict().keys()) if i < 15 ])
        for key, value in lora_state_dict.items():
            if key in model.state_dict():
                model.state_dict()[key].copy_(value)
            # else:
            #     logging.info(f"Doesn't find {key}" )
    return model, adapter_name
