# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc
import logging
import math
import os
import random
import sys
import types
from contextlib import contextmanager
from functools import partial

import torch
import torch.cuda.amp as amp
import torch.distributed as dist
from tqdm import tqdm

from .distributed.fsdp import shard_model
from .modules.model import WanModel
from .modules.t5 import T5EncoderModel
from .modules.vae import WanVAE
from .utils.fm_solvers import (
    FlowDPMSolverMultistepScheduler,
    get_sampling_sigmas,
    retrieve_timesteps,
)
from .utils.fm_solvers_unipc import FlowUniPCMultistepScheduler
from diffusers import FlowMatchEulerDiscreteScheduler

from typing import Optional, List, Union, List
from safetensors.torch import load_file
from peft import LoraConfig, inject_adapter_in_model, get_peft_model, PeftModel

def load_peft_adapter(
    model: torch.nn.Module,
    adapter_path: str,
    adapter_config: Optional[LoraConfig] = None,
    adapter_name: str = "default",
    strict: bool = False,
    rank: int = 0
) -> PeftModel:
    """
    Загружает PEFT адаптер в модель с обработкой различных форматов
    
    Args:
        model: Базовая модель
        adapter_path: Путь к адаптеру (.pt/.safetensors)
        adapter_config: Конфиг LoRA (если None, будет загружен из файла)
        adapter_name: Имя адаптера (для множественных адаптеров)
        strict: Строгая загрузка весов
    """
    try:
        # Загрузка конфига, если не предоставлен
        if adapter_config is None:
            adapter_config = LoraConfig.from_pretrained(adapter_path)
        
        # Инициализация PEFT модели
        if not isinstance(model, PeftModel):
            model = get_peft_model(
                model, adapter_config,
                adapter_name=adapter_name,
            )
        else:
            while adapter_name in model.peft_config:
                if adapter_name.endswith("_1"):
                    adapter_name[-1] = int(adapter_name[-1]) + 1
                else:
                    adapter_name += "_1"
            model.add_adapter(adapter_name, adapter_config)
        
        # Загрузка весов адаптера
        if adapter_path.endswith(".safetensors"):
            adapter_state_dict = load_file(adapter_path)
        else:
            adapter_state_dict = torch.load(adapter_path, map_location="cpu", weights_only=True)
        
        # Нормализация ключей (только там где есть lora)
        normalized_state_dict = {}
        num_i = 0
        for k, v in adapter_state_dict.items():
            if 'lora' in k:
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
                    
                if not rank and num_i < 20 : print(f'k={k} --> new_key={new_key}')
                normalized_state_dict[new_key] = v
                num_i += 1
    
        # Загрузка всех весов
        # load_result = model.load_state_dict(normalized_state_dict, strict=strict)
        # if len(load_result.missing_keys) > 0:
        #     logging.warning(f"Missing keys: {[k for i, k in enumerate(load_result.missing_keys) if ('lora' in k and i < 50)]}")
        # if len(load_result.unexpected_keys) > 0:
        #     logging.warning(f"Unexpected keys: {[k for i, k in enumerate(load_result.unexpected_keys) if (i < 50)] }")
        
        # Загружаем веса с преобразованием имен
        if not rank : print([k for i, k in enumerate(model.state_dict().keys()) if i < 15 ])
        for key, value in normalized_state_dict.items():
            # k base_model.model.time_embedding.0.lora_A.default.weight
            # model.state_dict(): 'base_model.model.time_embedding.0.lora_A.cfg_lora.weight'
            
            model.state_dict()[key].copy_(value)

        return model
        
    except Exception as e:
        logging.error(f"Error loading adapter {adapter_path}: {str(e)}")
        raise


class WanT2V:

    def __init__(
        self,
        config,
        checkpoint_dir,
        device_id=0,
        rank=0,
        t5_fsdp=False,
        dit_fsdp=False,
        use_usp=False,
        t5_cpu=False,
        model_ckpt_path=None,
        # for LoRas
        lora_paths: List[str] = [],
        lora_configs: Optional[Union[List[LoraConfig], LoraConfig]] = None,
        lora_weights: List[str] = None,
    ):
        r"""
        Initializes the Wan text-to-video generation model components.

        Args:
            config (EasyDict):
                Object containing model parameters initialized from config.py
            checkpoint_dir (`str`):
                Path to directory containing model checkpoints
            device_id (`int`,  *optional*, defaults to 0):
                Id of target GPU device
            rank (`int`,  *optional*, defaults to 0):
                Process rank for distributed training
            t5_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for T5 model
            dit_fsdp (`bool`, *optional*, defaults to False):
                Enable FSDP sharding for DiT model
            use_usp (`bool`, *optional*, defaults to False):
                Enable distribution strategy of USP.
            t5_cpu (`bool`, *optional*, defaults to False):
                Whether to place T5 model on CPU. Only works without t5_fsdp.
        """
        self.device = torch.device(f"cuda:{device_id}")
        self.config = config
        self.rank = rank
        self.t5_cpu = t5_cpu

        self.num_train_timesteps = config.num_train_timesteps
        self.param_dtype = config.param_dtype

        shard_fn = partial(shard_model, device_id=device_id)
        self.text_encoder = T5EncoderModel(
            text_len=config.text_len,
            dtype=config.t5_dtype,
            device=torch.device("cpu"),
            checkpoint_path=os.path.join(checkpoint_dir, config.t5_checkpoint),
            tokenizer_path=os.path.join(checkpoint_dir, config.t5_tokenizer),
            shard_fn=shard_fn if t5_fsdp else None,
        )

        self.vae_stride = config.vae_stride
        self.patch_size = config.patch_size
        self.vae = WanVAE(
            vae_pth=os.path.join(checkpoint_dir, config.vae_checkpoint),
            device=self.device,
        )

        logging.info(f"Creating WanModel from {checkpoint_dir}")
        self.model = WanModel.from_pretrained(checkpoint_dir)
        gc.collect()

        # noCFG LoRa
        self.nocfg_rank = 128
        self.nocfg_lora_modules = [
            "ffn.0", "ffn.2",
            "self_attn.q", "self_attn.k", "self_attn.v", "self_attn.o",
            "time_embedding.0", "time_embedding.2", "time_projection.1",
        ]
        self.nocfg_merge = True # False

        # Загрузка основного чекпоинта (если указан)
        if model_ckpt_path is not None:
            logging.info(f"Loading custom weights from {model_ckpt_path}")
            weights = torch.load(model_ckpt_path, map_location="cpu")
            
            # Проверяем, содержит ли чекпоинт LoRA адаптер
            if any("lora" in k.lower() for k in weights.keys()):
                model_ckpt_lora_config = LoraConfig(
                    r=self.nocfg_rank,
                    target_modules=self.nocfg_lora_modules,
                    exclude_modules="vace_blocks",
                    inference_mode=True, 
                )
                self.model = load_peft_adapter(
                    self.model,
                    model_ckpt_path,
                    adapter_config=model_ckpt_lora_config,
                    strict=False,
                    adapter_name="cfg_lora", rank=rank
                )
                # Merge lora into model
                if self.nocfg_merge:
                    self.model.merge_and_unload()
                    if hasattr(self.model, "base_model"):
                        self.model = getattr(self.model.base_model, "model", self.model.base_model)
                    if rank == 0:
                        torch.save(self.model.state_dict(), "/home/jovyan/dmitrienko/workspace/checkpoints/pretrained/wan21_t2v_nocfg.pt")
                    dist.barrier()
            else:
                self.model.load_state_dict(weights, strict=False)

        # Загрузка дополнительных LoRA адаптеров
        if len(lora_paths):
            if isinstance(lora_configs, list):
                if len(lora_configs) != len(lora_paths):
                    raise ValueError("Number of LORA configs must match number of LORA paths")
            else:
                lora_configs = [lora_configs] * len(lora_paths)

            for i, (path, lora_config) in enumerate(zip(lora_paths, lora_configs)):
                if path == '': continue
                if lora_config is None:
                    lora_weights = torch.load(path, map_location="cpu")
                    if any("time_projection." in k.lower() for k in lora_weights.keys()):
                        # Дефолтный конфиг, если не предоставлен
                        target_modules=[
                            "ffn.0", "ffn.2",
                            "self_attn.q", "self_attn.k","self_attn.v", "self_attn.o",
                            "time_embedding.0", "time_embedding.2", "time_projection.1",
                        ]
                    else:
                        target_modules=[
                            "ffn.0", "ffn.2",
                            "self_attn.q", "self_attn.k","self_attn.v", "self_attn.o",
                            "cross_attn.q", "cross_attn.k","cross_attn.v", "cross_attn.o",
                        ]
                    lora_config = LoraConfig(
                        r=128, target_modules=target_modules,
                        inference_mode=True,
                        exclude_modules="vace_blocks",
                    )
                self.model = load_peft_adapter(
                    self.model,
                    path,
                    adapter_config=lora_config,
                    adapter_name=f"lora_{i}",
                    # is_trainable=False,
                    strict=False, rank=rank
                )
        
        # Активация всех адаптеров (если их несколько)
        if len(lora_paths) > 0 and isinstance(self.model, PeftModel):
            all_adapters = [adapter_name for adapter_name in self.model.peft_config.keys()]
            logging.info(f"Activating all adapters: {all_adapters}")
            # self.model.set_adapter(all_adapters) TypeError: unhashable type: 'list'
            # self.model.enable_adapters() AttributeError: 'PeftModel' object has no attribute 'enable_adapters'
            lora_weights = [1]*len(lora_paths) if lora_weights is None else lora_weights
            self.adapter_name = "merge"
            self.model.add_weighted_adapter(
                adapters=all_adapters,
                weights=lora_weights,
                adapter_name=adapter_name,
                combination_type="linear",
                density=1
            )
            self.model.set_adapter(self.adapter_name)

        self.model.eval().requires_grad_(False)

        if use_usp:
            from xfuser.core.distributed import get_sequence_parallel_world_size

            from .distributed.xdit_context_parallel import (
                usp_attn_forward,
                usp_dit_forward,
            )

            for block in self.model.blocks:
                block.self_attn.forward = types.MethodType(
                    usp_attn_forward, block.self_attn
                )
            self.model.forward = types.MethodType(usp_dit_forward, self.model)
            self.sp_size = get_sequence_parallel_world_size()
        else:
            self.sp_size = 1

        if dist.is_initialized():
            dist.barrier()
        if dit_fsdp:
            self.model = shard_fn(self.model)
        else:
            self.model.to(self.device)

        self.sample_neg_prompt = config.sample_neg_prompt

    def generate(
        self,
        input_prompt,
        size=(1280, 720),
        frame_num=81,
        shift=5.0,
        sample_solver="unipc",
        sampling_steps=50,
        guide_scale=5.0,
        n_prompt="",
        seed=-1,
        offload_model=True,
    ):
        r"""
        Generates video frames from text prompt using diffusion process.

        Args:
            input_prompt (`str`):
                Text prompt for content generation
            size (tupele[`int`], *optional*, defaults to (1280,720)):
                Controls video resolution, (width,height).
            frame_num (`int`, *optional*, defaults to 81):
                How many frames to sample from a video. The number should be 4n+1
            shift (`float`, *optional*, defaults to 5.0):
                Noise schedule shift parameter. Affects temporal dynamics
            sample_solver (`str`, *optional*, defaults to 'unipc'):
                Solver used to sample the video.
            sampling_steps (`int`, *optional*, defaults to 40):
                Number of diffusion sampling steps. Higher values improve quality but slow generation
            guide_scale (`float`, *optional*, defaults 5.0):
                Classifier-free guidance scale. Controls prompt adherence vs. creativity
            n_prompt (`str`, *optional*, defaults to ""):
                Negative prompt for content exclusion. If not given, use `config.sample_neg_prompt`
            seed (`int`, *optional*, defaults to -1):
                Random seed for noise generation. If -1, use random seed.
            offload_model (`bool`, *optional*, defaults to True):
                If True, offloads models to CPU during generation to save VRAM

        Returns:
            torch.Tensor:
                Generated video frames tensor. Dimensions: (C, N H, W) where:
                - C: Color channels (3 for RGB)
                - N: Number of frames (81)
                - H: Frame height (from size)
                - W: Frame width from size)
        """
        # preprocess
        F = frame_num
        target_shape = (
            self.vae.model.z_dim,
            (F - 1) // self.vae_stride[0] + 1,
            size[1] // self.vae_stride[1],
            size[0] // self.vae_stride[2],
        )

        seq_len = (
            math.ceil(
                (target_shape[2] * target_shape[3])
                / (self.patch_size[1] * self.patch_size[2])
                * target_shape[1]
                / self.sp_size
            )
            * self.sp_size
        )

        if n_prompt == "":
            n_prompt = self.sample_neg_prompt
        seed = seed if seed >= 0 else random.randint(0, sys.maxsize)
        seed_g = torch.Generator(device=self.device)
        seed_g.manual_seed(seed)

        if not self.t5_cpu:
            self.text_encoder.model.to(self.device)
            context = self.text_encoder([input_prompt], self.device)
            context_null = self.text_encoder([n_prompt], self.device)
            if offload_model:
                self.text_encoder.model.cpu()
        else:
            context = self.text_encoder([input_prompt], torch.device("cpu"))
            context_null = self.text_encoder([n_prompt], torch.device("cpu"))
            context = [t.to(self.device) for t in context]
            context_null = [t.to(self.device) for t in context_null]

        noise = [
            torch.randn(
                target_shape[0],
                target_shape[1],
                target_shape[2],
                target_shape[3],
                dtype=torch.float32,
                device=self.device,
                generator=seed_g,
            )
        ]

        @contextmanager
        def noop_no_sync():
            yield

        no_sync = getattr(self.model, "no_sync", noop_no_sync)

        # evaluation mode
        with amp.autocast(dtype=self.param_dtype), torch.no_grad(), no_sync():

            if sample_solver == "unipc":
                sample_scheduler = FlowUniPCMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sample_scheduler.set_timesteps(
                    sampling_steps, device=self.device, shift=shift
                )
                timesteps = sample_scheduler.timesteps
            elif sample_solver == "dpm++":
                sample_scheduler = FlowDPMSolverMultistepScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler, device=self.device, sigmas=sampling_sigmas
                )
            elif sample_solver == 'euler':
                sample_scheduler = FlowMatchEulerDiscreteScheduler(
                    num_train_timesteps=self.num_train_timesteps,
                    shift=1,
                    use_dynamic_shifting=False,
                    # stochastic_sampling=stochastic_sampling
                )
                sampling_sigmas = get_sampling_sigmas(sampling_steps, shift)
                timesteps, _ = retrieve_timesteps(
                    sample_scheduler,
                    device=self.device,
                    sigmas=sampling_sigmas)
            else:
                raise NotImplementedError("Unsupported solver.")

            # sample videos
            latents = noise

            arg_c = {"context": context, "seq_len": seq_len}
            arg_null = {"context": context_null, "seq_len": seq_len}

            for _, t in enumerate(tqdm(timesteps)):
                latent_model_input = latents
                timestep = [t]

                timestep = torch.stack(timestep)
                # print(os.system('nvidia-smi'))
                self.model.to(self.device)
                # print('\nafter model on device\n' )
                # print( os.system('nvidia-smi'))
                noise_pred_cond = self.model(latent_model_input, t=timestep, **arg_c)[0]
                noise_pred_uncond = self.model(
                    latent_model_input, t=timestep, **arg_null
                )[0]

                noise_pred = noise_pred_uncond + guide_scale * (
                    noise_pred_cond - noise_pred_uncond
                )

                temp_x0 = sample_scheduler.step(
                    noise_pred.unsqueeze(0),
                    t,
                    latents[0].unsqueeze(0),
                    return_dict=False,
                    generator=seed_g,
                )[0]
                latents = [temp_x0.squeeze(0)]

            x0 = latents
            if offload_model:
                self.model.to("cpu")
            if self.rank == 0:
                videos = self.vae.decode(x0)

        del noise, latents
        del sample_scheduler
        if offload_model:
            gc.collect()
            torch.cuda.synchronize()
        if dist.is_initialized():
            dist.barrier()

        return videos[0] if self.rank == 0 else None
