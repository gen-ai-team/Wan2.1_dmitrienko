import torch
import torch.nn as nn
import torch.nn.functional as F
from safetensors.torch import load_file


class LoraHook:
    def __init__(self, lora_A, lora_B):
        self.lora_A = lora_A
        self.lora_B = lora_B

    def _set_lora_scale(self, lora_scale: float = 1.0):
        self.lora_scale = lora_scale

    def hook(self, module, input, output):
        lora_weights = torch.mm(self.lora_B, self.lora_A)
        return output + self.lora_scale * F.linear(input[0], lora_weights)


def load_model_from_network_storage(checkpoint_path):
    """
    Load a safetensors model from network storage by first copying to memory

    Args:
        checkpoint_path: Path to the safetensors file
    """
    print(f"Loading model from: {checkpoint_path}")

    # Read the entire file into memory
    print("Reading file into memory...")
    with open(checkpoint_path, "rb") as f:
        file_content = f.read()
    print(f"Read {len(file_content) / (1024*1024*1024):.2f}GB into memory")

    # Load using safetensors.torch.load
    try:
        print("Loading tensors...")
        import safetensors

        tensors = safetensors.torch.load(
            file_content,
        )
        print(f"Successfully loaded {len(tensors)} tensors")
        return tensors
    except Exception as e:
        print(f"Error loading tensors: {str(e)}")
        raise


class LoraModel:
    def __init__(self, model):
        self.model = model
        # self.lora_name_to_path = lora_name_to_path
        self.hooks = {}

        self.current_lora_name = None

    def init_lora_hooks(self, lora_path):

        state_dict = load_model_from_network_storage(
            lora_path,
            # device="cpu"
        )
        name = lora_path
        self.hooks[name] = {}
        for weight_name in state_dict.keys():
            layer_name = (
                weight_name.removeprefix("diffusion_model.")
                .removesuffix(".lora_A.weight")
                .removesuffix(".lora_B.weight")
            )
            lora_A = state_dict[f"diffusion_model.{layer_name}.lora_A.weight"]
            lora_B = state_dict[f"diffusion_model.{layer_name}.lora_B.weight"]
            self.hooks[name][layer_name] = LoraHook(lora_A, lora_B)

    # @property
    # def available_names(self):
    #     return list(self.lora_name_to_path.keys())

    @property
    def device(self):
        return self.model.device

    @property
    def dtype(self):
        return self.model.dtype

    def to(self, *args, **kwargs):
        self.model.to(*args, **kwargs)
        if self.current_lora_name is not None:
            for hook in self.hooks[self.current_lora_name].values():
                hook.lora_A = hook.lora_A.to(*args, **kwargs).to(self.model.dtype)
                hook.lora_B = hook.lora_B.to(*args, **kwargs).to(self.model.dtype)

    def apply_lora(self, lora_path, lora_scale: float = 1.0):
        self.init_lora_hooks(lora_path)
        # if name not in self.hooks:
        #     raise ValueError(
        #         f"LoRA configuration '{name}' not found in '{self.available_names}'."
        #     )

        self.remove_lora()

        for layer_name, hook in self.hooks[lora_path].items():
            module = self.get_module(layer_name)
            if module:
                hook._set_lora_scale(lora_scale)
                self.register_hook(module, hook)

        self.current_lora_name = lora_path

    def remove_lora(self):
        if self.current_lora_name is not None:
            for layer_name in self.hooks[self.current_lora_name].keys():
                module = self.get_module(layer_name)
                if module:
                    self.remove_hook(module)
                hook = self.hooks[self.current_lora_name][layer_name]
                hook.lora_A = hook.lora_A.to(device=torch.device("cpu"))
                hook.lora_B = hook.lora_B.to(device=torch.device("cpu"))
            torch.cuda.empty_cache()

        self.current_lora_name = None

    def get_module(self, name):
        parent_module = self.model
        name_parts = name.split(".")
        for part in name_parts[:-1]:
            parent_module = getattr(parent_module, part, None)
            if parent_module is None:
                return None
        return getattr(parent_module, name_parts[-1], None)

    def register_hook(self, module, hook):
        handle = module.register_forward_hook(hook.hook)
        setattr(module, "_lora_hook_handle", handle)

    def remove_hook(self, module):
        if hasattr(module, "_lora_hook_handle"):
            handle = getattr(module, "_lora_hook_handle")
            handle.remove()
            delattr(module, "_lora_hook_handle")

    def __call__(self, *args, **kwargs):
        return self.model(*args, **kwargs)
