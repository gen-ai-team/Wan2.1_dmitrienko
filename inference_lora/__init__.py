from .lora_model import LoraModel

lora_name_to_path = {
    "WANARCSH0T111_1": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan14b_i2v/arc_shot_1_832x480_lora32_lr1e-4_2gpus/20250321_19-27-37/epoch30/adapter_model.safetensors",
    "WANARCSH0T111": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan14b_i2v/arc_shot_v2_832x480_lora32_lr2e-5_4gpus/20250321_11-09-47/epoch56/adapter_model.safetensors",
    # "Чебурашка_64_200": "/home/jovyan/nkiselev/diffusion-pipe/checkpoints/wan/Чебурашка_64/20250319_16-43-07/epoch200/adapter_model.safetensors",
    # "Z00M1N": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan1.3b/zoom_in_1024_lr4e-5_4gpus/20250320_14-30-35/epoch40/adapter_model.safetensors",
    "Z00M1N": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan1.3b/zoom_in_strong_r64_832_lr5e-5_1gpus/20250322_08-38-38/epoch2/adapter_model.safetensors",
    "Z00M1N_144": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan1.3b/zoom_in_strong_r64_144_lr1e-4_1gpus/20250322_09-45-55/epoch5/adapter_model.safetensors",
    "Z00M1N_288": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan1.3b/zoom_in_strong_r64_144_lr1e-4_1gpus/20250322_10-40-07/epoch10/adapter_model.safetensors",
    "WANARCSH0T111_288": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan1.3b/arc_v2_r64_288_lr5e-5_1gpus/20250322_11-14-44/epoch5/adapter_model.safetensors",
    "WANARCSH0T111_288_v2": "/home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/wan1.3b/arc_v2_r64_288_lr1e-4_2gpus/20250322_14-13-55/epoch3/adapter_model.safetensors",
}

lora_name_to_tag = {
    "WANARCSH0T111": "WANARCSH0T111, 360 degrees shot, arcshot.",
    "WANARCSH0T111_1": "WANARCSH0T111, 360 degrees shot, arcshot.",
    "WANARCSH0T111_288": "WANARCSH0T111, 360 degrees shot, arcshot.",
    "Z00M1N": "Z00M1N, ZOOM IN.",
    "Z00M1N_144": "Z00M1N, ZOOM IN.",
    "Z00M1N_288": "Z00M1N, ZOOM IN.",
    "Чебурашка_64_200": "Cheburashka",
}


def add_lora_tag(prompt, lora_name):
    if lora_name is not None:
        return lora_name_to_tag[lora_name] + f" {prompt}"
    return prompt
