import argparse
import logging
import os, sys
from datetime import datetime
import random
from PIL import Image
import warnings
import torch
import torch.distributed as dist


import wan

from wan.utils.utils import cache_video, cache_image
from inference_lora import lora_name_to_path, add_lora_tag, LoraModel
from wan.configs import WAN_CONFIGS, SIZE_CONFIGS, MAX_AREA_CONFIGS, SUPPORTED_SIZES

EXAMPLE_PROMPT = {"t2v": {"prompt": "A beautiful landscape"}}
lora_name_to_tag = {
    "WANARCSH0T111": "WANARCSH0T111, 360 degrees shot, arcshot.",
    "Z00M1N": "Z00M1N, ZOOM IN.",
    "WANP4NR1GHT": "WANP4NR1GHT, camera pan right.",
    "opensource_arc": "r0t4tION orb1t",
    "r0t4tION_orb1t_r1ght": "r0t4tION orb1t r1ght",
    "r0t4tION_orb1t": "r0t4tION orb1t",
}

article_prompts = [
    "Two anthropomorphic cats in comfy boxing gear and bright gloves fight intensely on a spotlighted stage.",
    "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside.",
    # "A 3D-rendered New Year's celebration scene shows a crowd with their backs to the camera, gazing up at fireworks blooming in the night sky. The massive '2025' digits, composed of dazzling fireworks, illuminate the entire shy. People are dressed in various winter outfits, with different postures -some have their arms crossed over their chests, while others raise their phones to capture the moment. In the background, the silhouette of the city is visible, with skyscrapers and trees faintly outlined. At the instant the fireworks explode, the crowd falls into an unusual stillness, collectively welcoming the arrival of the new year: Long shot, viewed from a low-angle perspective.",
    "A surrealist digital illustration features a relaxed white cat wearing pink wireless headphones, immersed in its own musical world. Its pointy ears gently press against a pair of pink sunglasses, adding a touch of fashion. The cat is skillfully blowing a transparent bubblegum bubble with its front paw, the bubble swaying lightly in the air as it reaches its maximum size. The background is a soft, pure pink, creating a dreamlike atmosphere. The cat's eyes are slightly open, exuding an expression of contentment and enjoyment, as if the entire world has turned sweet. A close-up shot focuses on the moment the cat blows the bubble, capturing the serenity and playfulness of the scene. In a near view, from an eye-level perspective, the details of the cat's facial expressions and movements are emphasized.",
    "A summer ocean scenery video features an elegant woman wearing a light beige sleeveless top, seated in a retro wooden cabin. Behind her lies a shimmering azure sea and distant misty emerald islands. Her wavy golden hair flows freely in the wind, resting gently on her shoulders. Sunlight streams through the porthole, illuminating her warm smile as she faces the camera sideways, her elbow resting lightly on the seat in a relaxed and charming posture. Outside the window, a clear sky dotted with a few leisurely white clouds creates a serene and carefree atmosphere. The high-definition video captures every ray of sunlight and every detail of the waves, showcasing the romantic allure of summer sailing. A close-up shot of the woman facing the camera sideways is paired with expansive seascapes filmed using a wide-angle lens.",
    "A dreamlike garden scene in digital art style features an orange-and-black butterfly gracefully fluttering in the air, surrounded by several translucent pink butterflies. The vibrant pink flowers in full bloom have delicate petals, golden stamens, and fresh green leaves. The background of a blue shy filled with shimmering light creates a fairy-tale-like atmosphere. In a close-up shot with the camera slightly tilted upward, the interaction between the butterflies and the flowers is captured, filling the entire frame with life and energy.",
    "A dynamic video scene captures a passionate street artist immersed in playing the violin against a pure white background. He wears a wide-brimmed straw hat, its brim gently shading his bright eyes, and dresses in a simple white shint paired with retro brown trousers. A saxophone hangs sty lishly at his waist, adding a touch of effortless charm. The antist's fingers move nimbly across the strings, fully absorbed in the music, his face radiating a deep love and dedication for his craft. The surrounding blank space stretches infinitely, as if the entire world is immersed in his melody, tranquil and pure. In a medium shot, the camera captures the performer from the front, focusing on the moment of complete devotion.",
    # "A modern art piece in 3D rendering style features a pink single-seat sofa chair placed on a smooth floor: The design of the sofa chair is minimalist, with a subtle granular texture on its surface. The background is a clear blue sky dotted with a few white clouds, creating an overall bright and fresh color palette. With no people in the scene, the sofa chair stands out prominently. The shot is taken in a medium view, with a slight rotation of the camera emphasizing the details and texture of the sof a chair.",
    # "Retro cyberpunk style - Under flickering neon lights, a cyber warrior in a leather jacket passes through an abandoned electronics factory.",
    "The video shows a scene of a person in the post-apocalyptic world",
    # 'The video shows an aircraft flying above the clouds. The plane appears to be a private jet, identifiable by its sleek design and the registration number "N2000A" visible on the side of the fuselage. The aircraft is captured from a side angle, highlighting its wings and engines as it soars through the sky. The background consists of a mix of white and gray clouds, suggesting that the plane is at a high altitude. The lighting indicates that the scene might be taking place during either sunrise or sunset, as the light has a warm, golden hue.',
    # "Solar system showing the planets rotating around the sun. The video should show the orbital paths of each planet on their axis. The sun should have visible solar flares.",
]


# A hyper-realistic close-up of a, A hyper-detailed 3D render of

# article_prompts += [
#     "In a misty bamboo forest at dawn, a majestic ceramic duck sits atop a moss-covered stone, its glossy blue-and-white glaze reflecting the soft golden light. Delicate dewdrops cling to its surface, while the surrounding bamboo sways gently, creating a serene, almost mystical atmosphere. The background fades into layers of fog and faint silhouettes of distant mountains.",
#     "A hyper-detailed 3D render of a weathered bronze dog statue, its patina showing shades of turquoise and emerald green. It stands proudly on a marble pedestal in an abandoned library, sunlight streaming through broken stained-glass windows, illuminating dust particles in the air. Ancient books and scrolls litter the floor, hinting at forgotten knowledge.",
#     "A neon-lit cyberpunk alleyway at night, where a massive holographic octopus floats mid-air, its tentacles flickering with electric blue and purple pixels. Graffiti-covered walls and flickering neon signs in Japanese characters frame the scene, while rain-slicked pavement mirrors the glow. The atmosphere is futuristic yet gritty.",
#     "A striking black-and-white zebra figurine made of polished onyx and alabaster, placed on a sleek glass table in a modern art gallery. Spotlights highlight its contrasting stripes, casting sharp shadows on the minimalist white walls. Visitors’ blurred reflections can be seen in the glass, adding depth to the composition.",
#     "A carved jade monkey with mischievous glowing green eyes, perched on a gnarled tree branch in a lush jungle temple. Vines and orchids surround it, and golden sunlight filters through the canopy, creating dappled shadows on ancient stone ruins covered in moss. The air feels humid and alive with distant animal calls.",
#     "A surreal, oversized porcelain cup with intricate floral patterns, floating in the middle of a starry void. Tiny planets and constellations are reflected in its tea’s surface, which ripples slightly as if touched by an unseen force. The scene blends fantasy and cosmic wonder, evoking a dreamlike mood.",
#     "A close-up of a luminescent mechanical flower with delicate gold-plated petals, embedded in the wall of a steampunk laboratory. Tiny gears rotate inside its core, emitting a soft hum, while copper pipes and bubbling glass tubes fill the background. The warm glow of gas lamps enhances the retro-futuristic aesthetic.",
#     "A photorealistic portrait of a serene young girl with braided silver hair, her face partially obscured by a translucent veil embroidered with tiny stars. She stands in a moonlit garden of white roses, their petals glowing faintly. The background blurs into an endless twilight sky, blending realism with ethereal fantasy.",
#     "A stoic marble statue of a boy holding a flickering holographic lantern, positioned at the edge of a cliff overlooking a stormy ocean. Lightning flashes in the distance, illuminating his weathered features and the crashing waves below. The mix of classical sculpture and sci-fi elements creates a dramatic, cinematic feel.",
#     "A 3D animated render of a cheerful cartoon girl with round, sparkly eyes and rosy cheeks, wearing a cozy oversized sweater and a floppy beret. Her messy braids float slightly as if caught in a gentle breeze, and she’s holding a glowing jar of fireflies with both hands. The background is a sun-dappled forest meadow with pastel flowers and tiny, curious animals peeking out from the grass. Style: Pixar-meets-Ghibli, soft lighting, warm and inviting.",
#     "A hyper-realistic close-up of a joyful golden retriever puppy, its fur fluffy and sun-kissed, tongue lolling out in a happy pant. It’s sitting in a field of wildflowers, with a soft bokeh background of rolling green hills and a bright blue sky. The lighting is warm and golden, capturing every detail of its expressive eyes and wet nose.",
# ]


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Generate an image or video from a text prompt using LoRA"
    )
    parser.add_argument(
        "--ckpt_dir",
        type=str,
        default=None,
        help="The path to the checkpoint directory.",
    )
    parser.add_argument(
        "--task",
        type=str,
        default="t2v",
        choices=["t2v-14B", "t2v-1.3B", "i2v-14B"],
        help="Task type: text-to-video or text-to-image",
    )
    parser.add_argument(
        "--size",
        type=str,
        default="512*512",
        help="Output size",
    )
    parser.add_argument(
        "--save_dir", type=str, default=None, help="The path to save the generated image or video to."
    )
    parser.add_argument(
        "--sample_solver", type=str, default="unipc", choices=["unipc", "dpm++", "euler"], help="The solver used to sample."
    )
    parser.add_argument(
        "--frame_num",
        type=int,
        default=65,
        help="Number of frames for video generation",
    )
    parser.add_argument(
        "--sample_steps", type=int, default=40, help="Number of sampling steps"
    )
    parser.add_argument(
        "--crop_image", type=int, default=1, help="Crop the image for i2v"
    )
    parser.add_argument(
        "--trigger_word",
        type=str,
        default="r0t4tION orb1t ",
    )
    parser.add_argument(
        "--sample_guide_scale",
        type=float,
        default=5.0,
        help="Guidance scale for sampling",
    )
    parser.add_argument("--base_seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--offload_model", action="store_true", help="Offload model to save memory"
    )
    parser.add_argument(
        "--lora_dirs",
        type=str,
        nargs="+",
        default=[
            "/home/jovyan/dmitrienko/workspace/checkpoints/wan14b/spell_fps16_65frames_512_lora64_lr2e-5_gbs2_2gpus/20250328_14-29-39/epoch40/"
        ],
        help="List of LoRA names",
    )
    parser.add_argument(
        "--lora_scales",
        type=float,
        nargs="+",
        default=[1.0],
        help="List of LoRA scales (must match LoRA names)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=article_prompts,
        help="Prompt for generation",
    )
    parser.add_argument(
        "--sample_shift",
        type=float,
        default=5,
        help="Sampling shift factor for flow matching schedulers.",
    )
    parser.add_argument(
        "--image",
        type=str,
        nargs="+",
        default=None,
        help="The image to generate the video from.",
    )
    parser.add_argument(
        "--ulysses_size",
        type=int,
        default=1,
        help="The size of the ulysses parallelism in DiT.",
    )
    parser.add_argument(
        "--ring_size",
        type=int,
        default=1,
        help="The size of the ring attention parallelism in DiT.",
    )
    parser.add_argument(
        "--t5_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for T5.",
    )
    parser.add_argument(
        "--t5_cpu",
        action="store_true",
        default=False,
        help="Whether to place T5 model on CPU.",
    )
    parser.add_argument(
        "--dit_fsdp",
        action="store_true",
        default=False,
        help="Whether to use FSDP for DiT.",
    )
    parser.add_argument(
        "--model_ckpt_path",
        type=str,
        default=None,
        help="The diffusion model ckpt to load on.",
    )
    parser.add_argument(
        "--lora_paths",
        type=str,
        nargs="+",
        default=None,
        help="List of LoRA names",
    )
    parser.add_argument(
        "--nocfg_lora_weight",
        type=float,
        default=4.,
        help="weight for nocfg lora for mixing with camera loras")
    parser.add_argument(
        "--prompts_path",
        type=str,
        default='/home/jovyan/aigkargapoltseva/MovieGenVideoBench.txt',
        help="Path to prompts for an sbs gen mode (t2v)")
    parser.add_argument(
        "--n_prompts",
        type=int,
        default=5,
        help="Number of prompts from prompts_path")
    parser.add_argument(
        "--start_prompt_i",
        type=int,
        default=0,
        help="Number of start i of prompts from prompts_path")
    args = parser.parse_args()
    return args


def generate(args):
    rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    local_rank = int(os.getenv("LOCAL_RANK", 0))
    device = local_rank
    logging.basicConfig(level=logging.INFO)

    if world_size > 1:
        torch.cuda.set_device(local_rank)
        dist.init_process_group(
            backend="nccl", init_method="env://", rank=rank, world_size=world_size
        )
    else:
        assert not (
            args.t5_fsdp or args.dit_fsdp
        ), f"t5_fsdp and dit_fsdp are not supported in non-distributed environments."
        assert not (
            args.ulysses_size > 1 or args.ring_size > 1
        ), f"context parallel are not supported in non-distributed environments."

    if args.ulysses_size > 1 or args.ring_size > 1:
        assert (
            args.ulysses_size * args.ring_size == world_size
        ), f"The number of ulysses_size and ring_size should be equal to the world size."
        from xfuser.core.distributed import (
            initialize_model_parallel,
            init_distributed_environment,
        )

        init_distributed_environment(
            rank=dist.get_rank(), world_size=dist.get_world_size()
        )

        initialize_model_parallel(
            sequence_parallel_degree=dist.get_world_size(),
            ring_degree=args.ring_size,
            ulysses_degree=args.ulysses_size,
        )
    cfg = WAN_CONFIGS[args.task]
    if args.ulysses_size > 1:
        assert (
            cfg.num_heads % args.ulysses_size == 0
        ), f"`num_heads` must be divisible by `ulysses_size`."

    logging.info(f"Generation job args: {args}")
    logging.info(f"Generation model config: {cfg}")

    if dist.is_initialized():
        base_seed = [args.base_seed] if rank == 0 else [None]
        dist.broadcast_object_list(base_seed, src=0)
        args.base_seed = base_seed[0]

    # model_ckpt_path = "/home/jovyan/shares/SR008.fs2/novitskiy/WAN_flash/gathered_checkpoints/wan_cfg_synthetic_v1_step_1000.pt"
    lora = True if  args.model_ckpt_path is not None and 'wan_cfg_synthetic_lora_v1_step' in args.model_ckpt_path else False
    if "t2v" in args.task or "t2i" in args.task:
        logging.info("Creating WanT2V_peftlora pipeline.")
        pipe = wan.WanT2V_peftlora(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
            # lora=lora,
            model_ckpt_path=args.model_ckpt_path,
        )
        args.image = [None] * len(args.prompt)
    else:
        logging.info("Creating WanI2V pipeline.")
        pipe = wan.WanI2V(
            config=cfg,
            checkpoint_dir=args.ckpt_dir,
            device_id=device,
            rank=rank,
            t5_fsdp=args.t5_fsdp,
            dit_fsdp=args.dit_fsdp,
            use_usp=(args.ulysses_size > 1 or args.ring_size > 1),
            t5_cpu=args.t5_cpu,
        )
    
    if args.model_ckpt_path is not None and args.model_ckpt_path != '':
        args.sample_guide_scale = 1
        args.sample_solver = 'euler'

    # logging.info("compiling model")
    # pipe.model = torch.compile(pipe.model)
    # pipe.vae.model = torch.compile(pipe.vae.model)
    # pipe.text_encoder.model = torch.compile(pipe.text_encoder.model)
    
    if args.prompts_path.endswith('.txt'):
        with open(args.prompts_path, 'r') as f:
            prompts = f.readlines()[: args.start_prompt_i + args.n_prompts]
    elif args.prompts_path.endswith('.csv'):
        import pandas as pd
        prompts = pd.read_csv(args.prompts_path,  header=None)[0].to_list()[: args.start_prompt_i + args.n_prompts]

    if hasattr(pipe.model, "peft_config"):
        all_adapters = [adapter_name for adapter_name in pipe.model.peft_config.keys()] 
    else:
        # значение по дефолту, чтобы запустить цикл 
        all_adapters = [""]

    for lora_name in all_adapters:
        
        if (len(all_adapters) == 1 and lora_name != "" ) or (lora_name != "" and args.model_ckpt_path is None): 
            # Если всего 1 адаптер у модели, то включаем его. Если их больше то включаем только 2 сразу: nocfg_camera + 2я лора
            # Если args.model_ckpt_path is None то нет nocfg, включаем 1 лору
            pipe.model.set_adapter(lora_name)
            logging.info(f"active_adapters = {pipe.model.active_adapters}")
        elif len(all_adapters) > 1 and args.model_ckpt_path is not None:
            if lora_name not in ["nocfg_camera", "nocfg", ""]:
                pipe.model.set_adapter(["nocfg_camera", lora_name, ])
                logging.info(f"active_adapters = { pipe.model.active_adapters}")
            else:
                continue
        
        os.makedirs(args.save_dir, exist_ok=True)
                

        for i in range(args.start_prompt_i, len(prompts)):
            save_path = os.path.join(
                args.save_dir,
                f"{i}.mp4",
            )
            if os.path.exists(save_path): continue
            # logging.info(f"Generating for prompt: { prompts[i]}")
            if "t2v" in args.task or "t2i" in args.task:
                video = pipe.generate(
                    prompts[i],
                    size=SIZE_CONFIGS[args.size],
                    frame_num=args.frame_num,
                    shift=args.sample_shift,
                    sample_solver=args.sample_solver,
                    sampling_steps=args.sample_steps,
                    guide_scale=args.sample_guide_scale,
                    seed=args.base_seed + i,
                    offload_model=args.offload_model,
                )
                if video is not None:
                    video = video[None]
            
            if rank == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                safe_prompt = prompts[i].replace(" ", "_").replace("/", "_")[:150]
                suffix = ".png" if args.task == "t2i" else ".mp4"
                model_name = "wan"
                if 'vace' in args.ckpt_dir.lower() : model_name = 'vace'
                # if  model_ckpt_path is not None: model_name += '_cfg'
                
                
                # save_path += f"{safe_prompt}_{timestamp}{suffix}"

                logging.info(f"Saving result to {save_path}")
                cache_video(
                    video,
                    save_path,
                    nrow=1,
                    fps=cfg.sample_fps,
                    normalize=True,
                    value_range=(-1, 1),
                )

    logging.info("Generation finished.")


if __name__ == "__main__":
    args = _parse_args()
    generate(args)
