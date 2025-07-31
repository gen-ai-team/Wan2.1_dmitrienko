
# mapfile -t IMAGES < /home/jovyan/dmitrienko/workspace/diffusion-pipe_dmitrienko/Wan2_1/higg_images.txt
# mapfile -t PROMPTS < /home/jovyan/dmitrienko/workspace/diffusion-pipe_dmitrienko/Wan2_1/nastya_prompts.txt

mapfile -t PROMPTS < /home/jovyan/novitskiy/WAN_Train/data/MovieGenVideoBench.txt

# Check if counts match
# if [ ${#IMAGES[@]} -ne ${#PROMPTS[@]} ]; then
#   echo "Error: Number of images and prompts must match!"
#   exit 1
# fi

# Build arguments
IMAGE_ARGS=("--image")
PROMPT_ARGS=("--prompt" )
for i in "${!PROMPTS[@]}"; do
  if [[ "$i" -gt -1 && "$i" -lt 21  ]]; then # 17
    # IMAGE_ARGS+=( "${IMAGES[$i]}")
    PROMPT_ARGS+=("${PROMPTS[$i]}")
  fi
done

#  CUDA_VISIBLE_DEVICES=0 torchrun --master_port=1126 --nproc_per_node=1 generate_v2.py --task t2v-14B \
#   --ckpt_dir "/home/jovyan/nkiselev/gen-ai-team/Wan2.1/wan_weights/Wan2.1-T2V-14B" \
#     "${PROMPT_ARGS[@]}" \
#     --size 1280*720 --t5_cpu --ulysses_size 1 --ring_size 1  \
#     --frame_num 65 --sample_steps 25 --sample_guide_scale 5 --base_seed 5575 \
#     --lora_scales 1. --trigger_word "" --crop_image 0 \
#     --lora_dirs /home/jovyan/dmitrienko/workspace/checkpoints/wan14b_lora128_higg+films_480-720_shift5/dolly_in-speedup-x10_lr1e-4_bs4_1gpu/20250509_10-33-53/epoch25 \
 

 CUDA_VISIBLE_DEVICES=0,1 torchrun --master_port=1126 --nproc_per_node=2 generate_v2.py --task t2v-14B \
  --ckpt_dir "/home/jovyan/nkiselev/gen-ai-team/Wan2.1/wan_weights/Wan2.1-T2V-14B" \
    "${PROMPT_ARGS[@]}" \
    --size 832*480 --dit_fsdp --t5_fsdp --ulysses_size 2 --ring_size 1  \
    --frame_num 81 --sample_steps 50 --sample_guide_scale 1 --base_seed 5575 --sample_shift 3  --crop_image 0 \
    --lora_scales 1. --trigger_word "" --lora_dirs /home/jovyan/dmitrienko/workspace/outputs  \
    --model_ckpt_path /home/jovyan/shares/SR008.fs2/novitskiy/WAN_flash/gathered_checkpoints/wan_cfg_synthetic_lora_v1_step_7199.pt \
 
 # 832*480 1280*720 \
 #
 # The video shows an aircraft flying above the clouds. The plane appears to be a private jet, identifiable by its sleek design and the registration number "N2000A" visible on the side of the fuselage. The aircraft is captured from a side angle, highlighting its wings and engines as it soars through the sky. The background consists of a mix of white and gray clouds, suggesting that the plane is at a high altitude. The lighting indicates that the scene might be taking place during either sunrise or sunset, as the light has a warm, golden hue."
 # "The video features a woman standing outdoors at night, illuminated by a soft blue light that casts a serene and somewhat ethereal glow over the scene. She is wearing a light-colored dress and has long hair that flows freely. The background includes blurred lights and what appears to be a building or structure, suggesting an urban setting."
# /home/jovyan/nkiselev/gen-ai-team/Wan2.1/Wan2.1-I2V-14B-720P