# CUDA_VISIBLE_DEVICES=0 python generate.py  --task i2v-14B --size  832*480 \
#  --ckpt_dir /home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/pretrained/Wan2.1-I2V-14B-480P \
#  --base_seed 44  \
#  --frame_num 33 --sample_steps 20 \
#  --lora_name WANARCSH0T111 \
#  --image /home/jovyan/dmitrienko/workspace/diffusion-pipe_dmitrienko/submodules/Wan2_1/examples/cat_3d.jpg \
#  --prompt "The video depicts a cat, 3d model."

mapfile -t IMAGES < /home/jovyan/dmitrienko/workspace/diffusion-pipe_dmitrienko/Wan2_1/higg_images.txt
mapfile -t PROMPTS < /home/jovyan/dmitrienko/workspace/diffusion-pipe_dmitrienko/Wan2_1/higg_texts.txt

# Check if counts match
if [ ${#IMAGES[@]} -ne ${#PROMPTS[@]} ]; then
  echo "Error: Number of images and prompts must match!"
  exit 1
fi

# Build arguments
IMAGE_ARGS=("--image")
PROMPT_ARGS=("--prompt" )
for i in "${!IMAGES[@]}"; do
  if [[ "$i" -gt 13 && "$i" -lt 70  ]]; then # 17
    IMAGE_ARGS+=( "${IMAGES[$i]}")
    PROMPT_ARGS+=("${PROMPTS[$i]}")
  fi
done

 CUDA_VISIBLE_DEVICES=4,5 torchrun --master_port=1164 --nproc_per_node=2 generate_v2.py --task i2v-14B \
  --ckpt_dir /home/jovyan/shares/SR008.fs2/dmitrienko/checkpoints/pretrained/Wan2.1-I2V-14B-480P \
    "${IMAGE_ARGS[@]}" \
    "${PROMPT_ARGS[@]}" \
    --size 832*480 --dit_fsdp --t5_fsdp --ulysses_size 2 --ring_size 1  \
    --frame_num 65 --sample_steps 40 --sample_guide_scale 5 --base_seed 5575 \
    --lora_scales 1. --trigger_word "" --crop_image 0 \
    --lora_dirs /home/jovyan/dmitrienko/workspace/outputs \
 
 
 # The video shows an aircraft flying above the clouds. The plane appears to be a private jet, identifiable by its sleek design and the registration number "N2000A" visible on the side of the fuselage. The aircraft is captured from a side angle, highlighting its wings and engines as it soars through the sky. The background consists of a mix of white and gray clouds, suggesting that the plane is at a high altitude. The lighting indicates that the scene might be taking place during either sunrise or sunset, as the light has a warm, golden hue."

 # "Summer beach vacation style, a white cat wearing sunglasses sits on a surfboard. The fluffy-furred feline gazes directly at the camera with a relaxed expression. Blurred beach scenery forms the background featuring crystal-clear waters, distant green hills, and a blue sky dotted with white clouds. The cat assumes a naturally relaxed posture, as if savoring the sea breeze and warm sunlight. A close-up shot highlights the feline's intricate details and the refreshing atmosphere of the seaside."
 #"ARCSH0T111. The video features a woman standing outdoors at night, illuminated by a soft blue light that casts a serene and somewhat ethereal glow over the scene. She is wearing a light-colored dress and has long hair that flows freely. The background includes blurred lights and what appears to be a building or structure, suggesting an urban setting."
# /home/jovyan/nkiselev/gen-ai-team/Wan2.1/Wan2.1-I2V-14B-720P