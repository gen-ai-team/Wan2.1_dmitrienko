#------------------------ Pipeline ------------------------# --compile

source /home/user/conda/bin/activate
conda activate /home/jovyan/.mlspace/envs/dmitrienko_vace
cd /home/jovyan/dmitrienko/workspace/diffusion-pipe_dmitrienko/Wan2_1

torchrun --master_port=1224 --nproc_per_node=8 generate_v3.py  \
    --size 832*480 --dit_fsdp --t5_fsdp --ulysses_size 8 --ring_size 1 \
    --base_seed 5575 --frame_num 81  \
    --sample_steps 50 --sample_guide_scale 5.0 --sample_shift 5 \
    --start_prompt_i 0 \
    --n_prompts 100 \
    --prompts_path "/home/jovyan/dmitrienko/workspace/inputs/KandiVideoPrompts_Текст_промпта_на_английском,_бьютификация_GigaChat-Max.csv" \
    --prompt "an adorable kangaroo wearing purple overalls and cowboy boots taking a pleasant stroll in Antarctica during a winter storm" \
    --ckpt_dir  "/home/jovyan/shares/SR008.fs2/novitskiy/wan_weights/Wan2.1-T2V-14B" \
    --save_dir /home/jovyan/dmitrienko/workspace/outputs/WAN/original/t2v_50steps_81frames_5gs_5sh_seed5575 \
