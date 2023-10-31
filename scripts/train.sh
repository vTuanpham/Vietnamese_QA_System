CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "src/models/configs/config_defaultSingleGPU.yaml" train.py \
        --train_file "src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json" "src/data/features/final_storge_converted/WizardLM_WizardLM_evol_instruct_70k/WizardLM_20k_Filtered_translated.json" "src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k_translated.json" "src/data/features/final_storge_converted/vilm-lima-vi/vilm_lima-vi.json" "src/data/features/final_storge_converted/mt_eng_vietnamese/MTEngVietnamese.json"\
        --each_train_file_percentage 30 30 15 5 20 \
        --val_file "src/data/features/final_storge_converted/mt_eng_vietnamese/MTEngVietnamese.json" "src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k_translated.json" "src/data/features/final_storge_converted/vilm-lima-vi/vilm_lima-vi.json" "src/data/features/final_storge_converted/WizardLM_WizardLM_evol_instruct_70k/WizardLM_20k_Filtered_translated.json" \
        --lora_r 32 \
        --dataset_name "Instruction_tune_8k_e3_en-vi" \
        --model_name_or_path EleutherAI/gpt-neo-125m \
        --shard_model \
        --max_model_shard_size 200MB \
        --max_train_samples 1000 \
        --max_eval_samples 1000 \
        --train_batch_size 1 \
        --num_epochs  3 \
        --seed 56 \
        --lr 2e-4 \
        --warmup_steps 0 \
        --model_dtype bfloat16 \
        --lora_dropout 0.05 \
        --weight_decay 0.2 \
        --model_type CAUSAL_LM \
        --minimum_free_spaces 1 \
        --gradient_accumulation_steps 128 \
        --generative_eval_batch_size 1 \
        --max_eval_generative_samples 200 \
        --perplexity_eval_batch_size 1 \
        --max_eval_perplexity_samples 999 \
        --lora_alpha 8 \
        --optim_name PagedAdamW8bit \
        --enable_model_offload \
        --gradient_checkpointing \
        --do_eval \
        --use_8bit \
        --llm_int8_enable_fp32_cpu_offload \
        --max_model_shard_size 500MB \
        --do_perplexity_eval \
        --do_generative_eval \
        --target_modules 'k_proj' 'v_proj' 'q_proj' 'out_proj' 'c_fc' 'c_proj' \
        --model_max_length 1024 \
        --max_new_tokens 256 \
        --context_length 1024 \
        --response_template " %%%%%%% Response:" \
        --print_model_key \
        --deep_speed_inf \
        --lr_sheduler_name cosine \
        --merge_weight_eval \
        --auto_kernel_injection
#        --repetition_penalty 1.2 \
#        --no_repeat_ngram_size 3 \
#        --top_k 80 \
#        --top_p 0.96 \
#        --do_sample \
#        --temperature 0.6 \
#        --penalty_alpha 0.6 \
#        --use_flash_attention_2 \
#        --injection_policy '''{"gpt2.modeling_gpt2.GPT2Block": "replace_policy.HFGPT2LayerPolicy"}''' \
