CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "src/models/configs/config_defaultSingleGPU.yaml" train.py \
        --train_file "src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k.json" "src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k_translated.json" \
        --val_file "src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k.json" "src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k_translated.json" \
        --lora_r 64 \
        --dataset_name "Instruction_tune_8k_e3_en-vi" \
        --model_name_or_path EleutherAI/pythia-410m-deduped-v0 \
        --max_train_samples 8000 \
        --max_eval_samples 500 \
        --train_batch_size 2 \
        --num_epochs  3 \
        --seed 53 \
        --lr 5e-4 \
        --warmup_steps 120 \
        --model_dtype bfloat16 \
        --lora_dropout 0.02 \
        --weight_decay 0.15 \
        --model_type CAUSAL_LM \
        --minimum_free_spaces 1 \
        --gradient_accumulation_steps 9 \
        --generative_eval_batch_size 3 \
        --max_eval_generative_samples 10 \
        --perplexity_eval_batch_size 1 \
        --max_eval_perplexity_samples 499 \
        --lora_alpha 32 \
        --optim_name PagedAdamW8bit \
        --enable_model_offload \
        --gradient_checkpointing \
        --do_eval \
        --use_8bit \
        --llm_int8_enable_fp32_cpu_offload \
        --max_model_shard_size 500MB \
        --do_perplexity_eval \
        --do_generative_eval \
        --target_modules 'query_key_value' 'dense' 'dense_h_to_4h' 'dense_4h_to_h' \
        --model_max_length 768 \
        --max_new_tokens 256 \
        --context_length 1024 \
        --response_template " %%%%%%% Response:" \
        --print_model_key \
        --deep_speed_inf \
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
