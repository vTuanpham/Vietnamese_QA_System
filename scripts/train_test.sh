CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "src/models/configs/accelerate_ds_zero3_cpu_offload_config.yaml" train.py \
        --train_file "src/data/features/final_storge_converted/THUDM-webglm-qa/WebglmQA.json" \
        --each_train_file_percentage 100 \
        --val_file "src/data/features/final_storge_converted/THUDM-webglm-qa/WebglmQA.json" \
        --test_file "src/data/features/final_storge_converted/THUDM-webglm-qa/WebglmQA.json" \
        --lora_r 64 \
        --output_dir "./" \
        --with_tracking \
        --report_to wandb \
        --dataset_name "WebglmQA_tuned_test" \
        --model_name_or_path EleutherAI/gpt-neo-125m \
        --checkpoint_at_max_time 0.5 \
        --max_train_samples 10000 \
        --max_eval_samples 500 \
        --train_batch_size 2 \
        --num_epochs  3 \
        --seed 56 \
        --lr 1e-4 \
        --warmup_steps 0 \
        --model_dtype bfloat16 \
        --lora_dropout 0.03 \
        --weight_decay 0.15 \
        --model_type CAUSAL_LM \
        --minimum_free_spaces 1 \
        --gradient_accumulation_steps 256 \
        --generative_eval_batch_size 1 \
        --max_eval_generative_samples 20 \
        --perplexity_eval_batch_size 1 \
        --max_eval_perplexity_samples 499 \
        --lora_alpha 32 \
        --optim_name AdamW \
        --enable_model_offload \
        --gradient_checkpointing \
        --do_eval \
        --do_perplexity_eval \
        --do_generative_eval \
        --target_modules 'k_proj' 'v_proj' 'q_proj' 'c_fc' 'c_proj' 'lm_head' \
        --no_split_module_classes "GPTNeoBlock" "GPTNeoAttention" "GPTNeoSelfAttention" "LayerNorm" "LayerNorm" "GPTNeoMLP" "Linear" \
        --model_max_length 1256 \
        --max_new_tokens 256 \
        --context_length 1256 \
        --response_template " %%%%%%% Response:" \
        --print_model_key \
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
