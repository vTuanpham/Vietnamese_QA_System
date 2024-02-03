CUDA_VISIBLE_DEVICES=0 accelerate launch --config_file "src/models/configs/config_defaultSingleGPU.yaml" train.py \
        --train_file "src/data/features/final_storge_converted/THUDM-webglm-qa/WebglmQA.json" \
        --each_train_file_percentage 100 \
        --val_file "src/data/features/final_storge_converted/THUDM-webglm-qa/WebglmQA.json" \
        --test_file "src/data/features/final_storge_converted/THUDM-webglm-qa/WebglmQA.json" \
        --lora_r 64 \
        --output_dir "./" \
        --dataset_name "WebglmQA_tuned" \
        --model_name_or_path "EleutherAI/pythia-410m-deduped-v0" \
        --use_4bit \
        --max_train_samples 5000 \
        --max_eval_samples 500 \
        --train_batch_size 1 \
        --num_epochs  3 \
        --seed 56 \
        --lr 1e-4 \
        --warmup_steps 0 \
        --model_dtype bfloat16 \
        --lora_dropout 0.02 \
        --weight_decay 0.1 \
        --model_type CAUSAL_LM \
        --minimum_free_spaces 1 \
        --gradient_accumulation_steps 128 \
        --generative_eval_batch_size 1 \
        --max_eval_generative_samples 10 \
        --perplexity_eval_batch_size 1 \
        --max_eval_perplexity_samples 300 \
        --lora_alpha 128 \
        --optim_name PagedLion8bit \
        --enable_model_offload \
        --gradient_checkpointing \
        --do_eval \
        --llm_int8_enable_fp32_cpu_offload \
        --max_model_shard_size 500MB \
        --do_perplexity_eval \
        --do_generative_eval \
        --target_modules 'query_key_value' 'dense' 'dense_h_to_4h' 'dense_4h_to_h' \
        --modules_to_save 'embed_in' 'embed_out' \
        --no_split_module_classes "GPTNeoXLayer" "GPTNeoXAttention" "GPTNeoXMLP" "GPTNeoXRotaryEmbedding" \
        --model_max_length 1256 \
        --max_new_tokens 256 \
        --context_length 1256 \
        --response_template "%%%%%%% Response:" \
        --add_tokens_list "####### Instruction:" "%%%%%%% Response:" \
        --print_model_key \
        --lr_sheduler_name cosine \
        --merge_weight_eval \
        --repetition_penalty 1.2 \
        --no_repeat_ngram_size 3 \
        --auto_kernel_injection
#        --no_repeat_ngram_size 3 \
#        --top_k 80 \
#        --top_p 0.96 \
#        --do_sample \
#        --temperature 0.6 \
#        --penalty_alpha 0.6 \
#        --use_flash_attention_2 \
#        --injection_policy '''{"gpt2.modeling_gpt2.GPT2Block": "replace_policy.HFGPT2LayerPolicy"}''' \
