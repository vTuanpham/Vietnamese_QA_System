import argparse

from src.models.trainer import train


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument("--model_name_or_path", type=str, default="google/umt5-small", help="Model name or path")
    parser.add_argument("--model_type", type=str, default="SEQ_2_SEQ_LM", help="Type of model to train")
    parser.add_argument("--model_dtype", type=str, default="auto", help="Model torch_dtype")
    parser.add_argument("--print_model_key", action='store_true', help="Whether to print out model structure")

    peft_group = parser.add_argument_group("Parameters efficient arguments")
    peft_group.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    peft_group.add_argument("--lora_alpha", type=int, default=64, help="Alpha parameter for LoRA scaling")
    peft_group.add_argument("--lora_dropout", type=float, default=0.04, help="Dropout probability for LoRA layers")
    peft_group.add_argument("--target_modules", nargs='+', type=str,  default=None,
                            help="The target modules for lora")

    bitsandbytes_group = parser.add_argument_group("Bitsandbytes arguments")
    bitsandbytes_group.add_argument("--use_4bit", action='store_true', help="Activate 4-bit precision base model loading")
    bitsandbytes_group.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        help="Compute dtype for 4-bit base models")
    bitsandbytes_group.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (fp4 or nf4)")
    bitsandbytes_group.add_argument("--use_nested_quant", action='store_true',
                        help="Activate nested quantization for 4-bit base models (double quantization)")
    bitsandbytes_group.add_argument("--use_8bit", action='store_true', help="Activate 8-bit precision base model loading")

    parser.add_argument("--better_transformer", action='store_true', help="Enable flash attention")
    parser.add_argument("--Optim_name", type=str, default="PagedLion8bit", help="Name of optimizer in bnb lib")

    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--do_test", action='store_true', help="Flag to perform testing")
    parser.add_argument("--do_eval", action='store_true', help="Flag to perform evaluation")
    parser.add_argument("--do_perplexity_eval", action='store_true', help="Flag to enable perplexity computation, relevant when using casual-LM")
    parser.add_argument("--do_generate_eval", action="store_true", help="Flag to enable model.generate eval")

    parser.add_argument("--merge_weight_eval", action='store_true', help="Flag to enable merge weight from peft for faster eval")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Use gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--enable_model_offload", action='store_true', help="Enable model offload")
    parser.add_argument("--llm_int8_enable_fp32_cpu_offload", action='store_true', help="")

    dataloader_group = parser.add_argument_group("Dataloader Arguments")
    dataloader_group.add_argument("--dataset_name", type=str, default="Instruction_en-vn_mix", help="Dataset name")
    dataloader_group.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    dataloader_group.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    dataloader_group.add_argument("--text_column", type=str, default="prompt", help="Text column")
    dataloader_group.add_argument("--label_column", type=str, default="target", help="Label column")
    dataloader_group.add_argument("--block_size", type=int, default=128, help="")
    dataloader_group.add_argument("--train_file", nargs='+', type=str, default=[
        r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json",
        r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json",
        r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleanedFormated.json",
        r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleaned_translatedFormated.json"
    ], help="List of training files")

    dataloader_group.add_argument("--val_file", nargs='+', type=str, default=[
        r"src/data/features/final_storge_converted/WizardLM_WizardLM_evol_instruct_70k/WizardLM_70kFormated.json",
        r"src/data/features/final_storge_converted/WizardLM_WizardLM_evol_instruct_70k/WizardLM_70k_translatedFormated.json"
    ], help="List of validation files")

    dataloader_group.add_argument("--test_file", nargs='+', type=str, default=[
        r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleaned_translatedFormated.json",
        r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleanedFormated.json"
    ], help="List of test files")

    dataloader_group.add_argument("--max_train_samples", type=int, default=10000,
                                  help="Maximum number of training samples")
    dataloader_group.add_argument("--max_eval_samples", type=int, default=100,
                                  help="Maximum number of evaluation samples")
    dataloader_group.add_argument("--max_predict_samples", type=int, default=20,
                                  help="Maximum number of prediction samples")
    dataloader_group.add_argument("--config_type", type=str, default="AdvanceInstructSample", help="Configuration type")
    dataloader_group.add_argument("--no_preprocess_data", action="store_true", help="Whether to tokenized the data first"
                                                                                    "turn off this flag for large dataset")

    generation_group = parser.add_argument_group("Generation Arguments")
    generation_group.add_argument("--top_k", type=int, default=10, help="Top-k value (default: 10)")
    generation_group.add_argument("--no_sample", action="store_true", help="Enable sampling (default: True)")
    generation_group.add_argument("--no_repeat_ngram_size", type=int, default=3, help="No repeat n-gram size (default: 3)")
    generation_group.add_argument("--num_beams", type=int, default=5, help="Number of beams (default: 5)")
    generation_group.add_argument("--no_early_stopping", action="store_true", help="Enable early stopping (default: True)")
    generation_group.add_argument("--max_time", type=int, default=100, help="Max time (default: 100)")
    generation_group.add_argument("--penalty_alpha", type=float, default=1.2, help="Penalty alpha (default: 1.2)")
    generation_group.add_argument("--repetition_penalty", type=float, default=2.5, help="Repetition penalty (default: 2.5)")
    generation_group.add_argument("--temperature", type=float, default=1.5, help="Temperature (default: 1.5)")
    generation_group.add_argument("--no_truncation", action="store_true", help="Enable truncation (default: True)")
    generation_group.add_argument("--encoder_repetition_penalty", type=float, default=2.0, help="Encoder repetition penalty (default: 2.0)")
    generation_group.add_argument("--max_length", type=int, default=1024, help="Max length (default: 1024)")
    generation_group.add_argument("--max_new_tokens", type=int, default=128, help="Max new tokens (default: 128)")

    args = parser.parse_args()

    # Sanity check
    if args.use_8bit and args.use_4bit:
        raise "Can't use 8bit and 4bit quantization at the same time"

    return args


if __name__=="__main__":
    args = parse_arguments()
    train(args)
