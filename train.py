import argparse

from src.models.trainer import train


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument("--model_name_or_path", type=str, default="google/umt5-small", help="Model name or path")
    parser.add_argument("--model_type", type=str, default="SEQ_2_SEQ_LM", help="Type of model to train")
    parser.add_argument("--model_dtype", type=str, default="auto", help="Model torch_dtype")

    parser.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    parser.add_argument("--lora_alpha", type=int, default=64, help="Alpha parameter for LoRA scaling")
    parser.add_argument("--lora_dropout", type=float, default=0.04, help="Dropout probability for LoRA layers")
    parser.add_argument("--target_modules", nargs='+', type=str,  default=None,
                        help="The target modules for lora")

    parser.add_argument("--use_4bit", type=bool, default=False, help="Activate 4-bit precision base model loading")
    parser.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        help="Compute dtype for 4-bit base models")
    parser.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (fp4 or nf4)")
    parser.add_argument("--use_nested_quant", type=bool, default=True,
                        help="Activate nested quantization for 4-bit base models (double quantization)")
    parser.add_argument("--use_8bit", type=bool, default=False, help="Activate 8-bit precision base model loading")
    parser.add_argument("--better_transformer", type=bool, default=False, help="Enable flash attention")

    parser.add_argument("--Optim_name", type=str, default="PagedLion8bit", help="Name of optimizer in bnb lib")

    parser.add_argument("--dataset_name", type=str, default="Instruction_en-vn_mix", help="Dataset name")
    parser.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    parser.add_argument("--eval_batch_size", type=int, default=8, help="Evaluation batch size")
    parser.add_argument("--text_column", type=str, default="prompt", help="Text column")
    parser.add_argument("--label_column", type=str, default="target", help="Label column")
    parser.add_argument("--block_size", type=int, default=128, help="")

    parser.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--seed", type=int, default=43, help="Random seed")
    parser.add_argument("--do_test", type=bool, default=False, help="Flag to perform testing")
    parser.add_argument("--do_eval", type=bool, default=True, help="Flag to perform evaluation")
    parser.add_argument("--do_perplexity_eval", type=bool, default=True, help="Flag to enable perplexity computation, relevant when using casual-LM")
    parser.add_argument("--do_casual_eval", type=bool, default=True, help="Flag to enable Casual-LM inference eval")

    parser.add_argument("--gradient_checkpointing", type=bool, default=True, help="Use gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--enable_model_offload", type=bool, default=False, help="Enable model offload")
    parser.add_argument("--llm_int8_enable_fp32_cpu_offload", type=bool, default=False, help="")

    dataloader_group = parser.add_argument_group("Dataloader Arguments")
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
    args = parser.parse_args()

    # Sanity check
    if args.use_8bit and args.use_4bit:
        raise "Can't use 8bit and 4bit quantization at the same time"

    return args


if __name__=="__main__":
    args = parse_arguments()
    train(args)
