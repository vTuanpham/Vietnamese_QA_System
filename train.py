import argparse
import warnings
import json
from ast import literal_eval

from transformers import models

import deepspeed.module_inject as module_inject

from src.models.trainer import train


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here.")

    parser.add_argument("--model_name_or_path", type=str, default="google/umt5-small", help="Model name or path")
    parser.add_argument("--model_type", type=str, default="SEQ_2_SEQ_LM", help="Type of model to train")
    parser.add_argument("--model_dtype", type=str, default="auto", help="Model torch_dtype")
    parser.add_argument("--print_model_key", action='store_true', help="Whether to print out model structure")
    parser.add_argument("--shard_model", action="store_true", help="Sharded the model weight to fit on memory")
    parser.add_argument("--shard_model_merge", action="store_true", help="Shard the model to load and then merge weight to peft")
    parser.add_argument("--max_model_shard_size", type=str, default="1GB", help="Max size per model shard")

    peft_group = parser.add_argument_group("Parameters efficient arguments")
    peft_group.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    peft_group.add_argument("--lora_alpha", type=int, default=64, help="Alpha parameter for LoRA scaling")
    peft_group.add_argument("--lora_dropout", type=float, default=None, help="Dropout probability for LoRA layers")
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

    parser.add_argument("--merge_weight_eval", action='store_true', help="Flag to enable merge weight from peft for faster eval")
    parser.add_argument("--gradient_checkpointing", action='store_true', help="Use gradient checkpointing")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    parser.add_argument("--enable_model_offload", action='store_true', help="Enable model offload")
    parser.add_argument("--llm_int8_enable_fp32_cpu_offload", action='store_true', help="")

    dataloader_group = parser.add_argument_group("Dataloader Arguments")
    dataloader_group.add_argument("--dataset_name", type=str, default="Instruction_en-vn_mix", help="Dataset name")
    dataloader_group.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    dataloader_group.add_argument("--perplexity_eval_batch_size", type=int, default=8, help="Perplexity evaluation batch size")
    dataloader_group.add_argument("--generative_eval_batch_size", type=int, default=8, help="Generative evaluation batch size")
    dataloader_group.add_argument("--text_column", type=str, default="prompt", help="Text column")
    dataloader_group.add_argument("--label_column", type=str, default="target", help="Label column")
    dataloader_group.add_argument("--response_template", type=str, default=" %%%%%%% Response:\n", help="Response template prefix for DataCollatorForCompletionOnlyLM")
    dataloader_group.add_argument("--block_size", type=int, default=768, help="Block size for group text function")
    dataloader_group.add_argument("--do_group_texts", action="store_true", help="Do group text, great for pretraining phase")
    dataloader_group.add_argument("--model_max_length", type=int, default=1024, help="The model maximum length")
    dataloader_group.add_argument("--context_length", type=int, default=768, help="The model maximum context length")
    dataloader_group.add_argument("--train_file", nargs='+', type=str, default=[
        # r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json",
        r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json",
        r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleanedFormated.json",
        # r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleaned_translatedFormated.json"
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
    dataloader_group.add_argument("--do_perplexity_eval", action='store_true', help="Flag to enable perplexity computation, relevant when using casual-LM")
    dataloader_group.add_argument("--do_generative_eval", action="store_true", help="Flag to enable model.generate eval")

    generation_group = parser.add_argument_group("Generation Arguments")
    generation_group.add_argument("--top_k", type=int, default=5, help="Top-k value ")
    generation_group.add_argument("--top_p", type=float, default=0.7, help="Top-p value")
    generation_group.add_argument("--no_sample", action="store_true", help="Enable sampling (default: True)")
    generation_group.add_argument("--no_repeat_ngram_size", type=int, default=3, help="No repeat n-gram size (default: 3)")
    generation_group.add_argument("--num_beams", type=int, default=5, help="Number of beams (default: 5)")
    generation_group.add_argument("--no_early_stopping", action="store_true", help="Enable early stopping (default: True)")
    generation_group.add_argument("--max_time", type=int, default=None, help="Max time")
    generation_group.add_argument("--penalty_alpha", type=float, default=1.2, help="Penalty alpha (default: 1.2)")
    generation_group.add_argument("--repetition_penalty", type=float, default=2.5, help="Repetition penalty (default: 2.5)")
    generation_group.add_argument("--temperature", type=float, default=0.6, help="Temperature (default: 1.5)")
    generation_group.add_argument("--no_truncation", action="store_true", help="Enable truncation (default: True)")
    generation_group.add_argument("--encoder_repetition_penalty", type=float, default=2.0, help="Encoder repetition penalty (default: 2.0)")
    generation_group.add_argument("--max_length", type=int, default=1024, help="Max length (default: 1024)")
    generation_group.add_argument("--max_new_tokens", type=int, default=None, help="Max new tokens ")
    generation_group.add_argument("--deep_speed_inf", action="store_true", help="Enable deep speed inference")
    generation_group.add_argument("--use_default_gen_config", action="store_true", help="Use model's default gen config")
    generation_group.add_argument("--injection_policy", type=json.loads, default=None,
                                  help="Which layer module to add Tensor-parallelism and the name of linear layers"
                                       "(eg '''{t5.modeling_t5.T5Block: ('SelfAttention.o', 'EncDecAttention.o', 'DenseReluDense.wo')}'''"
                                       "")
    generation_group.add_argument("--auto_kernel_injection", action="store_true",
                                  help="Enable kernel injection for deepspeed inference")

    args = parser.parse_args()

    # Sanity check
    if args.use_8bit and args.use_4bit:
        raise "Can't use 8bit and 4bit quantization at the same time"

    if args.deep_speed_inf and args.num_beams > 1:
        raise "Deepspeed inference can't use num beams higher than 1"

    if args.deep_speed_inf and not args.merge_weight_eval:
        warnings.warn("Deepspeed inference is enable but merge weight eval is disable."
                      "This might slow down the generation a bit.")

    if not args.deep_speed_inf:
        warnings.warn("Deepspeed inference is disable, this might result in very slow inference")

    if args.injection_policy and not isinstance(args.injection_policy, dict):
        try:
            args.injection_policy = dict(literal_eval(args.injection_policy))
        except Exception as e:
            raise f"Invalid injection policy, please pass your args in the form of dict: {e}"

    if args.injection_policy:
        # The injection policy require the key to be a module class (Very annoying)
        for idx, module in enumerate(list(args.injection_policy.keys())[0].split(".")):
            if idx == 0: fetch_module = models
            fetch_module = getattr(fetch_module, module)

        try:
            # The injection policy value can be in the deepspeed module_inject module
            for idx, module in enumerate(list(args.injection_policy.values())[0].split(".")):
                if idx == 0: fetch_policy = module_inject
                fetch_policy = getattr(fetch_policy, module)
        except Exception as e:
            warnings.warn("Invalid injection policy for deepspeed, assuming the values is user specified"
                          f"Error message: {e}")
            fetch_policy = tuple(list(args.injection_policy.values())[0])
            pass

        args.injection_policy = {fetch_module: fetch_policy}
        print(f"Input injection policy: {args.injection_policy}")

    if args.injection_policy and args.auto_kernel_injection:
        warnings.warn("Cannot use both user specified injection policy and kernel injection"
                      "Setting injection_policy to None")
        args.injection_policy = None

    if not args.auto_kernel_injection and args.injection_policy and args.deep_speed_inf:
        warnings.warn(f"Be sure to use auto_kernel_injection first to check"
                      f" if {args.model_name_or_path} support kernel injection,"
                      f"the list of supported models are here:"
                      f" https://github.com/microsoft/DeepSpeed/blob/4ae3a3da0dfd19d7ab7a76e7c742ac12f44fc1c0/docs/_tutorials/automatic-tensor-parallelism.md")

    return args


if __name__=="__main__":
    args = parse_arguments()
    train(args)
