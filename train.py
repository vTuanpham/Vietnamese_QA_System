import argparse
import warnings
import json
from ast import literal_eval

from transformers import models, SchedulerType

import deepspeed.module_inject as module_inject

from src.data import QADataloader
from src.data.configs import AdvanceInstructSample
from src.models.trainer import train


def parse_arguments():
    parser = argparse.ArgumentParser(description="Your script description here.")

    model_group = parser.add_argument_group("Model configs and loading args")
    model_group.add_argument("--model_name_or_path", type=str, default="google/umt5-small", help="Model name or path")
    model_group.add_argument("--model_type", type=str, default="SEQ_2_SEQ_LM", help="Type of model to train")
    model_group.add_argument("--model_dtype", type=str, default="auto", help="Model torch_dtype")
    model_group.add_argument("--print_model_key", action='store_true', help="Whether to print out model structure")
    model_group.add_argument("--shard_model", action="store_true", help="Sharded the model weight to fit on memory")
    model_group.add_argument("--shard_model_merge", action="store_true", help="Shard the model to load and then merge weight to peft")
    model_group.add_argument("--max_model_shard_size", type=str, default="1GB", help="Max size per model shard")
    model_group.add_argument("--use_flash_attention_2", action="store_true", help="Enable flash attention 2 or not"
                                                                             "More info: https://huggingface.co/docs/transformers/perf_infer_gpu_one")
    model_group.add_argument("--better_transformer", action='store_true', help="Enable flash attention")
    model_group.add_argument("--no_split_module_classes", nargs='+', type=str, default=None,
                             help="A list of layer class names that should never be split across device "
                                  "(for instance any layer that has a residual connection).")

    peft_group = parser.add_argument_group("Parameters efficient arguments")
    peft_group.add_argument("--lora_r", type=int, default=8, help="LoRA attention dimension")
    peft_group.add_argument("--lora_alpha", type=int, default=64, help="Alpha parameter for LoRA scaling")
    peft_group.add_argument("--lora_dropout", type=float, default=None, help="Dropout probability for LoRA layers")
    peft_group.add_argument("--target_modules", nargs='+', type=str,  default=None,
                            help="The target modules for lora")
    peft_group.add_argument("--lora_bias", type=str, default="none", help="Bias type for Lora")
    peft_group.add_argument("--modules_to_save", nargs='+', type=str, default=None, help="List of modules apart from "
                                                                                         "LoRA layers to be set as trainable and saved in the final checkpoint.")

    bitsandbytes_group = parser.add_argument_group("Bitsandbytes arguments")
    bitsandbytes_group.add_argument("--use_4bit", action='store_true', help="Activate 4-bit precision base model loading")
    bitsandbytes_group.add_argument("--bnb_4bit_compute_dtype", type=str, default="bfloat16",
                        help="Compute dtype for 4-bit base models")
    bitsandbytes_group.add_argument("--bnb_4bit_quant_type", type=str, default="nf4", help="Quantization type (fp4 or nf4)")
    bitsandbytes_group.add_argument("--use_nested_quant", action='store_true',
                        help="Activate nested quantization for 4-bit base models (double quantization)")
    bitsandbytes_group.add_argument("--use_8bit", action='store_true', help="Activate 8-bit precision base model loading")

    training_group = parser.add_argument_group("Training configs group")
    training_group.add_argument("--log_weights_cpkt", action="store_true", help="Whether to log checkpoint to wandb")
    training_group.add_argument("--with_tracking", action="store_true", help="Whether to enable experiment trackers for logging.")
    training_group.add_argument("--output_dir", type=str, default=None, help="Where to store the final model.")
    training_group.add_argument(
        "--report_to",
        type=str,
        default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"`, `"comet_ml"` and `"clearml"`. Use `"all"` (default) to report to all integrations. '
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    training_group.add_argument("--optim_name", type=str, default="PagedLion8bit", help="Name of optimizer in bnb lib")
    training_group.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay")
    training_group.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    training_group.add_argument("--lr_sheduler_name", type=SchedulerType, default="linear",
                                choices=["linear", "cosine", "cosine_with_restarts", "polynomial", "constant", "constant_with_warmup"],
                                help="The scheduler type to use.")
    training_group.add_argument("--warmup_steps", type=int, default=20, help="Num warmup steps")
    training_group.add_argument("--num_epochs", type=int, default=5, help="Number of epochs")
    training_group.add_argument("--seed", type=int, default=43, help="Random seed")
    training_group.add_argument("--do_test", action='store_true', help="Flag to perform testing")
    training_group.add_argument("--do_eval", action='store_true', help="Flag to perform evaluation")

    training_group.add_argument("--merge_weight_eval", action='store_true', help="Flag to enable merge weight from peft for faster eval")
    training_group.add_argument("--gradient_checkpointing", action='store_true', help="Use gradient checkpointing")
    training_group.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Gradient accumulation steps")
    training_group.add_argument("--enable_model_offload", action='store_true', help="Enable model offload")
    training_group.add_argument("--minimum_free_spaces", type=int, default=1, help="Minimum free spaces to keep in GB")
    training_group.add_argument("--llm_int8_enable_fp32_cpu_offload", action='store_true', help="")
    training_group.add_argument("--resume_from_checkpoint", type=str,
                                default=None, help="If the training should continue from a checkpoint folder.")
    training_group.add_argument("--checkpointing_steps", type=str, default=None,
                                help="How often should we save state in steps for resume")
    training_group.add_argument("--checkpoint_at_max_time", type=float, default=None,
                                help="How often in hours should we save state in hours for resume")
    training_group.add_argument('--override_last_cpkt_step', action='store_true',
                                help="Override last cpkt step and epoch")
    training_group.add_argument('--convert_cpkt', action='store_true', help='Convert checkpoint into model and push to'
                                                                            'hub')

    dataloader_group = parser.add_argument_group("Dataloader Arguments")
    dataloader_group.add_argument("--dataset_name", type=str, default="Instruction_en-vn_mix", help="Dataset name")
    dataloader_group.add_argument("--train_batch_size", type=int, default=4, help="Training batch size")
    dataloader_group.add_argument("--perplexity_eval_batch_size", type=int, default=8, help="Perplexity evaluation batch size")
    dataloader_group.add_argument("--generative_eval_batch_size", type=int, default=8, help="Generative evaluation batch size")
    dataloader_group.add_argument("--text_column", type=str, default="prompt", help="Text column")
    dataloader_group.add_argument("--label_column", type=str, default="target", help="Label column")
    dataloader_group.add_argument("--response_template", type=str, default=" %%%%%%% Response:", help="Response template prefix for DataCollatorForCompletionOnlyLM")
    dataloader_group.add_argument("--add_tokens_list", nargs='+', type=str, default=None, help="List of special tokens to add to the tokenizer")
    dataloader_group.add_argument("--block_size", type=int, default=768, help="Block size for group text function")
    dataloader_group.add_argument("--do_group_texts", action="store_true", help="Do group text, great for pretraining phase")
    dataloader_group.add_argument("--model_max_length", type=int, default=1024, help="The model maximum length")
    dataloader_group.add_argument("--context_length", type=int, default=768, help="The model maximum context length")
    dataloader_group.add_argument("--train_file", nargs='+', type=str, default=None, help="List of training files")
    dataloader_group.add_argument("--each_train_file_percentage", nargs='+', type=int, default=None,
                                  help="The percentage weight of each train files")

    dataloader_group.add_argument("--val_file", nargs='+', type=str, default=None, help="List of validation files")

    dataloader_group.add_argument("--test_file", nargs='+', type=str, default=None, help="List of test files")

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
    dataloader_group.add_argument("--max_eval_generative_samples", type=int, default=50, help="Max generative examplew for manual evaluation")
    dataloader_group.add_argument("--do_generative_eval", action="store_true", help="Flag to enable model.generate eval")
    dataloader_group.add_argument("--max_eval_perplexity_samples", type=int, default=50, help="Max evaluation examples for perplexity evaluation")

    generation_group = parser.add_argument_group("Generation Arguments")
    generation_group.add_argument("--top_k", type=int, default=50, help="Top-k value ")
    generation_group.add_argument("--top_p", type=float, default=1.0, help="Top-p value")
    generation_group.add_argument("--do_sample", action="store_true", help="Enable sampling (default: True)")
    generation_group.add_argument("--no_repeat_ngram_size", type=int, default=0, help="No repeat n-gram size (default: 3)")
    generation_group.add_argument("--num_beams", type=int, default=1, help="Number of beams (default: 5)")
    generation_group.add_argument("--early_stopping", action="store_true", help="Enable early stopping (default: True)")
    generation_group.add_argument("--max_time", type=int, default=None, help="Max time")
    generation_group.add_argument("--penalty_alpha", type=float, default=None, help="Penalty alpha (default: 1.2)")
    generation_group.add_argument("--repetition_penalty", type=float, default=1.0, help="Repetition penalty (default: 2.5)")
    generation_group.add_argument("--temperature", type=float, default=1.0, help="Temperature (default: 1.5)")
    generation_group.add_argument("--no_truncation", action="store_true", help="Enable truncation (default: True)")
    generation_group.add_argument("--encoder_repetition_penalty", type=float, default=1.0, help="Encoder repetition penalty (default: 2.0)")
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
    if args.each_train_file_percentage:
        assert len(args.each_train_file_percentage) == len(args.train_file), "The each_train_file_percentage length must be " \
                                                                             "equal to the numbers of files in the train_file"
        assert sum(args.each_train_file_percentage) == 100, "The each_train_file_percentage arguments must be a list of int" \
                                                            "that add up to 100%"

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

    dataloader_args = {
        "model_name": args.model_name_or_path,
        "text_column": args.text_column,
        "target_column": args.label_column,
        "train_file": args.train_file,
        "each_train_file_percentage": args.each_train_file_percentage,
        "val_file": args.val_file,
        "test_file": args.test_file,
        "train_batch_size": args.train_batch_size,
        "perplexity_eval_batch_size": args.perplexity_eval_batch_size,
        "generative_eval_batch_size": args.generative_eval_batch_size,
        "seed": args.seed,
        "max_train_samples": args.max_train_samples,
        "max_eval_samples": args.max_eval_samples,
        "max_predict_samples": args.max_predict_samples,
        "config_type": AdvanceInstructSample,
        "task_type": args.model_type,
        "block_size": args.block_size,
        "no_preprocess_data": args.no_preprocess_data,
        "do_group_texts": args.do_group_texts,
        "do_perplexity_eval": args.do_perplexity_eval,
        "do_generative_eval": args.do_generative_eval,
        "model_max_length": args.model_max_length,
        "context_length": args.context_length,
        "response_template": args.response_template,
        "add_tokens_list": args.add_tokens_list,
        "max_eval_generative_samples": args.max_eval_generative_samples,
        "max_eval_perplexity_samples": args.max_eval_perplexity_samples
    }

    qa_dataloader = QADataloader(**dataloader_args)
    qa_dataloader_instance = qa_dataloader.__call__()

    train(args, qa_dataloader, qa_dataloader_instance)
