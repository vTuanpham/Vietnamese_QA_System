import datetime
import gc
import math
import os
import random
import shutil
import logging
from copy import deepcopy
import warnings

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import sys
sys.path.insert(0,r'./')
import psutil
import threading

import numpy as np

import torch
try:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
except Exception:
    raise "Please update your pytorch, this script require a version higher than 1.7 with cuda"
import deepspeed
import torch.nn as nn

from accelerate import Accelerator, infer_auto_device_map, dispatch_model
from accelerate.logging import get_logger
from accelerate.utils.memory import find_executable_batch_size
from accelerate.utils import DistributedType
from accelerate.state import AcceleratorState

from tqdm.auto import tqdm
import datasets, transformers
from transformers import \
    (AutoModelForCausalLM,
     AutoModelForSeq2SeqLM,
     get_scheduler,
     set_seed,
     BitsAndBytesConfig,
     GenerationConfig,
     AutoConfig,
     pipeline)
from transformers.trainer_pt_utils import get_parameter_names

import bitsandbytes as bnb
from peft import LoraConfig, TaskType, get_peft_model, PeftConfig, PeftModel, prepare_model_for_kbit_training

from src.data import QADataloader
from src.models.model_utils import poor_man_llm_load
from src.data.configs import AdvanceInstructSample, AdvanceQAExample


logger = get_logger(__name__)


# Converting Bytes to Megabytes
def b2mb(x):
    return int(x / 2**20)


def merge_adapter(base_model_name: str, peft_adapter: PeftModel,
                  adapter_save_path: str, adapter_name: str, main_process: bool,
                  model_type: str="CAUSAL_LM", model_dtype=None, better_transformer: bool=False,
                  shard_model: bool=False, max_memory: dict={0: "0.3GB"}, max_shard_size: str="500MB"):

    peft_adapter.save_pretrained(adapter_save_path,
                                 save_adapter=True,
                                 is_main_process=main_process)
    adapter_path_file = os.path.join(adapter_save_path, adapter_name)

    offload_config = {
        # "device_map": "auto",
        # "offload_folder": "offload_inf",
        "torch_dtype": model_dtype,
        "use_cache": True,
        "offload_state_dict": True,
        "low_cpu_mem_usage": True,
        "trust_remote_code":True,
        "max_memory": max_memory
    }

    if model_type == "CAUSAL_LM":
        if not shard_model:
            base_model = AutoModelForCausalLM.from_pretrained(base_model_name,
                                                              **offload_config
                                                              )
        else:
            base_model = poor_man_llm_load(base_model_name, model_type=model_type,
                                           model_dtype=model_dtype, max_shard_size=max_shard_size,
                                           additional_kwargs=offload_config)
    elif model_type == "SEQ_2_SEQ_LM":
        if not shard_model:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name,
                                                               **offload_config
                                                               )
        else:
            base_model = poor_man_llm_load(base_model_name, model_type=model_type,
                                           model_dtype=model_dtype, max_shard_size=max_shard_size,
                                           additional_kwargs=offload_config)

    if getattr(base_model, "quantization_method", None) == "gptq":
        warnings.warn(f"The model {base_model_name} is gptq quantized and cannot be merged to LORA layers.\n"
                      f"Returning the original adapter...")
        del base_model
        gc.collect()
        return peft_adapter

    if better_transformer:
        try:
            base_model = base_model.to_bettertransformer()
        except Exception as e:
            warnings.warn(f"This model type {base_model_name} is not yet "
                          f"support for BetterTransformer, please change model type if "
                          f"you still want to use it.\n Continue running without it...")
            warnings.warn(f"Error message: {e}")
            pass

    model_to_merge = PeftModel.from_pretrained(base_model,
                                               adapter_path_file,
                                               device_map="auto",
                                               )
    merged_model = model_to_merge.merge_and_unload(progressbar=True)
    del base_model, peft_adapter, model_to_merge
    gc.collect()

    return merged_model


# This context manager is used to track the peak memory usage of the process
class TorchTracemalloc:
    def __enter__(self):
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()  # reset the peak gauge to zero
        self.begin = torch.cuda.memory_allocated()
        self.process = psutil.Process()

        self.cpu_begin = self.cpu_mem_used()
        self.peak_monitoring = True
        peak_monitor_thread = threading.Thread(target=self.peak_monitor_func)
        peak_monitor_thread.daemon = True
        peak_monitor_thread.start()
        return self

    def cpu_mem_used(self):
        """get resident set size memory for the current process"""
        return self.process.memory_info().rss

    def peak_monitor_func(self):
        self.cpu_peak = -1

        while True:
            self.cpu_peak = max(self.cpu_mem_used(), self.cpu_peak)

            # can't sleep or will not catch the peak right (this comment is here on purpose)
            # time.sleep(0.001) # 1msec

            if not self.peak_monitoring:
                break

    def __exit__(self, *exc):
        self.peak_monitoring = False

        gc.collect()
        torch.cuda.empty_cache()
        self.end = torch.cuda.memory_allocated()
        self.peak = torch.cuda.max_memory_allocated()
        self.used = b2mb(self.end - self.begin)
        self.peaked = b2mb(self.peak - self.begin)

        self.cpu_end = self.cpu_mem_used()
        self.cpu_used = b2mb(self.cpu_end - self.cpu_begin)
        self.cpu_peaked = b2mb(self.cpu_peak - self.cpu_begin)


def train(training_args):
    accelerator = Accelerator(gradient_accumulation_steps=training_args.gradient_accumulation_steps,
                              project_dir="./")
    accelerator.print(f"{AcceleratorState()}")

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)
    if accelerator.is_local_main_process:
        datasets.utils.logging.set_verbosity_warning()
        transformers.utils.logging.set_verbosity_info()
    else:
        datasets.utils.logging.set_verbosity_error()
        transformers.utils.logging.set_verbosity_error()

    # LoRA attention dimension
    lora_r = training_args.lora_r
    # Alpha parameter for LoRA scaling
    lora_alpha = training_args.lora_alpha
    # Dropout probability for LoRA layers
    lora_dropout = training_args.lora_dropout

    # Activate 4-bit precision base model loading
    use_4bit = training_args.use_4bit
    # Compute dtype for 4-bit base models
    bnb_4bit_compute_dtype = training_args.bnb_4bit_compute_dtype
    # Quantization type (fp4 or nf4)
    bnb_4bit_quant_type = training_args.bnb_4bit_quant_type
    # Activate nested quantization for 4-bit base models (double quantization)
    use_nested_quant = training_args.use_nested_quant

    use_8bit = training_args.use_8bit

    model_name_or_path = training_args.model_name_or_path
    gradient_accumulation_steps = training_args.gradient_accumulation_steps
    lr_sheduler_name = training_args.lr_sheduler_name
    dataset_name = training_args.dataset_name
    train_batch_size = training_args.train_batch_size
    perplexity_eval_batch_size = training_args.perplexity_eval_batch_size
    generative_eval_batch_size = training_args.generative_eval_batch_size
    text_column = training_args.text_column
    label_column = training_args.label_column
    lr = training_args.lr
    num_epochs = training_args.num_epochs
    seed = training_args.seed
    do_test = training_args.do_test
    do_eval = training_args.do_eval
    gradient_checkpointing = training_args.gradient_checkpointing
    weight_decay = training_args.weight_decay
    target_modules = training_args.target_modules
    task_type = training_args.model_type
    block_size = training_args.block_size
    do_group_texts = training_args.do_group_texts
    model_max_length = training_args.model_max_length
    context_length = training_args.context_length
    better_transformer = training_args.better_transformer
    model_offload = training_args.enable_model_offload
    llm_int8_cpu_offload = training_args.llm_int8_enable_fp32_cpu_offload
    optim_name = training_args.optim_name
    model_dtype = training_args.model_dtype
    perplexity_eval = training_args.do_perplexity_eval
    generative_eval = training_args.do_generative_eval
    merge_weight_eval = training_args.merge_weight_eval
    print_model_key = training_args.print_model_key

    top_k = training_args.top_k
    do_sample = training_args.do_sample
    no_repeat_ngram_size = training_args.no_repeat_ngram_size
    num_beams = training_args.num_beams
    early_stopping = training_args.early_stopping
    max_time = training_args.max_time
    penalty_alpha = training_args.penalty_alpha
    repetition_penalty = training_args.repetition_penalty
    temperature = training_args.temperature
    no_truncation = training_args.no_truncation
    encoder_repetition_penalty = training_args.encoder_repetition_penalty
    max_length = training_args.max_length
    no_preprocess_data = training_args.no_preprocess_data
    max_new_tokens = training_args.max_new_tokens
    shard_model = training_args.shard_model
    max_model_shard_size = training_args.max_model_shard_size
    deep_speed_inf = training_args.deep_speed_inf
    top_p = training_args.top_p
    injection_policy = training_args.injection_policy
    auto_kernel_injection = training_args.auto_kernel_injection
    use_default_gen_config = training_args.use_default_gen_config
    shard_model_merge = training_args.shard_model_merge
    response_template = training_args.response_template
    minimum_free_spaces = training_args.minimum_free_spaces
    use_flash_attention_2 = training_args.use_flash_attention_2
    max_eval_generative_samples = training_args.max_eval_generative_samples
    max_eval_perplexity_samples = training_args.max_eval_perplexity_samples
    lora_bias = training_args.lora_bias
    modules_to_save = training_args.modules_to_save
    warmup_steps = training_args.warmup_steps
    no_split_module_classes = training_args.no_split_module_classes

    set_seed(seed)

    compute_dtype = getattr(torch, bnb_4bit_compute_dtype)
    if model_dtype != "auto":
        model_dtype = getattr(torch, model_dtype)
    task_type = getattr(TaskType, task_type)

    dataloader_args = {
        "accelerator": accelerator,
        "model_name": model_name_or_path,
        "text_column": text_column,
        "target_column": label_column,
        "train_file": training_args.train_file,
        "each_train_file_percentage": training_args.each_train_file_percentage,
        "val_file": training_args.val_file,
        "test_file": training_args.test_file,
        "train_batch_size": train_batch_size,
        "perplexity_eval_batch_size": perplexity_eval_batch_size,
        "generative_eval_batch_size": generative_eval_batch_size,
        "seed": seed,
        "max_train_samples": training_args.max_train_samples,
        "max_eval_samples": training_args.max_eval_samples,
        "max_predict_samples": training_args.max_predict_samples,
        "config_type": AdvanceInstructSample,
        "task_type": task_type,
        "block_size": block_size,
        "no_preprocess_data": no_preprocess_data,
        "do_group_texts": do_group_texts,
        "do_perplexity_eval": perplexity_eval,
        "do_generative_eval": generative_eval,
        "model_max_length": model_max_length,
        "context_length": context_length,
        "response_template": response_template,
        "max_eval_generative_samples": max_eval_generative_samples,
        "max_eval_perplexity_samples": max_eval_perplexity_samples
    }

    # Check GPU compatibility with bfloat16
    if compute_dtype == torch.float16 and use_4bit:
        major, _ = torch.cuda.get_device_capability()
        if major >= 8:
            accelerator.print("=" * 80)
            accelerator.print("Your GPU supports bfloat16: accelerate training with bf16=True")
            accelerator.print("=" * 80)

    if use_4bit:
        quant_config = BitsAndBytesConfig(
            load_in_4bit=use_4bit,
            bnb_4bit_use_double_quant=use_nested_quant,
            bnb_4bit_compute_type=compute_dtype,
            bnb_4bit_quant_type=bnb_4bit_quant_type,
            llm_int8_enable_fp32_cpu_offload=llm_int8_cpu_offload,
            llm_int8_threshold=6.0,
        )
    elif use_8bit:
        quant_config = BitsAndBytesConfig(
            load_in_8bit=use_8bit,
            llm_int8_enable_fp32_cpu_offload=llm_int8_cpu_offload,
            llm_int8_threshold=6.0,
        )
    else:
        quant_config = None
        warnings.warn("\n   No quantization is applied")

    peft_config = LoraConfig(
        task_type=task_type,
        inference_mode=False,
        r=lora_r, lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias=lora_bias,
        modules_to_save=modules_to_save
    )

    if not use_default_gen_config:
        try:
            generation_config, unused_config = GenerationConfig.from_pretrained(
                model_name_or_path, top_k=top_k, do_sample=do_sample, return_unused_kwargs=True,
                no_repeat_ngram_size=no_repeat_ngram_size, num_beams=num_beams, early_stopping=early_stopping,
                max_time=max_time, penalty_alpha=penalty_alpha, repetition_penalty=repetition_penalty, temperature=temperature,
                truncation=not no_truncation, encoder_repetition_penalty=encoder_repetition_penalty, max_length=max_length,
                max_new_tokens=max_new_tokens, top_p=top_p, use_cache=True, low_memory=True
            )
            if len(unused_config) > 0: accelerator.print(f"Unused config: {unused_config}")
        except Exception as e:
            warnings.warn(f"The model {model_name_or_path} does not have a generation config")
            warnings.warn(f"Error message: {e}")
            generation_config = GenerationConfig.from_dict(config_dict={
                "top_k": top_k, "do_sample": do_sample, "no_repeat_ngram_size": no_repeat_ngram_size, "num_beams": num_beams, "early_stopping": early_stopping,
                "max_time": max_time, "penalty_alpha": penalty_alpha, "repetition_penalty": repetition_penalty,  "max_new_tokens": max_new_tokens,
                "temperature": temperature, "encoder_repetition_penalty": encoder_repetition_penalty,
                "max_length": max_length, "truncation": not no_truncation, "top_p": top_p, "use_cache": True, "low_memory": True
            })
    else:
        generation_config = GenerationConfig.from_dict(config_dict={"min_new_tokens": 20,
                                                                    "max_length": context_length,
                                                                    "max_time": max_time})
    accelerator.print(f"Model generation config: {generation_config}")

    qa_dataloader = QADataloader(**dataloader_args)
    qa_dataloader_instance = qa_dataloader.__call__()

    tokenizer = qa_dataloader.tokenizer

    accelerator.print(" Print out a couple samples for tokenizer compatibility check for multilingual task")
    for idx, data in enumerate(iter(qa_dataloader_instance['test']['perplexity_eval'])):
        accelerator.print("\n==============================================================================\n")
        accelerator.print("\n Input: "+qa_dataloader.tokenizer.decode(data['input_ids'][0], skip_special_tokens=True))
        labels = data['labels'].cpu().numpy()
        labels = np.where(labels != -100, labels, qa_dataloader.tokenizer.pad_token_id)
        accelerator.print("\n Response:"+qa_dataloader.tokenizer.decode(labels[0], skip_special_tokens=True))
        accelerator.print("\n==============================================================================\n")
        if idx == 10: break

    try:
        config = AutoConfig.from_pretrained(
            model_name_or_path,
        )
    except Exception:
        warnings.warn(f"Model {model_name_or_path} does not have a config.json")
        config = None

    # System setup info
    free_in_GB = int(torch.cuda.mem_get_info()[0] / 1024 ** 3)
    max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - minimum_free_spaces}GB'
    n_gpus = torch.cuda.device_count()
    max_memory = {i: max_memory for i in range(n_gpus)}

    accelerator.print(f"System max memory: {max_memory}\n"
                      f"System num gpus: {n_gpus}\n"
                      f"System free in GB: {free_in_GB}")

    offload_config = {
        "device_map": "auto",
        "offload_folder": "offload",
        "offload_state_dict": True,
        "low_cpu_mem_usage": True,
        "max_memory": max_memory
    } if model_offload else {}

    full_model_config = {
        "quantization_config": quant_config,
        "trust_remote_code": True,
        "load_in_8bit": use_8bit,
        "load_in_4bit": use_4bit,
        "torch_dtype": model_dtype,
        "config": config,
    }

    if use_flash_attention_2: full_model_config["use_flash_attention_2"] = True

    if "gpt2" in model_name_or_path:
        full_model_config["scale_attn_by_inverse_layer_idx"] = True
        full_model_config["reorder_and_upcast_attn"] = True

    if model_offload: full_model_config = {**full_model_config, **offload_config}

    # creating model
    if task_type == "CAUSAL_LM":
        if not shard_model:
            base_model = AutoModelForCausalLM.from_pretrained(model_name_or_path,
                                                              **full_model_config)
        else:
            base_model = poor_man_llm_load(model_name_or_path, model_type=task_type,
                                           model_dtype=model_dtype, max_shard_size=max_model_shard_size,
                                           additional_kwargs=full_model_config)
    elif task_type == "SEQ_2_SEQ_LM":
        if not shard_model:
            base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path,
                                                              **full_model_config)
        else:
            base_model = poor_man_llm_load(model_name_or_path, model_type=task_type,
                                           model_dtype=model_dtype, max_shard_size=max_model_shard_size,
                                           additional_kwargs=full_model_config)
    accelerator.print(f"\n  Base model memory footprint: {base_model.get_memory_footprint()}\n")

    device_map = infer_auto_device_map(
        base_model,
        max_memory=max_memory,
        no_split_module_classes=no_split_module_classes,
        dtype=model_dtype
    )

    base_model = dispatch_model(base_model, device_map=device_map)

    # We resize the embeddings only when necessary to avoid index errors. If you are creating a model from scratch
    # on a small vocab and want a smaller embedding size, remove this test.
    embedding_size = base_model.get_input_embeddings().weight.shape[0]
    accelerator.print(f"Model embedding size: {embedding_size}")
    accelerator.print(f"Tokenizer vocab size: {len(tokenizer)}")
    if len(tokenizer) > embedding_size:
        base_model.resize_token_embeddings(len(tokenizer))

    if better_transformer:
        try:
            base_model = base_model.to_bettertransformer()
        except Exception as e:
            warnings.warn(f"This model type {model_name_or_path} is not yet "
                          f"support for BetterTransformer, please change model type if "
                          f"you still want to use it.\n Continue running without it...")
            warnings.warn(f"Error message: {e}")
            better_transformer = False
            pass

    # Please enable gradient_checkpointing at all cost, this will save your life
    if use_4bit or use_8bit or getattr(base_model, "quantization_method", None) == "gptq":
        accelerator.print(f"Preparation for kbit training...")
        base_model = prepare_model_for_kbit_training(base_model,
                                                     use_gradient_checkpointing=gradient_checkpointing) # Prepare model in peft already include gradient-checkpoint, freeze params
    elif gradient_checkpointing:
        base_model.gradient_checkpointing_enable()
    else:
        warnings.warn("You disable gradient checkpoint, this will result in vram consumtion")

    base_model.config.use_cache = False

    if print_model_key:
        accelerator.print(base_model)

    # TODO: For cast weights to fp32
    if modules_to_save and use_8bit or use_4bit:
        pass

    # model = torch.compile(model, mode="max-autotune")
    adapter = get_peft_model(base_model, peft_config=peft_config, adapter_name=dataset_name)
    if gradient_checkpointing: adapter.gradient_checkpointing_enable() # Double check!
    adapter.print_trainable_parameters()

    if print_model_key:
        accelerator.print(adapter)

    # optimizer
    decay_parameters = get_parameter_names(adapter, [nn.LayerNorm])
    decay_parameters = [name for name in decay_parameters if "bias" not in name]
    optimizer_grouped_parameters = [
        {
            "params": [p for n, p in adapter.named_parameters() if n in decay_parameters],
            "weight_decay": weight_decay,
        },
        {
            "params": [p for n, p in adapter.named_parameters() if n not in decay_parameters],
            "weight_decay": 0.0,
        },
    ]

    # bnb.optim.PagedAdamW8bit
    optimizer = getattr(bnb.optim, optim_name)(optimizer_grouped_parameters, lr=lr)
    accelerator.print(f"\nLoading {optim_name} from bits and bytes...")

    num_update_steps_per_epoch = math.ceil(len(qa_dataloader_instance['train']) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    # lr scheduler
    lr_scheduler = get_scheduler(
        lr_sheduler_name,
        optimizer=optimizer,
        num_warmup_steps=warmup_steps * gradient_accumulation_steps,
        num_training_steps=max_train_steps * gradient_accumulation_steps,
    )

    adapter, train_dataloader, optimizer, lr_scheduler = accelerator.prepare(
        adapter, qa_dataloader_instance['train'], optimizer, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(qa_dataloader_instance['train']) / gradient_accumulation_steps)
    max_train_steps = num_epochs * num_update_steps_per_epoch
    # Afterwards we recalculate our number of training epochs
    num_epochs = math.ceil(max_train_steps / num_update_steps_per_epoch)

    # On TPU, the tie weights in our model have been disconnected, so we need to restore the ties.
    if accelerator.distributed_type == DistributedType.TPU:
        adapter.tie_weights()

    if do_eval:
        if perplexity_eval:
            perplexity_eval_dataloader = accelerator.prepare(qa_dataloader_instance['eval']['perplexity_eval'])
        if generative_eval:
            generative_eval_dataloader = accelerator.prepare(qa_dataloader_instance['eval']['generative_eval'])
    else:
        logger.info("\nEvaluation turn off for this session")

    if do_test:
        test_dataloader = accelerator.prepare(qa_dataloader_instance['test'])
    else:
        logger.info("\nTest turn off for this session")

    if training_args.checkpointing_steps:
        # Register the LR scheduler
        accelerator.register_for_checkpointing(lr_scheduler)

        # Save the starting state
        accelerator.save_state()

    for epoch in tqdm(range(num_epochs), desc="Training progress"):
        with TorchTracemalloc() as tracemalloc:
            adapter.train()
            total_loss = 0
            for step, batch in enumerate(tqdm(train_dataloader, desc=f"Training progress epoch {epoch}")):
                with accelerator.accumulate(adapter):
                    outputs = adapter(**batch)
                    loss = outputs.loss
                    total_loss += loss.detach().float()
                    accelerator.backward(loss)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # Checks if the accelerator has performed an optimization step behind the scenes
                if accelerator.sync_gradients:
                    accelerator.print(total_loss / step)
                    del loss, outputs, batch

                # if isinstance(checkpointing_steps, int):
                #     if completed_steps % checkpointing_steps == 0:
                #         output_dir = f"step_{completed_steps}"
                #         if args.output_dir is not None:
                #             output_dir = os.path.join(args.output_dir, output_dir)
                #         accelerator.save_state(output_dir)
                # if completed_steps >= args.max_train_steps:
                #     break

        # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
        accelerator.print("GPU Memory before entering the train : {}".format(b2mb(tracemalloc.begin)))
        accelerator.print("GPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.used))
        accelerator.print("GPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.peaked))
        accelerator.print(
            "GPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.peaked + b2mb(tracemalloc.begin)
            )
        )

        accelerator.print("CPU Memory before entering the train : {}".format(b2mb(tracemalloc.cpu_begin)))
        accelerator.print("CPU Memory consumed at the end of the train (end-begin): {}".format(tracemalloc.cpu_used))
        accelerator.print("CPU Peak Memory consumed during the train (max-begin): {}".format(tracemalloc.cpu_peaked))
        accelerator.print(
            "CPU Total Peak Memory consumed during the train (max): {}".format(
                tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
            )
        )
        train_epoch_loss = total_loss / len(train_dataloader)
        train_ppl = torch.exp(train_epoch_loss)
        accelerator.print(f"{epoch=}: {train_ppl=} {train_epoch_loss=}")

        # TODO: Refactor evaluation
        if do_eval:
            cur_time = '_'.join(str(datetime.datetime.now().strftime('%Y-%m-%d %H-%M-%S')).split())
            if merge_weight_eval:
                if not getattr(base_model, "quantization_method", None) == "gptq":
                    accelerator.print(f"Merging model for faster inference...")
                    inference_model = merge_adapter(model_name_or_path,
                                                    peft_adapter=adapter,
                                                    adapter_save_path=f"src/models/adapters/{dataset_name}-e{epoch}-{cur_time}",
                                                    main_process=accelerator.is_main_process, adapter_name=dataset_name,
                                                    model_type=task_type,
                                                    model_dtype=model_dtype,
                                                    better_transformer=better_transformer,
                                                    shard_model=shard_model_merge,
                                                    max_memory={0: "0.3GB"},
                                                    max_shard_size=max_model_shard_size)
                else:
                    warnings.warn(
                        f"The model {model_name_or_path} is gptq quantized and cannot be merged to LORA layers.\n"
                        f"Skipping merge_weight_eval...")
            else:
                warnings.warn(f"Weight from peft not merged yet, this may result in slower inference")
                # deepcopy since using torch.autocast which will auto cast if there is a mismatch between lora adatper
                # and the base model dtype, deepcopy ensure that any modified inference_model dtype will not affect
                # the base model and adapter dtype
                inference_model = deepcopy(adapter)

            if deep_speed_inf:
                world_size = int(os.getenv('WORLD_SIZE', str(torch.cuda.device_count())))
                os.environ["RANK"] = "0"
                os.environ["LOCAL_RANK"] = "0"
                os.environ["WORLD_SIZE"] = str(torch.cuda.device_count())

                # The injection_policy shows two things:
                #   1. which layer module we need to add Tensor-Parallelism
                #   2. the name of several linear layers: a) attention_output (both encoder and decoder),
                #       and b) transformer output
                accelerator.print(f"Model type for inference: {type(inference_model)}")
                injection_config = {
                    "replace_with_kernel_inject": auto_kernel_injection,
                    "injection_policy": injection_policy
                } if auto_kernel_injection or injection_policy else {}

                inference_model = deepspeed.init_inference(
                    accelerator.unwrap_model(inference_model),
                    mp_size=world_size,
                    **injection_config
                )

            inference_model.eval()
            if task_type == "SEQ_2_SEQ_LM" and generative_eval:
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)
                eval_preds = []
                if generative_eval:
                    with TorchTracemalloc() as tracemalloc:
                        with torch.no_grad():
                            for idx, batch in enumerate(
                                    tqdm(generative_eval_dataloader, desc=f"Evaluating epoch {epoch} generative")):
                                # Pass dummy batch to avoid caffe error
                                if idx == 0 and accelerator.distributed_type != DistributedType.NO:
                                    inference_model(**batch)
                                batch = {k: v for k, v in batch.items() if k != "labels"}
                                outputs = inference_model.generate(
                                    **batch, generation_config=generation_config,
                                    synced_gpus=True if accelerator.distributed_type != DistributedType.NO else False,
                                    pad_token_id=tokenizer.pad_token_id
                                )  # synced_gpus=True for Distributed training
                                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                                preds = accelerator.gather_for_metrics(outputs).detach().cpu().numpy()
                                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

                    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
                    accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
                    accelerator.print(
                        "GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
                    accelerator.print(
                        "GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
                    accelerator.print(
                        "GPU Total Peak Memory consumed during the eval (max): {}".format(
                            tracemalloc.peaked + b2mb(tracemalloc.begin)
                        )
                    )

                    accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
                    accelerator.print(
                        "CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
                    accelerator.print(
                        "CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
                    accelerator.print(
                        "CPU Total Peak Memory consumed during the eval (max): {}".format(
                            tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
                        )
                    )

                try:
                    cur_dir = os.getcwd()
                    if '/' in model_name_or_path:
                        model_name = model_name_or_path.replace("/", "-")
                    else:
                        model_name = model_name_or_path
                    log_path = os.path.join(cur_dir, f"src/models/runs/logs/log_dir_e{epoch}_{model_name}_{cur_time}.txt")
                    with open(log_path, 'w') as log_file:
                        # Log info
                        log_file.write(f"\n       {epoch=}: {train_ppl=} {train_epoch_loss=}\n")
                        # log_file.write(f"\n       Accuracy: {accuracy}\n")
                        for i in range(0, 10):
                            idx = random.randint(0, len(eval_preds)-1)
                            accelerator.print(f"        Question: {qa_dataloader.dataset['eval'][idx][text_column]}\n")
                            accelerator.print(f"    Evaluation prediction: {eval_preds[idx]}\n")
                            accelerator.print(f"    Actual label: {qa_dataloader.dataset['eval'][idx][label_column]}\n")

                            log_file.write("===================================================================\n")
                            log_file.write(f"Question: {qa_dataloader.dataset['eval'][idx][text_column]}\n")
                            log_file.write(f"Evaluation prediction: {eval_preds[idx]}\n")
                            log_file.write(f"Actual label: {qa_dataloader.dataset['eval'][idx][label_column]}\n")
                            log_file.write("===================================================================\n")
                        log_file.write(f"\n     Training arguments: \n")
                        for key, value in vars(training_args).items():
                            log_file.write(f"\n {key}: {value} ")

                except Exception as e:
                    warnings.warn(f"Can't save config for this run {epoch}\n"
                                  f"Error message: {e}")
                    pass

            elif task_type == "CAUSAL_LM":
                torch.backends.cuda.enable_mem_efficient_sdp(False)
                torch.backends.cuda.enable_flash_sdp(False)
                eval_preds = []
                if generative_eval:
                    with TorchTracemalloc() as tracemalloc:
                        with torch.no_grad():
                            for idx, batch in enumerate(
                                    tqdm(generative_eval_dataloader, desc=f"Evaluating epoch {epoch} generative")):
                                # Pass dummy batch to avoid caffe error
                                if idx == 0 and accelerator.distributed_type != DistributedType.NO:
                                    inference_model(**batch)
                                batch = {k: v for k, v in batch.items() if k != "labels"}
                                with torch.no_grad():
                                    outputs = inference_model.generate(
                                        **batch, generation_config=generation_config,
                                        synced_gpus=True if accelerator.distributed_type != DistributedType.NO else False,
                                        pad_token_id=tokenizer.pad_token_id
                                    )  # synced_gpus=True for Distributed training
                                outputs = accelerator.pad_across_processes(outputs, dim=1, pad_index=tokenizer.pad_token_id)
                                preds = accelerator.gather_for_metrics(outputs).detach().cpu().numpy()
                                eval_preds.extend(tokenizer.batch_decode(preds, skip_special_tokens=True))

                    # Printing the GPU memory usage details such as allocated memory, peak memory, and total memory usage
                    accelerator.print("GPU Memory before entering the eval : {}".format(b2mb(tracemalloc.begin)))
                    accelerator.print(
                        "GPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.used))
                    accelerator.print(
                        "GPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.peaked))
                    accelerator.print(
                        "GPU Total Peak Memory consumed during the eval (max): {}".format(
                            tracemalloc.peaked + b2mb(tracemalloc.begin)
                        )
                    )

                    accelerator.print("CPU Memory before entering the eval : {}".format(b2mb(tracemalloc.cpu_begin)))
                    accelerator.print(
                        "CPU Memory consumed at the end of the eval (end-begin): {}".format(tracemalloc.cpu_used))
                    accelerator.print(
                        "CPU Peak Memory consumed during the eval (max-begin): {}".format(tracemalloc.cpu_peaked))
                    accelerator.print(
                        "CPU Total Peak Memory consumed during the eval (max): {}".format(
                            tracemalloc.cpu_peaked + b2mb(tracemalloc.cpu_begin)
                        )
                    )

                perplexity = 0
                if perplexity_eval:
                    inference_model.eval()
                    losses = []
                    for step, batch in enumerate(tqdm(perplexity_eval_dataloader,
                                                      desc=f"Evaluating epoch {epoch} perplexity")):
                        with torch.no_grad():
                            outputs = inference_model(**batch)

                        loss = outputs.loss
                        losses.append(accelerator.gather_for_metrics(loss.repeat(qa_dataloader.perplexity_eval_batch_size)))

                    losses = torch.cat(losses)
                    try:
                        eval_loss = torch.mean(losses)
                        perplexity = math.exp(eval_loss)
                    except OverflowError:
                        perplexity = float("inf")

                    accelerator.print(f"epoch {epoch}: perplexity: {perplexity} eval_loss: {eval_loss}")

                if generative_eval:
                    try:
                        cur_dir = os.getcwd()
                        if '/' in model_name_or_path:
                            model_name = model_name_or_path.replace("/", "-")
                        else:
                            model_name = model_name_or_path
                        log_path = os.path.join(cur_dir, f"src/models/runs/logs/log_dir_e{epoch}_{model_name}_{cur_time}.txt")
                        with open(log_path, 'w') as log_file:
                            # Log info
                            log_file.write(f"\n       {epoch=}: {train_ppl=} {train_epoch_loss=}\n")
                            log_file.write(f"\n       Perplexity: {perplexity}\n")
                            for idx in range(0, len(eval_preds)-1):
                                try:
                                    accelerator.print(f"        Question:\n {qa_dataloader.dataset['eval'][idx][text_column]}\n")
                                    accelerator.print(f"    Evaluation prediction:\n {eval_preds[idx]}\n")
                                    accelerator.print(f"    Actual label:\n {qa_dataloader.dataset['eval'][idx][label_column]}\n")

                                    log_file.write("===================================================================\n")
                                    log_file.write(f"Question:\n {qa_dataloader.dataset['eval'][idx][text_column]}\n")
                                    log_file.write(f"Evaluation prediction:\n {eval_preds[idx]}\n")
                                    log_file.write(f"Actual label:\n {qa_dataloader.dataset['eval'][idx][label_column]}\n")
                                    log_file.write("===================================================================\n")
                                except Exception as e:
                                    warnings.warn(f"Can't write config for prediction with idx {idx}\n"
                                                  f"Error message: {e}")
                                    pass
                            log_file.write(f"\n     Training arguments: \n")
                            for key, value in vars(training_args).items():
                                log_file.write(f"\n {key}: {value} ")
                    except Exception as e:
                        warnings.warn(f"Can't save config for this epoch {epoch}\n"
                                      f"Error message: {e}")

            del inference_model
            accelerator.print("Removing inference model offload_inf...")
            shutil.rmtree('offload_inf') if os.path.exists("offload_inf") else None
            gc.collect()

    accelerator.wait_for_everyone()
    # if better_transformer:
    #     adapter = adapter.reverse_bettertransformer()
    adapter.save_pretrained(dataset_name)
    adapter.push_to_hub(
        "1TuanPham/"
        + f"{dataset_name}_{model_name_or_path}_{peft_config.peft_type}_{peft_config.task_type}".replace("/", "_"),
        state_dict=accelerator.get_state_dict(adapter),
        use_auth_token=True,
    )
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    train()