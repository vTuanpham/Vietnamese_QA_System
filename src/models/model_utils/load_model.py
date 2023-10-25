import os
import tempfile
import torch
from transformers import AutoModelForCausalLM, AutoModelForSeq2SeqLM


def poor_man_llm_load(model_name: str, model_type: str,
                      model_dtype, max_shard_size: str="200MB",
                      additional_kwargs: dict=None):
    if model_dtype != "auto" and isinstance(model_dtype, str):
        model_dtype = getattr(torch, model_dtype)
    try:
        if model_type == "CAUSAL_LM":
            model = AutoModelForCausalLM.from_pretrained(model_name,
                                                         torch_dtype=model_dtype,
                                                         use_cache=True,
                                                         trust_remote_code=True,
                                                         low_cpu_mem_usage=True)
        elif model_type == "SEQ_2_SEQ_LM":
            model = AutoModelForSeq2SeqLM.from_pretrained(model_name,
                                                          torch_dtype=model_dtype,
                                                          use_cache=True,
                                                          trust_remote_code=True,
                                                          low_cpu_mem_usage=True
                                                          )
        else:
            raise f"Unsupported model type for: {model_type}"

        if "offload_folder" in additional_kwargs.keys():
            os.makedirs(additional_kwargs["offload_folder"]) if not os.path.exists(additional_kwargs["offload_folder"]) else None
        with tempfile.TemporaryDirectory() as tmp_dir:
            print("\nCreating temporary folder...\n")
            model.save_pretrained(tmp_dir, max_shard_size=max_shard_size)
            print('Temp Dir Path:', tmp_dir)
            print(sorted(os.listdir(tmp_dir)))
            print(f"Model {model_name} successfully dump at {tmp_dir}")
            if model_type == "CAUSAL_LM":
                sharded_model = AutoModelForCausalLM.from_pretrained(tmp_dir, **additional_kwargs)
            else:
                sharded_model = AutoModelForSeq2SeqLM.from_pretrained(tmp_dir, **additional_kwargs)
            return sharded_model
    except Exception as e:
        raise f"Unable to offload {model_name} with the " \
              f"following error: {e}"
