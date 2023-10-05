from transformers import pipeline
import transformers
import deepspeed
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "tomaxe/gpt-neo-2.7B-sharded"
peft_model_id = "1TuanPham/Instruction_en-vi_18k_tomaxe_gpt-neo-2.7B-sharded_LORA_CAUSAL_LM"

max_memory = f'{int(torch.cuda.mem_get_info()[0] / 1024 ** 3) - 2}GB'
n_gpus = torch.cuda.device_count()
max_memory = {i: max_memory for i in range(n_gpus)}

local_rank = int(os.getenv("LOCAL_RANK", '0'))
world_size = int(os.getenv('WORLD_SIZE', '1'))
os.environ["RANK"] = "0"
os.environ["LOCAL_RANK"] = "0"
os.environ["WORLD_SIZE"] = "1"

offload_config = {
    # "device_map": "auto",
    # "offload_folder": "offload_inf",
    "torch_dtype": torch.bfloat16,
    "use_cache": True,
    "offload_state_dict": True,
    "low_cpu_mem_usage": True,
    "trust_remote_code":True,
    "max_memory": max_memory,
}

model = AutoModelForCausalLM.from_pretrained(model_name,
                                                  **offload_config
                                                  )
tokenizer = AutoTokenizer.from_pretrained(model_name)
model.load_adapter(peft_model_id)
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# The injection_policy shows two things:
#   1. which layer module we need to add Tensor-Parallelism
#   2. the name of several linear layers: a) attention_output (both encoder and decoder),
#       and b) transformer output

pipe.model = deepspeed.init_inference(
    pipe.model,
    mp_size=world_size,
    dtype=torch.bfloat16
  )

pipe.device = torch.device(f'cuda:{local_rank}')
while True:
  prompt = input("Enter prompt: \n")
  if "[END]" in prompt: break
  output = pipe(prompt, do_sample=True, top_p=0.8, max_new_tokens=256,
                temperature=0.6, max_length=2048, early_stopping=True,
                repetition_penalty=2.5, no_repeat_ngram_size=3)

  if not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0:
      print(output)