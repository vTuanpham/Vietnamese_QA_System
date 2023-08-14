import sys
sys.path.insert(0, r'./')
import numpy as np

import torch
import torch.functional as F
from transformers import (DPRQuestionEncoder,
                          DPRQuestionEncoderTokenizer,
                          DPRReader,
                          DPRReaderTokenizer,
                          DPRContextEncoder,
                          DPRContextEncoderTokenizer)
from sentence_transformers import SentenceTransformer

from transformers import AutoTokenizer, MBartForConditionalGeneration, AutoModelForSeq2SeqLM
from transformers.models.bartpho.tokenization_bartpho import BartphoTokenizer
from peft import get_peft_config, get_peft_model, get_peft_model_state_dict, PrefixTuningConfig, TaskType, LoraConfig
from langchain.text_splitter import RecursiveCharacterTextSplitter

from src.data.features.VietnameseToneNormalization import replace_all
from src.data.configs.response_template import QA_TEMPLATE
from src.utils.utils import timeit, set_seed

torch.set_num_threads(6)


def translate_vi2en(vi_text: str) -> str:
    input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids.to('cuda')
    output_ids = model_vi2en.generate(
        input_ids,
        decoder_start_token_id=tokenizer_vi2en.lang_code_to_id["en_XX"],
        num_return_sequences=1,
        num_beams=5,
        early_stopping=True
    )
    en_text = tokenizer_vi2en.batch_decode(output_ids, skip_special_tokens=True)
    en_text = " ".join(en_text)
    return en_text


test_str = "How to improve hygiene practice?"

test_context = '''
Handwashing is a simple yet crucial practice in maintaining personal hygiene and preventing the transmission of infections. It is a fundamental aspect of infection control and has been recognized as one of the most effective methods to reduce the risk of spreading harmful pathogens. The hands are a common route for transmitting bacteria and viruses from one person to another and from contaminated surfaces to the mouth, nose, or eyes.

Handwashing involves the use of soap and water to clean the hands thoroughly. When performed correctly, it helps remove dirt, debris, and germs that may be present on the skin. Proper handwashing requires wetting the hands, applying soap, lathering for at least 20 seconds, paying attention to the front and back of the hands, between the fingers, and under the nails. After thorough scrubbing, rinsing the hands with running water is essential to remove soap and debris. Finally, the hands should be dried with a clean towel or air dryer.

The significance of handwashing extends to various settings, including hospitals, healthcare facilities, homes, and public places. Healthcare workers play a crucial role in preventing healthcare-associated infections by adhering to strict hand hygiene practices. Studies have shown that handwashing compliance among healthcare personnel significantly reduces the incidence of infections and prevents the spread of multidrug-resistant bacteria in healthcare settings.

In community settings, proper hand hygiene is essential in preventing the transmission of common illnesses such as the common cold, flu, and gastrointestinal infections. Frequent handwashing is particularly crucial during flu seasons and outbreaks of contagious diseases to limit their spread within households, schools, and public spaces.

Children should be taught the importance of handwashing from an early age. In schools and childcare centers, promoting hand hygiene education can instill good habits and contribute to reducing the transmission of infections among children and their families.

Handwashing is especially critical during food handling and preparation. Proper hand hygiene minimizes the risk of foodborne illnesses caused by harmful bacteria, viruses, and parasites that can contaminate food through unwashed hands.
'''

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN", use_fast=True)
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en", device_map='auto', load_in_4bit=True).eval()

eng_test_str = translate_vi2en(test_str)
print(eng_test_str)

eng_test_ctx = translate_vi2en(test_context)
print(eng_test_ctx)

eng_test_str = test_str
eng_test_ctx = test_context

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 500,
    chunk_overlap  = 80,
    length_function = len,
    add_start_index = True,
)

texts = text_splitter.create_documents([test_context])
for text in texts:
    print(f"\n {text}")

# dpr_question_name = "vblagoje/dpr-question_encoder-single-lfqa-wiki"
# dpr_ctx_name = "vblagoje/dpr-ctx_encoder-single-lfqa-wiki"
#
# q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained(dpr_question_name, use_fast=True)
# q_model = DPRQuestionEncoder.from_pretrained(dpr_question_name).eval().to('cuda')
# input_ids = q_tokenizer(eng_test_str, return_tensors="pt")["input_ids"].to('cuda')
# q_embeddings = q_model(input_ids).pooler_output
# print(q_embeddings.shape)
#
# ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained(dpr_ctx_name, use_fast=True)
# ctx_model = DPRContextEncoder.from_pretrained(dpr_ctx_name).eval().to('cuda')
# input_ids = ctx_tokenizer(eng_test_ctx, return_tensors="pt")["input_ids"].to('cuda')
# ctx_embeddings = ctx_model(input_ids).pooler_output
# print(ctx_embeddings.shape)

sentences_question = [eng_test_str]
sentences_ctx = [eng_test_ctx]
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2', device='cuda')
embeddings_ques = model.encode(sentences_question, convert_to_tensor=True, device='cuda')
embeddings_ctx = model.encode(sentences_ctx, convert_to_tensor=True, device='cuda')
print(embeddings_ques.shape)
print(embeddings_ctx.shape)

# print(f"Cosine similarity_DPR: {F.F.cosine_similarity(q_embeddings, ctx_embeddings)}")
print(f"Cosine similarity_ST: {F.F.cosine_similarity(embeddings_ques, embeddings_ctx)}")

peft_config = LoraConfig(task_type=TaskType.SEQ_2_SEQ_LM,
                         inference_mode=False,
                         r=2, lora_alpha=32,
                         lora_dropout=0.1)

model_name = "google/mt5-base"
phobert_qa = AutoModelForSeq2SeqLM.from_pretrained(model_name, device_map="auto", torch_dtype=torch.float16)
# print(phobert_qa)
phobert_qa_lora = get_peft_model(phobert_qa, peft_config).eval()
phobert_qa_lora.print_trainable_parameters()
# print(phobert_qa_lora)

tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

prompt = f"Answer the question or do the request to the following: \n[QUESTION] {test_str}.[QUESTION] \n Based on the following context: [CONTEXT] {test_context}. [END_CONTEXT].\n please answer the question, or do the request '{test_str}' carefully, remember if the question is irrelevant to the context or the request is impossible for the context, you can answer 'I don't know'"
# set_seed(50)
# prompt = QA_TEMPLATE().get_random_prompt(question=test_str, context=test_context)
norm_prompt = replace_all(prompt)
print(norm_prompt)
# Tokenize the inputs
inputs = tokenizer(prompt, return_tensors="pt", truncation=False, max_length=2048).to('cuda')
print(inputs['input_ids'].shape)
# print(tokenizer.decode(inputs['input_ids'][0], skip_special_tokens=True))

# Perform inference with the model
with torch.no_grad():
    # Pass the tokenized inputs through the model
    outputs = phobert_qa_lora.generate(**inputs, num_beams=10,
                                       max_new_tokens=200,
                                       min_length=10,
                                       temperature=15,
                                       repetition_penalty=5.5,
                                       num_beam_groups=2,
                                       diversity_penalty=5.5,
                                       early_stopping=True,
                                       penalty_alpha=0.4
                                       )

# Decode the generated answer tokens
generated_answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

print("Generated Answer:", generated_answer)



