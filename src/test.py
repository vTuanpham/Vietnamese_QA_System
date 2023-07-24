import sys
sys.path.insert(0, r'./')
import torch
import torch.functional as F
from transformers import (DPRQuestionEncoder,
                          DPRQuestionEncoderTokenizer,
                          DPRReader,
                          DPRReaderTokenizer,
                          DPRContextEncoder,
                          DPRContextEncoderTokenizer)
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


def translate_vi2en(vi_text: str) -> str:
    input_ids = tokenizer_vi2en(vi_text, return_tensors="pt").input_ids
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


test_str = "Làm thế nào vũ trụ có thể là vô hạn?"

test_context = "Một trong những câu hỏi khó trả lời là nếu vũ trụ có giới hạn. Con người chỉ biết một phần nhỏ của Vũ trụ, nhưng dường như nó là vô hạn. Điều đó có thể được không? Trên thực tế, nhiều nhà thiên văn học cho rằng vũ trụ đang giãn nở, vì vậy về mặt kỹ thuật, nó sẽ không phải là vô hạn mà là hữu hạn. Từ quan điểm của nhiều người, có vẻ khó tin và thực tế, thậm chí là tưởng tượng."

tokenizer_vi2en = AutoTokenizer.from_pretrained("vinai/vinai-translate-vi2en", src_lang="vi_VN")
model_vi2en = AutoModelForSeq2SeqLM.from_pretrained("vinai/vinai-translate-vi2en")

eng_test_str = translate_vi2en(test_str)
print(eng_test_str)

eng_test_ctx = translate_vi2en(test_context)
print(eng_test_ctx)

q_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-multiset-base")
q_model = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-multiset-base")
input_ids = q_tokenizer(eng_test_str, return_tensors="pt")["input_ids"]
q_embeddings = q_model(input_ids).pooler_output
print(q_embeddings.shape)

ctx_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
ctx_model = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-multiset-base")
input_ids = ctx_tokenizer(eng_test_ctx, return_tensors="pt")["input_ids"]
ctx_embeddings = ctx_model(input_ids).pooler_output
print(ctx_embeddings.shape)

print(f"Cosine similarity: {F.F.cosine_similarity(q_embeddings, ctx_embeddings)}")

