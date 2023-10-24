import json
import sys
import random
sys.path.insert(0,r'./')
import os
from tqdm.auto import tqdm

from typing import List, Dict, Union
from functools import partialmethod

from datasets import load_dataset

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample


PARSER_TYPE = "MTEngVietnamese"


class MTEngVietnamese(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_translate=False)
        self.target_config = AdvanceInstructSample

    def read(self):
        super(MTEngVietnamese, self).read()
        self.data_read = load_dataset("mt_eng_vietnamese", 'iwslt2015-vi-en')

        return None

    def convert(self):
        super(MTEngVietnamese, self).convert()

        data_converted = []
        translation_en2vi_prefix = [
            "Translate this sentence to Vietnamese for me:\n",
            "Can you translate this English sentence to Vietnamese?",
            "Translate this message to Vietnamese:",
            "What does this say in Vietnamese?\n",
            "Please help me with this translation:",
            "I need a Vietnamese translation for the following:",
            "In Vietnamese, how would you say:\n",
            "Could you provide a Vietnamese translation for:",
            "I'm looking for the Vietnamese version of:",
            "Translate this English text to Vietnamese:",
            "I'd like to get this sentence translated into Vietnamese:\n",
            "How do you say this in Vietnamese?\n",
            "Convert this to Vietnamese, please:",
            "Translate the following to Vietnamese:",
            "In Vietnamese, the phrase should be:",
            "Could you do a Vietnamese translation for:",
            "I need a Vietnamese version of this:\n",
        ]
        translation_vi2en_prefix = [
            "Dịch câu này sang tiếng Anh giúp tôi:\n",
            "Bạn có thể dịch câu tiếng Việt này sang tiếng Anh được không?\n",
            "Dịch tin nhắn này sang tiếng Anh:",
            "Cái này nói tiếng Anh là gì?",
            "Hãy giúp tôi dịch câu nói này sang tiếng anh:",
            "Tôi cần một bản dịch tiếng Anh cho cái này:\n",
            "Bằng tiếng Anh, câu này sẽ như thế nào:",
            "Bạn có thể cung cấp bản dịch tiếng Anh cho:",
            "Tôi đang tìm phiên bản tiếng Anh của:",
            "Dịch đoạn văn tiếng Việt này sang tiếng Anh giúp tôi:",
            "Có thể dịch đoạn tiếng Việt này sang tiếng Anh giúp tôi được không?",
            "Dịch thư này sang tiếng Anh:",
            "Cái này nói tiếng Anh làm sao?",
            "Xin bạn giúp tôi dịch đoạn này sang tiếng Anh:",
            "Tôi cần một bản dịch tiếng Anh cho đoạn sau:\n",
            "Bằng tiếng Anh, đoạn văn này sẽ như thế nào:",
            "Bạn có thể cung cấp bản dịch tiếng Anh cho:",
            "Tôi đang tìm phiên bản tiếng Anh của đoạn văn này:\n",
        ]
        translation_system_prompt = [
            "You're an AI assistant with expertise in translation.",
            "As a translation specialist AI, you can help with language conversions.",
            "Your area of expertise lies in translation services.",
            "Specializing in translation, you're here to assist with language conversions.",
            "You excel in the field of translation and language conversion.",
            "You are a language translation expert AI.",
            "Your specialization is in translation across different languages.",
            "Your primary skill is in translating text from one language to another.",
            "As an AI language translation expert, you can help with translation requests.",
            "Your primary focus is on facilitating language translation tasks.",
            "You're here to make language translation easy and efficient.",
            "As a translation expert AI, you're dedicated to helping with language conversions.",
            "You specialize in breaking language barriers through translation.",
            "You excel at bridging communication gaps by providing translation services.",
            "Your expertise is in transforming text from one language to another.",
            "You're equipped to handle a wide range of translation needs.",
            "Language translation is your forte, and you're here to assist.",
            "You're well-versed in the art of translating languages.",
            "You have a deep understanding of language translation and can assist with various requests.",
        ]
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                translate_task = "en2vi" if bool(random.getrandbits(1)) else "vi2en"
                data_dict = {}
                # Randomly assign generic system prompt to data
                data_dict['system_prompt'] = random.choice(translation_system_prompt)
                data_dict['qas_id'] = self.id_generator(size=4) +"_"+ translate_task

                if translate_task == "en2vi":
                    en2vi_prefix = random.choice(translation_en2vi_prefix)
                    data_dict['question_text'] = en2vi_prefix + " " + data['translation']["en"]
                    data_dict['orig_answer_texts'] = data['translation']["vi"]

                if translate_task == "vi2en":
                    vi2en_prefix = random.choice(translation_vi2en_prefix)
                    data_dict['question_text'] = vi2en_prefix + " " + data['translation']["vi"]
                    data_dict['orig_answer_texts'] = data['translation']["en"]

                data_dict['answer_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    mt_eng_vietnamese_parser = MTEngVietnamese(r"src/data/features/final_storge_converted/mt_eng_vietnamese/dummy.txt",
                                               r"src/data/features/final_storge_converted/mt_eng_vietnamese")
    mt_eng_vietnamese_parser.read()
    mt_eng_vietnamese_parser.convert()
    mt_eng_vietnamese_parser.save
