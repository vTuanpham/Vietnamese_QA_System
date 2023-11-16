import json
import sys
import random

import numpy as np

sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample


PARSER_TYPE = "ZaloAIMath_train_dev"


class ZaloAIMath(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         )
        self.target_config = AdvanceInstructSample

    @staticmethod
    def split_two(lst, ratio=[0.5, 0.5]):
        assert (np.sum(ratio) == 1.0)  # makes sure the splits make sense
        train_ratio = ratio[0]
        # note this function needs only the "middle" index to split, the remaining is the rest of the split
        indices_for_splittin = [int(len(lst) * train_ratio)]
        train, test = np.split(lst, indices_for_splittin)
        return train, test

    def read(self):
        super(ZaloAIMath, self).read()
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)

        self.data_read = json_data["data"]

        return None

    def convert(self):
        super(ZaloAIMath, self).convert()

        math_qa_system_prompts = [
            "You're an AI math wizard, ready to solve complex problems. Keep the answer short",
            "As a math expert AI, you're here to tackle mathematical challenges. Keep the answer short",
            "Your specialty lies in mathematics, and you're here to assist with problem-solving. Keep the answer short",
            "Specializing in math, you excel in solving numerical puzzles and equations. Keep the answer short",
            "You are a math problem-solving expert AI. Keep the answer short",
            "Your primary skill is in solving math problems of all kinds. Keep the answer short",
            "As an AI math expert, you can help with various mathematical inquiries. Keep the answer short",
            "Your primary focus is on facilitating mathematical problem-solving tasks. Keep the answer short",
            "You're here to make math problem-solving easy and efficient. Keep the answer short",
            "As a math expert AI, you're dedicated to helping with numerical challenges. Keep the answer short",
            "You specialize in unraveling mathematical mysteries and equations. Keep the answer short",
            "You excel at cracking numbers and equations of all types. Keep the answer short",
            "Your expertise is in solving mathematical problems and equations. Keep the answer short",
            "You're equipped to handle a wide range of mathematical needs and inquiries. Keep the answer short",
            "Mathematics is your forte, and you're here to assist with problem-solving. Keep the answer short",
            "You're well-versed in the art of solving mathematical challenges. Keep the answer short",
            "You have a deep understanding of mathematics and can assist with various math questions. Keep the answer short",
            "Your mission is to simplify math for users. Keep the answer short",
            "Math is your playground, and you're here to make it easier for everyone. Keep the answer short"
        ]

        data_converted = []
        for data in tqdm(self.data_read, desc=f"Converting data"):
            data_dict = {}
            data_dict['system_prompt'] = random.choice(math_qa_system_prompts)
            data_dict['qas_id'] = data["id"]
            data_dict['question_text'] = data['question']
            data_dict['question_text'] += "\nĐây là các lựa chọn, hãy chọn một đáp án duy nhất: \n"
            for choice in data["choices"]:
                data_dict['question_text'] += f"\n{choice}"

            if 'explanation' in data:
                data_dict['orig_answer_texts'] = f"{data['explanation']}\n\n"
                data_dict['orig_answer_texts'] += f"{data['answer']}"
            else:
                data_dict['orig_answer_texts'] = f"{data['answer']}"

            data_dict['answer_lengths'] = None
            data_converted.append(data_dict)

        self.converted_data = self.split_two(data_converted, ratio=[0.9, 0.1])[1]

        pass


if __name__ == '__main__':
    zalo_ai_parser = ZaloAIMath(r"src/data/features/final_storge_converted/zaloAI-math/math_train.json",
                                r"src/data/features/final_storge_converted/zaloAI-math")
    zalo_ai_parser.read()
    zalo_ai_parser.convert()
    zalo_ai_parser.save
