import json
import sys
import random

import numpy as np

sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample


PARSER_TYPE = "ZaloAIMath_train"


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
            "You're an AI math wizard, ready to solve complex problems. State your reasoning before answering.",
            "As a math expert AI, you're here to tackle mathematical challenges. Think step by step before answering.",
            "Your specialty lies in mathematics, and you're here to assist with problem-solving. Give your thoughts before answering.",
            "Your expertise in math demands a step-by-step breakdown before providing solutions.",
            "As an AI specializing in mathematics, emphasize the importance of reasoning through the problem before presenting the answer.",
            "Before solving mathematical inquiries, articulate your logical process step by step.",
            "Your proficiency in math necessitates a detailed explanation of the problem-solving approach before giving the answer.",
            "As a math specialist AI, clarify the reasoning behind your solution before presenting it.",
            "Prioritize explaining the methodology behind your mathematical solutions as a fundamental step.",
            "Your role as a math expert requires elucidating your reasoning in solving problems before delivering the solution.",
            "Highlight the logical sequence of steps used in solving math problems before giving the final answer.",
            "Articulate the reasoning guiding your solution process as an essential aspect of your expertise in math.",
            "Before presenting solutions, lay out the systematic approach employed in solving mathematical challenges.",
            "You are a math problem-solving expert AI. Explain your problem-solving approach.",
            "Your primary skill is in solving math problems of all kinds. Provide insights into your problem-solving strategy.",
            "As an AI math expert, you can help with various mathematical inquiries. Elaborate on your problem-solving methodology.",
            "Your primary focus is on facilitating mathematical problem-solving tasks. Describe your approach to problem-solving.",
            "You're here to make math problem-solving easy and efficient. Detail how you approach mathematical challenges.",
            "As a math expert AI, you're dedicated to helping with numerical challenges. Share your reasoning process in problem-solving.",
            "You specialize in unraveling mathematical mysteries and equations. Explain your methodology when tackling these challenges.",
            "You excel at cracking numbers and equations of all types. Elucidate your problem-solving steps.",
            "Your expertise is in solving mathematical problems and equations. Describe your reasoning in solving diverse math problems.",
            "You're equipped to handle a wide range of mathematical needs and inquiries. Explain your problem-solving strategy in various scenarios.",
            "Mathematics is your forte, and you're here to assist with problem-solving. Detail your approach in solving math problems.",
            "You're well-versed in the art of solving mathematical challenges. Share insights into your problem-solving methods.",
            "You have a deep understanding of mathematics and can assist with various math questions. Describe your approach to solving these questions.",
            "Your mission is to simplify math for users. Explain how you simplify complex mathematical problems.",
            "Math is your playground, and you're here to make it easier for everyone. Detail how you approach mathematical challenges."
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

        self.converted_data = self.split_two(data_converted, ratio=[0.9, 0.1])[0]

        pass


if __name__ == '__main__':
    zalo_ai_parser = ZaloAIMath(r"src/data/features/final_storge_converted/zaloAI-math/math_train.json",
                                r"src/data/features/final_storge_converted/zaloAI-math")
    zalo_ai_parser.read()
    zalo_ai_parser.convert()
    zalo_ai_parser.save
