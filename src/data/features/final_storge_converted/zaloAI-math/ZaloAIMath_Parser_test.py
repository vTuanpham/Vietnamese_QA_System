import json
import sys
import random
sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample


PARSER_TYPE = "ZaloAIMath_test"


class ZaloAIMath(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         )
        self.target_config = AdvanceInstructSample

    def read(self):
        super(ZaloAIMath, self).read()
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)

        self.data_read = json_data["data"]

        return None

    def convert(self):
        super(ZaloAIMath, self).convert()

        math_qa_system_prompts = [
            "You're an AI math wizard, ready to solve complex problems.",
            "As a math expert AI, you're here to tackle mathematical challenges.",
            "Your specialty lies in mathematics, and you're here to assist with problem-solving.",
            "Specializing in math, you excel in solving numerical puzzles and equations.",
            "You are a math problem-solving expert AI.",
            "Your primary skill is in solving math problems of all kinds.",
            "As an AI math expert, you can help with various mathematical inquiries.",
            "Your primary focus is on facilitating mathematical problem-solving tasks.",
            "You're here to make math problem-solving easy and efficient.",
            "As a math expert AI, you're dedicated to helping with numerical challenges.",
            "You specialize in unraveling mathematical mysteries and equations.",
            "You excel at cracking numbers and equations of all types. How can I assist?",
            "Let's dive into some math problems! What mathematical question would you like me to answer?",
            "Your expertise is in solving mathematical problems and equations.",
            "You're equipped to handle a wide range of mathematical needs and inquiries.",
            "Mathematics is your forte, and you're here to assist with problem-solving.",
            "You're well-versed in the art of solving mathematical challenges.",
            "You have a deep understanding of mathematics and can assist with various math questions.",
            "Your mission is to simplify math for users. What do you need help with?",
            "Math is your playground, and you're here to make it easier for everyone."
        ]

        data_converted = []
        for data in tqdm(self.data_read, desc=f"Converting data"):
            data_dict = {}
            data_dict['system_prompt'] = random.choice(math_qa_system_prompts)
            data_dict['qas_id'] = data["id"]
            data_dict['question_text'] = data['question']
            for choice in data["choices"]:
                data_dict['question_text'] += f"\n{choice}"

                data_dict['orig_answer_texts'] = "TEST DON'T HAVE ANSWER"
            data_dict['answer_lengths'] = None
            data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    zalo_ai_parser = ZaloAIMath(r"src/data/features/final_storge_converted/zaloAI-math/math_test.json",
                                r"src/data/features/final_storge_converted/zaloAI-math")
    zalo_ai_parser.read()
    zalo_ai_parser.convert()
    zalo_ai_parser.save
