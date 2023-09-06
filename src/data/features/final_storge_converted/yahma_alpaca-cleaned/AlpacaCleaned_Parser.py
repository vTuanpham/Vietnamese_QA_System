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
from src.data.configs import AdvanceInstructSample, QA_TEMPLATE


PARSER_TYPE = "AlpacaCleaned"


class AlpacaCleaned(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_translate=False)
        self.target_config = AdvanceInstructSample
        self.target_fields = ['question_text', 'orig_answer_texts']

    def read(self):
        super(AlpacaCleaned, self).read()
        self.data_read = load_dataset("yahma/alpaca-cleaned")

        return None

    def convert(self):
        super(AlpacaCleaned, self).convert()

        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                # Randomly assign generic system prompt to data
                data_dict['system_prompt'] = QA_TEMPLATE().get_generic_system_prompt(random.randint(1, 20)) if bool(random.getrandbits(1)) else ""
                data_dict['qas_id'] = self.id_generator()
                data_dict['question_text'] = data['instruction'] + " " + data['input']

                data_dict['orig_answer_texts'] = data['output']
                data_dict['answer_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    alpaca_cleaned_parser = AlpacaCleaned(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\yahma_alpaca-cleaned\dummy.txt",
                                          r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\yahma_alpaca-cleaned")
    alpaca_cleaned_parser.read()
    alpaca_cleaned_parser.convert()
    alpaca_cleaned_parser.save
