import json
import sys
sys.path.insert(0,r'./')
import os
from tqdm.auto import tqdm

from typing import List, Dict, Union
from functools import partialmethod

from datasets import load_dataset

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample


PARSER_TYPE = "OpenOrca"


class OpenOrcaParser(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_translate=True)
        self.target_config = AdvanceInstructSample
        self.target_fields = ['question_text', 'orig_answer_texts']

    def read(self):
        super(OpenOrcaParser, self).read()
        self.data_read = load_dataset("Open-Orca/OpenOrca")

        return None

    def convert(self):
        super(OpenOrcaParser, self).convert()

        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                data_dict['system_prompt'] = data['system_prompt']
                data_dict['qas_id'] = data['id']
                data_dict['question_text'] = data['question']

                data_dict['orig_answer_texts'] = data['response']
                data_dict['answer_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted[:3000]

        pass


if __name__ == '__main__':
    open_orca_parser = OpenOrcaParser(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\Open-Orca_OpenOrca\dummy.txt",
                                      r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\Open-Orca_OpenOrca")
    open_orca_parser.read()
    open_orca_parser.convert()
    open_orca_parser.save
