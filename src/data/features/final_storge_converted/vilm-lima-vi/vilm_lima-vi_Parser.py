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


PARSER_TYPE = "vilm_lima-vi"


class VilmLimaVi(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE)
        self.target_config = AdvanceInstructSample

    def read(self):
        super(VilmLimaVi, self).read()
        self.data_read = load_dataset("vilm/lima-vi")

        return None

    def convert(self):
        super(VilmLimaVi, self).convert()
        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                # Randomly assign generic system prompt to data
                data_dict['system_prompt'] = QA_TEMPLATE().get_generic_system_prompt(random.randint(1, 20)) if bool(random.getrandbits(1)) else ""
                data_dict['qas_id'] = self.id_generator(size=6)
                data_dict['question_text'] = data['question']
                data_dict['orig_answer_texts'] = data['answer']
                data_dict['answer_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    vilm_lima_vi_parser = VilmLimaVi(r"src/data/features/final_storge_converted/vilm-lima-vi/dummy.txt",
                                     r"src/data/features/final_storge_converted/vilm-lima-vi")
    vilm_lima_vi_parser.read()
    vilm_lima_vi_parser.convert()
    vilm_lima_vi_parser.save
