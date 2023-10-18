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


PARSER_TYPE = "WizardLM_20k_Filtered"


class WizardLM70k(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_translate=True,
                         no_translated_code=True,
                         max_example_per_thread=300,
                         large_chunks_threshold=3000
                         )
        self.target_config = AdvanceInstructSample
        self.target_fields = ['question_text', 'orig_answer_texts']

    def read(self):
        super(WizardLM70k, self).read()
        self.data_read = load_dataset("WizardLM/WizardLM_evol_instruct_70k")

        return None

    def convert(self):
        super(WizardLM70k, self).convert()

        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                # Randomly assign generic system prompt to data
                data_dict['system_prompt'] = QA_TEMPLATE().get_generic_system_prompt(random.randint(1, 20)) if bool(random.getrandbits(1)) else ""
                data_dict['qas_id'] = self.id_generator()
                data_dict['question_text'] = data['instruction']

                data_dict['orig_answer_texts'] = data['output']
                data_dict['answer_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted[:20000]

        pass


if __name__ == '__main__':
    wizardlm_parser = WizardLM70k(r"src/data/features/final_storge_converted/WizardLM_WizardLM_evol_instruct_70k/dummy.txt",
                                  r"src/data/features/final_storge_converted/WizardLM_WizardLM_evol_instruct_70k")
    wizardlm_parser.read()
    wizardlm_parser.convert()
    wizardlm_parser.save
