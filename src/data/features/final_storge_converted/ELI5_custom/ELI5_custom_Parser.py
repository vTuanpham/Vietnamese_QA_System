import json
import sys
import random
import string
sys.path.insert(0,r'./')
import os
from tqdm.auto import tqdm
from typing import List, Dict, Union
from functools import partialmethod

from datasets import load_dataset

from src.data.features import DataParser


PARSER_TYPE = "ELI5_custom_Parser"


class ELI5Parser(DataParser):
    def __init__(self, file_path: str, output_path: str, max_ctxs: int=100):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_ctx_augmentation=True,
                         do_translate=True)
        self.max_ctxs = max_ctxs

    def read(self):
        super(ELI5Parser, self).read()
        self.data_read = load_dataset("rusano/ELI5_custom")

        return None

    # @staticmethod
    # def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    #     return ''.join(random.choice(chars) for _ in range(size))

    def convert(self):
        super(ELI5Parser, self).convert()
        data_converted = []
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                data_dict['doc_tokens'] = data['context']
                data_dict['doc_tokens'] = super().inject_random_ctx(data_dict['doc_tokens'])
                data_dict['qas_id'] = self.id_generator(size=8)
                data_dict['question_text'] = data['question']

                data_dict['is_impossible'] = None
                data_dict['orig_answer_texts'] = data['answer']
                data_dict['answer_lengths'] = None
                data_dict['is_trivial'] = None
                data_dict['docs_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    eli5_parser = ELI5Parser(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\ELI5_mult_answers_en\dummy.txt",
                             r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\ELI5_mult_answers_en",
                             max_ctxs=100)
    eli5_parser.read()
    eli5_parser.convert()
    eli5_parser.save