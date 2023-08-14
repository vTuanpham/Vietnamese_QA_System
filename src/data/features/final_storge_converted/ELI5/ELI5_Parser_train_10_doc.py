import json
import sys
sys.path.insert(0,r'./')
import os
from tqdm.auto import tqdm

from typing import List, Dict, Union
from functools import partialmethod

from src.data.features import DataParser


PARSER_TYPE = "ELI5_Parser_train_10_doc"


class ELI5Parser(DataParser):
    def __init__(self, file_path: str, output_path: str, max_ctxs: int=100):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_ctx_augmentation=False,
                         do_translate=True,
                         max_example_per_thread=2000,
                         large_chunks_threshold=20000)
        self.max_ctxs = max_ctxs

    def read(self):
        super(ELI5Parser, self).read()
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)
        data_read = []
        for data in tqdm(json_data, desc="Reading data"):
            data_read.append(data)

        self.data_read = data_read
        return None

    def convert(self):
        super(ELI5Parser, self).convert()
        data_converted = []
        for data in tqdm(self.data_read, desc="Converting data"):
            data_dict = {}
            data_dict['doc_tokens'] = data['ctxs'][:self.max_ctxs]
            # data_dict['doc_tokens'] = super().inject_random_ctx(data_dict['doc_tokens'])
            data_dict['qas_id'] = data['question_id']
            data_dict['question_text'] = data['question']

            data_dict['is_impossible'] = None
            data_dict['orig_answer_texts'] = data['answers'][0] if data['answers'] else None
            data_dict['answer_lengths'] = None
            data_dict['is_trivial'] = None
            data_dict['docs_lengths'] = None
            data_converted.append(data_dict)

        self.converted_data = data_converted[:10]

        pass


if __name__ == '__main__':
    eli5_parser = ELI5Parser(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\ELI5\ELI5_train_10_doc.json",
                             r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\ELI5",
                             max_ctxs=10)
    eli5_parser.read()
    eli5_parser.convert()
    eli5_parser.save