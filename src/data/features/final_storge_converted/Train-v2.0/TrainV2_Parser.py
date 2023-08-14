import json
import sys
sys.path.insert(0,r'./')
import os
from tqdm.auto import tqdm

from typing import List, Dict, Union
from functools import partialmethod

from src.data.features import DataParser


PARSER_TYPE = "Train_V2"


class TrainV2Parser(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_ctx_augmentation=True)

    def read(self):
        super(TrainV2Parser, self).read()
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)['data']
        data_read = []
        for data in tqdm(json_data, desc="Reading from file"):
            data_read.append(data)

        self.data_read = data_read
        return None

    def convert(self):
        super(TrainV2Parser, self).convert()

        data_converted = []
        for data in tqdm(self.data_read, desc="Converting data"):
            data_dict = {}
            for paragraph in data['paragraphs']:
                data_dict['doc_tokens'] = [paragraph['context']]
                data_dict['doc_tokens'] = super().inject_random_ctx(data_dict['doc_tokens'])
                for qa in paragraph["qas"]:
                    data_dict['qas_id'] = qa['id']
                    data_dict['question_text'] = qa['question']

                    data_dict['is_impossible'] = qa['is_impossible']
                    answers = qa['answers']
                    data_dict['orig_answer_texts'] = answers[0]["text"] if answers else None
                    data_dict['answer_lengths'] = None
                    data_dict['is_trivial'] = None
                    data_dict['docs_lengths'] = None
                    data_converted.append(data_dict)
                    data_dict = {}

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    trainv2_parser = TrainV2Parser(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\Train-v2.0\train-v2.0.json",
                                   r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\Train-v2.0")
    trainv2_parser.read()
    trainv2_parser.convert()
    trainv2_parser.save