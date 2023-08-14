import json
import sys
sys.path.insert(0,r'./')
import os
from tqdm.auto import tqdm

from typing import List, Dict, Union
from functools import partialmethod

from src.data.features import DataParser


PARSER_TYPE = "Train_IR"


class TrainIRParser(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path, parser_type=PARSER_TYPE)

    def read(self):
        super(TrainIRParser, self).read()
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)
        data_read = []
        for data in tqdm(json_data):
            data_read.append(data)

        self.data_read = data_read
        return None

    def convert(self):
        super(TrainIRParser, self).convert()

        data_converted = []
        for data in tqdm(self.data_read):
            data_dict = {}
            if data['label']:
                data_dict['qas_id'] = data['id']
                data_dict['question_text'] = data['question']
                data_dict['answer_lengths'] = None
                data_dict['orig_answer_texts'] = data['text'] if data['label'] else None
                data_dict['is_impossible'] = not data['label']
                data_dict['is_trivial'] = True
                data_dict['docs_lengths'] = None
                data_dict['doc_tokens'] = []
                data_converted.append(data_dict)
        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    test = TrainIRParser(r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\Train_IR\train_IR.json",
                         r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\Train_IR")
    test.read()
    test.convert()
    test.save
