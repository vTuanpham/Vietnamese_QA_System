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


PARSER_TYPE = "Ctx_injector"


class CTXInjector(DataParser):
    """This class is for data that is already converted
    to AdvanceQAsample(translated) and needs to inject more context"""
    def __init__(self, file_path: str, output_path: str, max_ctxs: int=100):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_ctx_augmentation=True,
                         do_translate=False)
        self.max_ctxs = max_ctxs

    def read(self):
        super(CTXInjector, self).read()
        with open(self.file_path, encoding='utf-8') as jfile:
            json_data = json.load(jfile)
        data_read = []
        for data in tqdm(json_data, desc="Reading data"):
            data_read.append(data)

        self.data_read = data_read
        return None

    def convert(self):
        super(CTXInjector, self).convert()
        data_injected = []
        for data in tqdm(self.data_read, desc="Injecting data"):
            data['doc_tokens'] = super().inject_random_ctx(data['doc_tokens'])
            data_injected.append(data)

        self.converted_data = data_injected

        pass


if __name__ == '__main__':
    ctx_injector = CTXInjector(r"/content/output/ELI5_Mult_Parser_translated.json",
                             r"/content/output",
                             max_ctxs=100)
    ctx_injector.read()
    ctx_injector.convert()
    ctx_injector.save