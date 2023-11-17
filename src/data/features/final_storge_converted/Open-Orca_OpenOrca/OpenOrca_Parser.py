import json
import sys
sys.path.insert(0,r'./')
import os
from tqdm.auto import tqdm

from datasets import load_dataset

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample


PARSER_TYPE = "OpenOrca"


class OpenOrcaParser(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_translate=True,
                         no_translated_code=True)
        self.target_config = AdvanceInstructSample
        self.target_fields = ['question_text', 'orig_answer_texts']
        self.max_examples = 80000

    def read(self):
        super(OpenOrcaParser, self).read()
        stream_data = load_dataset("Open-Orca/OpenOrca", streaming=True, keep_in_memory=False)
        progress_bar = tqdm(total=self.max_examples, desc="Loading data")
        self.data_read = []
        for idx, data in enumerate(stream_data['train']):
            if idx <= self.max_examples:
                self.data_read.append(data)
                progress_bar.update(1)
            else:
                break

        return None

    def convert(self):
        super(OpenOrcaParser, self).convert()
        data_converted = []
        for data in tqdm(self.data_read, desc=f"Converting data"):
            data_dict = {}
            data_dict['system_prompt'] = data['system_prompt']
            data_dict['qas_id'] = data['id']
            data_dict['question_text'] = data['question']

            data_dict['orig_answer_texts'] = data['response']
            data_dict['answer_lengths'] = None
            data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    open_orca_parser = OpenOrcaParser(r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/dummy.txt",
                                      r"src/data/features/final_storge_converted/Open-Orca_OpenOrca")
    open_orca_parser.read()
    open_orca_parser.convert()
    open_orca_parser.save
