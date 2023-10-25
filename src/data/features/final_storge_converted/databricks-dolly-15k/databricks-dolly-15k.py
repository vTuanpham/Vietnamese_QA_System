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


PARSER_TYPE = "databricks_dolly15k"


class DataBricksDolly15k(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE,
                         do_translate=True,
                         no_translated_code=True,
                         max_example_per_thread=300,
                         large_chunks_threshold=3000)
        self.target_config = AdvanceInstructSample
        self.target_fields = ['question_text', 'orig_answer_texts']

    def read(self):
        super(DataBricksDolly15k, self).read()
        self.data_read = load_dataset("databricks/databricks-dolly-15k")

        return None

    def convert(self):
        super(DataBricksDolly15k, self).convert()
        data_converted = []
        docs_prefix = [
            "Context:",
            "Docs:",
            "Here are some info:",
            "Documents:",
            "Relevant info:",
            "This might help:",
            "Here are some documents:",
            "Relevant documents:",
            "Some relevant documents:",
            "Additional Resources:",
            "Supporting Information:",
            "Explore Further:",
            "Supplementary Docs:",
            "Related Documentation:",
            "In-Depth Details:",
            "Reference Material:",
            "For Your Reference:",
            "Background Information:",
            "Dive into Details:",
            "Check Out These Docs:",
            "Detailed Information:",
            "For Your Consideration:",
            "For a Deeper Understanding:"
        ]
        for split in self.data_read:
            for data in tqdm(self.data_read[split], desc=f"Converting {split} data"):
                data_dict = {}
                # Randomly assign generic system prompt to data
                data_dict['system_prompt'] = QA_TEMPLATE().get_generic_system_prompt(random.randint(1, 20)) if bool(random.getrandbits(1)) else ""
                data_dict['qas_id'] = self.id_generator(size=4) + f"_{data['category']}"

                if len(data['context']) != 0:
                    doc_prefix = random.choice(docs_prefix)
                    data_dict['question_text'] = data['instruction'] + "\n " + f" {doc_prefix} \n" + data['context']
                else:
                    data_dict['question_text'] = data['instruction']

                data_dict['orig_answer_texts'] = data['response']
                data_dict['answer_lengths'] = None
                data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    databricks_dolly_parser = DataBricksDolly15k(r"src/data/features/final_storge_converted/databricks-dolly-15k/dummy.txt",
                                                 r"src/data/features/final_storge_converted/databricks-dolly-15k")
    databricks_dolly_parser.read()
    databricks_dolly_parser.convert()
    databricks_dolly_parser.save
