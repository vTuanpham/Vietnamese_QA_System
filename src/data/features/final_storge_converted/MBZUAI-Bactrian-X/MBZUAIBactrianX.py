import random
import sys
sys.path.insert(0,r'./')
from tqdm.auto import tqdm

from datasets import load_dataset

from src.data.features import DataParser
from src.data.configs import AdvanceInstructSample, QA_TEMPLATE


PARSER_TYPE = "BactrianX"


class BactrianXParser(DataParser):
    def __init__(self, file_path: str, output_path: str):
        super().__init__(file_path, output_path,
                         parser_type=PARSER_TYPE)
        self.target_config = AdvanceInstructSample
        self.max_examples = 67017

    def read(self):
        super(BactrianXParser, self).read()
        stream_data = load_dataset("MBZUAI/Bactrian-X", "vi",
                                   streaming=True,
                                   keep_in_memory=False)
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
        super(BactrianXParser, self).convert()
        data_converted = []
        for data in tqdm(self.data_read, desc=f"Converting data"):
            data_dict = {}
            data_dict['system_prompt'] = QA_TEMPLATE().get_generic_system_prompt(random.randint(1, 20)) if bool(random.getrandbits(1)) else ""
            data_dict['qas_id'] = data['id']
            data_dict['question_text'] = data['instruction']
            if 'input' in data:
                data_dict['question_text'] += f"\n{data['input']}"

            data_dict['orig_answer_texts'] = data['output']
            data_dict['answer_lengths'] = None
            data_converted.append(data_dict)

        self.converted_data = data_converted

        pass


if __name__ == '__main__':
    bactrian_x_parser = BactrianXParser(r"src/data/features/final_storge_converted/MBZUAI-Bactrian-X/dummy.txt",
                                        r"src/data/features/final_storge_converted/MBZUAI-Bactrian-X")
    bactrian_x_parser.read()
    bactrian_x_parser.convert()
    bactrian_x_parser.save
