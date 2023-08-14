import json
import math
import random
import sys
sys.path.insert(0, r'./')

import datasets
import torch

from typing import Optional, Dict, List, Union, Set
from os.path import join
from datasets import DatasetDict, load_dataset, concatenate_datasets
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from configs import AdvanceQAExample


class AdvanceQa(Dataset):
    def __init__(self, json_file_paths: List[str], num_examples):
        num_examples_each = math.floor(num_examples/len(json_file_paths))
        self.full_json_data = []
        for json_path in json_file_paths:
            with open(json_path, encoding='utf-8') as jfile:
                json_data = json.load(jfile)
                self.full_json_data += json_data[:num_examples_each]

    def __len__(self) -> int:
        return len(self.full_json_data)

    def __getitem__(self, idx):
        try:
            advance_qapair = AdvanceQAExample(**self.full_json_data[idx])
        except KeyError:
            raise f"Missing keys to fill for {self.full_json_data[idx]} in item {idx}"

        return advance_qapair


class QADataloader:
    def __init__(self,
                 model_name: str,
                 text_column: str,
                 target_column: str,
                 train_file: Union[str, List[str]],
                 val_file: Optional[Union[str, List[str]]]=None,
                 test_file: Optional[Union[str, List[str]]]=None,
                 batch_size: int = 8,
                 seed: int = 42,
                 translate: bool=False,
                 use_fast_tokenizer: bool=True,
                 max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None,
                 max_predict_samples: Optional[int] = None
                 ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.text_column = text_column
        self.target_column = target_column
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size
        self.translate = translate

        self.seed = seed
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples

    def __call__(self, *args, **kwargs) -> Union[Set[DataLoader],Set]:
        dataloaders = {}
        if self.train_file is not None:
            print('\nLoading train datasets' + '.' * 10)
            train_dataset = self.load_data('train', self.train_file)
            if self.max_train_samples is not None:
                train_dataset = train_dataset.select(range(self.max_train_samples))
            dataloaders['train'] = self.get_dataloader(train_dataset, shuffle_flag=True)

        if self.val_file is not None:
            print('\nLoading validation datasets' + '.' * 10)
            eval_dataset = self.load_data('val', self.val_file)
            if self.max_eval_samples is not None:
                eval_dataset = eval_dataset.select(range(self.max_eval_samples))
            dataloaders['eval'] = self.get_dataloader(eval_dataset)

        if self.test_file is not None:
            print('\nLoading test datasets' + '.' * 10)
            test_dataset = self.load_data('test', self.test_file)
            if self.max_predict_samples is not None:
                test_dataset = test_dataset.select(range(self.max_predict_samples))
            dataloaders['test'] = self.get_dataloader(test_dataset)

        return dataloaders

    def load_data(self, key: str, data_file: List[str]) -> DatasetDict:
        """
        Loads a dataset from a file on disk and returns it as a dictionary of Dataset objects.

        Args:
            key (str): The key to assign to the loaded dataset in the returned dictionary of Dataset objects.
            data_file (Union[str, List[str]]): The path or paths to the data file(s) to load. If multiple is True, data_file
                                                should be a list of file paths. Otherwise, it should be a single file path.
            mutiple (bool): A flag that indicates whether the data_file argument is a list of multiple file paths.

        Returns:
            A dictionary of Dataset objects that represents the loaded dataset. If mutiple is True, the function
            concatenates the datasets from the multiple files before returning them. Otherwise, it returns a single
            dataset loaded from the data_file path.
        """

        dataset_list = []
        for file in data_file:
            data_files = {key: file}
            extension = file.split(".")[-1]
            dataset_list.append(load_dataset(extension, data_files=data_files, split=key))
        dataset = concatenate_datasets(dataset_list)
        dataset.shuffle(self.seed)
        return dataset

    def dynamic_collate(self, batch):
        """
        A collate function that tokenizes the inputs and targets, and applies dynamic padding and truncation
        based on the maximum length in the batch.

        Args:
            batch (list): A list of examples, where each example is a dictionary with a text column and a target column.

        Returns:
            dict: A dictionary with the input IDs, attention masks, and target IDs with attention masks where tokens are padded,
            and the target IDs are masked to exclude padded values.
        """

        inputs = [example[self.text_column] for example in batch]
        targets = [example[self.target_column] for example in batch]

        if self.translate:
            inputs = translate_en2vi(inputs, self.device)
            targets = translate_en2vi(targets, self.device)

        print(inputs)
        print(targets)

        inp_tokens = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        tgt_tokens = self.tokenizer.batch_encode_plus(
            targets,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        target_ids = tgt_tokens["input_ids"]
        target_mask = tgt_tokens["attention_mask"].bool()
        target_ids = target_ids.masked_fill(~target_mask, -100)

        return {"input_ids": inp_tokens["input_ids"],
                "attention_mask": inp_tokens["attention_mask"],
                "labels": target_ids}

    def get_dataloader(self, dataset, shuffle_flag: bool = False) -> DataLoader:
        """
        :param dataset: (Dataset): dataset from which to load the data.
        :param shuffle_flag: set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
        :return: a dataset
        """
        sampler = RandomSampler(data_source=dataset,generator=self.generator) if shuffle_flag else SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler= sampler,
                                collate_fn=self.dynamic_collate,
                                batch_size=self.batch_size,
                                drop_last=True
                                )

        return dataloader


if __name__ == "__main__":
    # dataloader_args = {
    #     "model_name": "google/flan-t5-base",
    #     "text_column": "prompt",
    #     "target_column": "response",
    #     "train_file": [r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\storage\dummy\dummy.json"],
    #     "batch_size": 2,
    #     "seed": 42,
    #     "translate": True,
    #     "max_train_samples": 10
    # }
    #
    # dataloaders = StateDataloader(**dataloader_args)
    # dataloaders = dataloaders.__call__()
    #
    # for batch in iter(dataloaders['train']):
    #     print(batch)
    idx = random.randint(0, 2000)
    qa_dataset = AdvanceQa(json_file_paths=[r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\DPRMarxism_QA\src\data\features\final_storge_converted\Train-v2.0\train-v2.0_converted.json"],
                           num_examples=2000)
    print(qa_dataset[idx])
    print(qa_dataset[idx].get_dict)
    print(qa_dataset[idx].get_dict_str)
    print(qa_dataset[idx].get_example(is_training=True))

