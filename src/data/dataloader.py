import gc
import json
import os
import math
import random
import sys
import copy
sys.path.insert(0, r'./')
from os.path import join

from tqdm.auto import tqdm
from typing import Optional, Dict, List, Union, Set

import numpy as np

import torch
import torch.distributed as dist
import datasets
from torch.utils.data import RandomSampler, SequentialSampler, DistributedSampler
from torch.utils.data.dataloader import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.configs import AdvanceQAExample, AdvanceInstructSample


class AdvanceQa(Dataset):
    def __init__(self, json_file_paths: List[str], num_examples,
                 config_type: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample,
                 get_example: bool = False):
        num_examples_each = math.floor(num_examples/len(json_file_paths))
        self.full_json_data = []
        self.config_type = config_type
        self.get_example = get_example
        for json_path in json_file_paths:
            assert os.path.isfile(json_path), f"Invalid data path for {json_path}"
            try:
                file_name = os.path.basename(json_path)
                extension = json_path.split(".")[-1]
                print(f"Loading {num_examples_each} from {file_name}...")
                iterable_json_data = load_dataset(extension, data_files=json_path,
                                                  streaming=True, keep_in_memory=False)
                for idx, data in enumerate(iter(iterable_json_data['train'])):
                    if idx > num_examples_each:
                        break
                    self.full_json_data.append(data)
                print(f"Finished loading from {file_name} with total loaded {len(self.full_json_data)} examples")
                del iterable_json_data,
                gc.collect()
            except Exception as e:
                raise f"An error occurred while reading the data: {e}"

    def __len__(self) -> int:
        return len(self.full_json_data)

    def __getitem__(self, idx):
        try:
            advance_qapair = self.config_type(**self.full_json_data[idx])
        except KeyError:
            raise f"Missing keys to fill for {self.full_json_data[idx]} in item {idx}"

        return advance_qapair.get_example(is_training=True) if self.get_example else advance_qapair


class QADataloader:
    def __init__(self,
                 model_name: str,
                 text_column: str,
                 target_column: str,
                 train_file: Union[str, List[str]],
                 val_file: Optional[Union[str, List[str]]]=None,
                 test_file: Optional[Union[str, List[str]]]=None,
                 batch_size: int = 8,
                 num_worker: int = 1,
                 seed: int = 42,
                 use_fast_tokenizer: bool=True,
                 return_dataset: bool=False,
                 max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None,
                 max_predict_samples: Optional[int] = None,
                 config_type: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample
                 ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=use_fast_tokenizer)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.text_column = text_column
        self.target_column = target_column
        self.config_type = config_type
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.batch_size = batch_size

        self.return_dataset = return_dataset
        self.seed = seed
        self.num_worker = num_worker
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples

    def __call__(self, *args, **kwargs) -> Union[Set[DataLoader],Set]:
        dataloaders = {}
        datasets = {}
        if self.train_file is not None:
            print('\nLoading train datasets' + '.' * 10)
            train_dataset = self.load_data(self.train_file, self.max_train_samples)
            dataloaders['train'] = self.get_dataloader(train_dataset, shuffle_flag=True)
            if self.return_dataset:
                datasets['train'] = copy.deepcopy(train_dataset)
                datasets['train'].get_example = True

        if self.val_file is not None:
            print('\nLoading validation datasets' + '.' * 10)
            eval_dataset = self.load_data(self.val_file, self.max_eval_samples)
            dataloaders['eval'] = self.get_dataloader(eval_dataset)
            if self.return_dataset:
                datasets['eval'] = copy.deepcopy(eval_dataset)
                datasets['eval'].get_example = True

        if self.test_file is not None:
            print('\nLoading test datasets' + '.' * 10)
            test_dataset = self.load_data(self.test_file, self.max_predict_samples)
            dataloaders['test'] = self.get_dataloader(test_dataset)
            if self.return_dataset:
                datasets['test'] = copy.deepcopy(test_dataset)
                datasets['test'].get_example = True

        return (dataloaders, datasets) if self.return_dataset else dataloaders

    def load_data(self, data_files: List[str], num_example: int=10000) -> AdvanceQa:
        """
        Loads a dataset from a file on disk and returns it as a dictionary of Dataset objects.

        Args:
            data_file (List[str]): The path or paths to the data file(s) to load. If multiple is True, data_file
                                                should be a list of file paths. Otherwise, it should be a single file path.
            num_example (int): Max number of example in the final dataset that was loaded from each file

        Returns:
            A dataset object loaded from all data_files that was divided equally
        """
        if num_example:
            dataset = AdvanceQa(json_file_paths=data_files, num_examples=num_example, config_type=self.config_type)
        else:
            dataset = AdvanceQa(json_file_paths=data_files, config_type=self.config_type)

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
        if not isinstance(batch[0], dict):
            inputs = [example.get_example(is_training=self.train_file is not None)[self.text_column] for example in batch]
            targets = [example.get_example(is_training=self.train_file is not None)[self.target_column] for example in batch]
        else:
            inputs = [example[self.text_column] for example in batch]
            targets = [example[self.target_column] for example in batch]

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

    @staticmethod
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    def get_dataloader(self, dataset, shuffle_flag: bool = False) -> DataLoader:
        """
        :param dataset: (Dataset): dataset from which to load the data.
        :param shuffle_flag: set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
        :return: a dataset
        """
        sampler = RandomSampler(data_source=dataset, generator=self.generator) if shuffle_flag else SequentialSampler(dataset)
        # if torch.cuda.is_available():
        #     sampler = DistributedSampler(dataset=dataset, shuffle=shuffle_flag, seed=self.seed)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                collate_fn=self.dynamic_collate,
                                batch_size=self.batch_size,
                                drop_last=False,
                                pin_memory=torch.cuda.is_available(),
                                worker_init_fn=self.seed_worker,
                                )

        return dataloader


if __name__ == "__main__":
    dataloader_args = {
        "model_name": "google/flan-t5-small",
        "text_column": "prompt",
        "target_column": "target",
        "train_file": [r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\Open-Orca_OpenOrca\OpenOrca_translated.json",
                       r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\Open-Orca_OpenOrca\OpenOrca.json",
                       r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\yahma_alpaca-cleaned\AlpacaCleaned.json",
                       r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\yahma_alpaca-cleaned\AlpacaCleaned_translated.json"],
        "batch_size": 8,
        "seed": 42,
        "max_train_samples": 450,
        "config_type": AdvanceInstructSample
    }

    idx = random.randint(0, 400)
    qa_dataset = AdvanceQa(json_file_paths=[r"C:\Users\Tuan Pham\Desktop\Study\SelfStudy\venv2\Vietnamese_QA_System\src\data\features\final_storge_converted\Open-Orca_OpenOrca\OpenOrca_translatedFormated.json"],
                           num_examples=400,
                           config_type=AdvanceInstructSample)
    # print(qa_dataset[idx])
    # print(qa_dataset[idx].get_dict)
    # print(qa_dataset[idx].get_dict_str)
    # print(qa_dataset[idx].get_example(is_training=True))

    qa_dataloader = QADataloader(**dataloader_args)
    qa_dataloader_instance = qa_dataloader.__call__()
    for idx, data in enumerate(iter(qa_dataloader_instance['train'])):
        # print("\n"+qa_dataloader.tokenizer.decode(data['input_ids'][0], skip_special_tokens=True))
        labels = data['labels'].cpu().numpy()
        labels = np.where(labels != -100, labels, qa_dataloader.tokenizer.pad_token_id)
        # print("\n"+qa_dataloader.tokenizer.decode(labels[0], skip_special_tokens=True))
        if idx == 100: break
