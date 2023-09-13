import gc
import os
import math
import random
import sys
import copy
sys.path.insert(0, r'./')

from tqdm.auto import tqdm
from typing import Optional, List, Union, Set

import numpy as np

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader, Dataset

from datasets import load_dataset
from transformers import AutoTokenizer

from src.data.configs import AdvanceQAExample, AdvanceInstructSample


class AdvanceQa(Dataset):
    def __init__(self, json_file_paths: List[str], num_examples,
                 config_type: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample,
                 get_example: bool = True, split: str='train'):
        num_examples_each = math.floor(num_examples/len(json_file_paths))
        self.full_json_data = []
        self.config_type = config_type
        self.get_example = get_example
        for json_path in tqdm(json_file_paths, desc=f"Loading {split} data"):
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
                    if get_example:
                        try:
                            config_data = self.config_type(**data).get_example(is_training=True)
                        except KeyError:
                            raise f"Missing keys to fill for {config_data} in item {idx} in {file_name}"
                    else:
                        config_data = data
                    self.full_json_data.append(config_data)
                print(f"Finished loading from {file_name} with total loaded {len(self.full_json_data)} examples")
                del iterable_json_data,
                gc.collect()
            except Exception as e:
                raise f"An error occurred while reading the data: {e}"

    def __len__(self) -> int:
        return len(self.full_json_data)

    def __getitem__(self, idx):

        if not self.get_example:
            try:
                config_data = self.config_type(**self.full_json_data[idx])
            except KeyError:
                raise f"Missing keys to fill for {config_data} in item {idx}"
            return config_data
        return self.full_json_data[idx]


class QADataloader:
    def __init__(self,
                 model_name: str,
                 text_column: str,
                 target_column: str,
                 train_file: Union[str, List[str]],
                 val_file: Optional[Union[str, List[str]]]=None,
                 test_file: Optional[Union[str, List[str]]]=None,
                 train_batch_size: int = 8,
                 eval_batch_size: int=16,
                 test_batch_size: int=16,
                 num_worker: int = 1,
                 seed: int = 42,
                 use_fast_tokenizer: bool=True,
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
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.test_batch_size = test_batch_size

        self.seed = seed
        self.num_worker = num_worker
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        self.max_predict_samples = max_predict_samples

    def __call__(self, *args, **kwargs) -> Union[Set[DataLoader],Set]:
        dataloaders = {}
        if self.train_file is not None:
            print('\nLoading train datasets' + '.' * 10)
            train_dataset = self.load_data(self.train_file, self.max_train_samples)
            dataloaders['train'] = self.get_dataloader(train_dataset,
                                                       shuffle_flag=True,
                                                       batch_size=self.train_batch_size)

        if self.val_file is not None:
            print('\nLoading validation datasets' + '.' * 10)
            eval_dataset = self.load_data(self.val_file, self.max_eval_samples)
            dataloaders['eval'] = self.get_dataloader(eval_dataset,
                                                      batch_size=self.eval_batch_size)

        if self.test_file is not None:
            print('\nLoading test datasets' + '.' * 10)
            test_dataset = self.load_data(self.test_file, self.max_predict_samples)
            dataloaders['test'] = self.get_dataloader(test_dataset,
                                                      batch_size=self.test_batch_size)

        self.dataset = {'train': train_dataset, 'eval': eval_dataset, 'test': test_dataset}

        return dataloaders

    def load_data(self, data_files: List[str],
                  num_example: int=10000,
                  split: str='train',
                  get_example: bool=True) -> AdvanceQa:
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
            dataset = AdvanceQa(json_file_paths=data_files,
                                num_examples=num_example,
                                config_type=self.config_type,
                                split=split,
                                get_example=get_example)
        else:
            dataset = AdvanceQa(json_file_paths=data_files,
                                config_type=self.config_type,
                                split=split,
                                get_example=get_example)

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

    def get_dataloader(self, dataset, shuffle_flag: bool = False, batch_size: int=1) -> DataLoader:
        """
        :param dataset: (Dataset): dataset from which to load the data.
        :param shuffle_flag: set to ``True`` to have the data reshuffled
                at every epoch (default: ``False``).
        :return: a dataset
        """
        sampler = RandomSampler(data_source=dataset,
                                generator=self.generator) if shuffle_flag else SequentialSampler(dataset)
        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                collate_fn=self.dynamic_collate,
                                batch_size=batch_size,
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
        "train_file": [r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json",
                       r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json",
                       r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleanedFormated.json",
                       r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleaned_translatedFormated.json"],
        "batch_size": 8,
        "seed": 42,
        "max_train_samples": 450,
        "config_type": AdvanceInstructSample
    }

    idx = random.randint(0, 400)
    qa_dataset = AdvanceQa(json_file_paths=[r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json"],
                           num_examples=400,
                           config_type=AdvanceInstructSample,
                           get_example=False)
    print(qa_dataset[idx])
    print(qa_dataset[idx].get_dict)
    print(qa_dataset[idx].get_dict_str)
    print(qa_dataset[idx].get_example(is_training=True))

    qa_dataloader = QADataloader(**dataloader_args)
    qa_dataloader_instance = qa_dataloader.__call__()
    for idx, data in enumerate(iter(qa_dataloader_instance['train'])):
        print("\n"+qa_dataloader.tokenizer.decode(data['input_ids'][0], skip_special_tokens=True))
        labels = data['labels'].cpu().numpy()
        labels = np.where(labels != -100, labels, qa_dataloader.tokenizer.pad_token_id)
        print("\n"+qa_dataloader.tokenizer.decode(labels[0], skip_special_tokens=True))
        if idx == 30: break
