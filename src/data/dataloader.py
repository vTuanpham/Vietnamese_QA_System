import gc
import os
import math
import random
import warnings
from itertools import chain
import sys
from functools import partialmethod
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
    def __init__(self, json_file_paths: List[str], num_examples, task_type: str,
                 config_type: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample,
                 get_example: bool = True, split: str='train'):
        num_examples_each = math.floor(num_examples/len(json_file_paths))
        assert task_type, "Please specified task type"
        self.task_type = task_type
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
                            config_data = self.config_type(**data).get_example(is_training=True,
                                                                               task_type=self.task_type)
                        except KeyError:
                            raise f"Missing keys to fill for {config_data} in item {idx} in {file_name}"
                    else:
                        config_data = data
                    self.full_json_data.append(config_data)
                print(f"Finished loading from {file_name} with total loaded {len(self.full_json_data)} examples")
                del iterable_json_data,
                gc.collect()
            except IOError as e:
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
                 task_type: str,
                 train_file: Union[str, List[str]],
                 val_file: Optional[Union[str, List[str]]]=None,
                 test_file: Optional[Union[str, List[str]]]=None,
                 train_batch_size: int = 8,
                 eval_batch_size: int=16,
                 test_batch_size: int=16,
                 block_size: int=128,
                 num_worker: int = 1,
                 seed: int = 42,
                 use_fast_tokenizer: bool=True,
                 max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None,
                 max_predict_samples: Optional[int] = None,
                 config_type: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample
                 ) -> None:

        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=use_fast_tokenizer,
                                                       trust_remote_code=True,
                                                       max_model_length=768)
        self.tokenizer.padding_side = "right"  # Fix weird overflow issue with fp16 training
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.text_column = text_column
        self.target_column = target_column
        self.config_type = config_type
        self.task_type= task_type
        self.block_size = block_size
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

        if self.block_size is None:
            block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                warnings.warn(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
            block_size = 1024
        else:
            if self.block_size > self.tokenizer.model_max_length:
                warnings.warn(
                    f"The block_size passed ({self.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            block_size = min(self.block_size, self.tokenizer.model_max_length)

    def __call__(self, *args, **kwargs) -> Union[Set[DataLoader],Set]:
        dataloaders = {}
        self.dataset = {}
        if self.train_file is not None:
            print('\nLoading train datasets' + '.' * 10)
            train_dataset = self.load_data(self.train_file, self.max_train_samples)
            dataloaders['train'] = self.get_dataloader(train_dataset,
                                                       shuffle_flag=True,
                                                       batch_size=self.train_batch_size)
            self.dataset['train'] = train_dataset

        if self.val_file is not None:
            print('\nLoading validation datasets' + '.' * 10)
            eval_dataset = self.load_data(self.val_file, self.max_eval_samples)
            dataloaders['eval'] = self.get_dataloader(eval_dataset,
                                                      batch_size=self.eval_batch_size)
            self.dataset['eval'] = eval_dataset

        if self.test_file is not None:
            print('\nLoading test datasets' + '.' * 10)
            test_dataset = self.load_data(self.test_file, self.max_predict_samples)
            dataloaders['test'] = self.get_dataloader(test_dataset,
                                                      batch_size=self.test_batch_size)
            self.dataset['test'] = test_dataset

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
                                task_type=self.task_type,
                                split=split,
                                get_example=get_example)
        else:
            dataset = AdvanceQa(json_file_paths=data_files,
                                config_type=self.config_type,
                                task_type=self.task_type,
                                split=split,
                                get_example=get_example)

        # Log a few random samples from the training set:
        for index in random.sample(range(len(dataset)), 3):
            print(f"Sample {index} of the training set: {dataset[index]}.")

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

        inp_tokens = self.tokenizer.batch_encode_plus(
            inputs,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )
        if self.task_type == "SEQ_2_SEQ_LM":
            targets = [example[self.target_column] for example in batch]
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
        elif self.task_type == "CAUSAL_LM":
            labels = inp_tokens["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            return {"input_ids": inp_tokens["input_ids"],
                    "labels": labels
                    }
        else:
            raise f"Unsupport task type for {self.task_type}"

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
        "model_name": "gpt2",
        "text_column": "prompt",
        "target_column": "target",
        "train_file": [r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json",
                       # r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json",
                       # r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleanedFormated.json",
                       r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleaned_translatedFormated.json"],
        "train_batch_size": 8,
        "seed": 42,
        "max_train_samples": 450,
        "config_type": AdvanceInstructSample,
        "task_type": "CASUAL_LM"
    }

    idx = random.randint(0, 400)
    qa_dataset = AdvanceQa(json_file_paths=[r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json"],
                           num_examples=400,
                           config_type=AdvanceInstructSample,
                           get_example=False,
                           task_type="CAUSAL_LM")
    # print(qa_dataset[idx])
    # print(qa_dataset[idx].get_dict)
    # print(qa_dataset[idx].get_dict_str)
    print(qa_dataset[idx].get_example(is_training=True, task_type="CAUSAL_LM"))

    qa_dataloader = QADataloader(**dataloader_args)
    qa_dataloader_instance = qa_dataloader.__call__()
    for idx, data in enumerate(iter(qa_dataloader_instance['train'])):
        print("\n"+qa_dataloader.tokenizer.decode(data['input_ids'][0], skip_special_tokens=True))
        labels = data['labels'].cpu().numpy()
        labels = np.where(labels != -100, labels, qa_dataloader.tokenizer.pad_token_id)
        print("\n"+qa_dataloader.tokenizer.decode(labels[0], skip_special_tokens=True))
        if idx == 30: break
