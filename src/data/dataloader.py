import gc
import logging
import os
import math
import random
import warnings
import sys
from itertools import chain

from accelerate import Accelerator

sys.path.insert(0, r'./')

from tqdm.auto import tqdm
from typing import Optional, List, Union, Set, Any, Dict

import numpy as np

import torch
from torch.utils.data import RandomSampler, SequentialSampler
from torch.utils.data.dataloader import DataLoader, Dataset

from datasets import load_dataset
from datasets import Dataset as hfDataset
from transformers import AutoTokenizer, DataCollatorForSeq2Seq, DataCollatorForLanguageModeling, default_data_collator
from trl import DataCollatorForCompletionOnlyLM


from src.data.configs import AdvanceQAExample, AdvanceInstructSample


class AdvanceQa(Dataset):
    def __init__(self, json_file_paths: List[str], task_type: str,
                 config_type: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample,
                 get_example: bool = True, split: str='train', num_examples: int = 100000,
                 do_perplexity_eval: bool = False, do_generative_eval: bool = False,
                 tokenizer: AutoTokenizer = None, max_seq_length: int = 1024):
        num_examples_each = math.floor(num_examples/len(json_file_paths))
        assert task_type, "Please specified task type"
        self.task_type = task_type
        self.full_json_data = []
        self.config_type = config_type
        self.get_example = get_example
        for json_path in tqdm(json_file_paths, desc=f"Loading {split} data"):
            assert os.path.isfile(json_path), f"Invalid data path for {json_path}"
            num_examples_each_file = num_examples_each
            try:
                file_name = os.path.basename(json_path)
                extension = json_path.split(".")[-1]
                print(f"Loading {num_examples_each_file} from {file_name}...")
                iterable_json_data = load_dataset(extension, data_files=json_path,
                                                  streaming=True, keep_in_memory=False)
                for idx, data in enumerate(iter(iterable_json_data['train'])):
                    if idx > num_examples_each_file:
                        break
                    if get_example:
                        try:
                            config_data = self.config_type(**data).get_example(is_training=split == 'train',
                                                                               task_type=self.task_type,
                                                                               do_perplexity_eval=do_perplexity_eval,
                                                                               do_generative_eval=do_generative_eval)
                            # Check if config data exceeds maximum length or not,
                            # This check is for DataCollatorForCompletionOnlyLM
                            if split == 'train' and task_type == 'CAUSAL_LM':
                                config_data_tokenzied = tokenizer(config_data['prompt'])
                                if len(config_data_tokenzied['input_ids']) > max_seq_length:
                                    warnings.warn(f"Example {idx} in {file_name} skipped due to length exceeded "
                                                 f"max seq length")
                                    num_examples_each_file += 1
                                    continue
                        except KeyError as e:
                            raise f"Missing keys to fill for {config_data} in item {idx} in {file_name}" \
                                  f"Error message: {e}"
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
                 accelerator,
                 model_name: str,
                 text_column: str,
                 target_column: str,
                 task_type: str,
                 train_file: Union[str, List[str]],
                 val_file: Optional[Union[str, List[str]]]=None,
                 test_file: Optional[Union[str, List[str]]]=None,
                 train_batch_size: int = 8,
                 generative_eval_batch_size: int=16,
                 max_eval_generative_samples: Optional[int] = None,
                 max_eval_perplexity_samples: Optional[int] = None,
                 perplexity_eval_batch_size: int=6,
                 test_batch_size: int=16,
                 block_size: int=768,
                 model_max_length: int=1024,
                 context_length: int=768,
                 num_worker: int = 1,
                 seed: int = 42,
                 use_fast_tokenizer: bool=True,
                 no_preprocess_data: bool=False,
                 do_perplexity_eval: bool=False,
                 do_generative_eval: bool=False,
                 do_group_texts: bool=False,
                 response_template: str=" %%%%%%% Response:",
                 max_train_samples: Optional[int] = None,
                 max_eval_samples: Optional[int] = None,
                 max_predict_samples: Optional[int] = None,
                 config_type: Union[AdvanceQAExample, AdvanceInstructSample] = AdvanceQAExample
                 ) -> None:

        self.accelerator = accelerator
        self.model_max_length = model_max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name,
                                                       use_fast=use_fast_tokenizer,
                                                       trust_remote_code=True,
                                                       max_model_length=self.model_max_length,
                                                       # GPT-2 is a model with absolute position embeddings so it’s
                                                       # usually advised to pad the inputs on the right rather than the left.
                                                       padding_side="left" if task_type == "CAUSAL_LM" and "gpt2" not in model_name else "right")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        self.text_column = text_column
        self.response_template = response_template
        self.target_column = target_column
        self.config_type = config_type
        self.task_type= task_type
        self.block_size = block_size
        self.context_length = context_length
        self.no_preprocess_data = no_preprocess_data
        self.do_group_texts = do_group_texts
        self.do_perplexity_eval = do_perplexity_eval
        self.do_generative_eval = do_generative_eval
        if no_preprocess_data:
            warnings.warn(f"\n Preprocessing data disable, this may result in vram accumulation overtime"
                          f"Please consider enable if the size of your dataset is smaller than 1000k or your setup"
                          f"have lowram\n")
        self.train_file = train_file
        self.val_file = val_file
        self.test_file = test_file
        self.train_batch_size = train_batch_size
        self.generative_eval_batch_size = generative_eval_batch_size
        self.perplexity_eval_batch_size = perplexity_eval_batch_size
        self.test_batch_size = test_batch_size

        self.seed = seed
        self.num_worker = num_worker
        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

        self.max_train_samples = max_train_samples
        self.max_eval_samples = max_eval_samples
        if not max_eval_generative_samples:
            self.max_eval_generative_samples = max_eval_samples
        else:
            assert max_eval_samples > max_eval_generative_samples, "Max eval generative samples can't be larger than the" \
                                                                   "whole eval dataset"
            self.max_eval_generative_samples = max_eval_generative_samples
        if not max_eval_perplexity_samples:
            self.max_eval_perplexity_samples = max_eval_samples
        else:
            assert max_eval_samples > max_eval_perplexity_samples, "Max eval perplexity samples can't be larger than the" \
                                                                   "whole eval dataset"
            self.max_eval_perplexity_samples = max_eval_perplexity_samples
        self.max_predict_samples = max_predict_samples

        # TODO: Investigate block size string concatenation for efficient training
        if self.block_size is None:
            self.block_size = self.tokenizer.model_max_length
            if block_size > 1024:
                warnings.warn(
                    "The chosen tokenizer supports a `model_max_length` that is longer than the default `block_size` value"
                    " of 1024. If you would like to use a longer `block_size` up to `tokenizer.model_max_length` you can"
                    " override this default with `--block_size xxx`."
                )
            self.block_size = 1024
        else:
            if self.block_size > self.tokenizer.model_max_length:
                warnings.warn(
                    f"The block_size passed ({self.block_size}) is larger than the maximum length for the model"
                    f"({self.tokenizer.model_max_length}). Using block_size={self.tokenizer.model_max_length}."
                )
            self.block_size = min(self.block_size, self.tokenizer.model_max_length)

    def __call__(self, *args, **kwargs) -> Union[Set[DataLoader],Set]:
        dataloaders = {}
        self.dataset = {}
        if self.train_file is not None:
            print('\nLoading train datasets' + '.' * 10)
            train_dataset = self.load_data(self.train_file, self.max_train_samples, split='train')
            dataloaders['train'] = self.get_dataloader(train_dataset if self.no_preprocess_data else self.preprocess_data(train_dataset, split='train'),
                                                       shuffle_flag=True,
                                                       batch_size=self.train_batch_size)
            self.dataset['train'] = train_dataset

        if self.val_file is not None:
            dataloaders['eval'] = {}
            print('\nLoading validation datasets' + '.' * 10)
            eval_dataset = self.load_data(self.val_file,
                                          self.max_eval_samples,
                                          split='eval',
                                          do_perplexity_eval=self.do_perplexity_eval,
                                          do_generative_eval=self.do_generative_eval)
            if self.do_generative_eval or self.task_type == "SEQ_2_SEQ_LM":
                eval_dataset_input = random.sample(list(eval_dataset), self.max_eval_generative_samples)
                eval_dataset_input = eval_dataset_input if self.no_preprocess_data else self.preprocess_data(eval_dataset_input)
                dataloaders['eval']['generative_eval'] = self.get_dataloader(eval_dataset_input,
                                                                             batch_size=self.generative_eval_batch_size)
            if self.do_perplexity_eval and not self.task_type == "SEQ_2_SEQ_LM":
                eval_dataset_input = eval_dataset[:self.max_eval_perplexity_samples]
                eval_dataset_input = eval_dataset_input if self.no_preprocess_data else self.preprocess_data(eval_dataset_input,
                                                                                        perplexity_eval=self.do_perplexity_eval)
                dataloaders['eval']['perplexity_eval'] = self.get_dataloader(eval_dataset_input,
                                                                             batch_size=self.perplexity_eval_batch_size)
            self.dataset['eval'] = eval_dataset

        if self.test_file is not None:
            dataloaders['test'] = {}
            print('\nLoading test datasets' + '.' * 10)
            test_dataset = self.load_data(self.test_file,
                                          self.max_predict_samples,
                                          split='test',
                                          do_perplexity_eval=self.do_perplexity_eval,
                                          do_generative_eval=self.do_generative_eval)

            if self.do_generative_eval or self.task_type == "SEQ_2_SEQ_LM":
                dataloaders['test']['generative_eval'] = self.get_dataloader(test_dataset if self.no_preprocess_data else self.preprocess_data(test_dataset),
                                                          batch_size=self.test_batch_size)
            if self.do_perplexity_eval and not self.task_type == "SEQ_2_SEQ_LM":
                dataloaders['test']['perplexity_eval'] = self.get_dataloader(test_dataset if self.no_preprocess_data else self.preprocess_data(test_dataset,
                                                                                                                                       perplexity_eval=self.do_perplexity_eval),
                                                                     batch_size=self.test_batch_size)

            self.dataset['test'] = test_dataset

        gc.collect()

        return dataloaders

    def load_data(self, data_files: List[str], num_example: int=10000,
                  split: str='train', get_example: bool=True,
                  do_perplexity_eval: bool=False, do_generative_eval: bool=False) -> AdvanceQa:
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
                                get_example=get_example,
                                do_perplexity_eval=do_perplexity_eval,
                                do_generative_eval= do_generative_eval,
                                tokenizer=self.tokenizer,
                                max_seq_length=self.model_max_length
                                )
        else:
            dataset = AdvanceQa(json_file_paths=data_files,
                                config_type=self.config_type,
                                task_type=self.task_type,
                                split=split,
                                get_example=get_example,
                                do_perplexity_eval=do_perplexity_eval,
                                do_generative_eval=do_generative_eval,
                                tokenizer=self.tokenizer,
                                max_seq_length=self.model_max_length
                                )

        # Log a few random samples from the training set:
        for index in random.sample(range(len(dataset)), 3):
            print(f"Sample {index} of the training set: {dataset[index]}.")

        return dataset # Parse from pytorch dataset to hugging face dataset ecosystem

    def preprocess_data(self, dataset, split=None, perplexity_eval: bool=False):
        with self.accelerator.main_process_first():
            # tokenized_dataset = dataset.map(lambda data: self.tokenize_function(data,
            #                                                                     split=split,
            #                                                                     perplexity_eval=perplexity_eval),
            #                                 desc=f"Running tokenizer on dataset",
            #                                 remove_columns=dataset.column_names,
            #                                 batched=True,
            #                                 num_proc=1)
            tokenized_dataset = list(map(lambda data: self.tokenize_function(data, split, perplexity_eval), dataset))

            if self.task_type == "CAUSAL_LM" and self.do_group_texts:
                return tokenized_dataset.map(self.group_texts,
                                             desc=f"Grouping texts in chunks of {self.block_size}",
                                             batched=False,
                                             num_proc=1)

            return tokenized_dataset

    def dynamic_collate(self, batch):
        """
        A collate function that tokenizes the inputs and targets, and applies dynamic padding and truncation
        based on the maximum length in the batch.

        Args:
            batch (list): A list of examples, where each example is a dictionary with a text column and a target column.

        Returns:
            dict: A dictionary with the input IDs, attention masks, and target IDs with attention masks where tokens are
            padded and the target IDs are masked to exclude padded values.
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
                    "attention_mask": inp_tokens["attention_mask"],
                    "labels": labels
                    }
        else:
            raise f"Unsupported task type for {self.task_type}"

    def tokenize_function(self, data: hfDataset, split: str=None,
                          perplexity_eval: bool=False):
        if not perplexity_eval:
            inputs = data[self.text_column]
        elif self.task_type == "CAUSAL_LM":
            inputs = data["perplexity"]
        else:
            warnings.warn(f"Cannot do perplexity eval on {self.task_type}")
            pass

        inp_tokens = self.tokenizer(
            inputs,
            # padding=False,
            return_special_tokens_mask=True,
            truncation="longest_first",
            max_length=self.model_max_length if split == "train" or perplexity_eval else self.context_length,
            # return_overflowing_tokens=True,
            # return_length=True,
        )

        if self.task_type == "SEQ_2_SEQ_LM":
            targets = data[self.target_column]
            tgt_tokens = self.tokenizer(
                targets,
                # padding=True,
                # return_tensors="pt",
                truncation="longest_first",
                return_special_tokens_mask=True,
                max_length=self.model_max_length if split == "train" else self.context_length
            )
            target_ids = tgt_tokens["input_ids"]
            target_mask = tgt_tokens["attention_mask"].bool()
            target_ids = target_ids.masked_fill(~target_mask, -100)

            return {"input_ids": inp_tokens["input_ids"],
                    "attention_mask": inp_tokens["attention_mask"],
                    "labels": target_ids}

        elif self.task_type == "CAUSAL_LM":
            return inp_tokens
        else:
            raise f"Unsupported task type for {self.task_type}"

    # Main data processing function that will concatenate all texts from our dataset and generate chunks of block_size.
    def group_texts(self, examples):
        # Concatenate all texts.
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])
        # We drop the small remainder, and if the total_length < block_size  we exclude this batch and return an empty dict.
        # We could add padding if the model supported it instead of this drop, you can customize this part to your needs.
        total_length = (total_length // self.block_size) * self.block_size
        # Split by chunks of max_len.
        result = {
            k: [t[i : i + self.block_size] for i in range(0, total_length, self.block_size)]
            for k, t in concatenated_examples.items()
        }
        result["labels"] = result["input_ids"].copy()
        return result

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
        :batch_size: The batch size of the dataset
        :return: a dataloder object

        Args:
            batch_size:

        """
        sampler = RandomSampler(data_source=dataset,
                                generator=self.generator) if shuffle_flag else SequentialSampler(dataset)

        if self.no_preprocess_data:
            collate_function = self.dynamic_collate
        elif self.task_type == "CAUSAL_LM":
            collate_function = DataCollatorForCompletionOnlyLM(self.response_template,
                                                               tokenizer=self.tokenizer,
                                                               mlm=False,
                                                               )
            # collate_function = DataCollatorForLanguageModeling(tokenizer=self.tokenizer, mlm=False)
        elif self.task_type == "SEQ_2_SEQ_LM":
            collate_function = DataCollatorForSeq2Seq(self.tokenizer)
        else:
            raise f"Unsupported task type for {self.task_type}"

        self.accelerator.print(f"Collate function {collate_function}")

        dataloader = DataLoader(dataset,
                                sampler=sampler,
                                collate_fn=collate_function,
                                batch_size=batch_size,
                                drop_last=False, # Keep this false for no model print evaluation mismatch
                                pin_memory=torch.cuda.is_available(),
                                worker_init_fn=self.seed_worker,
                                )

        return dataloader


if __name__ == "__main__":
    dataloader_args = {
        "accelerator": Accelerator(),
        "model_name": "EleutherAI/gpt-neo-125m",
        "text_column": "prompt",
        "target_column": "target",
        "train_file": [
                       r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json",
                       r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json",
                       r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleanedFormated.json",
                       r"src/data/features/final_storge_converted/yahma_alpaca-cleaned/AlpacaCleaned_translatedFormated.json"
                        ],
        "val_file": [
            # r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json",
            r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json"
        ],
        "train_batch_size": 8,
        "perplexity_eval_batch_size": 6,
        "generative_eval_batch_size": 8,
        "seed": 42,
        "max_train_samples": 450,
        "max_eval_samples": 200,
        "config_type": AdvanceInstructSample,
        "task_type": "CAUSAL_LM",
        "no_preprocess_data": False,
        "do_group_texts": False,
        "do_perplexity_eval": True,
        "do_generative_eval": True
    }

    # qa_dataset = AdvanceQa(json_file_paths=[
    #                                         r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrca_translatedFormated.json",
    #                                         r"src/data/features/final_storge_converted/Open-Orca_OpenOrca/OpenOrcaFormated.json"],
    #                        num_examples=5000,
    #                        config_type=AdvanceInstructSample,
    #                        get_example=False,
    #                        task_type="CAUSAL_LM")
    # print(qa_dataset[idx])
    # print(qa_dataset[idx].get_dict)
    # print(qa_dataset[idx].get_dict_str)
    # for i in range(0, 20):
    #     idx = random.randint(0, 5000)
    #     # prompt = qa_dataset[idx].get_example(is_training=True, task_type="CAUSAL_LM")
    #     # if len(prompt['prompt']) < 768 and len(prompt['prompt']) > 512:
    #     #     print(prompt)
    #     prompt = qa_dataset[idx]
    #     print(prompt)

    qa_dataloader = QADataloader(**dataloader_args)
    qa_dataloader_instance = qa_dataloader.__call__()
    print(qa_dataloader.dataset)

    for idx, data in enumerate(iter(qa_dataloader_instance['eval']['generative_eval'])):
        print("\n"+qa_dataloader.tokenizer.decode(data['input_ids'][0], skip_special_tokens=True))
        labels = data['labels'].cpu().numpy()
        labels = np.where(labels != -100, labels, qa_dataloader.tokenizer.pad_token_id)
        print("\n"+qa_dataloader.tokenizer.decode(labels[0], skip_special_tokens=True))
        if idx == 10: break

    for idx, data in enumerate(iter(qa_dataloader_instance['eval']['perplexity_eval'])):
        print("\n"+qa_dataloader.tokenizer.decode(data['input_ids'][0], skip_special_tokens=True))
        labels = data['labels'].cpu().numpy()
        labels = np.where(labels != -100, labels, qa_dataloader.tokenizer.pad_token_id)
        print("\n"+qa_dataloader.tokenizer.decode(labels[0], skip_special_tokens=True))
        if idx == 2: break
