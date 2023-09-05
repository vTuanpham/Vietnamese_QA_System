import json
import sys
import random
sys.path.insert(0,r'./')
import pprint
from pprint import PrettyPrinter
from typing import List, Dict
from dataclasses import dataclass, field, asdict, fields


@dataclass
class AdvanceInstructSample:
    """
    A single training/test example for the Instruct dataset.
    """
    qas_id: str
    system_prompt: str

    question_text: str

    orig_answer_texts: str = None
    answer_lengths: int = None

    # example_template = QA_TEMPLATE()

    def __post_init__(self) -> None:
        # Post validate
        self.answer_lengths = len(self.orig_answer_texts) if self.orig_answer_texts is not None else None

    def __str__(self) -> str:
        return self.__repr__

    @property
    def __repr__(self) -> str:
        s = ""
        s += f"\n Question id: {self.qas_id}"
        s += f"\n System prompt: {self.system_prompt}"
        s += f"\n Question: {self.question_text}"
        if self.orig_answer_texts:
            s += f"\n Answer text: {self.orig_answer_texts}"
            s += f"\n Answer length: {self.answer_lengths}"

        return s

    @property
    def get_dict(self) -> Dict:
        return asdict(self)

    @staticmethod
    def get_keys() -> List[str]:
        all_fields = fields(AdvanceInstructSample)
        return [v.name for v in all_fields]

    @property
    def get_dict_str(self, indent: int=4) -> None:
        pp = pprint.PrettyPrinter(indent=indent)
        pp.pprint(self.get_dict)

    def get_example(self,
                    inputs_column: str="prompt",
                    targets_column: str="target",
                    system_prefix: str="System prompt:",
                    question_prefix: str="Question:",
                    is_training: bool=True
                    ) -> Dict:

        system_msg = system_prefix + ' ' + self.system_prompt
        question_msg = question_prefix + ' ' + self.question_text
        prompt = system_msg + ' ' + question_msg
        label = self.orig_answer_texts

        return {inputs_column: prompt,
                targets_column: label}


if __name__ == "__main__":
    python_question_contexts = [
        "To comment a single line in Python, you use the '#' symbol. Anything after the '#' on the same line is considered a comment and is ignored by the Python interpreter.",
        "Python 2 and Python 3 are two different versions of the Python programming language. Python 3 introduced several backward-incompatible changes to the language to improve its design and fix inconsistencies.",
        "To generate a random number in Python, you can use the 'random' module. Import the module and then use functions like 'random.random()' or 'random.randint()' depending on your requirements.",
        "The 'if __name__ == '__main__':' statement is used to determine whether the Python script is being run as the main program or if it is being imported as a module into another program.",
        "To open and read a file in Python, you can use the 'open()' built-in function in combination with various file modes. After reading, close the file using the 'close()' method of the file object.",
        "The 'range()' function generates a sequence of numbers in Python. It can be used in 'for' loops or to create lists of numbers within a specified range.",
        "Decorators in Python are a powerful way to modify or extend the behavior of functions or methods without changing their actual code. They use the '@' symbol and can be used to add functionality like logging, caching, etc.",
        "Exception handling in Python is done using 'try', 'except', 'else', and 'finally' blocks. It allows you to gracefully handle errors and exceptions that may occur during program execution.",
        "Lists and tuples are both used to store collections of items in Python, but lists are mutable (can be changed), whereas tuples are immutable (cannot be changed).",
        "A virtual environment in Python is a self-contained directory that contains its own Python interpreter and installed packages. It allows you to work on different projects with different dependencies without conflicts.",
        "To remove duplicates from a list in Python, you can convert it to a 'set' and then back to a 'list', as sets automatically remove duplicate elements.",
        "In Python lists, 'append()' adds an element to the end of the list, 'extend()' appends elements from another iterable, and 'insert()' inserts an element at a specified index.",
        "To iterate over a dictionary in Python, you can use a 'for' loop, which by default, iterates over the keys. You can also use the 'items()' method to loop over both keys and values.",
        "A lambda function in Python is a small anonymous function defined using the 'lambda' keyword. It can have any number of arguments but can only have one expression.",
        "You can find the length of a string in Python using the built-in 'len()' function, which returns the number of characters in the string.",
        "The 'pass' statement in Python is a null operation; it does nothing. It is used as a placeholder where syntactically some code is required, but you want to skip its execution.",
        "In Python, you create a class using the 'class' keyword, and you define methods (functions within the class) to perform actions or return information about the object.",
        "Shallow copy creates a new object, but it references the same elements as the original object. Deep copy creates a completely independent copy of the object and its elements.",
        "To sort a dictionary by its values in Python, you can use the 'sorted()' function with a lambda function as the 'key' argument or utilize the 'collections' module.",
        "The 'zip()' function in Python is used to combine multiple iterables into a single iterable of tuples. Each tuple contains elements from corresponding positions in the input iterables."
    ]

    # print(AdvanceQAExample.straighten_docs(python_question_contexts))

    example8 = AdvanceInstructSample(qas_id="8", question_text="What do cats eat?",
                                    doc_tokens=[],
                                    orig_answer_texts="meat and fish", is_trivial=True)

    print(example8)
    print(example8.get_example(is_training=True))
    print(example8.get_dict)

    example6 = AdvanceInstructSample(qas_id="6", question_text="What is the meaning of existence?",
                                    doc_tokens=["The meaning of existence is uncertain.",
                                                "Throughout history, philosophers and theologians have debated the purpose of life and the nature of existence."],
                                    orig_answer_texts="Dying", is_trivial=False)
    print(example6)
    print(example6.get_example(is_training=True))
    print(example6.get_dict)
    print(example6.get_dict_str)
    print(example6.get_keys())