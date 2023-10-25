import json
import os.path
import sys
from typing import Union, List
sys.path.insert(0, r'./')


def reformat_data(data_paths: List[str], added_string: str="Formated"):
    "Format json data to supported data type for pyarrow"

    for file in data_paths:
        assert os.path.isfile(file), f"Please provide the correct path, No path exist for {file}"
        with open(file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            file_name = os.path.basename(file).split(".")[0]
            formated_file = file.replace(file_name+".", file_name+added_string+".")
            with open(formated_file, "w", encoding="utf-8") as f:
                for item in json_data:
                    json.dump(item, f, ensure_ascii=False)
                    f.write("\n")
        print(f"Finished converted {file}")


if __name__=="__main__":
    reformat_data([r"src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k.json",
                   r"src/data/features/final_storge_converted/databricks-dolly-15k/databricks_dolly15k_translated.json"])