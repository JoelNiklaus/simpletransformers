import os
import re
from pathlib import Path

from datasets import load_dataset


def prepare_data(base_folder, overwrite_cache=False, debug=False):
    folder = os.path.join(base_folder, "data")
    if Path(folder).exists() and not overwrite_cache:
        print("data folder already exists. Set the flag overwrite_cache to True to download the data again.")
        return

    Path(folder).mkdir(parents=True, exist_ok=True)

    prepare_cuad(debug, folder)
    prepare_posture50k(debug, folder)


def prepare_posture50k(debug, folder):
    splits = ["train", "dev", "test"]
    posture50k = load_dataset('json', data_files={split: f"TRProceduralPosture50K/opinion_{split}.txt"
                                                  for split in splits})
    for split in splits:
        rows = posture50k[split]['sections']
        if debug:
            rows = rows[:100]
        output_file = "test" if split == "test" else "train"  # save everything else to train
        with open(f"data/{output_file}.txt", 'a+') as output:
            for row in rows:
                text = ""
                for section in row:
                    text += section["headtext"] + "\n".join(section["paragraphs"]) + "\n"
                    text = re.sub(r'\n+', '\n', text).strip()
                    output.write(str(text) + '\n')
        print(f"Saved the data to the folder {folder}")


def prepare_cuad(debug, folder):
    dataset = load_dataset("cuad")
    for split in ['train', 'test']:
        text = dataset[split]['context']
        if debug:
            text = text[:100]
        with open(f"data/{split}.txt", 'w+') as output:
            for row in text:
                row = re.sub(r'\n+', '\n', row).strip()
                output.write(str(row) + '\n')
        print(f"Saved the data to the folder {folder}")
