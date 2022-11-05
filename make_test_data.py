import csv
from datasets import load_dataset

def create_test_data(task: str, langs: list, huggingface_labels: dict):
    """ Creates test data using huggingface
    Args:
        task (str): dataset/task to be downloaded
        langs (list): list of all the langs for which the dataset needs to be loaded
        huggingface_labels (dict): dictionary of labels from int to string

    Returns:
        None
    """
    for lang in langs:
        dataset = load_dataset(task, lang)
        file_name = f"./download/{task}/test-{lang}.tsv"
        with open(file_name, "w", encoding="utf-8-sig") as f:
            writer = csv.writer(f, delimiter="\t")
            for data in dataset['test']:
                line = list(data.values())
                line[2] = huggingface_labels[line[2]]
                writer.writerow(line)

def main():
    langs = ['ar', 'bg', 'de', 'el', 'en', 'es', 'fr', 'hi', 'ru', 'sw', 'th', 'tr', 'ur', 'vi', 'zh']
    task = "xnli"
    huggingface_labels = {
        0: "entailment",
        1: "neutral",
        2: "contradiction"
    }
    create_test_data(task, langs, huggingface_labels)

if __name__ == '__main__':
    main()

