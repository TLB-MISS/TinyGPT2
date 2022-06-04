from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm

def main():
    parser = ArgumentParser()
    parser.add_argument("--dataset_dir", type=Path, required=True)
    args = parser.parse_args()

    train_dataset_filename = args.dataset_dir / "train_dataset.json"
    test_dataset_filename = args.dataset_dir / "test_dataset.json"
    train_metric_filename = args.dataset_dir / "train_metrics.json"
    test_metric_filename = args.dataset_dir / "test_metrics.json"

    train_metrics = json.loads(train_metric_filename.read_text())
    test_metrics = json.loads(test_metric_filename.read_text())
    num_train_examples = train_metrics['num_examples']
    num_test_examples = test_metrics['num_examples']
    seq_len = train_metrics['max_seq_len']
	
	# check train dataset
    num_train_line = 0
    with train_dataset_filename.open('r') as f:
        for line in tqdm(f, desc="Loading Train Dataset", unit=" lines"):
            example = json.loads(line)
            input_ids = example['input_ids']
            attention_mask = example['attention_mask']
            assert len(input_ids) == seq_len
            assert len(attention_mask) == seq_len
            num_train_line += 1
    assert num_train_line == num_train_examples
    print("Train dataset is valid!")

    # check test dataset
    num_test_line = 0
    with test_dataset_filename.open('r') as f:
        for line in tqdm(f, desc="Loading Test Dataset", unit=" lines"):
            example = json.loads(line)
            input_ids = example['input_ids']
            attention_mask = example['attention_mask']
            assert len(input_ids) == seq_len
            assert len(attention_mask) == seq_len
            num_test_line += 1
    assert num_test_line == num_test_examples
    print("Test dataset is valid!")

if __name__ == '__main__':
    main()