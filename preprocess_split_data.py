"""
Preprocess the openwebtext dataset from HuggingFace's dataset library
"""
import torch
from datasets import load_from_disk
from transformers import GPT2TokenizerFast
from argparse import ArgumentParser
from pathlib import Path
import json

def main():
	parser = ArgumentParser()
	parser.add_argument("--output_dir", type=Path, required=True)
	parser.add_argument("--dataset_dir", type=str, required=True)
	args = parser.parse_args()

	# download the unprocessed dataset
	dataset = load_from_disk(args.dataset_dir)
	tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

	# Process the dataset and split it into train and test subsets
	dataset = dataset.map(lambda e: tokenizer(e['text']))
	dataset = dataset['train']
	print(dataset)

	dataset = dataset.remove_columns('text')
	shuffled_dataset = dataset.shuffle(seed=42)
	print(shuffled_dataset)
	dataset=shuffled_dataset.train_test_split(test_size=0.05)
	print(dataset)

	train_dataset=dataset['train']
	test_dataset=dataset['test']

	print(train_dataset)
	print(test_dataset)

	# metric info
	num_train_examples = len(train_dataset)
	num_test_examples = len(test_dataset)
	max_seq_len = 1024

	# Write the processed dataset into files
	# Specify your own path to save the files
	#test_path = "/home/ubuntu/openwebtext_seq_512_no_pad_filtered/val"

	'''
	num_shards=64
	for i in range(0, num_shards):
    	shard_test=test_dataset.shard(num_shards=num_shards, index=i)
    	name=f"{test_path}/test_dataset_512_filtered_{i}"
    	print(name)
    	print(shard_test)
    	shard_test.to_json(f"{name}.json", orient="records", lines=True)
	'''
	'''
	num_shards=512
	for i in range(0, num_shards):
	    name=f"{corpus_path}/train_dataset_512_filtered_{i}"
	    print(name)
	    shard=train_dataset.shard(num_shards=num_shards, index=i)
	    print(shard)
	    shard.to_json(f"{name}.json", orient="records", lines=True)
	'''
	# no shard
	corpus_path_train = f"{str(args.output_dir)}/train_dataset.json"
	corpus_path_test = f"{str(args.output_dir)}/test_dataset.json"
	train_dataset.to_json(corpus_path_train, orient="records", lines=True)
	test_dataset.to_json(corpus_path_test, orient="records", lines=True)
	
	metrics_train_filename = args.output_dir / "train_metrics.json"
	with metrics_train_filename.open('w') as metrics_train_file:
		metrics = {
			"num_examples": num_train_examples,
			"max_seq_len": max_seq_len
		}
		metrics_train_file.write(json.dumps(metrics))
	metrics_test_filename = args.output_dir / "test_metrics.json"
	with metrics_test_filename.open('w') as metrics_test_file:
		metrics = {
			"num_examples": num_test_examples,
			"max_seq_len": max_seq_len
		}
		metrics_test_file.write(json.dumps(metrics))

if __name__ == '__main__':
    main()