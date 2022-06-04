import torch
from argparse import ArgumentParser
from pathlib import Path
import json
from tqdm import tqdm
import gc

def main():
	parser = ArgumentParser()
	parser.add_argument("--output_dir", type=Path, required=True)
	parser.add_argument("--dataset_dir", type=Path, required=True)
	args = parser.parse_args()

	train_dataset_filename = args.dataset_dir / "train_dataset.json"
	test_dataset_filename = args.dataset_dir / "test_dataset.json"
	epoch_train_filename = args.output_dir / "train_dataset.json"
	epoch_test_filename = args.output_dir / "test_dataset.json"
	metrics_train_filename = args.output_dir / "train_metrics.json"
	metrics_test_filename = args.output_dir / "test_metrics.json"
	train_shard = 2000000 # for reduce memory consumption
	test_shard = 2000000
	max_seq_len = 1024
	num_expectation_train_instances = 0
	num_expectation_test_instances = 0
	num_train_instances = 0
	num_test_instances = 0

	# make train dataset
	input_ids = []
	attention_mask = []
	with train_dataset_filename.open('r') as f:
		num_line = 0
		for line in tqdm(f, desc="Loading Train Dataset", unit=" lines"):
			example = json.loads(line)
			input_ids.extend(example['input_ids'])
			attention_mask.extend(example['attention_mask'])
			num_line += 1

			if num_line % train_shard == 0:
				print("len of input_ids:", len(input_ids), "  len of attention_mask:", len(attention_mask))
				assert len(input_ids) == len(attention_mask)
				train_shard_seq_len = len(input_ids)
				train_dataset = {'input_ids': input_ids, 'attention_mask': attention_mask}
				train_dataset = [{k: t[i : i + max_seq_len] for k, t in train_dataset.items()} for i in range(0, train_shard_seq_len, max_seq_len)]
				if sum([len(train_dataset[-1][k])!=max_seq_len for k in train_dataset[-1].keys()]) > 0:
					input_ids = train_dataset[-1]['input_ids']
					attention_mask = train_dataset[-1]['attention_mask']
					train_dataset = train_dataset[0:-1]
				else:
					input_ids = []
					attention_mask = []
				with epoch_train_filename.open('a') as f:
					for example in train_dataset:
						f.write(json.dumps(example) + '\n')
						num_train_instances += 1
				num_expectation_train_instances += train_shard_seq_len // max_seq_len
				print("expectation of train instances:", num_expectation_train_instances, "  real train instances:", num_train_instances)
		# last shard
		print("len of input_ids:", len(input_ids), "  len of attention_mask:", len(attention_mask))
		assert len(input_ids) == len(attention_mask)
		train_shard_seq_len = len(input_ids)
		if train_shard_seq_len >= max_seq_len:
			train_dataset = {'input_ids': input_ids, 'attention_mask': attention_mask}
			train_dataset = [{k: t[i : i + max_seq_len] for k, t in train_dataset.items()} for i in range(0, train_shard_seq_len, max_seq_len)]
			if sum([len(train_dataset[-1][k])!=max_seq_len for k in train_dataset[-1].keys()]) > 0:
				train_dataset = train_dataset[0:-1]
			with epoch_train_filename.open('a') as f:
				for example in train_dataset:
					f.write(json.dumps(example) + '\n')
					num_train_instances += 1
			num_expectation_train_instances += train_shard_seq_len // max_seq_len
			print("expectation of train instances:", num_expectation_train_instances, "  real train instances:", num_train_instances)

	with metrics_train_filename.open('w') as f:
		metrics = {
			"num_examples": num_train_instances,
			"max_seq_len": max_seq_len
		}
		f.write(json.dumps(metrics))
	
	del(train_dataset)
	gc.collect()
	
	# make test dataset
	input_ids = []
	attention_mask = []
	with test_dataset_filename.open('r') as f:
		num_line = 0
		for line in tqdm(f, desc="Loading Test Dataset", unit=" lines"):
			example = json.loads(line)
			input_ids.extend(example['input_ids'])
			attention_mask.extend(example['attention_mask'])
			num_line += 1

			if num_line % test_shard == 0:
				print("len of input_ids:", len(input_ids), "  len of attention_mask:", len(attention_mask))
				assert len(input_ids) == len(attention_mask)
				test_shard_seq_len = len(input_ids)
				test_dataset = {'input_ids': input_ids, 'attention_mask': attention_mask}
				test_dataset = [{k: t[i : i + max_seq_len] for k, t in test_dataset.items()} for i in range(0, test_shard_seq_len, max_seq_len)]
				if sum([len(test_dataset[-1][k])!=max_seq_len for k in test_dataset[-1].keys()]) > 0:
					input_ids = test_dataset[-1]['input_ids']
					attention_mask = test_dataset[-1]['attention_mask']
					test_dataset = test_dataset[0:-1]
				else:
					input_ids = []
					attention_mask = []
				with epoch_test_filename.open('a') as f:
					for example in test_dataset:
						f.write(json.dumps(example) + '\n')
						num_test_instances += 1
				num_expectation_test_instances += test_shard_seq_len // max_seq_len
				print("expectation of test instances:", num_expectation_test_instances, "  real test instances:", num_test_instances)
		# last shard
		print("len of input_ids:", len(input_ids), "  len of attention_mask:", len(attention_mask))
		assert len(input_ids) == len(attention_mask)
		test_shard_seq_len = len(input_ids)
		if test_shard_seq_len >= max_seq_len:
			test_dataset = {'input_ids': input_ids, 'attention_mask': attention_mask}
			test_dataset = [{k: t[i : i + max_seq_len] for k, t in test_dataset.items()} for i in range(0, test_shard_seq_len, max_seq_len)]
			if sum([len(test_dataset[-1][k])!=max_seq_len for k in test_dataset[-1].keys()]) > 0:
				test_dataset = test_dataset[0:-1]
			with epoch_test_filename.open('a') as f:
				for example in test_dataset:
					f.write(json.dumps(example) + '\n')
					num_test_instances += 1
			num_expectation_test_instances += test_shard_seq_len // max_seq_len
			print("expectation of test instances:", num_expectation_test_instances, "  real test instances:", num_test_instances)

	with metrics_test_filename.open('w') as f:
		metrics = {
			"num_examples": num_test_instances,
			"max_seq_len": max_seq_len
		}
		f.write(json.dumps(metrics))
	
	

if __name__ == '__main__':
    main()