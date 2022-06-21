from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TinyGPT2Model
import torch
from tqdm import tqdm
import argparse
from pathlib import Path
import json

def main():
    parser = argparse.ArgumentParser()


    parser.add_argument("--target_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--tokenizer_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--task_name",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--task_dir",
                        default=None,
                        type=Path)
    parser.add_argument("--max_seq_len",
                        default=128,
                        type=int)
    args = parser.parse_args()

    device = "cuda"
    if 'gpt2' in args.target_model:
        model = GPT2LMHeadModel.from_pretrained(args.target_model).to(device)
    else:
        model = TinyGPT2Model.from_pretrained(args.target_model, fit_size=768).to(device)
    tokenizer = GPT2TokenizerFast.from_pretrained(args.tokenizer_model)

    if args.task_name == "wiki":
        test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    elif args.task_name == "lambada":
        test = load_dataset("lambada", split="test")
        encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")
    elif args.task_name == "ptb":
        test = load_dataset("ptb_text_only", split="test")
        encodings = tokenizer("\n\n".join(test["sentence"]), return_tensors="pt")

    if args.task_name == "webtext":
        if args.task_dir is None:
            raise ValueError("Need path of webtext test directory. Use args.task_dir")
        test_file = args.task_dir / "test_dataset.json"
        input_ids = []
        with test_file.open() as f:
            for i, line in enumerate(tqdm(f, desc="Testing examples")):
                line = line.strip()
                example = json.loads(line)
                input_ids.extend(example['input_ids'])
                
        input_ids = torch.tensor(input_ids).reshape((1,-1))
        max_length = args.max_seq_len
    else:
        max_length = model.config.n_positions
    stride = max_length

    nlls = []
    if args.task_name != "webtext":
        for i in tqdm(range(0, encodings.input_ids.size(1), stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, encodings.input_ids.size(1))
            trg_len = end_loc - i  # may be different from stride on last loop
            input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
            target_ids = input_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(input_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len
            nlls.append(neg_log_likelihood)
    else:
        for i in tqdm(range(0, input_ids.shape[1], stride)):
            begin_loc = max(i + stride - max_length, 0)
            end_loc = min(i + stride, input_ids.shape[1])
            trg_len = end_loc - i
            sample_ids = input_ids[:, begin_loc:end_loc].to(device)
            target_ids = sample_ids.clone()
            target_ids[:, :-trg_len] = -100

            with torch.no_grad():
                outputs = model(sample_ids, labels=target_ids)
                neg_log_likelihood = outputs[0] * trg_len    
            nlls.append(neg_log_likelihood)

    ppl = torch.exp(torch.stack(nlls).sum() / end_loc)
    print(ppl.item())

if __name__ == "__main__":
    main()
