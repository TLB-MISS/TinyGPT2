import torch
import os
from transformers import GPT2LMHeadModel, GPT2TokenizerFast, TinyGPT2Model, GPT2Config

tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

model_t = GPT2LMHeadModel.from_pretrained("gpt2")
student_model_config = GPT2Config.from_json_file(os.path.join("student_head_config", "config.json"))
model_s = TinyGPT2Model.from_scratch(student_model_config, fit_size=768)

input_ids = tokenizer.encode('I enjoy walking with my cute dog', return_tensors='pt')
greedy_output_t = model_t.generate(input_ids, max_length=50)
greedy_output_s = model_s.generate(input_ids, max_length=50)

print("GPT2 Output:\n")
print(tokenizer.decode(greedy_output_t[0], skip_special_tokens=True))

print("\n\nTinyGPT2 Output:\n")
print(tokenizer.decode(greedy_output_s[0], skip_special_tokens=True))
