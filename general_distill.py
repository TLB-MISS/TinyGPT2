from __future__ import absolute_import, division, print_function

import argparse
import logging
import os
import random
import sys
import json
import datetime
import shutil

import numpy as np
import torch
from collections import namedtuple
from pathlib import Path
from torch.utils.data import (DataLoader, RandomSampler,Dataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange
from torch.nn import MSELoss

from transformers import (
    CONFIG_NAME,
    WEIGHTS_NAME,
    GPT2Model,
    GPT2LMHeadModel,
    TinyGPT2Model,
    GPT2Config,
    set_seed,
)
from learning_rates import AnnealingLR

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

InputFeatures = namedtuple("InputFeatures", "input_ids attention_mask")

def convert_example_to_features(example, max_seq_length):
    input_ids = example["input_ids"]
    attention_mask = example["attention_mask"]

    if len(input_ids) != len(attention_mask):
        logger.info('input_ids: {}\nattention_mask: {}'.format(input_ids, attention_mask))
        attention_mask = [1] * len(input_ids)
    
    if len(input_ids) > max_seq_length:
        logger.info('len(tokens): {}'.format(len(input_ids)))
        logger.info('max_seq_length: {}'.format(max_seq_length))
        input_ids = input_ids[:max_seq_length]
        attention_mask = attention_mask[:max_seq_length]

    input_array = np.zeros(max_seq_length, dtype=np.int32)
    input_array[:len(input_ids)] = input_ids

    attention_array = np.zeros(max_seq_length, dtype=np.bool_)
    attention_array[:len(attention_mask)] = attention_mask

    features = InputFeatures(input_ids=input_array,
                             attention_mask=attention_array)
    return features


class PregeneratedDataset(Dataset):
    def __init__(self, training_path, local_rank, max_seq_len=1024, working_dir=None, reduce_memory=False):
        #self.epoch = epoch
        #self.data_epoch = int(epoch % num_data_epochs)
        logger.info('training_path: {}'.format(training_path))
        data_file = training_path / "train_dataset.json"
        metrics_file = training_path / "train_metrics.json"
        #data_file = training_path / "epoch_{}.json".format(self.data_epoch)
        #metrics_file = training_path / "epoch_{}_metrics.json".format(self.data_epoch)

        logger.info('data_file: {}'.format(data_file))
        logger.info('metrics_file: {}'.format(metrics_file))

        assert data_file.is_file() and metrics_file.is_file()
        metrics = json.loads(metrics_file.read_text())
        num_samples = metrics['num_examples']
        seq_len = max_seq_len
        self.temp_dir = None
        self.working_dir = working_dir
        self.local_rank = local_rank

        if reduce_memory:
            cache_dir = os.path.join(self.working_dir, "cache")
            if self.local_rank in [-1, 0]:
                if not os.path.exists(cache_dir):
                    os.makedirs(cache_dir)
            # barrier until making cache_dir
            if self.local_rank != -1:
                torch.distributed.barrier()
            
            if self.local_rank in [-1, 0]:
                input_ids = np.memmap(filename=Path(cache_dir)/'input_ids.memmap',
                                    mode='w+', dtype=np.int32, shape=(num_samples, seq_len))
                attention_mask = np.memmap(filename=Path(cache_dir)/'attention_mask.memmap',
                                    shape=(num_samples, seq_len), mode='w+', dtype=np.bool_)
            # barrier until local rank 0 process make file
            if self.local_rank != -1:
                torch.distributed.barrier()

            if self.local_rank not in [-1, 0]:
                input_ids = np.memmap(filename=Path(cache_dir)/'input_ids.memmap',
                                    mode='r', dtype=np.int32, shape=(num_samples, seq_len))
                attention_mask = np.memmap(filename=Path(cache_dir)/'attention_mask.memmap',
                                    mode='r', dtype=np.bool_, shape=(num_samples, seq_len))
        else:
            input_ids = np.zeros(shape=(num_samples, seq_len), dtype=np.int32)
            attention_mask = np.zeros(shape=(num_samples, seq_len), dtype=np.bool_)

        # barrier until setting ids for all process
        if self.local_rank != -1:
            torch.distributed.barrier()

        if not reduce_memory or self.local_rank in [-1, 0]:
            #logging.info("Loading training examples for epoch {} in local rank {}".format(epoch, self.local_rank))
            logging.info("Loading training examples for local rank {}".format(self.local_rank))
            with data_file.open() as f:
                for i, line in enumerate(tqdm(f, total=num_samples, desc="Training examples")):
                    line = line.strip()
                    example = json.loads(line)
                    features = convert_example_to_features(example, seq_len)
                    input_ids[i] = features.input_ids
                    attention_mask[i] = features.attention_mask
        
        # barrier until local rank 0 process load data 
        if self.local_rank != -1:
            torch.distributed.barrier()

        self.num_samples = num_samples
        self.seq_len = seq_len
        self.input_ids = input_ids
        self.attention_mask = attention_mask

    def __len__(self):
        return self.num_samples

    def __getitem__(self, item):
        return (torch.tensor(self.input_ids[item].astype(np.int64)),
                torch.tensor(self.attention_mask[item].astype(np.int64)))
    
    # not work
    def __exit__(self, exc_type, exc_val, traceback):
        cache_dir = os.path.join(self.working_dir, "cache")
        if os.path.exists(cache_dir):
            shutil.rmtree(cache_dir)

def get_learning_rate_scheduler(optimizer, args, max_steps):

    # Add linear learning rate scheduler.
    if args.lr_decay_iters is not None:
        num_iters = args.lr_decay_iters
    else:
        num_iters = max_steps
    num_iters = max(1, num_iters)
    init_step = 0
    warmup_iter = int(args.warmup * num_iters)
    plateau_iter = warmup_iter + int(args.plateau * num_iters)
    lr_scheduler = AnnealingLR(
        optimizer=optimizer,
        local_rank=args.local_rank,
        start_lr=args.learning_rate,
        warmup_iter=warmup_iter,
        plateau_iter=plateau_iter,
        total_iters=num_iters,
        decay_style=args.lr_decay_style,
        last_iter=init_step,
    )

    return lr_scheduler

# soft cross-entropy loss between the student network’s logits against the teacher’s logits
# Distilling the Knowledge in a Neural Network(Hinton et al., 2015)
def soft_cross_entropy(predicts, targets):
    student_likelihood = torch.nn.functional.log_softmax(predicts, dim=-1)
    targets_prob = torch.nn.functional.softmax(targets, dim=-1)
    return (- targets_prob * student_likelihood).mean()

def main():
    parser = argparse.ArgumentParser()

    # Directory parameters
    parser.add_argument("--pregenerated_data",
                        type=Path,
                        required=True)
    parser.add_argument("--teacher_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--student_model",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--output_dir",
                        default=None,
                        type=str,
                        required=True)
    parser.add_argument("--working_dir",
                        default='.',
                        type=str)
    
    # lr scheduler parameters
    parser.add_argument("--learning_rate",
                        default=5e-5,
                        type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument("--lr_decay_iters",
                        default=None,
                        type=int,
                        help="number of iterations to decay learning rate over")
    parser.add_argument("--warmup",
                        default=0.01,
                        type=float,
                        help="Proportion of training to perform linear learning rate warmup for. "
                             "E.g., 0.1 = 10%% of training.")
    parser.add_argument("--plateau",
                        default=0.04,
                        type=float,
                        help="Percentage of total iterations to keep at max if using plateau lr")
    parser.add_argument("--lr_decay_style",
                        default="cosine",
                        choices=["constant", "linear", "cosine", "exponential", "plateau"],
                        type=str,
                        help="Learning rate decay function")

    # Other parameters
    parser.add_argument("--max_seq_length",
                        default=1024,
                        type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. \n"
                             "Sequences longer than this will be truncated, and sequences shorter \n"
                             "than this will be padded.")
    parser.add_argument("--reduce_memory",
                        action="store_true",
                        help="Store training data as on-disc memmaps to massively reduce memory usage")
    parser.add_argument("--train_batch_size",
                        default=512,
                        type=int,
                        help="Total batch size for training.")
    parser.add_argument('--weight_decay',
                        '--wd',
                        default=1e-2,
                        type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--temperature',
                        default=1.0,
                        type=float,
                        help='temperature'),
    parser.add_argument("--num_train_epochs",
                        default=1,
                        type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--no_cuda",
                        action='store_true',
                        help="Whether not to use CUDA when available")
    parser.add_argument("--local_rank",
                        type=int,
                        default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed',
                        type=int,
                        default=42,
                        help="random seed for initialization")
    parser.add_argument('--gradient_accumulation_steps',
                        type=int,
                        default=1,
                        help="Number of updates steps to accumulate before performing a backward/update pass.")
    parser.add_argument('--continue_train',
                        action='store_true',
                        help='Whether to train from checkpoints')

    # Saving arguments
    parser.add_argument('--eval_step',
                        type=int,
                        default=1000)
    parser.add_argument('--model_save_step',
                        type=int,
                        default=10000)

    args = parser.parse_args()
    logger.info('args:{}'.format(args))

    checkpoint = None
    start_epoch = 0
    start_step = 0
    if args.continue_train:
        checkpoint = torch.load(os.path.join(args.student_model,"optim_info.tar"))
        start_epoch = int(checkpoint['epoch']) + 1
        start_step = int(checkpoint['global_step'])

    samples_per_epoch = []
    for i in range(start_epoch, args.num_train_epochs):
        #epoch_file = args.pregenerated_data / "epoch_{}.json".format(i)
        #metrics_file = args.pregenerated_data / "epoch_{}_metrics.json".format(i)
        epoch_file = args.pregenerated_data / "train_dataset.json"
        metrics_file = args.pregenerated_data / "train_metrics.json"
        if epoch_file.is_file() and metrics_file.is_file():
            metrics = json.loads(metrics_file.read_text())
            samples_per_epoch.append(metrics['num_examples'])
        else:
            raise ValueError("epoch file ({}) or metrics file ({}) doesn't exist.".format(epoch_file, metrics_file))
    
    if args.local_rank == -1 or args.no_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        n_gpu = torch.cuda.device_count()
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        n_gpu = 1
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        torch.distributed.init_process_group(backend='nccl', timeout=datetime.timedelta(seconds=7200),)

    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.local_rank in [-1, 0] else logging.WARN)

    logger.info("device: {} n_gpu: {}, distributed training: {}".format(
        device, n_gpu, bool(args.local_rank != -1)))

    if args.gradient_accumulation_steps < 1:
        raise ValueError("Invalid gradient_accumulation_steps parameter: {}, should be >= 1".format(
            args.gradient_accumulation_steps))

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if n_gpu > 0:
        torch.cuda.manual_seed_all(args.seed)

    if os.path.exists(args.output_dir) and os.listdir(args.output_dir) and (not args.continue_train):
        raise ValueError("Output directory ({}) already exists and is not empty.".format(args.output_dir))
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_batch_size = args.train_batch_size
    total_train_examples = 0
    num_train_optimization_steps = 0
    for i in range(args.num_train_epochs - start_epoch):
        # The modulo takes into account the fact that we may loop over limited epochs of data
        samples = samples_per_epoch[i % len(samples_per_epoch)]
        total_train_examples += samples
        if args.local_rank == -1:
            print("samples:", samples)
            print()
            num_train_optimization_steps += int(int(samples / train_batch_size) / args.gradient_accumulation_steps)
        else:
            num_train_optimization_steps += int(int(samples / (train_batch_size * torch.distributed.get_world_size())) \
                                                    / args.gradient_accumulation_steps)

    student_model_config = GPT2Config.from_json_file(os.path.join(args.student_model, CONFIG_NAME))
    teacher_model_config = GPT2Config.from_json_file(os.path.join(args.teacher_model, CONFIG_NAME))
    max_seq_len = student_model_config.n_positions
    fit_size = teacher_model_config.n_embd
    if args.continue_train:
        student_model = TinyGPT2Model.from_pretrained(args.student_model, fit_size=fit_size)
    else:
        student_model = TinyGPT2Model.from_scratch(student_model_config, fit_size=fit_size)
    # Use "GPT2LMHeadModel" to investigate the effect of distillation of prediction layer on distillation performance. 
    # If the prediction layer does not have a good effect on the distillation performance, replace it with "GPT2Model".
    # teacher_model = GPT2Model.from_pretrained(args.teacher_model)
    teacher_model = GPT2LMHeadModel.from_pretrained(args.teacher_model)

    student_model.to(device)
    teacher_model.to(device)

    if args.local_rank != -1:
        teacher_model = torch.nn.parallel.DistributedDataParallel(teacher_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
        student_model = torch.nn.parallel.DistributedDataParallel(student_model, device_ids=[args.local_rank], output_device=args.local_rank, find_unused_parameters=True)
    elif n_gpu > 1:
        student_model = torch.nn.DataParallel(student_model)
        teacher_model = torch.nn.DataParallel(teacher_model)

    if args.local_rank in [-1, 0]:
        size = 0
        for n, p in student_model.named_parameters():
            logger.info('n: {}'.format(n))
            logger.info('p: {}'.format(p.nelement()))
            size += p.nelement()

        logger.info('Total parameters: {}'.format(size))

    # Prepare optimizer
    param_optimizer = list(student_model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = (
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    )
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, lr=args.learning_rate, weight_decay=args.weight_decay)
    if args.continue_train: 
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # Prepaer lr scheduler
    lr_scheduler = get_learning_rate_scheduler(optimizer, args, num_train_optimization_steps)
    if args.continue_train: 
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler_state_dict'])

    loss_mse = MSELoss()

    if args.local_rank in [-1, 0]:
        batch_size = train_batch_size * torch.distributed.get_world_size() if args.local_rank != -1 else train_batch_size
        logging.info("***** Running training *****")
        logging.info("  Num examples = {}".format(total_train_examples))
        logging.info("  Batch size = %d", batch_size)
        logging.info("  gradient_accumulation_steps = %d", args.gradient_accumulation_steps)
        logging.info("  Num steps = %d", num_train_optimization_steps)

    global_step = start_step
    for epoch in trange(start_epoch, args.num_train_epochs, desc="Epoch"):
        epoch_dataset = PregeneratedDataset(training_path=args.pregenerated_data, local_rank=args.local_rank,
                                            max_seq_len=max_seq_len, working_dir=args.working_dir, reduce_memory=args.reduce_memory)
        if args.local_rank == -1:
            train_sampler = RandomSampler(epoch_dataset)
        else:
            train_sampler = DistributedSampler(epoch_dataset, seed=args.seed)
        train_dataloader = DataLoader(epoch_dataset, sampler=train_sampler, batch_size=train_batch_size, pin_memory=True)
        
        tr_loss = 0.
        tr_att_loss = 0.
        tr_rep_loss = 0.
        tr_logit_loss = 0.
        student_model.train()
        nb_tr_steps = 0
        with tqdm(total=len(train_dataloader), desc="Epoch {}".format(epoch)) as pbar:
            for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration", ascii=True)):
                batch = tuple(t.to(device) for t in batch)

                input_ids, attention_mask = batch
                
                if input_ids.size()[0] != train_batch_size:
                    continue

                att_loss = 0.
                rep_loss = 0.
                logit_loss = 0.

                student_output = student_model(input_ids=input_ids, attention_mask=attention_mask)
                student_logits, student_reps, student_atts = student_output.logits, student_output.hidden_states, student_output.attentions

                teacher_output = teacher_model(input_ids=input_ids, attention_mask=attention_mask)
                teacher_logits, teacher_reps, teacher_atts = teacher_output.logits, teacher_output.hidden_states, teacher_output.attentions
                teacher_reps = [teacher_rep.detach() for teacher_rep in teacher_reps]  # speedup 1.5x
                teacher_atts = [teacher_att.detach() for teacher_att in teacher_atts]
                teacher_logits = teacher_logits.detach()

                teacher_layer_num = len(teacher_atts)
                student_layer_num = len(student_atts)
                # assert teacher_layer_num % student_layer_num == 0
                layers_per_block = int(teacher_layer_num / student_layer_num)
                new_teacher_atts = [teacher_atts[i * layers_per_block + layers_per_block - 1]
                                    for i in range(student_layer_num)]

                for student_att, teacher_att in zip(student_atts, new_teacher_atts):
                    student_att = torch.where(student_att <= -1e2, torch.zeros_like(student_att).to(device),
                                              student_att)
                    teacher_att = torch.where(teacher_att <= -1e2, torch.zeros_like(teacher_att).to(device),
                                              teacher_att)
                    att_loss += loss_mse(student_att, teacher_att)

                new_teacher_reps = [teacher_reps[i * layers_per_block] for i in range(student_layer_num + 1)]

                for student_rep, teacher_rep in zip(student_reps, new_teacher_reps):
                    rep_loss += loss_mse(student_rep, teacher_rep)

                if student_model_config.use_prediction_distill:
                    logit_loss += soft_cross_entropy(student_logits / args.temperature,
                                                        teacher_logits / args.temperature)
                else:
                    student_logits = student_logits.detach()

                if n_gpu > 1:
                    # mean() to average on multi-gpu parallel training
                    att_loss = att_loss.mean()
                    rep_loss = rep_loss.mean()
                    if student_model_config.use_prediction_distill:
                        logit_loss = logit_loss.mean()
                if args.gradient_accumulation_steps > 1:
                    att_loss = att_loss / args.gradient_accumulation_steps
                    rep_loss = rep_loss / args.gradient_accumulation_steps
                    if student_model_config.use_prediction_distill:
                        logit_loss = logit_loss / args.gradient_accumulation_steps
                    
                loss = att_loss + rep_loss
                if student_model_config.use_prediction_distill:
                    loss += logit_loss
                loss = rep_loss
                loss.backward()

                tr_att_loss += att_loss.item()
                tr_rep_loss += rep_loss.item()
                if student_model_config.use_prediction_distill:
                    tr_logit_loss += logit_loss.item()
                tr_loss += loss.item()

                nb_tr_steps += 1
                pbar.update(1)

                mean_loss = tr_loss * args.gradient_accumulation_steps / nb_tr_steps
                mean_att_loss = tr_att_loss * args.gradient_accumulation_steps / nb_tr_steps
                mean_rep_loss = tr_rep_loss * args.gradient_accumulation_steps / nb_tr_steps
                if student_model_config.use_prediction_distill:
                    mean_logit_loss = tr_logit_loss * args.gradient_accumulation_steps / nb_tr_steps

                if (step + 1) % args.gradient_accumulation_steps == 0:
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_step += 1
                    if args.local_rank in [-1, 0]:
                        if global_step % args.eval_step == 0:
                            result = {}
                            result['global_step'] = global_step
                            result['loss'] = mean_loss
                            result['att_loss'] = mean_att_loss
                            result['rep_loss'] = mean_rep_loss
                            if student_model_config.use_prediction_distill:
                                result['logit_loss'] = mean_logit_loss
                            output_eval_file = os.path.join(args.output_dir, "log.txt")
                            with open(output_eval_file, "a") as writer:
                                logger.info("***** Eval results *****")
                                for key in sorted(result.keys()):
                                    logger.info("  %s = %s", key, str(result[key]))
                                    writer.write("%s = %s\n" % (key, str(result[key])))
                            logging.info("lr : {}".format(lr_scheduler.get_lr()))
                        
                        if global_step % args.model_save_step == 0:
                            # Save a trained model
                            model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                            logging.info("** ** * Saving fine-tuned model ** ** * ")
                            # Only save the model it-self
                            model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                            output_model_file = os.path.join(args.output_dir, model_name)
                            output_config_file = os.path.join(args.output_dir, CONFIG_NAME)

                            torch.save(model_to_save.state_dict(), output_model_file)
                            model_to_save.config.to_json_file(output_config_file)

            if args.local_rank in [-1, 0]:
                model_name = "step_{}_{}".format(global_step, WEIGHTS_NAME)
                logging.info("** ** * Saving fine-tuned model ** ** * ")
                model_to_save = student_model.module if hasattr(student_model, 'module') else student_model

                output_model_file = os.path.join(args.output_dir, model_name)
                output_config_file = os.path.join(args.output_dir, CONFIG_NAME)
                # save optimizer & lr_scheduler state for continue_train
                optim_name = "optim_info.tar"
                output_optim_file = os.path.join(args.output_dir, optim_name) 

                torch.save(model_to_save.state_dict(), output_model_file)
                model_to_save.config.to_json_file(output_config_file)
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'optimizer_state_dict': optimizer.state_dict(),
                    'lr_scheduler_state_dict': lr_scheduler.state_dict(),
                }, output_optim_file)
                logging.info("global step : {},   lr : {}".format(global_step, lr_scheduler.get_lr()))

            # remove cache directory when before starting next epoch
            if args.local_rank in [-1, 0]:
                cache_dir = os.path.join(args.working_dir, "cache")
                if os.path.exists(cache_dir):
                    shutil.rmtree(cache_dir)
            # barrier until removing cache_dir
            if args.local_rank != -1:
                torch.distributed.barrier()

if __name__ == "__main__":
    main()
