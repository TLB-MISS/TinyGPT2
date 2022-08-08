# TinyGPT2

This is TinyGPT2 repo which is compression model of GPT2 using TinyBERT technologies.
Here, I focus only on GD(General distillation).
Please refer [here](https://github.com/huawei-noah/Pretrained-Language-Model/tree/master/TinyBERT) for TinyBERT repo.

## Differences from TinyBERT

### 1. Attention objective

Unlike in BERT, the attention matrix of GPT2 is a lower triangular matrix. Therefore, if the sequence length is set to $l$, the attention objective function should be as follows:

![image](https://user-images.githubusercontent.com/77427895/183375478-e3fec76a-0645-4868-a8b4-9cd75f3ba74d.png)

### 2. Prediction objective

In BERT, the main downstream task should be learned by detaching the PLM head and attaching a new classification head. However, in GPT2, the main downstream task is to compute zero-shot PPL as maintaining the PLM head. Therefore, in TinyGPT2, unlike TinyBERT, it is necessary to learn prediction objective in GD.

Additionally, I modify the bug of the `soft_cross_entropy` function in TinyBERT.

### 3. Objective scaling

BERT has Post-LN structures, so the scale of FFN residual block output is stable, but GPT2 is not because it has Pre-LN structures. Therefore, the FFN loss of TinyGPT2 has a larger scale than the TinyBERT's. Meanwhile, the scale of attention loss is not significantly affected by the position of the LN compared to FFN loss. This is because the attention matrix of attention objective is not a residual block output, but an intermediate output that calculates the residual block output. Therefore, appropriate scaling is required.

## Environment settings

Please see preliminary.txt.

## Pretrain

```
NCCL_ASYNC_ERROR_HANDLING=1 NCCL_P2P_DISABLE=1 NCCL_IB_DISABLE=1 NCCL_SOCKET_IFNAME={SOCKET NAME} python3 -m torch.distributed.launch --nproc_per_node={# OF PROCESSES PER NODE} --nnodes={# OF NODES} --node_rank={NODE RANK} --master_addr={ADDR} --master_port={PORT} general_distill.py --pregenerated_data={CORPUS DIR} --teacher_model=gpt2 --student_model=student_prediction_config --reduce_memory --train_batch_size={BSZ} --gradient_accumulation_steps={ACCUM STEPS} --learning_rate=2.5e-4 --warmup=0.01 --output_dir={OUTPUT DIR} --working_dir={CACHE DIR} --eval_step=10
```

## Finetune

```
python3 run_clm.py --model_name_or_path {MODEL DIR} --dataset_name {DATASET} --dataset_config_name {OPTIONAL} --per_device_train_batch_size {BSZ FOR TRAIN} --per_device_eval_batch_size {BSZ FOR EVAL} --do_train --do_eval --output_dir {OUTPUT DIR} --tokenizer_name gpt2
```

## Performance

PPL
```
python3 calc_ppl.py --tokenizer_model=gpt2 --target_model={MODEL DIR} --task_name={DATASET}
```

Generate sentences
```
python3 generate_exp.py
```
