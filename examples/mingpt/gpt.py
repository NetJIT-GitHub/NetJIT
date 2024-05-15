import os

import torch
from torch.distributed import init_process_group
from torch.utils.data import random_split

from char_dataset import CharDataset, DataConfig
from model import GPT, GPTConfig, OptimizerConfig, create_optimizer

LOCAL_RANK = int(os.environ["LOCAL_RANK"])
WORLD_SIZE = int(os.environ["WORLD_SIZE"])
WORLD_RANK = int(os.environ["RANK"])


def ddp_setup():
    init_process_group(backend="nccl", world_size=WORLD_SIZE, rank=WORLD_RANK)
    torch.cuda.set_device(LOCAL_RANK)


def get_train_objs(gpt_cfg: GPTConfig, opt_cfg: OptimizerConfig, data_cfg: DataConfig):
    dataset = CharDataset(data_cfg)
    train_len = int(len(dataset) * data_cfg.train_split)
    print(train_len)
    train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])

    gpt_cfg.vocab_size = dataset.vocab_size
    gpt_cfg.block_size = dataset.block_size
    model = GPT(gpt_cfg)
    optimizer = create_optimizer(model, opt_cfg)

    return model, optimizer, train_set, test_set
