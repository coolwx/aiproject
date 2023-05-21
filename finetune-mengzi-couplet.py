#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import random as rd
import json
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from argparse import ArgumentParser
from tqdm import tqdm
import math
import time
from datetime import datetime
import os
from os.path import join, exists
import transformers
import pickle
import sys
import numpy as np
import logging
import warnings


# In[2]:


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


parser = ArgumentParser("T5 persona")
parser.add_argument("--dumped_token",type=str, default='/kaggle/input/t5-mengzi-new/couplet_json_new/')
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--kaggle",type=str, default='kaggle')
parser.add_argument('--log_path', default='./train.log', type=str)
parser.add_argument('--epochs', default=100, type=int, required=False, help='训练的最大轮次')
parser.add_argument('--lr', default=4e-5, type=float, required=False, help='学习率')
parser.add_argument('--eps', default=1.0e-09, type=float, required=False, help='衰减率')
parser.add_argument('--warmup_steps', default=400, type=int, required=False, help='warmup steps ')
parser.add_argument('--gradient_accumulation_steps', default=4, type=int, required=False, help='梯度积累')
parser.add_argument('--max_grad_norm', default=2.0, type=float, required=False)
parser.add_argument('--save_model_path', default='./model', type=str, required=False,
                        help='模型输出路径')
parser.add_argument('--pretrained_model', default='', type=str, required=False,
                        help='预训练的模型的路径')
parser.add_argument('--optimizer_state', default=None, type=str, required=False,
                        help='optimizer的路径')
parser.add_argument('--log_step', default=100, type=int, required=False, help='多少步汇报一次loss')
parser.add_argument('--num_workers', type=int, default=2, help="dataloader加载数据时使用的线程数量")

args = parser.parse_args(["--kaggle", 'kaggle'])

def prepare_data_batch(batch):
    response_input_ids = batch['response']['input_ids']
    response_input_ids[response_input_ids == 0] = -100
    return batch['query']['input_ids'], batch['query']['attention_mask'], response_input_ids


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., :].contiguous().view(-1)
    _, logit = logit.max(dim=-1)
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

class ConvAI2Dataset(torch.utils.data.Dataset):
    def __init__(self, queries, labels):
        self.queries = queries
        self.labels = labels
     
        
    def __getitem__(self, idx):
        query = {
            key: torch.tensor(val[idx])
            for key, val in self.queries.items()
        }
        response = {
            key: torch.tensor(val[idx])
            for key, val in self.labels.items()
        }
        return { 'query': query, 'response': response }

    def __len__(self):
        return len(self.labels['input_ids'])
    
def prepare_data_batch(batch):
    response_input_ids = batch['response']['input_ids']
    response_input_ids[response_input_ids == 0] = -100
    return batch['query']['input_ids'], batch['query']['attention_mask'], response_input_ids


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., :].contiguous().view(-1)
    _, logit = logit.max(dim=-1)
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word

def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])
 
    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
 
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
 
    return logger

def load_dataset(logger,args):
    logger.info(f"Load tokenized train & val dataset from {args.dumped_token}.")
    path = args.dumped_token

    with open(path + 'train_query.json') as train_query, open(path + 'test_query.json') as val_query:
        print("Load train_query")
        tmp = train_query.readline()
        train_query_tokenized = json.loads(tmp)
        print("Load val_query")
        tmp = val_query.readline()
        val_query_tokenized = json.loads(tmp)

    with open(path + 'train_response.json') as train_response, open(path + 'test_response.json') as val_response:
        print("Load train_response")
        tmp = train_response.readline()
        train_response_tokenized = json.loads(tmp)
        print("Load val_response")
        tmp = val_response.readline()
        val_response_tokenized = json.loads(tmp)
        
        
    train_dataset = ConvAI2Dataset(train_query_tokenized,
                                   train_response_tokenized)
    train_loader = DataLoader(train_dataset,
                              batch_size=args.batch_size,
                              shuffle=True,
                              drop_last=True,
                              num_workers=args.num_workers
                              )

    val_dataset = ConvAI2Dataset(val_query_tokenized,
                                 val_response_tokenized)
    val_loader = DataLoader(val_dataset,
                            batch_size=args.batch_size,
                            shuffle=True,
                            drop_last=True)
    return train_loader,val_loader

def train_epoch(model, train_dataloader, optimizer, scheduler, logger,
                epoch, args):
    model.train()
    epoch_start_time = datetime.now()
    total_loss = 0  

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, batch in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids, attention_mask, labels = prepare_data_batch(batch)
            input_ids, attention_mask,  labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
            
            outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
            logits = outputs.logits
            loss = outputs.loss

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=-100)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.detach().item()
            
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                scheduler.step()

            if (batch_idx + 1) % args.log_step == 0:
                logger.info(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1, epoch + 1, loss.detach().item()* args.gradient_accumulation_steps , batch_acc, scheduler.get_lr()))

            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                logger.info("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                logger.info(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    logger.info(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    logger.info('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    logger.info('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss

def validate_epoch(model, validate_dataloader, logger, epoch, args):
    logger.info("start validating")
    model.eval()
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, batch in enumerate(validate_dataloader):
                input_ids, attention_mask,  labels = prepare_data_batch(batch)
                input_ids, attention_mask,  labels = input_ids.to(device), attention_mask.to(device), labels.to(device)
                outputs = model(input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels)
                logits = outputs.logits
                loss = outputs.loss

                total_loss += loss.detach().item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            logger.info(
                "validate epoch {}: loss {}".format(epoch+1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            logger.info('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            logger.info("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            logger.info(str(exception))
            raise exception

def train(model, logger, train_dataloader, validate_dataloader, args):
    
    t_total = len(train_dataloader) // args.gradient_accumulation_steps * args.epochs
    logger.info(f't_total:{t_total}')
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay) ], 'weight_decay': 1e-2},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay) ], 'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters,lr=args.lr, eps=args.eps)
    if args.optimizer_state is not None:
        optimizer.load_state_dict(torch.load(args.optimizer_state))
        
    scheduler = transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=args.warmup_steps, num_training_steps=t_total)
    logger.info('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(args.epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler,
            logger=logger, epoch=epoch, args=args)
        train_losses.append(train_loss)
        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader,
            logger=logger, epoch=epoch, args=args)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            logger.info('saving current best model for epoch {}'.format(epoch + 1))
        model_path = join(args.save_model_path, 'best_model_in_epoch{}'.format(epoch + 1))
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        model_to_save = model.module if hasattr(model, 'module') else model
        if epoch % 12 == 0:
            model_to_save.save_pretrained(model_path)
            torch.save(optimizer.state_dict(), './optimizer')
            

    logger.info('training finished')
    logger.info("train_losses:{}".format(train_losses))
    logger.info("validate_losses:{}".format(validate_losses))
def main():
    logger = get_logger(args.log_path)
    logger.info('start training!')
    logger.info('using device:{}'.format(device))

    # 创建模型的输出目录
    if not os.path.exists(args.save_model_path):
        os.mkdir(args.save_model_path)

    # 创建模型
    if args.pretrained_model:  # 加载预训练模型
        model = AutoModelForSeq2SeqLM.from_pretrained(args.pretrained_model)
    else:  # 初始化模型
        model = AutoModelForSeq2SeqLM.from_pretrained(("Langboat/mengzi-t5-base"))
    model = model.to(device)

    # 计算模型参数数量
    num_parameters = 0
    parameters = model.parameters()
    for parameter in parameters:
        num_parameters += parameter.numel()
    logger.info('number of model parameters: {}'.format(num_parameters))

    # 记录参数设置
    logger.info("args:{}".format(args))

    # 加载训练集和验证集
    # ========= Loading Dataset ========= #
    train_loader, val_loader = load_dataset(logger, args)
    train(model, logger, train_loader, val_loader, args)


# In[3]:


main()

