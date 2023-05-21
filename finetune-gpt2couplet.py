#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import BertTokenizer
from transformers import BertTokenizerFast
import argparse
import pickle
from tqdm import tqdm
from transformers import GPT2TokenizerFast, GPT2LMHeadModel
import logging
import numpy as np
import json
from tqdm import tqdm
import math
import time
import torch
import torch.nn.functional as F
import torch.optim as optim
from datetime import datetime
import os
from torch.utils.data import Dataset, DataLoader
from os.path import join, exists
from torch.nn import CrossEntropyLoss
from tqdm import tqdm
from torch.nn import DataParallel
import transformers
import pickle
import sys
# from pytorchtools import EarlyStopping
from sklearn.model_selection import train_test_split
# from data_parallel import BalancedDataParallel
from transformers import GPT2TokenizerFast, GPT2LMHeadModel, GPT2Config
from transformers import BertTokenizerFast
import torch.nn.utils.rnn as rnn_utils
import numpy as np

def preprocess():
    """
    对原始语料进行tokenize，将每段对话处理成如下形式："[CLS]utterance1[SEP]utterance2[SEP]utterance3[SEP]"
    """
    # 设置参数

    vocab_path = 'model/vocab.txt'
    train_data_in_path = "data/train/in.txt"
    train_data_out_path = "data/train/out.txt"
    test_data_in_path = "data/test/in.txt"
    test_data_out_path = "data/test/out.txt"
    train_save_path = 'data/duilian_train.pkl'
    test_save_path = 'data/duilian_test.pkl'

    # 初始化tokenizer
    tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]",
                                  cls_token="[CLS]")
    sep_id = tokenizer.sep_token_id
    cls_id = tokenizer.cls_token_id
    #处理训练集
    print("preprocessing data, save path:{}".format(train_save_path))
    print("开始处理训练集")
    train_data = []
    train_data_len = []
    with open(train_data_in_path, "r", encoding="utf-8") as train_in_file:
        with open(train_data_out_path, "r", encoding="utf-8") as train_out_file:
            train_in_data = train_in_file.readlines()
            train_out_data = train_out_file.readlines()
            all_data = zip(train_in_data, train_out_data)
            for (in_line, out_line) in list(all_data):
                in_line = in_line.replace(" ", "").replace("\n", "")+"-"
                out_line = out_line.replace(" ", "").replace("\n", "")
                input_ids = [cls_id]#以[CLS]作为开头
                input_ids += tokenizer.encode(in_line, add_special_tokens=False)
                input_ids += tokenizer.encode(out_line, add_special_tokens=False)
                input_ids.append(sep_id)
                train_data.append(input_ids)
                train_data_len.append(len(input_ids))
    len_mean = np.mean(train_data_len)
    len_median = np.median(train_data_len)
    len_max = np.max(train_data_len)
    print("最长对联长度:{}，平均对联长度:{}".format(len_max,len_mean))
    with open(train_save_path, "wb") as f:
        pickle.dump(train_data, f)
    print('completed...')
    print("there are {} 对联 in training dataset".format(len(train_data)))
    #处理测试集
    print("preprocessing data, save path:{}".format(test_save_path))
    print("开始处理测试集")
    test_data = []
    test_data_len = []
    with open(test_data_in_path, "r", encoding="utf-8") as test_in_file:
        with open(test_data_out_path, "r", encoding="utf-8") as test_out_file:
            test_in_data = test_in_file.readlines()
            test_out_data = test_out_file.readlines()
            all_data = zip(test_in_data, test_out_data)
            for (in_line, out_line) in list(all_data):
                in_line = in_line.replace(" ", "").replace("\n", "")+"-"
                out_line = out_line.replace(" ", "").replace("\n", "")
                input_ids = [cls_id]  # 以[CLS]作为开头
                input_ids += tokenizer.encode(in_line, add_special_tokens=False)
                input_ids += tokenizer.encode(out_line, add_special_tokens=False)
                input_ids.append(sep_id)
                test_data.append(input_ids)
                test_data_len.append(len(input_ids))
    len_mean = np.mean(test_data_len)
    len_median = np.median(test_data_len)
    len_max = np.max(test_data_len)
    print("最长对联长度:{}，平均对联长度:{}".format(len_max, len_mean))
    with open(test_save_path, "wb") as f:
        pickle.dump(train_data, f)
    print('completed...')
    print("there are {} 对联 in test dataset".format(len(test_data)))




def collate_fn(batch):
    input_ids = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=0)  ###pad成相同长度
    labels = rnn_utils.pad_sequence(batch, batch_first=True, padding_value=-100)
    return input_ids, labels


class MyDataset(Dataset):
    """

    """

    def __init__(self, input_list, max_len):
        self.input_list = input_list
        self.max_len = max_len

    def __getitem__(self, index):
        input_ids = self.input_list[index]
        input_ids = input_ids[:self.max_len]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        return input_ids

    def __len__(self):
        return len(self.input_list)


def load_dataset():
    """
    加载训练集和验证集，这里直接划分问答的训练集为train/val，就不使用验证集了
    """
    print("loading training dataset and validating dataset")
    train_path = '/kaggle/input/coupledata/duilian_train.pkl'
    test_path = '/kaggle/input/coupledata/duilian_test.pkl'
    ##对话数目为1425170
    with open(train_path, "rb") as f:
        train_input_list = pickle.load(f)
    with open(test_path,"rb") as f:
        test_input_list = pickle.load(f)
        

    # 划分训练集与验证集，这里数据量太大，这里先使用10w进行训练
    # test
    # input_list_train = input_list_train[:24]
    # input_list_val = input_list_val[:24]
    max_len = 70  ###150
    val_num = 600000
    input_list_train = train_input_list[:val_num]
    input_list_val = train_input_list[val_num:val_num+20000]
    # test
    # input_list_train = input_list_train[:24]
    # input_list_val = input_list_val[:24]
    train_dataset = MyDataset(input_list_train, max_len)
    val_dataset = MyDataset(input_list_val, max_len)
    return train_dataset, val_dataset


def calculate_acc(logit, labels, ignore_index=-100):
    logit = logit[..., :-1, :].contiguous().view(-1, logit.size(-1))
    labels = labels[..., 1:].contiguous().view(-1)

    _, logit = logit.max(dim=-1)  # 对于每条数据，返回最大的index
    # 进行非运算，返回一个tensor，若labels的第i个位置为pad_id，则置为0，否则为1
    non_pad_mask = labels.ne(ignore_index)
    n_correct = logit.eq(labels).masked_select(non_pad_mask).sum().item()
    n_word = non_pad_mask.sum().item()
    return n_correct, n_word


ignore_index = -100
gradient_accumulation_steps = 4
max_grad_norm = 2
batch_size = 64  ##64
num_workers = 4
save_model_path = 'model'
patience = 0
epochs = 6
lr = 2.6e-5  ###2.6e-5
eps = 1.0e-09
warmup_steps_ratio = 0.1  ##400 for 32w
vocab_path = '/kaggle/input/gpt2base/gpt2通用中文模型/vocab.txt'
pretrained_model = '/kaggle/input/gpt2base/gpt2通用中文模型'
model_config = '/kaggle/input/gpt2base/gpt2通用中文模型/config.json'

device = 'cuda'


def train_epoch(model, train_dataloader, optimizer, scheduler, epoch):
    model.train()
    # device = 'cuda'
    epoch_start_time = datetime.now()
    total_loss = 0  # 记录下整个epoch的loss的总和

    # epoch_correct_num:每个epoch中,output预测正确的word的数量
    # epoch_total_num: 每个epoch中,output预测的word的总数量
    epoch_correct_num, epoch_total_num = 0, 0

    for batch_idx, (input_ids, labels) in enumerate(train_dataloader):
        # 捕获cuda out of memory exception
        try:
            input_ids = input_ids.to(device)
            labels = labels.to(device)
            outputs = model.forward(input_ids, labels=labels)
            logits = outputs.logits
            loss = outputs.loss
            loss = loss.mean()

            # 统计该batch的预测token的正确数与总数
            batch_correct_num, batch_total_num = calculate_acc(logits, labels, ignore_index=ignore_index)
            # 统计该epoch的预测token的正确数与总数
            epoch_correct_num += batch_correct_num
            epoch_total_num += batch_total_num
            # 计算该batch的accuracy
            batch_acc = batch_correct_num / batch_total_num

            total_loss += loss.item()
            if gradient_accumulation_steps > 1:
                loss = loss / gradient_accumulation_steps

            loss.backward()
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

            # 进行一定step的梯度累计之后，更新参数
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # 更新参数
                optimizer.step()
                # 更新学习率
                scheduler.step()
                # 清空梯度信息
                optimizer.zero_grad()

            if (batch_idx + 1) % 10 == 0 or batch_idx == 0:
                print(
                    "batch {} of epoch {}, loss {}, batch_acc {}, lr {}".format(
                        batch_idx + 1, epoch + 1, loss.item() * gradient_accumulation_steps, batch_acc,
                        scheduler.get_lr()))
            #             if (batch_idx +1) % 100 == 0:
            #                 f.write(f'{epoch+1}, {batch_idx+1}, {loss.item() * gradient_accumulation_steps}, {batch_acc}, {scheduler.get_lr()}\n')
            del input_ids, outputs

        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print("WARNING: ran out of memory")
                if hasattr(torch.cuda, 'empty_cache'):
                    torch.cuda.empty_cache()
            else:
                print(str(exception))
                raise exception

    # 记录当前epoch的平均loss与accuracy
    epoch_mean_loss = total_loss / len(train_dataloader)
    epoch_mean_acc = epoch_correct_num / epoch_total_num
    print(
        "epoch {}: loss {}, predict_acc {}".format(epoch + 1, epoch_mean_loss, epoch_mean_acc))

    # save model
    #     print('saving model for epoch {}'.format(epoch + 1))
    #     model_path = join('model', 'epoch{}_bs{}_accu{}_tepoch{}'.format(epoch + 1,batch_size,gradient_accumulation_steps, epochs))
    #     if not os.path.exists(model_path):
    #         os.mkdir(model_path)
    #     model_to_save = model.module if hasattr(model, 'module') else model
    #     model_to_save.save_pretrained(model_path)
    print('epoch {} finished'.format(epoch + 1))
    epoch_finish_time = datetime.now()
    print('time for one epoch: {}'.format(epoch_finish_time - epoch_start_time))

    return epoch_mean_loss


def validate_epoch(model, validate_dataloader, epoch):
    print("start validating")
    model.eval()
    # device = 'cuda'
    epoch_start_time = datetime.now()
    total_loss = 0
    # 捕获cuda out of memory exception
    try:
        with torch.no_grad():
            for batch_idx, (input_ids, labels) in enumerate(validate_dataloader):
                input_ids = input_ids.to(device)
                labels = labels.to(device)
                outputs = model.forward(input_ids, labels=labels)
                logits = outputs.logits
                loss = outputs.loss
                loss = loss.mean()

                total_loss += loss.item()
                del input_ids, outputs

            # 记录当前epoch的平均loss
            epoch_mean_loss = total_loss / len(validate_dataloader)
            print(
                "validate epoch {}: loss {}".format(epoch + 1, epoch_mean_loss))
            epoch_finish_time = datetime.now()
            print('time for validating one epoch: {}'.format(epoch_finish_time - epoch_start_time))
            return epoch_mean_loss
    except RuntimeError as exception:
        if "out of memory" in str(exception):
            print("WARNING: ran out of memory")
            if hasattr(torch.cuda, 'empty_cache'):
                torch.cuda.empty_cache()
        else:
            print(str(exception))
            raise exception


def train(model, train_dataset, validate_dataset):
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn,
        drop_last=True
    )
    validate_dataloader = DataLoader(validate_dataset, batch_size=batch_size, shuffle=True,
                                     num_workers=num_workers, collate_fn=collate_fn, drop_last=True)
    #     early_stopping = EarlyStopping(patience, verbose=True, save_path=save_model_path)
    t_total = len(train_dataloader) // gradient_accumulation_steps * epochs
    print(f'total_steps: {t_total}, warmup_steps: {warmup_steps_ratio * t_total}')
    optimizer = transformers.AdamW(model.parameters(), lr=lr, eps=eps)
    # scheduler = transformers.WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=warmup_steps_ratio * t_total, num_training_steps=t_total
    )

    print('starting training')

    # 用于记录每个epoch训练和验证的loss
    train_losses, validate_losses = [], []
    # 记录验证集的最小loss
    best_val_loss = 10000
    # 开始训练
    for epoch in range(epochs):
        # ========== train ========== #
        train_loss = train_epoch(
            model=model, train_dataloader=train_dataloader,
            optimizer=optimizer, scheduler=scheduler, epoch=epoch)
        train_losses.append(train_loss)

        # ========== validate ========== #
        validate_loss = validate_epoch(
            model=model, validate_dataloader=validate_dataloader, epoch=epoch)
        validate_losses.append(validate_loss)

        # 保存当前困惑度最低的模型，困惑度低，模型的生成效果不一定会越好
        if validate_loss < best_val_loss:
            best_val_loss = validate_loss
            print('saving current best model for epoch {}'.format(epoch + 1))
            model_path = join(save_model_path, 'min_ppl_model'.format(epoch + 1))
            if not os.path.exists(model_path):
                os.mkdir(model_path)
            model_to_save = model.module if hasattr(model, 'module') else model
            model_to_save.save_pretrained(model_path)

        #  如果patience=0,则不进行early stopping
        if patience == 0:
            continue
    #         early_stopping(validate_loss, model)
    #         if early_stopping.early_stop:
    #             print("Early stopping")
    #             break
    print('training finished')
    print("train_losses:{}".format(train_losses))
    print("validate_losses:{}".format(validate_losses))

#preprocess()

###训练代码
tokenizer = BertTokenizerFast(vocab_file=vocab_path, sep_token="[SEP]", pad_token="[PAD]", cls_token="[CLS]")
if not os.path.exists(save_model_path):
    os.mkdir(save_model_path)
if pretrained_model:  # 加载预训练模型
    model = GPT2LMHeadModel.from_pretrained(pretrained_model)
    print(f'loaded model in {pretrained_model}')
else:  # 初始化模型
    model_config = GPT2Config.from_json_file(model_config)
    model = GPT2LMHeadModel(config=model_config)
model = model.to(device)
print('model config:\n{}'.format(model.config.to_json_string()))
# 计算模型参数数量
num_parameters = 0
parameters = model.parameters()
for parameter in parameters:
    num_parameters += parameter.numel()
print('number of model parameters: {}'.format(num_parameters))
###load dataset
train_dataset, validate_dataset = load_dataset()
###train
train(model, train_dataset, validate_dataset)







