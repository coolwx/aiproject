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

def read_duilian_split(train_length = 2700,valid_length=250):
    with open("/kaggle/input/json-data/1500_couplet_data_modified.json","r",encoding="utf-8") as f:
        with open("/kaggle/input/json-data/3000_couplet_data_modified.json","r",encoding="utf-8") as f2:
            a = json.load(f)
            b = json.load(f2)
            a = a+b
            query = []
            response = []
            for item in a:
                if "question" in item.keys():
                    query.append(item['question'])
                    response.append(item['anwser']+"。"+item['axplanation'])
            return query[:train_length], response[:train_length], query[train_length:train_length + valid_length], response[
                                                                                                               train_length:train_length + valid_length]


def preprocess(args):
    print(f"Reading {args.dataset_type} dataset...")
    train_query, train_response,test_query, test_response = read_duilian_split()

    assert len(train_query) == len(train_response)
    assert len(test_query) == len(test_response)

    print("Dataset loaded.")

    print("Tokenize...")

    tokenizer = AutoTokenizer.from_pretrained("Langboat/mengzi-t5-base")

    print("Tokenize query...")
    train_query_tokenized = tokenizer(train_query,
                                      truncation=True,
                                      padding=True,
                                      max_length=args.max_source_length)
    train_query_tokenized = {
        key: val
        for key, val in train_query_tokenized.items()
    }

    test_query_tokenized = tokenizer(test_query,
                                     truncation=True,
                                     padding=True,
                                     max_length=args.max_source_length)
    test_query_tokenized = {
        key: val
        for key, val in test_query_tokenized.items()
    }

    print("Tokenize response...")
    train_response_tokenized = tokenizer(train_response,
                                         truncation=True,
                                         padding=True,
                                         max_length=args.max_target_length)
    train_response_tokenized = {
        key: val
        for key, val in train_response_tokenized.items()
    }

    test_response_tokenized = tokenizer(test_response,
                                        truncation=True,
                                        padding=True,
                                        max_length=args.max_target_length)
    test_response_tokenized = {
        key: val
        for key, val in test_response_tokenized.items()
    }

    path = './couplet_json_new/'

    print(f"Saving tokenized dict at {path}")
    os.makedirs(path, exist_ok=True)

    with open(path + 'train_query.json', 'w') as train_query:
        print("Dump train_query")
        print(len(train_query_tokenized['input_ids']))
        json.dump(train_query_tokenized, train_query)
    with open(path + 'test_query.json', 'w') as test_query:
        print("Dump test_query")
        print(len(test_query_tokenized['input_ids']))
        json.dump(test_query_tokenized, test_query)

    with open(path + 'train_response.json', 'w') as train_response:
        print("Dump train_response")
        print(len(train_response_tokenized['input_ids']))
        json.dump(train_response_tokenized, train_response)
    with open(path + 'test_response.json', 'w') as test_response:
        print("Dump test_response")
        print(len(test_response_tokenized['input_ids']))
        json.dump(test_response_tokenized, test_response)


def mypreprocess():
    parser = ArgumentParser("Transformers EncoderDecoderModel Preprocessing")
    parser.add_argument(
        "--trainset",
        type=str,
        default=
        "../input/couplet-origin")

    parser.add_argument("--max_source_length", type=int, default=128)
    parser.add_argument("--max_target_length", type=int, default=256)

    parser.add_argument("--dataset_type",
                        type=str,
                        default='coupledata',
                        )

    args = parser.parse_args(["--dataset_type", 'coupledata'])

    preprocess(args)

mypreprocess()

