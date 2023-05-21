# -*- coding: utf-8 -*-
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
from transformers import AutoTokenizer, AutoModel
tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True)
model = AutoModel.from_pretrained("THUDM/chatglm-6b-int4", trust_remote_code=True).half().cuda()
def preprocess(split_dir, output_path,number):
    json_list = []
    train_data_in_path = os.path.join(split_dir,"in.txt")
    train_data_out_path =  os.path.join(split_dir,"out.txt")
    i = 0
    with open(train_data_in_path, "r", encoding="utf-8") as train_in_file:
        with open(train_data_out_path, "r", encoding="utf-8") as train_out_file:
            train_in_data = train_in_file.readlines()
            train_out_data = train_out_file.readlines()
            all_data = zip(train_in_data, train_out_data)
            for (in_line, out_line) in list(all_data):
                i+=1
                if i%100 == 0:
                    print("100 times")
                if i >= number:
                    break
                in_line = in_line.replace(" ", "").replace("\n", "")
                out_line = out_line.replace(" ", "").replace("\n", "")
                question = "下面给出一副对联，"+in_line+","+out_line+","+"请回答他是一副关于什么的对联"
                response, history = model.chat(tokenizer, question, history=[])
                json_list.append(
                    {
                        "input":question,
                        "output":response
                    }
                )
    print(json_list)
    with open('1500_couplet_data.json', 'w',encoding="utf-8") as train_query:
        print("Dump train_query")
        json.dump(json_list, train_query,ensure_ascii=False)

preprocess("/kaggle/input/couplet-origin","data/",1500)