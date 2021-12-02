import argparse
import configparser
import json
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from preprocess.read_data import *
from utils.get_model import get_model
from utils.metric import *
from preprocess.read_data import *

warnings.filterwarnings('ignore')

os.environ["TOKENIZERS_PARALLELISM"] = "true"


parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', default='./config/default.config')
parser.add_argument('--model_path', default='./save_model/model.ckpt')
parser.add_argument('--test_path', default='./data/test.csv')
parser.add_argument('--show_bar',action='store_true', default=False)
args = parser.parse_args()


config = configparser.RawConfigParser()
config.read(args.config)


print("load data from",args.test_path)

test_data = process_csv(args.test_path)

print("test data %d " %(len(test_data['labels'])))
test_data = read_data(config, args, False, **test_data)

mymodel=get_model(config,args, usegpu=True)
mymodel = torch.nn.DataParallel(mymodel).cuda()
model_dict = torch.load(args.model_path).module.state_dict()
print("load model from", args.model_path)
mymodel.module.load_state_dict(model_dict)


acc=0.0
recall=0.0
f1=0.0
pre=0.0
mymodel.eval()


for j,batch in enumerate(test_data):

    input_ids, attention_mask, labels = [Variable(elem.cuda()) for elem in batch]

    with torch.no_grad():
        pred = mymodel(input_ids, attention_mask, is_training=False)
        pred = pred.cpu().detach().cpu().numpy()
        labels = labels.cpu().detach().cpu().numpy()

        metric_ = metric(pred, labels)
        acc += metric_[0]
        pre += metric_[1]
        recall += metric_[2]
        f1 += metric_[3]

print("test Acc:%f Precision:%f Recall:%f F1:%f" %(acc/len(test_data),pre/len(test_data),recall/len(test_data),f1/len(test_data)))
