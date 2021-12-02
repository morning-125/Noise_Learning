import argparse
import configparser
import os
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

from preprocess.read_data import *
from utils.common import *
from utils.get_model import *
from utils.metric import *

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def train_step(config, args, mymodel, optimizer, loss_func, train_data, valid_data):
    epochs = config.getint("train", "epoch")
    warmup_epochs = config.getint("train", "warmup_epoch")
    
    pre_best=0.0

    for epoch in range(1, epochs + 1):
        train_loss = 0.0
        train_acc = 0.0
        train_recall = 0.0


        bar1 = None
        if args.show_bar:
            bar1 = get_progressbar(epoch, epochs, len(train_data), 'train ')

        for i, data in enumerate(train_data):
            # import pdb
            # pdb.set_trace()
            input_ids, attention_mask, labels = [Variable(elem.cuda()) for elem in data]
            optimizer.zero_grad()

            out = mymodel(input_ids, attention_mask, is_training=True)

            loss = loss_func(out, labels)

            train_loss += loss.mean().item()

            # import pdb
            # pdb.set_trace()

            # if epoch > warmup_epochs:
            #     ordered_label_errors = get_noise_indices(
            #         s=labels.cpu().detach().numpy(),
            #         psx=out.cpu().detach().numpy(),
            #         sorted_index_method='normalized_margin', # Orders label errors
            #     )
            #     print(ordered_label_errors)

            loss.mean().backward()
            optimizer.step()

            out = out.cpu().detach().numpy()
            labels = labels.cpu().detach().numpy()
            train_metric = metric(out,labels)
            train_acc += train_metric[0]
            train_recall += train_metric[2]

            if args.show_bar:
                bar1.dynamic_messages.loss = train_loss / (i + 1)
                bar1.dynamic_messages.acc = train_acc / (i + 1)
                bar1.dynamic_messages.recall = train_recall / (i + 1)
                bar1.update(i + 1)
        
        if bar1:
            bar1.finish()

        print("train %d/%d epochs Loss:%f, Acc:%f, Recall:%f" \
        %(epoch, epochs, train_loss / (i + 1), train_acc / (i + 1), train_recall / (i + 1)))
        
        if valid_data is None:
            continue


        val_loss = 0.0
        val_acc = 0.0
        val_recall = 0.0
        mymodel.eval()

        for j, batch in enumerate(valid_data):
            val_input_ids, val_attention_mask, val_labels = [Variable(elem.cuda()) for elem in batch]

            with torch.no_grad():
                pred = mymodel(val_input_ids, val_attention_mask, is_training=False)
                val_loss += loss_func(pred, val_labels).mean()
                pred = pred.cpu().detach().cpu().numpy()
                val_labels = val_labels.cpu().detach().cpu().numpy()
                val_metric = metric(pred, val_labels)
                val_acc += val_metric[0]
                val_recall += val_metric[2]

        print("valid %d/%d epochs Loss:%f, Acc:%f, Recall:%f" \
        %(epoch, epochs, val_loss / len(valid_data), val_acc / len(valid_data), val_recall / (len(valid_data))))
        if args.pick_ckpt == "best_acc" and pre_best < val_acc:
            pre_best = val_acc
            torch.save(mymodel, args.model_dir+"_best.ckpt")
            print('save best model',args.model_dir+"_best.ckpt")
    
    torch.save(mymodel, args.model_dir+"_last.ckpt")
    print('save last model',args.model_dir+"_last.ckpt")



warnings.filterwarnings('ignore')

parser = argparse.ArgumentParser()
parser.add_argument('--config', '-c', default='./config/default.config')
parser.add_argument('--model_dir', default='./save_model/model.ckpt')
parser.add_argument('--train_path', default='./data/train.csv')
parser.add_argument('--test_path', default='./data/test.csv')
parser.add_argument('--noise_ratio',type=float,default=0.0)
parser.add_argument('--noise_type',type=str,default="sym")
parser.add_argument('--pick_ckpt',type=str,default='best')
parser.add_argument('--show_bar',action='store_true', default=False)
args = parser.parse_args()


config = configparser.RawConfigParser()
config.read(args.config)

print("use gpu device =",os.environ["CUDA_VISIBLE_DEVICES"])

print("load data from",args.train_path)

all_data = process_csv(args.train_path)
train_data,valid_data = split_data(all_data)
print("train data %d , valid data %d" %(len(train_data['labels']),len(valid_data['labels'])))
train_data = read_data(config, args, True, **train_data)
valid_data = read_data(config, args, False,**valid_data)


mymodel=get_model(config, args, usegpu=True)
mymodel=torch.nn.DataParallel(mymodel).cuda()

loss_func=nn.NLLLoss(reduce=False)
optimizer=optim.Adam(mymodel.parameters(),lr=config.getfloat("train", "learning_rate"))
train_step(config, args, mymodel, optimizer, loss_func, train_data, valid_data=valid_data)



