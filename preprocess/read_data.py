import torch
from torch.utils.data import Dataset,DataLoader
from transformers import AutoTokenizer
from transformers import BertTokenizer
import pandas as pd
import numpy as np
from utils.common import *

class DataToDataset(Dataset):
    def __init__(self,text,label):
        self.text=text
        self.label=label
            
    def __len__(self):
        return len(self.label)
            
    def __getitem__(self,index):
        return self.text['input_ids'][index],self.text['attention_mask'][index],self.label[index]


def read_data(config, args, is_train, labels, titles, texts):
    full_text = [i+"[SEP]"+j for (i,j) in zip(titles,texts)]
    label_id = [i-1 for i in labels]
    n_class = max(label_id)+1
    config.set("data","label_class", n_class)
    if is_train:
        label_id = flip_label(label_id, n_class, args.noise_type, args.noise_ratio)
    tokenizer = get_tokenizer_instance(config.get("model","pre_bert"))

    token_id=tokenizer(full_text,padding=True,truncation=True,max_length=config.getint("data","sentence_len"),return_tensors='pt')
    label_id=torch.tensor(label_id)

    #封装数据
    datasets=DataToDataset(token_id,label_id)

    BATCH_SIZE=config.getint("data","batch_size")

    data_loader=DataLoader(dataset=datasets,batch_size=BATCH_SIZE,shuffle=True,num_workers=5)

    return data_loader


def read_data_cl(config, args, is_train, labels, titles, texts):
    full_text = [i+"[SEP]"+j for (i,j) in zip(titles,texts)]
    label_id = [i-1 for i in labels]
    n_class = max(label_id)+1
    config.set("data","label_class", n_class)
    if is_train:
        label_id = flip_label(label_id, n_class, args.noise_type, args.noise_ratio)
    tokenizer = get_tokenizer_instance(config.get("model","pre_bert"))

    token_id=tokenizer(full_text,padding=True,truncation=True,max_length=config.getint("data","sentence_len"),return_tensors='pt')

    input_id = np.array(token_id["input_ids"])
    attmsk_id = np.array(token_id["attention_mask"])
    X = np.concatenate((input_id, attmsk_id), axis=1)
    y = np.array(label_id)
    return X, y
   

def process_csv(data_path):
    file = pd.read_csv(data_path,header=None)
    file = file.sample(frac=1)
    lst = file.values.tolist()
    labels = [i[0] for i in lst]
    titles = [i[1] for i in lst]
    texts  =[i[2] for i in lst]
    data = {}
    data["labels"] = labels
    data["titles"] = titles
    data["texts"] = texts
    return data

def split_data(data, train_ratio=0.9):
    # split data into train and valid set
    num = len(data["labels"])
    train = {
        "labels": data["labels"][:int(train_ratio * num)],
        "titles": data["titles"][:int(train_ratio * num)],
        "texts": data["texts"][:int(train_ratio * num)]
    }
    valid = {
        "labels": data["labels"][int(train_ratio * num):],
        "titles": data["titles"][int(train_ratio * num):],
        "texts": data["texts"][int(train_ratio * num):]
    }
    
    return train, valid

def flip_label(y, n_class, pattern, ratio, one_hot=False):
    # y: true label, one hot
    # pattern: 'pair' or 'sym'
    # p: float, noisy ratio
    
    # convert one hot label to int
    # import pdb 
    # pdb.set_trace()
    if one_hot:
        y = np.argmax(y,axis=1)#[np.where(r==1)[0][0] for r in y]
    
    #filp label
    # print(y)
    for i in range(len(y)):
        if pattern=='sym':
            p1 = ratio/(n_class-1)*np.ones(n_class)
            p1[y[i]] = 1-ratio
            y[i] = np.random.choice(n_class,p=p1)
        elif pattern=='asym':
            y[i] = np.random.choice([y[i],(y[i]+1)%n_class],p=[1-ratio,ratio])            
            
    #convert back to one hot
    if one_hot:
        y = np.eye(n_class)[y]
    # print(y)
    return y



