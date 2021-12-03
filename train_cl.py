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

from sklearn.base import BaseEstimator

from cleanlab.classification import LearningWithNoisyLabels

os.environ["TOKENIZERS_PARALLELISM"] = "true"

class TrainDataset(Dataset):
    def __init__(self, text_att, label) -> None:
        super().__init__()
        self.text, self.attmsk = np.hsplit(text_att, 2)
        self.label = label
    
    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        return self.text[index], self.attmsk[index], self.label[index]


class PredDataset(Dataset):
    def __init__(self, text_att) -> None:
        super().__init__()
        self.text, self.attmsk = np.hsplit(text_att, 2)
    
    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        return self.text[index], self.attmsk[index]


class MyModel(BaseEstimator):
    def __init__(
        self,
        model,
        optimizer,
        loss_func,
        valid_x=None,
        valid_y=None,
        batch_size=32,
        epochs=10,
        log_interval=100
    ):
        self.model = model
        self.optimizer = optimizer
        self.loss_func = loss_func
        self.batch_size = batch_size
        self.epochs = epochs
        self.log_interval=log_interval
        self.valid_x = valid_x
        self.valid_y = valid_y


    def fit(self, X, y, sample_weight=None):
        train_set = TrainDataset(X, y)
        train_loader = DataLoader(
            dataset=train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=5
        )
        for epoch in range(1, self.epochs + 1):
            self.model.train()
            for batch_idx, (data, attmsk, target) in enumerate(train_loader):
                data, attmsk, target = Variable(data).cuda(), Variable(attmsk).cuda(), Variable(target).cuda()
                self.optimizer.zero_grad()
                output = self.model(data, attmsk, is_training=True)
                loss = self.loss_func(output, target)
                loss.backward()
                self.optimizer.step()
                if self.log_interval is not None and \
                    batch_idx % self.log_interval == 0:
                    print(
                        'TrainEpoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                            epoch, batch_idx * len(data), len(X),
                            100. * batch_idx / len(train_loader),
                            loss.item()),
                    )
            print("Train Acc:", self.score(X, y)[0])
        if self.valid_x is not None and self.valid_y is not None:
            print("Valid Acc:", self.score(self.valid_x, self.valid_y))
        print()


    def predict(self, X):
        probs = self.predict_proba(X)
        return probs.argmax(axis=1)


    def predict_proba(self, X):
        pred_set = PredDataset(X)
        pred_loader = DataLoader(
            dataset=pred_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=5
        )
        self.model.eval()

        outputs = []
        for text, attmsk in pred_loader:
            with torch.no_grad():
                text = Variable(text)
                attmsk = Variable(attmsk)
                output = self.model(text, attmsk, is_training=False)
                outputs.append(output)

        outputs = torch.cat(outputs, dim=0)
        out = outputs.cpu().numpy()
        pred = np.exp(out) # the loss func should be nllloss
        return pred


    def score(self, X, y, sample_weight=None):
        pred = self.predict(X)
        val_metric = metric_cl(pred, y, sample_weight)
        return val_metric


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
train_data, valid_data = split_data(all_data)
X_train, y_train = read_data_cl(config, args, True, **train_data)
X_valid, y_valid = read_data_cl(config, args, False, **valid_data)

basemodel=get_model(config, args, usegpu=True)
basemodel=torch.nn.DataParallel(basemodel).cuda()

loss_func=nn.NLLLoss()
optimizer=optim.Adam(basemodel.parameters(), lr=config.getfloat("train", "learning_rate"))

clf = MyModel(
    model=basemodel,
    optimizer=optimizer,
    loss_func=loss_func,
    batch_size=config.getint("data", "batch_size"),
    epochs=10,
    log_interval=100,
    valid_x=X_valid,
    valid_y=y_valid
)
# clf.fit(X_train, y_train)
lnl = LearningWithNoisyLabels(clf=clf)
lnl.fit(X_train, y_train)
res = lnl.score(X_valid, y_valid)
print("Result:", res)