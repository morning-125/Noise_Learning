import torch
import torch.nn as nn

from utils.common import *

class BertClassifier(nn.Module):
    def __init__(self, config, args, usegpu=True):
        super(BertClassifier, self).__init__()

        self.encoder = get_model_instance("bert_base")
        for parm in self.encoder.parameters():
            parm.requires_grad=True

        self.classifier1 = torch.nn.Linear(768, config.getint("model","hidden_size"))
        self.classifier2 = torch.nn.Linear(config.getint("model","hidden_size"), config.getint("data","label_class"))
        self.dropout = torch.nn.Dropout(config.getfloat("train","dropout"))
        self.log_softmax=torch.nn.LogSoftmax(dim=1)


    def forward(self, input_id, attention_mask, is_training=False):
        x = self.encoder(input_id, attention_mask)[0]
        x = x[:,0,:]
        x = self.classifier1(x)
        x = torch.nn.Tanh()(x)

        if is_training:
            x = self.dropout(x)

        x = self.classifier2(x)
        x = self.log_softmax(x)

        return x