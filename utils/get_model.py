from model import *

match_list = {
    'FFBert': BertClassifier
}


def get_model(config, args, usegpu):
    model_name=config.get("model","model_name")
    if model_name in match_list.keys():
        net = match_list[model_name](config, args, usegpu)
        return net
