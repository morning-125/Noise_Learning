import progressbar as pbar
from progressbar import DynamicMessage as DM
from transformers import BertModel,AutoModel,AutoTokenizer,BertTokenizer,BartForConditionalGeneration
from ckip_transformers.nlp import CkipWordSegmenter, CkipPosTagger, CkipNerChunker


def get_progressbar(epoch,epochs, total, train):
    widgets = [pbar.CurrentTime(),' ',
        train,str(epoch),'|',str(epochs),' ',
        pbar.Percentage(), ' ', pbar.Bar('#'), ' ', 
        DM('loss'),' ',DM('acc'),' ',DM('recall'),' ',
        pbar.Timer(), ' ', pbar.ETA(), pbar.FileTransferSpeed()]
    bar = pbar.ProgressBar(widgets=widgets, maxval=total)

    return bar

def get_model_instance(name):
    if name == "bert_base":
        return  AutoModel.from_pretrained("/data1/qd/noise_master/pre_train_models/bert-base-uncased")
    if name == "legal_bert":
        return  BertModel.from_pretrained("/data1/qd/save_model/chinese-legal-electra-small-discriminator")
    if name == "roberta":
        return  AutoModel.from_pretrained("/data1/qd/save_model/chinese-roberta-wwm-ext")
    if name == "lawformer":
        return  AutoModel.from_pretrained("/data1/qd/save_model/Lawformer")
    if name == "xlnet":
        return  BertModel.from_pretrained("/data1/qd/save_model/chinese-xlnet-base")     

def get_tokenizer_instance(name):
    if name == "bert_base":
        return AutoTokenizer.from_pretrained("/data1/qd/noise_master/pre_train_models/bert-base-uncased")
    if name == "legal_bert":
        return BertTokenizer.from_pretrained("/data1/qd/save_model/chinese-legal-electra-small-discriminator")
    if name == "roberta":
        return AutoTokenizer.from_pretrained("/data1/qd/save_model/chinese-roberta-wwm-ext")
    if name == "lawformer":
        return AutoTokenizer.from_pretrained("/data1/qd/save_model/chinese-roberta-wwm-ext")
    if name == "xlnet":
        return AutoTokenizer.from_pretrained("/data1/qd/save_model/chinese-xlnet-base")


def pad(text, text_ner):
    out = []
    for sentence, sentence_ner in zip(text, text_ner):
        sentence_ner = eval(sentence_ner)
        sentence = [i for i in sentence]
        for entity in sentence_ner:
            left = entity[2][0]
            right = entity[2][1]
            sentence[left] = '[PAD]'
            for i in range(left + 1, right):
                sentence[i] = ''
        out.append(''.join(sentence))
    return out

def mask(text, text_ner):
    out = []
    for sentence, sentence_ner in zip(text, text_ner):
        sentence_ner = eval(sentence_ner)
        sentence = [i for i in sentence]
        for entity in sentence_ner:
            left = entity[2][0]
            right = entity[2][1]
            for i in range(left, right):
                sentence[i] = '[MASK]'
        out.append(''.join(sentence))
    return out




# def ner(content):
#     ner_driver = CkipNerChunker(level=3, device=4)
#     ner = ner_driver(input_text=content, show_progress=0)
#     return ner

# content = ["我是一名学生，我今天吃了一顿西餐"]
# x = ner(content)
# import pdb
# pdb.set_trace()