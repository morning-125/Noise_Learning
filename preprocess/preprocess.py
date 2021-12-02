import pandas as pd
# file = pd.read_csv('/data1/qd/noise_master/dbpedia_csv/train.csv',header=None)

data_path = '/data1/qd/noise_master/ag_news/ag_news_csv/train.csv'
small_data = '/data1/qd/noise_master/ag_news/ag_news_csv/small.csv'

def cut_data(data_path,ratio=1):
    f = pd.read_csv(data_path,header=None)
    shuffle_csv = f.sample(frac=ratio)
    return shuffle_csv

small = cut_data(data_path,0.05)
# print(small.head(10))
small.to_csv(small_data,header=None,index=False)
# f = pd.read_csv(small_data,header=None)
# tt = f.values.tolist()
# print(tt[::])

