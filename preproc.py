# first and foremost: what should the attention matrix relate?
# in translation (seq_len, seq_len) attents words to each other
# ex. "the dog ran fast" by "the dog ran fast" -> "ran" attents to "dog"

# for stocks (seq_len, seq_len) should still work where a seq could be like:
# "Date,Open,High,Low,Close,Adj Close,Volume" by "Date,Open,High,Low,Close,Adj Close,Volume"
# but this will get not that much info since why would the highest price on a day relate to anything
#   else but the highest price on that day? 

# so append something like a headline at the end?
# "Date,Open,High,Low,Close,Adj Close,Volume" + "full headline embedding"
# shorter: ("d,o,h,l,c,ac,v" + "hl_embed") by ("d,o,h,l,c,ac,v" + "hl_embed")
# which should then possibly relate words in the hl_embed to possible values within the stock data

# in other words, we want to attent a headline to a stock price

# and now our model shape, using transformer terminology, becomes:
# (seq_len + stock_data, seq_len + stock_data)

import pandas as pd
import torch
import os
import matplotlib.pyplot as plt

from newsapi import NewsApiClient
from dotenv import load_dotenv
import os


# TODO: I think we HAVE to include the stock data in the matmuls leading up to MHA but
#   not sure how to do this yet since for words we just tokenize, but "d,o,h,l,c,ac,v"
#   are just numbers, not words, so how do I append single float values next to the 
#   positional encodings?
#   TODO: - just to try something, I will see if I can just use a vector of the same value
#           where len matches d_model

def preproc():

    # directory setup:
    # TODO: move these to config.py
    data_dir = 'data'
    subdirs = ['train', 'test', 'validation']

    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    for subdir in subdirs:
        subdir_path = os.path.join(data_dir, subdir)
        if not os.path.exists(subdir_path):
            os.makedirs(subdir_path)


# =======================================================================
# STOCK DATA PREPROC
# NOTE: will organize into different functionts or files later
# =======================================================================



    # NOTE: d_model stays the same. seq_len is a bit longer
    # take input sentences, embed as normal:

    # take stock datapoints, each value becomes vector of length d_model
    # NOTE: output.csv is generated automatically from get_csv.js
    #       so COULD put into config but not really necessary right now
    df = pd.read_csv("output.csv") # Date,Open,High,Low,Close,Adj Close,Volume

    df = df.dropna()
    # remove lines that do not contain relevant data
    df = df[~df.apply(lambda row: row.astype(str).str.contains('Dividend').any(), axis=1)] 
    df = df[~df.apply(lambda row: row.astype(str).str.contains('Splits').any(), axis=1)]


    # save the dates to create positional encodings
    dates = []
    for date in df['Date']:
        dates.insert(0, date)
    print(dates[:5])
    print(len(dates))
    
    # build data_points tensor
    # NOTE: there is probably a faster way to do this but for code clarity I did it this way
    data_points = []
    for index, row in df.iterrows():
        data_point = []
        data_point.append(float(row['Open']))
        data_point.append(float(row['High']))
        data_point.append(float(row['Low']))
        data_point.append(float(row['Close']))
        data_point.append(float(row['Adj Close']))
        data_point.append(float(row['Volume']))



        # Append the data point to the data_points list
        data_points.insert(0, data_point) # using insert at 0 to reverse data so training starts at beginning

    categories = ['Open', 'High', 'Low', 'Close', 'Adj Close']
    for i, category in enumerate(categories):
        plt.plot([data_point[i] for data_point in data_points], label=category)
    plt.xlabel('Time')
    plt.ylabel('Value')
    plt.title('Stock Data Over Time')
    plt.legend()
    plt.show()


    # data_points = (data_points - data_points.mean(dim=0)) / data_points.std(dim=0)
    ret_tensor = torch.tensor(data_points, dtype=torch.float32)
    print(ret_tensor[:5])
    print(ret_tensor.shape)


    # normalize the tensor
    ret_tensor = (ret_tensor - ret_tensor.mean(dim=0)) / ret_tensor.std(dim=0)
    print("normalized:")
    print(ret_tensor[:5])

    # split the tensor into train, test, validation subsets
    train_size = int(0.7 * len(ret_tensor))
    test_size = int(0.2 * len(ret_tensor))
    val_size = len(ret_tensor) - train_size - test_size

    train_tensor, test_tensor, val_tensor = torch.utils.data.random_split(ret_tensor, [train_size, test_size, val_size])

    # Save the tensors to their respective directories
    torch.save(train_tensor, os.path.join(data_dir, 'train', 'normalized_stock_data.pt'))
    torch.save(test_tensor, os.path.join(data_dir, 'test', 'normalized_stock_data.pt'))
    torch.save(val_tensor, os.path.join(data_dir, 'validation', 'normalized_stock_data.pt'))

    # data_points is 1258 by 6
    # NOTE: this is about 3.5 years, meaning about 547 days of accuracy are lost
    #       possibly due to their being no data on those days (weekends, holidays, notrade days etc.)




# =======================================================================
# HEADLINE DATA PREPROC
# =======================================================================






# below directly from https://newsapi.org/docs/client-libraries/python
def get_headlines():


    # Init
    load_dotenv()
    api_key = os.getenv('NEWS_API_KEY')
    newsapi = NewsApiClient(api_key=api_key)

    # /v2/top-headlines
    top_headlines = newsapi.get_top_headlines(q='bitcoin',
                                            sources='bbc-news,the-verge',
                                            category='business',
                                            language='en',
                                            country='us')

    # /v2/everything
    all_articles = newsapi.get_everything(q='bitcoin',
                                        sources='bbc-news,the-verge',
                                        domains='bbc.co.uk,techcrunch.com',
                                        from_param='2017-12-01',
                                        to='2017-12-12',
                                        language='en',
                                        sort_by='relevancy',
                                        page=2)

    # /v2/top-headlines/sources
    sources = newsapi.get_sources()





p = preproc()