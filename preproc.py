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
import random


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
    stock_df = pd.read_csv("data/Stock_data.csv") # Date,Open,High,Low,Close,Adj Close,Volume

    print(len(stock_df))

    stock_df = stock_df.dropna()
    # remove lines that do not contain relevant data
    stock_df = stock_df[~stock_df.apply(lambda row: row.astype(str).str.contains('Dividend').any(), axis=1)] 
    stock_df = stock_df[~stock_df.apply(lambda row: row.astype(str).str.contains('Splits').any(), axis=1)]

    print(f"Number of items in stock_df: {len(stock_df)}")
# =======================================================================
# get headlines into a df
# =======================================================================

    # Load headline data
    headlines_df = pd.read_csv('data/CNN_articles.csv')

    # extract headline and date from articles
    headlines_map = {}
    for index, row in headlines_df.iterrows():
        date_published = row['Date published'].split(' ')[0]
        headline = row['Headline'].replace(' - CNN', '')
        if date_published not in headlines_map:
            headlines_map[date_published] = []
        headlines_map[date_published].append(headline)


    # build data_points tensor
# TODO: there is something wrong with this and I dont know why
#       - the loop runs 5133 times (tested with a loopcounter)
#       - the number of rows parsed is 5133
#       - the size of combined matrix is 5133
#       - BUT, the indices go from 0 to 5138??
#       - NOTE: the len of the uncleaned stock_df is 5139 BUT
#               i checked above, the length of that becomes 5133
#               after cleening and isnt touched afterwards
    combined = []
    # not sure if ill need a min and max later but saving just in case
    min_val = 999999
    max_val = -999999
    for index, row in stock_df.iterrows():
        numerical_data = []
        # ensure they are all floats
        numerical_data.append(float(row['Open']))
        numerical_data.append(float(row['High']))
        numerical_data.append(float(row['Low']))
        numerical_data.append(float(row['Close']))
        numerical_data.append(float(row['Adj Close']))

        min_val = min(min_val, *numerical_data) # NOTE: asterisk lets each value in numerical_data get passed ot min function
        max_val = max(max_val, *numerical_data)


        # normalize
        # TODO: fix normalization, attempting below causes datapoints like
        # [... 0.5000000000000011, 1.0, 0.0, 0.5500000000000002, 0.5000000000000011 ...]
        # or using small epsilon in denominator:
        # [... 0.4999999750000024, 0.9999999500000025, 0.0, 0.5499999725000015, 0.4999999750000024 ...]
        # epsilon = 1e-8
        # numerical_data = [(x - min(numerical_data)) / (max(numerical_data) - min(numerical_data) + epsilon) for x in numerical_data]

        # use the date to get a headline with that date
        random_dated_headline = ""
        date = row['Date']
        if date in headlines_map:
            random_dated_headline = random.choice(headlines_map[date])
        else:
            random_dated_headline = "No headline available"

        # appending the volume seperately since its much higher than the small values for Open...Adj Close
        combined.append([index] + [date] + numerical_data + [float(row['Volume'])] + [random_dated_headline])
        



    print(f"Number of rows parsed: {len(combined)}")
    print(f"min: {min_val}, max: {max_val}")
    print(combined[:5])
    middle_index = len(combined) // 2
    print(combined[middle_index - 2:middle_index + 3])
    print(combined[-5:])


    # place the headlines in the string sections of each data_point
    # ex:
    # [5132, '2025-01-10', '195.41', 197.62, 191.6, 193.17, 193.17, 18566759.0, '<headline goes here>']


# NOTE: have to pay like 500$ to get access to anything actually useful through newsapi
#       aka. im not gonna do that

# below directly from https://newsapi.org/docs/client-libraries/python
# def get_headlines():

#     # Init
#     load_dotenv()
#     api_key = os.getenv('NEWS_API_KEY')
#     newsapi = NewsApiClient(api_key=api_key)

#     # /v2/top-headlines
#     top_headlines = newsapi.get_top_headlines(q='bitcoin',
#                                             sources='bbc-news,the-verge',
#                                             category='business',
#                                             language='en',
#                                             country='us')

#     # /v2/everything
#     all_articles = newsapi.get_everything(q='bitcoin',
#                                         sources='bbc-news,the-verge',
#                                         domains='bbc.co.uk,techcrunch.com',
#                                         from_param='2017-12-01',
#                                         to='2017-12-12',
#                                         language='en',
#                                         sort_by='relevancy',
#                                         page=2)

#     # /v2/top-headlines/sources
#     sources = newsapi.get_sources()





p = preproc()