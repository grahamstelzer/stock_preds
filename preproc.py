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
import tensorflow as tf




# TODO: I think we HAVE to include the stock data in the matmults leading up to MHA but
#   not sure how to do this yet since for words we just tokenize, but "d,o,h,l,c,ac,v"
#   are just numbers, not words, so how do I append single float values next to the 
#   positional encodings?
#   TODO: - just to try something, I will see if I can just use a vector of the same value
#           where len matches d_model


def preproc():
    # NOTE: d_model stays the same. seq_len is a bit longer
    # take input sentences, embed as normal:

    # take stock datapoints, each value becomes vector of length d_model
    df = pd.read_csv("output.csv") # Date,Open,High,Low,Close,Adj Close,Volume

    # print(df['Date'])
    dates = []
    for date in df['Date']:
        dates.insert(0, date)
    # print(dates)
    


    data_points = []
    for o, h, l, c, a_c, v in df[['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']].values:
        # TODO fix this, all values end up as strings instead of flaots
        print(o.type)
        data_point = [o, h, l, c, a_c, v]
        dp_tensor = torch.tensor(data_point)
        data_points.append(dp_tensor)


    print(data_points)
    # data_points is 1262 by 6 


    # NOTE: should move this have within the model since we should save the resize tensor?
    #       not actually sure though, since we just need encoding per dp
    #       might not need to be learned
    for item in data_points:
        resize_tensor = torch.randn((6,512))
        item = item * resize_tensor

    print(data_points)




    # append stock datapoints to end of the input sentences
    
    # split data:


    
    # input matrix will then be ((seq_len_hline + # stock_labels), d_model)





p = preproc()