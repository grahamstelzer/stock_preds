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

    # append stock datapoints to end of the input sentences
    
    # split data:


    
    # input matrix will then be ((seq_len_hline + # stock_labels), d_model)



