(included in preproc.py)
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
