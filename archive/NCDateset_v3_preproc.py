import os
import pandas as pd

path = "archive/News_Category_Dataset_v3.json"

# CSV data like such: 
# {"link": "..." ,                 | probably not needed
#  "headline": "...",              | 
#  "category": "...",              | TODO: sort by?
#  "short_descriptions": "...",    | 
#  "authors": "...",               | TODO: possible "and" or "," with multiple authors
#  "date": "..."}                  | TODO: find first and last

all_data = pd.read_csv(path)
all_data = all_data.dropna()

print(f"num lines = {all_data}")

# map all items by date:
# date_map = {}
# for index, row in all_data.iterrows():
#     sublist = []
#     sublist.append(row['headline'])
#     sublist.append(row['category'])
#     sublist.append(row['short_descriptions'])
#     sublist.append(row['authors'])

#     date_map[row['data']] = 
