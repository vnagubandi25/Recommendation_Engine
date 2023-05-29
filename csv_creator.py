import pandas as pd
import linecache as lc
import time
from pandas.io.json import json_normalize

df = pd.DataFrame()
start = time.time()
data = []

for x in range(1, 1587990):
    line = lc.getline('/data/venka/cse272/hw2/data/Automotive2.json',x)
    line = eval(line)
    data.append(line)

new_rows = pd.DataFrame(data)
df = pd.concat([df, new_rows], axis=0)

final_df = df.drop(['style'], axis=1).join(json_normalize(df['style']))
final_df = final_df[final_df['verified'] != False]


checker = final_df.head(10000)
checker.index = pd.RangeIndex(len(checker.index))
checker.to_csv("/data/venka/cse272/hw2/data/checker.csv")

final_df.index = pd.RangeIndex(len(final_df.index))
final_df.to_csv("/data/venka/cse272/hw2/data/all_data.csv")

print(checker.head(100))

model_df = final_df[["reviewerID","asin","overall"]]
model_df.index = pd.RangeIndex(len(model_df.index))
model_df.to_csv("/data/venka/cse272/hw2/data/model_data.csv")