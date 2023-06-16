import os
import numpy as np
import pandas as pd


root = '/data/gaf/SBandBasic'
dates = []
nums = []
date_list = sorted(os.listdir(root))
for date in date_list:
    file_list = sorted(os.listdir(os.path.join(root, date)))
    num = 0
    for file_ in file_list:
        path = os.path.join(root, date, file_)
        num += 1
    nums.append(num)
    dates.append(date)

dates, nums = np.array(dates), np.array(nums)
cumsum = np.cumsum(nums)

df = pd.DataFrame(np.array([dates, nums, cumsum]).T, columns=['date', 'num', 'cumsum'])
df.to_csv('results/count.csv', index=False)
