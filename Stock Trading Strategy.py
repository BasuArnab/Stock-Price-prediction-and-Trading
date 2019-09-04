import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("BA.csv",index_col=0)
dataset.dropna(inplace=True)

dataset.corr()
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

# Importing the dataset
dataset['MA10'] = dataset['close'].rolling(10).mean()
dataset['MA500'] = dataset['close'].rolling(500).mean()

dataset=dataset[500:]

plt.plot(np.array(dataset['MA10']), color = 'red',label='MA10')
plt.plot(np.array(dataset['MA500']), color = 'blue',label='MA500')
plt.legend(loc='upper left')
plt.show()

#Now this represents 6000 rows of data
#inorder to invest from say the some 300 days
ma10=dataset['MA10'][4107:4407]
plt.plot(ma10, color = 'red',label='MA10 for 300 days')
plt.plot(np.array(dataset['MA500'][4107:4407]), color = 'blue',label='MA500 for 300 days')
plt.legend(loc='upper left')
plt.show()