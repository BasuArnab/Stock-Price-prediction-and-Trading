import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv("BA.csv",index_col=0)
dataset.describe()
dataset.dropna(inplace=True)

dataset=dataset/100
dataset['volume']=dataset['volume']/100000

dataset.corr()
from pandas.plotting import scatter_matrix
scatter_matrix(dataset)
plt.show()

def load_data(stock, seq_len):
    amount_of_features = len(stock.columns)
    data = stock.as_matrix() #pd.DataFrame(stock)
    sequence_length = seq_len + 1
    result = []
    for index in range(len(data) - sequence_length):
        result.append(data[index: index + sequence_length])
        
    result = np.array(result)
    row = round(0.9 * result.shape[0])
    print(row)
    train = result[:int(row), :]
    x_train = train[:, :-1]
    y_train = train[:, 0][:,0]
    y_train=np.roll(y_train,-1,axis=0)
    x_test = result[int(row):, :-1]
    y_test = result[int(row):, 0][:,0]
    y_test=np.roll(y_test,-1,axis=0)
    x_train, y_train, x_test, y_test=x_train[:-1,:], y_train[:-1], x_test[:-1,:], y_test[:-1]
    x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1], amount_of_features))
    x_test = np.reshape(x_test, ( x_test.shape[0], x_test.shape[1], amount_of_features))  

    return [x_train, y_train, x_test, y_test]

window = 30
X_train, y_train, X_test, y_test = load_data(dataset, window)



from keras.models import Sequential
from keras.layers.core import Dense, Dropout
from keras.layers.recurrent import LSTM
def build_model():
        model = Sequential()
        model.add(LSTM(128,  input_shape=(30,6), return_sequences=True))
        model.add(Dropout(0.2))
        model.add(LSTM(64, input_shape=(30,6),return_sequences=False))
        model.add(Dropout(0.2))
        model.add(Dense(16,init='uniform',activation='relu'))        
        model.add(Dense(1,init='uniform',activation='relu'))
        model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
        return model


model = build_model()

model.fit(X_train,y_train,batch_size=32,epochs=5,validation_split=0.1)


predictions=model.predict(X_train)
plt.plot(y_train,color='blue', label='y_train')
plt.plot(predictions,color='red', label='prediction')
plt.legend(loc='upper left')
plt.show()


#Look that predictive power !!!!

predictions=model.predict(X_test)
plt.plot(y_test,color='blue', label='y_test')
plt.plot(predictions,color='red', label='prediction')
plt.legend(loc='upper left')
plt.show()

#Much Disappointment
#Why has this happened ?????

pd.DataFrame(y_train).describe()
pd.DataFrame(y_test).describe()
dataset['adjusted close'].plot()
#Our mean is 0.5, 75% of the dataset is below 0.66 
#How is it possible for our model to understand this sudden rise in prices with so little data
#Goes on to show 2 things
#How major a factor does well curated dataset play in Deep Learning
#The real reason why stock prices are damn hard to predict

#It never learnt that prices could go that far high
#even if we include it in the dataset, due to their extremely small numbers, they will be treated as outliers
#But if we take into account the essence of the model
#lotting both individually side by side shows that our model has actuallu understood the relations between columns and the essence of time
#Just can't imagine that high of a price ! XD
predictions=model.predict(X_test)
plt.subplot(2, 1, 1)
plt.plot(y_test[90:150],color='blue', label='y_test')
plt.legend(loc='upper left')
plt.subplot(2, 1, 2)
plt.plot(predictions[90:150],color='red', label='prediction')
plt.legend(loc='upper left')
plt.show()