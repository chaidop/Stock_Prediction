# univariate lstm example
# source: https://machinelearningmastery.com/how-to-develop-lstm-models-for-time-series-forecasting/
from numpy import *
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
import keras.backend as K
import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time
from sklearn.preprocessing import MinMaxScaler

# function to calculate the efficiency of the network
def efficiency(true, predicted):
    error = 0
    #unscale the data to get the unscaled error, put next two rows in comment to see scaled error
    true = scaler.inverse_transform(true)
    predicted = scaler.inverse_transform(predicted)

    for i in range(len(true)):
        error = error + abs(true[i] - predicted[i])

    error = error/len(true)
    print("ERROR : ", error)

    return error


# split a univariate sequence into samples
def split_sequence(sequence, n_steps):
    X, y = list(), list()
    for i in range(len(sequence)):
        # find the end of this pattern
        end_ix = i + n_steps
        # check if we are beyond the sequence
        if end_ix > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y = sequence[i:end_ix], sequence[end_ix]
        X.append(seq_x)
        y.append(seq_y)
    return array(X), array(y)

#convert excel to csv and extract the 'Close' values
read_file = pd.read_excel("Stock_Price_Training_Data.xlsx")
read_file.to_csv (r'Stock_Price_Training_Data.csv', index = None, header=True)
df = pd.read_csv('Stock_Price_Training_Data.csv', usecols=[4], engine='python')

dataset = df.values
dateset = dataset.astype('float32')
print("dataset", dataset, dataset.shape)

# scale the data
scaler = MinMaxScaler(feature_range=(-1, 1))
dataset = scaler.fit_transform(dataset)


# define input sequence
# choose a number of time steps
n_steps = 30
#number of predicted values
nxt_steps = 30
# split into samples
X, y = split_sequence(dataset, n_steps)

n_features = 1
# reshape from [samples, timesteps] into [samples, timesteps, features]
X = X.reshape((X.shape[0], X.shape[1], n_features))

#split the training set and test set to 80% and 20% of the initial dataset accordingly
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, shuffle=False)

#starting to count the time for the training process
start_time = time.time()

# define model
model = Sequential()
model.add(LSTM(30, activation='relu', input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
# fit model
model.fit(X_train, y_train, epochs=200, verbose=0)
model.summary()

# print the weights (uncomment it to check)

weights = model.get_weights()
print("WEIGHTS ARE \n",weights)
print("LEARNING RATE = ", K.eval(model.optimizer.lr))

#stop the timer and print the total training time
print("--- LSTM Train Complete in %s seconds ---" % (time.time() - start_time))

# demonstrate training set prediction
start_time2 = time.time()
yhat = model.predict(X_train, verbose=0)
print("--- Training Prediction Complete in %s seconds ---" % (time.time() - start_time2))
# calculate efficiency
trainScore = efficiency(y_train, yhat)
print('Train Score: %.2f MSE' % (trainScore))
print("Train Shape ", X_train.shape)
#plot the training data
plt.plot(range(0,len(yhat)), scaler.inverse_transform(yhat), color='purple')
plt.grid()
plt.plot(scaler.inverse_transform(y_train), color='orange')
plt.title('Training prediction VS real')
plt.legend(['predicition', 'real'], loc='upper left')
plt.show()

# shift train predictions for plotting
trainPredictPlot = empty_like(dataset)
trainPredictPlot[:, :] = nan
trainPredictPlot[n_steps:len(y_train) + n_steps , :] = yhat


######################################################################################

#Test set prediction
start_time3 = time.time()
yhat = model.predict(X_test, verbose=0)
print("--- Test Prediction Complete in %s seconds ---" % (time.time() - start_time3))
# calculate efficiency
testScore = efficiency(y_test, yhat)
print('Test Score: %.2f MSE' % (testScore))
print("Test Shape ", X_test.shape)

#plot test data
plt.plot(range(0,len(yhat)), scaler.inverse_transform(yhat))
plt.grid()
plt.plot(scaler.inverse_transform(y_test), color='pink')
plt.title('Test prediction VS real')
plt.legend(['predicition', 'real'], loc='upper left')
plt.show()

testPredictPlot = empty_like(dataset)
testPredictPlot[:, :] = nan
testPredictPlot[len(y_train) + n_steps :len(dataset) +nxt_steps -1 , :] = yhat

#plot training and test together
plt.plot(scaler.inverse_transform(trainPredictPlot), color='pink')
plt.grid()
plt.plot(scaler.inverse_transform(testPredictPlot))
plt.title('All predictions')
plt.show()
#plot training and test together and real
plt.plot(scaler.inverse_transform(dataset))
plt.plot(scaler.inverse_transform(trainPredictPlot), color='purple')
plt.grid()
plt.plot(scaler.inverse_transform(testPredictPlot), color='pink')
plt.title('All predictions VS real')
plt.show()

####################################################################################################
# predict next 30
start_time4 = time.time()
X_next_30 = list()
print("Len(dataset)-n_steps: ",len(dataset)-n_steps,"\n Len(dataset) ",len(dataset))
seq_x = dataset[len(dataset)-n_steps:len(dataset)]
X_next_30.append(seq_x)
X_next_30 = array(X_next_30)
X_next_30 = X_next_30.reshape((X_next_30.shape[0], X_next_30.shape[1], n_features))
yhat_next = model.predict(X_next_30, verbose=0)
seq_y = list()
seq_y.append(yhat_next)
print("X_next_30: \n", X_next_30, X_next_30.shape)


print("====================================")
for i in range(nxt_steps-1):
    seq_x = concatenate((seq_x, yhat_next))
    seq_x = seq_x[1:len(seq_x)]
    X_next_30 = list()
    X_next_30.append(seq_x)
    X_next_30 = array(X_next_30)
    X_next_30 = X_next_30.reshape((X_next_30.shape[0], X_next_30.shape[1], n_features))
    yhat_next = model.predict(X_next_30)
    seq_y.append(yhat_next)



seq_y = array(seq_y)
print("--- Next 30 Complete in %s seconds ---" % (time.time() - start_time4))
seq_y = seq_y.reshape((seq_y.shape[0], n_features))
plt.plot(scaler.inverse_transform(seq_y))
plt.grid()
plt.title('Next 30 predicitons')
plt.show()
print("All 30 predictions are : ", scaler.inverse_transform(seq_y))