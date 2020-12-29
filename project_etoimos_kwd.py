import numpy
import math
import datetime
from datetime import date
from keras.models import load_model
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

def addtoLastRow(dataset,valuetoAdd):
    newLastRow = []
    a = dataset[-1]  # to a exei thn 1h granmmh
    print(a.shape," ==============================IN APPEND a is ", a)
    for i in range(1,a.shape[1]):
        newLastRow.append(a[0][i])
    print(valuetoAdd.shape, " Value While a is ", valuetoAdd[0])
    x = valuetoAdd[0]
    newLastRow.append(x)
    row_to_be_added = numpy.array(newLastRow)
    row_to_be_added = row_to_be_added.reshape(1, row_to_be_added.shape[0])
    print(row_to_be_added.shape, " LAST ROW While a is ", row_to_be_added)
    return row_to_be_added


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return numpy.array(dataX), numpy.array(dataY)


# stock = input('Enter stock symbol')

df = pd.read_csv('Stock_Price_Training_Data.csv', usecols=[4], engine='python', skipfooter=3)
dataset = df.values
dateset = dataset.astype('float32')
print(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size
train, test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# posa steps istorikou krataei, dld gia eisodo x[0:lookback-1] provlepei thn eksodo x[lookback]
look_back = 10
# posa steps tha kanw prediction
next_pred = 30
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

# model
model = Sequential()
model.add(LSTM(10, input_shape=(1, look_back)))
model.add(Dense(1))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])

model.fit(trainX, trainY, epochs=10, batch_size=1, verbose=2)
# save model
model.save('STOCK-LSTM.h5')

# make predictions
trainPredict = model.predict(trainX)
print("trainX shape ",trainX.shape,"\ntrainPredict ",trainPredict.shape,"\ndataset ",dateset.shape)
testPredict = model.predict(testX)
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform([trainY])
testPredict = scaler.inverse_transform(testPredict)
testY = scaler.inverse_transform([testY])
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[0], trainPredict[:, 0]))
print('Train Score: %.2f MSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[0], testPredict[:, 0]))
print('Test Score: %.2f MSE' % (testScore))
# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back, :] = trainPredict
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1, :] = testPredict
# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
#plt.show()

# we will use model to get the 30 predictions in the future
predictX2 = testX
print("Predictions ", predictX2, predictX2.shape)
model = load_model('STOCK-LSTM.h5')
for i in range(next_pred):
    pr = model.predict(predictX2)
    row_to_be_added = addtoLastRow(predictX2, pr[-1])
    row_to_be_added = numpy.reshape(row_to_be_added,(row_to_be_added.shape[0], 1, row_to_be_added.shape[1]))
    predictX2 = numpy.delete(predictX2, 0, 0)
    predictX2 = numpy.vstack((predictX2, row_to_be_added))
pr = model.predict(predictX2)

# Delete the unwanted extra rows that are not the 30 future predictions
#for i in range(testX.shape[0]):
    #pr = numpy.delete(pr, 0, 0)

print("predictX2 shape ",predictX2.shape,"\npr ",pr.shape)
# invert predictions
pr = scaler.inverse_transform(pr)
# shift 30 predictions for plotting
predictX2Plot = numpy.empty_like(dataset)
print("predictX2Plot shape ",predictX2Plot.shape,"\npp ",pr.shape)

predictX2Plot[:, :] = numpy.nan
#predictX2Plot[len(trainPredict) + (look_back * 2) + 1:len(dataset) - 1 - next_pred, :] = pr
# plot baseline and predictions
#x_values = numpy.linspace(lower, upper, n_points)

# Plot the 30 predicted values
plt.plot(range(len(trainPredict)+ (look_back * 2) +next_pred +1,len(trainPredict)+ (look_back * 2) +next_pred +1+len(pr[:, 0])),pr[:, 0])
plt.grid()
plt.show()


last_val = testPredict[-1]
last_val_scaled = last_val / last_val
t = numpy.reshape(last_val_scaled, (1, 1, 1))
print(last_val, "scaled", last_val_scaled, " t", t)
next_val = model.predict(numpy.reshape(last_val_scaled, (1, 1, 1)))
print("Last Day Value:", numpy.asscalar(last_val))
print("Next Day Value:", numpy.asscalar(last_val * next_val))
