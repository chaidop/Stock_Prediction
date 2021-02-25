import numpy
import time
import pandas as pd
import tensorflow as tf
import keras.backend as K
from keras.models import Sequential
from keras.layers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def efficiency(true, predicted):
    error = 0
    for i in range(len(true)):
        error = error + abs(true[i] - predicted[i])

    error = error/len(true)
    print("ERROR : ", error)

    return error

def addtoLastRow(dataset,valuetoAdd):
    newLastRow = []
    a = dataset[-1]  # to a exei thn 1h granmmh
    for i in range(1,a.shape[1]):
        newLastRow.append(a[0][i])
    x = valuetoAdd[0]
    newLastRow.append(x)
    row_to_be_added = numpy.array(newLastRow)
    row_to_be_added = row_to_be_added.reshape(1, row_to_be_added.shape[0])
    return row_to_be_added

def create_pred_for_plot(dataset):
    dataX = []
    # kanw epanalamvanomena append to teleutaio stoixeio kathe row se ena teliko numpy array
    for i in range(dataset.shape[0]):
        a = dataset[i][0]
        dataX.append(a)
    for i in range(1,dataset.shape[1] ):
        a = dataset[dataset.shape[0]-1][i]
        dataX.append(a)

    return numpy.array(dataX)

#dinei enan pinaka eisodou dataX, opou kathe row einai mia timeseries look_back mhkous
# kai ena dataY opou kathe row einai to output tou diktyou gia tis epomenes +step_size times
def create_dataset(dataset, look_back=0, step_size=0):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back -step_size + 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        b = dataset[i + look_back:(i + look_back + step_size), 0]
        dataY.append(b)

    return numpy.array(dataX), numpy.array(dataY)

def create_dataset_ofAll(dataset, look_back=0):
    dataX = []
    for i in range(len(dataset)- look_back + 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)

    return numpy.array(dataX)
# stock = input('Enter stock symbol')

df = pd.read_csv('Stock_Price_Training_Data.csv', usecols=[4], engine='python')
dataset = df.values
dateset = dataset.astype('float32')
print(dataset)
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
all_size = int(len(dataset))
train_size = int(len(dataset) * 0.80)
test_size = len(dataset) - train_size

all, train, test = dataset[:, :], dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# posa steps istorikou krataei, dld gia eisodo x[0:lookback-1] provlepei thn eksodo x[lookback]
look_back = 30
# posa steps tha kanw prediction
next_pred = 30

X_lookback, ylookback = create_dataset(dataset, look_back, next_pred)
X_all = create_dataset_ofAll(dataset,look_back)

# Splitting the dataset into the Training set and Test set
trainX, testX, trainY, testY = train_test_split(X_lookback, ylookback, test_size=0.2, random_state=0, shuffle=False)


all = numpy.reshape(all, (all.shape[0], 1, all.shape[1]))
X_all = numpy.reshape(X_all, (X_all.shape[0], 1, X_all.shape[1]))
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))

op =tf.keras.optimizers.SGD(learning_rate=0.01)# for gradient cliping use parameter clipnorm=1.0

# model
model = Sequential()
model.add(LSTM(30, input_shape=(1, look_back), dropout=0.0,recurrent_dropout=0.2))#activation='relu',return_sequences=True gia na enwthei me to 2o LSTM layer
model.add(Dropout(.2))
model.add(Dense(units=next_pred, kernel_initializer='uniform', activation='relu'))#or linear
model.compile(loss='mean_squared_error', metrics=['accuracy'])#optimizer is 'adam' or op that has SGD(opt is worse)(, optimizer='adam')
start_time = time.time()
# Fitting the LSTM to the Training set
history = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=200, batch_size=1, verbose=2)
model.summary()

# evaluate the model
train_mse = model.evaluate(trainX, trainY, verbose=0)
test_mse = model.evaluate(testX, testY, verbose=0)

print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
# plot loss during training
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

print("Train: , Test: ", (train_mse, test_mse))


print("--- LSTM Fit Complete in %s seconds ---" % (time.time() - start_time))
print("\n")
weights = model.get_weights()
print("WEIGHTS ARE \n",weights,"with length: ", len(weights))
print("LEARNING RATE = ", K.eval(model.optimizer.lr))


# make predictions
start_time2 = time.time()
trainPredict = model.predict(numpy.array(trainX))
print("--- Training Prediction Complete in %s seconds ---" % (time.time() - start_time2))

start_time3 = time.time()
testPredict = model.predict(numpy.array(testX))
print("--- Test Prediction Complete in %s seconds ---" % (time.time() - start_time3))
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform(testY)
# calculate root mean squared error
trainScore = efficiency(trainY[:,0], trainPredict[:, 0])
testScore = efficiency(testY[:,0], testPredict[:, 0])

# fix data for plotting
trainPredict_mod = create_pred_for_plot(trainPredict)
testPredict_mod = create_pred_for_plot(testPredict)

# rebuild the Structure
dataset_test_total = pd.DataFrame()

trainPredict_mod = numpy.reshape(trainPredict_mod,(trainPredict_mod.shape[0],1))
testPredict_mod = numpy.reshape(testPredict_mod,(testPredict_mod.shape[0],1))

# shift train predictions for plotting
trainPredictPlot = numpy.empty_like(dataset)
trainPredictPlot[:, :] = numpy.nan
trainPredictPlot[look_back:len(trainPredict) + look_back +next_pred -1, :] = trainPredict_mod
# shift test predictions for plotting
testPredictPlot = numpy.empty_like(dataset)
testPredictPlot[:, :] = numpy.nan
testPredictPlot[len(trainPredict) + look_back :len(dataset) +next_pred -1 , :] = testPredict_mod

# plot baseline and predictions
plt.plot(scaler.inverse_transform(dataset))
plt.title('Real Stock Price')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.grid()
plt.show()
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.title('TrainPredict and TestPredict Stock Price')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.legend()
plt.grid()
plt.show()

# we will use model to get the 30 predictions in the future
predictX2 = testX

ad = len(all)
All_data_values_pred = numpy.array(all[ad-look_back:])

reshaped_row = numpy.reshape(All_data_values_pred,(1,1,look_back))

#every row of allpred has 30 predictions (columns)
#to plot that, we only take the last prediction from every row and plot this
#with exception to the forst and last(the predictions that do not belong in the data set)
# row which we plot for all their 30 elements
start_time4 = time.time()
allpred = model.predict(numpy.array(X_all))
print("--- Extra 30 predictions Complete in %s seconds ---" % (time.time() - start_time4))

last30pred = allpred[-1][:]
allpred_mod = create_pred_for_plot(allpred)
plt.plot(scaler.inverse_transform(allpred_mod.reshape(-1,1)))
plt.grid()
plt.title("All predictions plus the extra 30 unknown")##final
plt.show()

pr = model.predict(predictX2)


# plot baseline and predictions
# all predictions plus the 30 forecasted
# invert predictions
pr = scaler.inverse_transform(pr)
last30pred = numpy.reshape(last30pred,(1,last30pred.shape[0]))
last30pred = scaler.inverse_transform(last30pred)
pr_mod = create_pred_for_plot(pr)
last30pred_mod = create_pred_for_plot(last30pred)

# because some vlues of the test prediction collide with some values in the train predictions, we use the vector
# in input_1d that concatenates the last next_pred-1 colliding values of the trainin predictions and uses the first
# next_pred test predictions
inputs_1d = numpy.append(trainPredict_mod[:trainPredict_mod.shape[0]-1-next_pred], pr_mod)
inputs_1d = numpy.append(inputs_1d, last30pred)


dataset_pred_thirty = pd.DataFrame()
dataset_pred_thirty['predicted'] = inputs_1d
dataset_pred_thirty['still_predicted'] = inputs_1d
s = scaler.fit_transform(dataset_pred_thirty)
all_thirty_predictions = scaler.inverse_transform(s)


last30pred_mod = numpy.reshape(last30pred_mod,(last30pred_mod.shape[0],1))


all = numpy.reshape(all,(all.shape[0]))
dataset_all_real_values = pd.DataFrame()
dataset_all_real_values['real'] = all
dataset_all_real_values['still_real'] = all
dataset_all_real_values = scaler.inverse_transform(dataset_all_real_values)

allpred_mod = numpy.reshape(allpred_mod,(allpred_mod.shape[0]))
dataset_all_pred_values = pd.DataFrame()
dataset_all_pred_values['real'] = allpred_mod
dataset_all_pred_values['still_real'] = allpred_mod
dataset_all_pred_values = scaler.inverse_transform(dataset_all_pred_values)
scaler.inverse_transform(dataset_all_pred_values)


# plot real vs all predicted
plt.plot(range(look_back,len(dataset_all_pred_values[:,0])+look_back),dataset_all_pred_values[:,0], color='red', label='Predicted Stock Price')
plt.plot(dataset_all_real_values[:,0], color='blue', label='Real Stock Price')
plt.title('Stock Price Prediction vs Real')
plt.xlabel('Index')
plt.ylabel('Stock Price')
#plt.savefig('lstm_predictedVSreal.png')
plt.legend()
plt.grid()
plt.show()


print('Train Score: %.2f MSE' % (trainScore))
print('Test Score: %.2f MSE' % (testScore))
