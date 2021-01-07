import numpy
import math
import time
import datetime
from datetime import date
from keras.models import load_model
from keras.models import *
import pandas as pd
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import *
from keras.optimizers import *
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from visual import plot

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

def create_pred_for_plot(dataset):
    dataX = []
    print("dataset ", dataset, dataset.shape)
    # kanw epanalamvanomena append to teleutaio stoixeio kathe row se ena teliko numpy array
    for i in range(dataset.shape[0]):
        # dataset[i].size
        #print("------------------dataset[",i,"][0] ",dataset[i][0])
        a = dataset[i][0]
        dataX.append(a)
    for i in range(1,dataset.shape[1] ):
        # dataset[i].size
        #print("===dataset[241][",i,"] ",dataset[dataset.shape[0]-1][i])
        a = dataset[dataset.shape[0]-1][i]
        dataX.append(a)

    return numpy.array(dataX)

#dinei enan pinaka eisodou dataX, opou kathe row einai mia timeseries look_back mhkous
# kai ena dataY opou kathe row einai to output tou diktyou gia tis epomenes +step_size times
def create_dataset(dataset, look_back=0, step_size=0):
    dataX, dataY = [], []
    print("DATASET \n", dataset)
    print("DATASET[1228] DATASET[1229] \n", dataset[1227],dataset[1228])
    for i in range(len(dataset) - look_back -step_size + 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        #print("DATAx \n", dataX, "\n a \n", a)
        b = dataset[i + look_back:(i + look_back + step_size), 0]
        dataY.append(b)
        #print("DATaY \n", dataY, "\n b \n", b)
    #print("XLOOKBACK \n",dataX,"\n TargetY \n",dataY)
    return numpy.array(dataX), numpy.array(dataY)

def create_dataset_ofAll(dataset, look_back=0):
    dataX = []
    for i in range(len(dataset)- look_back + 1):
        a = dataset[i:(i + look_back), 0]
        print(i," a ",dataset[i:(i + look_back), 0])
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
print("Sizes all, train and test ", all_size, train_size, test_size)

all, train, test = dataset[:, :], dataset[0:train_size, :], dataset[train_size:len(dataset), :]

# posa steps istorikou krataei, dld gia eisodo x[0:lookback-1] provlepei thn eksodo x[lookback]
look_back = 10
# posa steps tha kanw prediction
next_pred = 30

X_lookback, ylookback = create_dataset(dataset, look_back, next_pred)
X_all = create_dataset_ofAll(dataset,10)
print("X_all\n", X_all, X_all.shape)

print("X_lookback ylookback\n", X_lookback, X_lookback.shape, ylookback, ylookback.shape)
# Splitting the dataset into the Training set and Test set
trainX, testX, trainY, testY = train_test_split(X_lookback, ylookback, test_size=0.2, random_state=0, shuffle=False)
print("TrainY testY\n", trainY, trainY.shape, testY, testY.shape)
print("trainX testX\n", trainX, trainX.shape, testX, testX.shape)

#trainX, trainY = create_dataset(train, look_back,next_pred)
#testX, testY = create_dataset(test, look_back,next_pred)
all = numpy.reshape(all, (all.shape[0], 1, all.shape[1]))
X_all = numpy.reshape(X_all, (X_all.shape[0], 1, X_all.shape[1]))
trainX = numpy.reshape(trainX, (trainX.shape[0], 1, trainX.shape[1]))
testX = numpy.reshape(testX, (testX.shape[0], 1, testX.shape[1]))
print("SHAPES all, train and test\n", all.shape, trainX.shape, testX.shape)

op =tf.keras.optimizers.SGD(learning_rate=0.01)# for gradient cliping use parameter clipnorm=1.0

# model
model = Sequential()
model.add(LSTM(200, input_shape=(1, look_back),dropout=0.0,recurrent_dropout=0.2))
model.add(Dropout(.2))
model.add(Dense(units=next_pred, kernel_initializer='uniform', activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['accuracy'])#optimizer is 'adam' or op that has SGD(opt is worse)
start_time = time.time()
# Fitting the LSTM to the Training set
history = model.fit(trainX, trainY, validation_data=(testX, testY),epochs=5, batch_size=1, verbose=2)
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
#print("WEIGHTS ARE \n",weights,"with length: ", len(weights))
# save model
model.save('STOCK-LSTM.h5')

# make predictions
trainPredict = model.predict(numpy.array(trainX))
print("trainX shape ",trainX.shape,"\ntrainPredict ",trainPredict.shape,"\ndataset ",dateset.shape)
testPredict = model.predict(numpy.array(testX))
# invert predictions
trainPredict = scaler.inverse_transform(trainPredict)
print("all \n", all, all.shape)
trainY = scaler.inverse_transform(trainY)
testPredict = scaler.inverse_transform(testPredict)

testY = scaler.inverse_transform(testY)
# calculate root mean squared error
trainScore = math.sqrt(mean_squared_error(trainY[:,0], trainPredict[:, 0]))
print('Train Score: %.2f MSE' % (trainScore))
testScore = math.sqrt(mean_squared_error(testY[:,0], testPredict[:, 0]))
print('Test Score: %.2f MSE' % (testScore))

# fix data for plotting
print("trainPredict \n", trainPredict, trainPredict.shape)
trainPredict_mod = create_pred_for_plot(trainPredict)
print("TRAINPREDICT_MOD SHAPE ",trainPredict_mod,trainPredict_mod.shape)

print("testPredict \n", testPredict, testPredict.shape)
testPredict_mod = create_pred_for_plot(testPredict)
print("TESTPREDICT_MOD SHAPE ",testPredict_mod,testPredict_mod.shape)

# rebuild the Structure
dataset_test_total = pd.DataFrame()
#dataset_test_total['real'] = trainPredict_mod
#dataset_test_total['predicted'] = testPredict_mod
#plot(predicted=dataset_test_total[:, 1], real=dataset_test_total[:, 0])
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
print("Predictions ", predictX2, predictX2.shape)
model = load_model('STOCK-LSTM.h5')

#for i in range(next_pred):
    #pr = model.predict(predictX2[-1:,-1:,:])
    #row_to_be_added = addtoLastRow(predictX2, pr[-1])
    #row_to_be_added = numpy.reshape(row_to_be_added,(row_to_be_added.shape[0], 1, row_to_be_added.shape[1]))
    #predictX2 = numpy.delete(predictX2, 0, 0)
    #predictX2 = numpy.vstack((predictX2, row_to_be_added))
    #print("------->",i," pred ",pr)
ad = len(all)
All_data_values_pred = numpy.array(all[ad-look_back:])
print("===================================================")
print("all length: ", ad, "All_data_values_pred",All_data_values_pred,All_data_values_pred.shape)
reshaped_row = numpy.reshape(All_data_values_pred,(1,1,look_back))
print("LAST ROW ",reshaped_row, reshaped_row.shape)
print("X_all ",X_all, X_all.shape)
allpred = model.predict(numpy.array(X_all))
print("allpred",allpred,allpred.shape)
last30pred = allpred[-1][:]
print("LAST 30 pred ",last30pred,last30pred.shape)
allpred_mod = create_pred_for_plot(allpred)
plt.plot(allpred_mod)
plt.grid()
plt.title("All predictions plus the extra 30 unknown!!!")
plt.show()

pr = model.predict(predictX2)


print("predictX2 shape ",predictX2.shape,"\npr ",pr.shape)

# plot baseline and predictions
#x_values = numpy.linspace(lower, upper, n_points)
# all predictions plus the 30 forecasted
# invert predictions
pr = scaler.inverse_transform(pr)
last30pred = numpy.reshape(last30pred,(1,last30pred.shape[0]))
last30pred = scaler.inverse_transform(last30pred)
pr_mod = create_pred_for_plot(pr)
last30pred_mod = create_pred_for_plot(last30pred)
print("pr_mod \n",pr_mod,pr_mod.shape,"last30pred_mod \n",last30pred_mod,last30pred_mod.shape)

# because some vlues of the test prediction collide with some values in the train predictions, we use the vector
# in input_1d that concatenates the last next_pred-1 colliding values of the trainin predictions and uses the first
# next_pred test predictions
inputs_1d = numpy.append(trainPredict_mod[:trainPredict_mod.shape[0]-1-next_pred], pr_mod)
inputs_1d = numpy.append(inputs_1d, last30pred)

#inputs_1d = numpy.append(inputs_1d, last30pred_mod)
print("trainPredict_mod \n",trainPredict_mod,trainPredict_mod.shape,"pr_mod \n",pr_mod,pr_mod.shape)
print("INPUT_1D \n",inputs_1d,inputs_1d.shape)

dataset_pred_thirty = pd.DataFrame()
dataset_pred_thirty['predicted'] = inputs_1d
dataset_pred_thirty['still_predicted'] = inputs_1d
print("prediction dataframe ", dataset_pred_thirty)
s = scaler.fit_transform(dataset_pred_thirty)
all_thirty_predictions = scaler.inverse_transform(s)

## Visualising the results
print("predicted inversed ", all_thirty_predictions[:, 1],all_thirty_predictions[:, 1].shape)
## shift by +lookback th predictions for plotting
plt.plot(range(look_back,look_back+len(all_thirty_predictions[:, 1])), all_thirty_predictions[:, 1])
plt.title('All predicted Stock Price')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.grid()
plt.show()

# shift 30 last predictions for plotting
#last30predPlot = numpy.empty_like(dataset)
#last30predPlot[:, :] = numpy.nan
last30pred_mod = numpy.reshape(last30pred_mod,(last30pred_mod.shape[0],1))
print("dataset shape trainpredict shape",len(dataset),len(trainPredict))
#last30predPlot[len(dataset) :(len(dataset) +next_pred ) , :]= last30pred_mod
plt.plot(trainPredictPlot)
plt.plot(testPredictPlot)
plt.plot(range(len(dataset) ,len(dataset)+ next_pred ),last30pred_mod)

plt.title('All predicted Stock Price 2')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.grid()
plt.show()

# Plot the test predicted values
plt.plot(range(len(trainPredict)+ (look_back * 2) +next_pred +1,len(trainPredict)+ (look_back * 2) +next_pred +1+len(pr[:, 0])),pr[:, 0])
plt.title('TestPredict Stock Price')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.grid()
plt.show()

all = numpy.reshape(all,(all.shape[0]))
dataset_all_real_values = pd.DataFrame()
dataset_all_real_values['real'] = all
dataset_all_real_values['still_real'] = all
dataset_all_real_values = scaler.inverse_transform(dataset_all_real_values)
print("real dataframe ", dataset_all_real_values)

allpred_mod = numpy.reshape(allpred_mod,(allpred_mod.shape[0]))
dataset_all_pred_values = pd.DataFrame()
dataset_all_pred_values['real'] = allpred_mod
dataset_all_pred_values['still_real'] = allpred_mod
dataset_all_pred_values = scaler.inverse_transform(dataset_all_pred_values)
scaler.inverse_transform(dataset_all_pred_values)
print("============= pred dataframe ==========\n", dataset_all_pred_values)


# plot real vs all predicted
#plt.plot(range(look_back,len(all_thirty_predictions[:,1])+look_back),all_thirty_predictions[:,1], color='red', label='Predicted Stock Price')
plt.plot(range(look_back,len(dataset_all_pred_values[:,0])+look_back),dataset_all_pred_values[:,0], color='red', label='Predicted Stock Price')
plt.plot(dataset_all_real_values[:,0], color='blue', label='Real Stock Price')
plt.title('Stock Price Prediction vs Real')
plt.xlabel('Index')
plt.ylabel('Stock Price')
plt.savefig('lstm_predictedVSreal.png')
plt.legend()
plt.grid()
plt.show()

#plot(predicted=all_thirty_predictions[:, 1], real=dataset_all_real_values[:,0])