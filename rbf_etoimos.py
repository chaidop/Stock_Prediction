# from https://github.com/mohabmes/StockNN/blob/master/RBF/RBF-stock.py

#######!!!!!!!  NOTES:
### koitaw kai ston prohgoumeno kwdika pou eixa 1 input_dim an allaksw ton kwdika gia to 30 prediction
### me auton se auto to arxeio
import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
from keras.initializers import Initializer
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from rbflayer import RBFLayer, InitCentersRandom
from keras.models import load_model
from err import error_count, calc_diff
from visual import plot
from sklearn.svm import SVR
import matplotlib.pyplot as plt
import time


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), 0]
        dataX.append(a)
        dataY.append(dataset[i + look_back, 0])
    return np.array(dataX), np.array(dataY)


def addtoLastRow(dataset,valuetoAdd):
    newLastRow = []
    a = dataset[-1]  # to a exei thn 1h granmmh
    #print(a.shape," ==============================IN APPEND a is ", a)
    for i in range(1,a.shape[1]):
        newLastRow.append(a[0][i])
    #print(valuetoAdd.shape, " Value While a is ", valuetoAdd[0][0])
    x = valuetoAdd[0][0]
    newLastRow.append(x)
    row_to_be_added = np.array(newLastRow)
    row_to_be_added = row_to_be_added.reshape(1, row_to_be_added.shape[0])
    #print(row_to_be_added.shape, " LAST ROW While a is ", row_to_be_added)
    return row_to_be_added


# gia to real_predictions, kanw append ola ta #lookback vectors se ena vector numpy array
def append_values(dataset):
    dataX = []
    a = dataset[0]  # to a exei thn 1h granmmh
    # print(a.shape," IN APPEND a is ", a)
    for i in range(a.shape[1]):
        dataX.append(a[0][i])
    # kanw epanalamvanomena append to teleutaio stoixeio kathe row se ena teliko numpy array
    for i in range(1, dataset.shape[0]):
        # dataset[i].size
        # print("dataset[i].size ",dataset[i].size)
        a = dataset[i][-1][-1]
        dataX.append(a)
    # print("dataX ",dataX)
    return np.array(dataX)


# gia na kanei plot ta predict me ta real values
def create_real_for_plot(dataset, predicted_stock_price):
    dataX = []
    # kanw epanalamvanomena append to teleutaio stoixeio kathe row se ena teliko numpy array
    for i in range(1, dataset.shape[0]):
        # dataset[i].size
        # print("dataset[i].size ",dataset[i].size)
        a = dataset[i][-1][0]
        dataX.append(a)
    # print("dataX ",dataX)
    dataX.append(predicted_stock_price)
    return np.array(dataX)


# gia na kanei TO PRED VALUES SE ENA VECTOR NUMPY
def create_pred_for_plot(dataset):
    dataX = []
    # kanw epanalamvanomena append to teleutaio stoixeio kathe row se ena teliko numpy array
    for i in range(dataset.shape[0]):
        # dataset[i].size
        # print("dataset[i].size ",dataset[i].size)
        a = dataset[i][0][0]
        dataX.append(a)
    # print("dataX ",dataX)
    return np.array(dataX)


def predict_prices(dates, prices, x):
    dates = np.reshape(dates, (len(dates), 1))
    svr_lin = SVR(kernel='linear', C=1e3)
    svr_poly = SVR(kernel='poly', C=1e3, degree=2)
    svr_rbf = SVR(kernel='rbf', C=1e3, gamma=0.1)
    start_time = time.time()
    svr_lin.fit(dates, prices)
    print("--- Linear Fit Complete in %s seconds ---" % (time.time() - start_time))
    print("\n")

    # The Polynomial fitting did not complete within a reasonable time, therefore commenting it out.
    # svr_poly.fit(dates,prices)
    # print("Polynomial Fit Complete")
    start_time = time.time()
    svr_rbf.fit(dates, prices)
    print("--- RBF Fit Complete in %s seconds ---" % (time.time() - start_time))
    print("\n")

    rbf_prediction = svr_rbf.predict(x)[0],
    linear_prediction = svr_lin.predict(x)[0]

    print("RBF Prediction is : ", rbf_prediction)
    print("\n")
    print("Linear Prediction is : ", linear_prediction)

    plt.scatter(dates, prices, color='black', label='Data')
    plt.plot(dates, svr_rbf.predict(dates), color='red', label='RBF model')
    plt.plot(dates, svr_lin.predict(dates), color='blue', label='Linear model')
    # plt.plot(dates,svr_poly.predict(dates), color = 'red', label = 'Polynomial model')
    plt.xlabel('Date')
    plt.ylabel('Price')
    plt.title('Support Vector Regression')
    plt.legend()
    plt.show()
    return rbf_prediction, linear_prediction


############ Data Preprocessing ############
# Importing the dataset
ds = pd.read_csv('Stock_Price_Training_Data.csv')

df = pd.read_csv('Stock_Price_Training_Data.csv', dayfirst=True)
print(df.head())
print('\n Data Types:')
print(df.dtypes)
dates = df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
print("to_datetime is \n", dates)

dataset = ds.iloc[:, [4, 4]].values
all_entries = int(len(dataset))
print("all entries", all_entries)
X = ds.iloc[:all_entries - 1, 4].values
y = ds.iloc[1:all_entries, 4].values
print(X)
# Feature Scaling
scaler = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:all_entries - 1, 0]
y = dataset_scaled[1:all_entries, 1]
print("X ALL", X, "\n y", y, "\n with size x = ", X.shape, " and y = ", y.shape)
lookback = 55
print("X[lookback]", X[lookback-1])
X_lookback, ylookback = create_dataset(dataset_scaled, lookback)
print("LOOKBACK", X_lookback, "\n y", ylookback, "\n with size x = ", X_lookback.shape, " and y = ", ylookback.shape)
# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X_lookback, ylookback, test_size=0.2, random_state=0, shuffle=False)
# X_train, X_test, y_train, y_test = train_test_split(X, y, teist_size = 0.2, random_state = 0)

# Sizes of dataset, train_ds, test_ds
X_all = X_lookback
y_all = ylookback
dataset_sz = X.shape[0]
all_sz = X_all.shape[0]
train_sz = X_train.shape[0]
test_sz = X_test.shape[0]
print("train_sz: ", train_sz, "\ntest_sz: ", test_sz, "\nall_sz: ", all_sz, "\ndataset_sz: ", dataset_sz)
print("X_train \n", X_train)
# X_train = np.reshape(X_train, (train_sz, 1))
# y_train = np.reshape(y_train, (train_sz, 1))
print("shape 1 Xtrain ", X_train.shape, "X_trrain 941th element ", X_train[940])

X_all = np.reshape(X_all, (X_all.shape[0], 1, X_all.shape[1]))
X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

print("AFTER RESHAPE train_sz: ", X_train.shape, "\ntest_sz: ", X_test.shape, "\nall_sz: ", X_all.shape)


print("X_train after transform \n", X_train)
print("reshaped shape 1 Xtrain ", X_train.shape, "X_trrain 941 element ", X_train[940])
print("y_train with size ", y_train.shape)

############ Building the RBF ############
# Initialising the RBF
regressor = Sequential()

# Adding the input layer and the first layer and Drop out Regularization
regressor.add(
    RBFLayer(60, input_dim=lookback, initializer=InitCentersRandom(X_train[0]), betas=2.0, input_shape=(1, lookback)))
regressor.add(Dropout(.2))

# Adding the output layer
regressor.add(Dense(units=1, kernel_initializer='uniform', activation='linear'))

# Compiling the RBF
regressor.compile(optimizer='adam', loss='mean_squared_error')
regressor.summary()
# Fitting the RBF to the Training set
regressor.fit(X_train, y_train, batch_size=1, epochs=5, shuffle=False)

############ Save & load Trained Model ############
# Save Trained Model
regressor.save('TICKER-RBF.h5')

# deletes the existing model
# del regressor

# load Trained Model
# regressor = load_model('TICKER-RBF', custom_objects={'RBFLayer':RBFLayer})

############ Predict & Test the Model ############
real_stock_price = np.array(X_test)
inputs = real_stock_price
predicted_stock_price = regressor.predict(inputs)
# rebuild the Structure
dataset_test_total = pd.DataFrame()
print("real_stock_prices ", real_stock_price, "size ", real_stock_price.shape, "\ninputs ", inputs, "size ",
      inputs.shape, "\n predicted_stock_price ", predicted_stock_price, "size ", predicted_stock_price.shape, )

real_stock_price_oneVector = append_values(real_stock_price)
print("real_stock_prices_oneVector ", real_stock_price_oneVector, "size ", real_stock_price_oneVector.shape, "\n")

real_stock_price_mod = create_real_for_plot(real_stock_price, ylookback[-1])
print("real mod shape ", real_stock_price_mod.shape)
dataset_test_total['real'] = real_stock_price_mod
# h teleutaia timh tou predicted_stock_price den yparxei sto set real,(epeidh einai ena +1 step prediction ektos set)
# opote gia na exoun idies diastaseis, vazw append sto real thn teleutaia timh tou predicted
print("LAST PRED ", predicted_stock_price[-1])

predicted_stock_price_mod = create_pred_for_plot(predicted_stock_price)
print("pred_stock_prices_oneVector ", predicted_stock_price_mod, "size ", predicted_stock_price_mod.shape, "\n")
dataset_test_total['predicted'] = predicted_stock_price_mod
print("Dataset_test_total ", dataset_test_total)

# real data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total)

# count of Wrong predicted value after applying treshold
err_cnt = error_count(predicted_stock_price[:, 0], predicted_stock_price[:, 1], toler_treshold=5.0)

# Calc difference between real data price and predicted price
diff_rate = calc_diff(predicted_stock_price[:, 0], predicted_stock_price[:, 1])
# show the inputs and predicted outputs
for i in range(len(predicted_stock_price[:, 1])):
    print("X=%s, Predicted=%s" % (predicted_stock_price[i, 1], predicted_stock_price[i, 0]))
print("Error count: ", err_cnt, "\n diff rate: ", diff_rate, "\n")
## Visualising the results
plot(predicted=predicted_stock_price[:, 1])
plot(real=predicted_stock_price[:, 0])
plot(predicted=predicted_stock_price[:, 1], real=predicted_stock_price[:, 0])

# MSE
mse = mean_squared_error(predicted_stock_price[:, 0], predicted_stock_price[:, 1])
print("MSE: ", mse)


############ Visualizing the results ############
print("#############################################################")
# prin thn allagh, ola ta X_all kai y_all htan X kai y

inputs = np.array(X_all)
all_real_price = np.array(ylookback)
print("all real price ", all_real_price,all_real_price.shape)
all_predicted_price = regressor.predict(X_all)
all_predicted_price_mod = create_pred_for_plot(all_predicted_price)
print("all stock prediction ", all_predicted_price, all_predicted_price.shape)

print("all inputs ", inputs)
size_of_in = inputs.size
print("input shape ", size_of_in)

list_of_predictions = []
#inputs = all_predicted_price
print("Inputs ", inputs, inputs.shape)
############## Predict the 30 prices in the future with #lookback input########
## inputs is a window of 55 giving you the 55 + 1 prediction
for k in range(30):
    thirty_predicted_price = regressor.predict(inputs)
    print(k, " predicted ", thirty_predicted_price[-1])
    #inputs = np.append(inputs, thirty_predicted_price[-1])
    #inputs = np.delete(inputs, 0 , 0)
    #print("============Inputs ", inputs, inputs.shape)
    #inputs = inputs[1:size_of_in]
    row_to_be_added = addtoLastRow(inputs, thirty_predicted_price[-1])
    row_to_be_added = np.reshape(row_to_be_added,(row_to_be_added.shape[0], 1, row_to_be_added.shape[1]))
    inputs = np.vstack((inputs, row_to_be_added))
    list_of_predictions.append(thirty_predicted_price[-1])
# thirty_predicted_price = np.reshape(thirty_predicted_price, (thirty_predicted_price.shape[0], 1))
print("THIRTY DIMENSIONS ", thirty_predicted_price.shape)
print("predicted 30 ", thirty_predicted_price[0][0][1172:1202])

dataset_pred_real = pd.DataFrame()
dataset_pred_real['real'] = all_real_price
dataset_pred_real['predicted'] = all_predicted_price_mod
print("dataset_pred_real dataframe ", dataset_pred_real)

# real test data price VS. predicted price
all_prices = scaler.inverse_transform(dataset_pred_real)

inputs_1d = append_values(inputs)
dataset_pred_thirty = pd.DataFrame()
dataset_pred_thirty['predicted'] = inputs_1d
dataset_pred_thirty['still_predicted'] = inputs_1d
print("prediction dataframe ", dataset_pred_thirty)

all_thirty_predictions = scaler.inverse_transform(dataset_pred_thirty)
# predicted_price = predict_prices(dates,inputs,31)

## Visualising the results
print("predicted inversed ", all_thirty_predictions[:, 1])
plot(predicted=all_thirty_predictions[:, 1])
plot(real=all_prices[:, 0])
plot(predicted=all_prices[:, 1], real=all_prices[:, 0])

# MSE
mse = mean_squared_error(all_prices[:, 0], all_prices[:, 1])
print("MSE: ", mse)

