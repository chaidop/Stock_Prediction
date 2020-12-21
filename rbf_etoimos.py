import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import keras
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
print (df.head())
print ('\n Data Types:')
print (df.dtypes)
dates = df['Date']=pd.to_datetime(df['Date'], dayfirst=True)
print("to_datetime is ",dates)


dataset = ds.iloc[:, [4,4]].values
all_entries = int(len(dataset))
print("all entries", all_entries)
X = ds.iloc[:all_entries-1, 4].values
y = ds.iloc[1:all_entries, 4].values
print(X)
# Feature Scaling
scaler  = MinMaxScaler(feature_range=(0, 1))
dataset_scaled = scaler.fit_transform(dataset)

X = dataset_scaled[:, 0]
y = dataset_scaled[:, 1]

# Splitting the dataset into the Training set and Test set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Sizes of dataset, train_ds, test_ds
dataset_sz = X.shape[0]
train_sz = X_train.shape[0]
test_sz = X_test.shape[0]

X_train = np.reshape(X_train, (train_sz, 1))
y_train = np.reshape(y_train, (train_sz, 1))

############ Building the RBF ############
# Initialising the RBF
regressor = Sequential()

# Adding the input layer and the first layer and Drop out Regularization
regressor.add(RBFLayer(100, input_dim=100,initializer=InitCentersRandom(X_train),betas=2.0, input_shape=(1,100)))
regressor.add(Dropout(.2))

# Adding the output layer
regressor.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'linear'))

# Compiling the RBF
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.summary()
# Fitting the RBF to the Training set
regressor.fit(X_train, y_train, batch_size = 30, epochs = 100,shuffle=False)


############ Save & load Trained Model ############
# Save Trained Model
regressor.save('TICKER-RBF.h5')

# deletes the existing model
#del regressor

# load Trained Model
#regressor = load_model('TICKER-RBF', custom_objects={'RBFLayer':RBFLayer})

############ Predict & Test the Model ############
real_stock_price = np.array(X_test)
inputs = real_stock_price
predicted_stock_price = regressor.predict(inputs)
# rebuild the Structure
dataset_test_total = pd.DataFrame()
dataset_test_total['real'] = real_stock_price
dataset_test_total['predicted'] = predicted_stock_price

# real data price VS. predicted price
predicted_stock_price = scaler.inverse_transform(dataset_test_total)

# count of Wrong predicted value after applying treshold
err_cnt = error_count(predicted_stock_price[:, 0], predicted_stock_price[:, 1], toler_treshold = 5.0)

# Calc difference between real data price and predicted price
diff_rate = calc_diff(predicted_stock_price[:, 0], predicted_stock_price[:, 1])

############ Visualizing the results ############
inputs = np.array(X)

all_real_price = np.array(y)
print("all real price ",all_real_price)
all_predicted_price = regressor.predict(inputs)
print("all_predicted_price DIMENSIONS ", all_predicted_price.shape)

print("all stock prediction ",all_predicted_price)
print("all inputs ",inputs)
size_of_in = inputs.size
print("input shape ",size_of_in)

list_of_predictions =[]
inputs = all_predicted_price

############## Predict the 30 prices in the future ########
for k in range(30):
    thirty_predicted_price = regressor.predict(inputs)
    print(k," predicted ", thirty_predicted_price[-1])
    inputs = np.append(inputs, thirty_predicted_price[-1])
    inputs = inputs[1:size_of_in]
    list_of_predictions.append(thirty_predicted_price[-1])
#thirty_predicted_price = np.reshape(thirty_predicted_price, (thirty_predicted_price.shape[0], 1))
print("THIRTY DIMENSIONS ",thirty_predicted_price.shape)
print("predicted 30 ",thirty_predicted_price)


dataset_pred_real = pd.DataFrame()
dataset_pred_real['real'] = all_real_price
dataset_pred_real['predicted'] = all_predicted_price
print("dataset_pred_real dataframe ",dataset_pred_real)


# real test data price VS. predicted price
all_prices = scaler.inverse_transform(dataset_pred_real)

dataset_pred_thirty = pd.DataFrame()
dataset_pred_thirty['predicted'] = inputs
dataset_pred_thirty['still_predicted'] = inputs
print("prediction dataframe ",dataset_pred_thirty)


all_thirty_predictions = scaler.inverse_transform(dataset_pred_thirty)
#predicted_price = predict_prices(dates,inputs,31)

## Visualising the results
print("predicted inversed ",all_thirty_predictions[:, 1])
plot(predicted=all_thirty_predictions[:, 1])
plot(real=all_prices[:, 0])
plot(predicted=all_prices[:,1], real=all_prices[:, 0])


# MSE
mse = mean_squared_error(predicted_stock_price[:, 0], predicted_stock_price[:, 1])
print("MSE: ",mse)