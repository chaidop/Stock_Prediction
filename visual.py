import math
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def plot(predicted=None, real=None):
    if predicted is not None and real is not None:
        plt.plot(real[:], color='red', label='Real Stock Price')
        plt.plot(predicted[:], color='blue', label='Predicted Stock Price')
        plt.title('Stock Price Prediction')
        plt.xlabel('Index')
        plt.ylabel('Stock Price')
        plt.savefig('stock_price_predicted_real.png')
        plt.legend()
        plt.show()

    elif predicted is not None:
        plt.plot(predicted[:], color='blue', label='Predicted Stock Price')
        plt.title('Predicted Stock Prices')
        plt.xlabel('Index')
        plt.ylabel('Stock Price')
        plt.savefig('Predicted_stock_price.png')
        plt.show()

    elif real is not None:
        plt.plot(real[:], color='red', label='Real Stock Price')
        plt.title('Real Stock Prices')
        plt.xlabel('Index')
        plt.ylabel('Stock Price')
        plt.savefig('Real_stock_price.png')
        plt.show()
