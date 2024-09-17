import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import *
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures
from sklearn.metrics import r2_score

ohe = OneHotEncoder()

# # Fit and transform the 'Gender' column
# encoded_data = ohe.fit_transform(df[['Gender']])
#
# # Create a DataFrame with the encoded columns
# encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(['Gender']))
#
# # Concatenate the original DataFrame with the encoded DataFrame
# df = pd.concat([df, encoded_df], axis=1).drop('Gender', axis=1)


file_path = r"F:\Personal\JORGRO\ML_practice\ML_Course\raw_data\baseballplayer.csv"


def readFile(file_path):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[:, 0].values
    y = dataset.iloc[:, -1].values
    plt.scatter(X, y, color='red')
    plt.title('Angle vs Distance')

    plt.xlabel('Angle')
    plt.ylabel('Distance')
    plt.show()

    return X, y

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

def model_train(X_train, X_test, y_train, y_test):
    X_train = np.reshape(X_train, (-1, 1))
    y_train = np.reshape(y_train, (-1, 1))

    X_test = np.reshape(X_test, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    poly_reg = PolynomialFeatures(degree=4)
    tran_X = poly_reg.fit_transform(X_train)
    est_X = poly_reg.fit_transform(X_test)

    regressor = LinearRegression()
    regressor.fit(tran_X, y_train)

    y_pred = regressor.predict(est_X)

    r_square = r2_score(y_test, y_pred)
    print('The R Square explanation power is :', r_square)

    return X_train, X_test, y_train, y_test, y_pred, regressor, poly_reg

def visualize_result(X, y, X_train, X_test, y_train, y_test, y_pred, lin_regressor, poly_reg):
    X = np.reshape(X, (-1, 1))
    plt.scatter(X, y, color='red')
    plt.plot(X, lin_regressor.predict(poly_reg.fit_transform(X)), color='green')
    #plt.plot(X_test[:, -1], lin_regressor.predict(X_test), color='yellow')
    plt.title('Area price of land')
    plt.xlabel('Area of land in thousands of SQFT')
    plt.ylabel('Price of the land in million USD')
    plt.show()
    print('viz')


if __name__ == '__main__':
    X, y = readFile(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)

    X_train, X_test, y_train, y_test, y_pred, lin_regressor , poly_reg= model_train(X_train, X_test, y_train, y_test)

    visualize_result(X, y, X_train, X_test, y_train, y_test, y_pred, lin_regressor, poly_reg)