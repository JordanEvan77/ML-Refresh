import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import tkinter as tk
from tkinter import *

file_path = r"F:\Personal\JORGRO\ML_practice\ML_Course\raw_data\landprice.csv"

def readFile(file_path):
    dataset = pd.read_csv(file_path)
    X = dataset.iloc[0:23, 0].values
    y = dataset.iloc[0:23, 1].values
    return X, y

def split_dat(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

def model_train(X_train, X_test, y_train, y_test):
    X_train1 = np.reshape(X_train, (-1, 1))
    y_train1 = np.reshape(y_train, (-1, 1))
    X_test1 = np.reshape(X_test, (-1, 1))
    y_test1 = np.reshape(y_test, (-1, 1))

    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train1, y_train1)
    X_test1 = np.append(X_test1, 160)
    X_test1 = np.reshape(X_test1, (-1, 1))
    y_test1 = np.append(y_test1, 160)
    y_test1 = np.reshape(y_test1, (-1, 1))
    pred_y = lin_regressor.predict(X_test1)

    r_square = r2_score(y_test1, pred_y)
    print('The R Square explanation power is :', r_square)

    area_test = np.reshape(37, (1,1))
    pred_price = lin_regressor.predict(area_test)
    print('pred_price:', pred_price, lin_regressor.coef_)
    return X_train1, X_test1, y_train1, y_test1, pred_y, lin_regressor

def visualize_result(X_train1, X_test1, y_train1, y_test1, pred_y, lin_regressor):
    plt.scatter(X_train1, y_train1, color='blue')
    plt.scatter(X_test1, y_test1, color='red')
    plt.plot(X_test1, pred_y, color='green')
    plt.title('Area price of land')
    plt.xlabel('Area of land in thousands of SQFT')
    plt.ylabel('Price of the land in million USD')
    plt.show()
    print('viz')

def simple_pred(window, lin_regressor, area):
    area = int(area)
    pred_price = lin_regressor.predict(np.reshape(area, (1,1)))

    label = Label(window, text=str(pred_price), fg='blue', font=('courier', 15))
    label.pack()

def gui_easy(lin_regressor):
    window = Tk()
    window.geometry("600x600")
    window.title("Template Window")
    label = Label(window, text='Area of land, 1,000 sqft', fg='red', font=('Courier', 15))
    label.pack()
    area = StringVar()
    area.set("")
    entry = Entry(window, textvariable=area, fg='green', width=10, font=('courier', 15))
    entry.pack()
    predbutton = Button(window, text='Predict', fg='red', command=lambda: simple_pred(window, lin_regressor, area.get()), height=2, width=15)
    predbutton.pack()
    window.mainloop()

if __name__ == '__main__':
    X, y = readFile(file_path)
    X_train, X_test, y_train, y_test = split_dat(X, y)
    X_train, X_test, y_train, y_test, pred_y, lin_regressor = model_train(X_train, X_test, y_train, y_test)
    visualize_result(X_train, X_test, y_train, y_test, pred_y, lin_regressor)
    gui_easy(lin_regressor)
