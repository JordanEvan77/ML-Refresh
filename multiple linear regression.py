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
from sklearn.preprocessing import OneHotEncoder
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


file_path = r"F:\Personal\JORGRO\ML_practice\ML_Course\raw_data\landprice2.csv"

def clean ():
    dataset = pd.read_csv(r'F:\Personal\JORGRO\ML_practice\ML_Course\raw_data\landprice2.csv')
    print('variables', dataset.columns)
    le = LabelEncoder()
    dataset['City_num'] = le.fit_transform(dataset['City'])

    print(dataset[['Area', 'Distance', 'Crime Rate', 'Price', 'City_num']].corr())
    # we would only keep variables [] because of their relationship with each other and land price.
    # varialbes [Distance and area] are too correlated, so we would only keep 1
    # crime rate is interesting, but barely related. city number is more impactful than crime rate

    # OHE is preferred>:
    #Label Encoding is useful when the categories have an inherent order or rank. In contrast, One Hot Encoding is
    # # useful when there is no inherent order or rank among the categories.

def readFile(file_path):
    dataset = pd.read_csv(file_path)
    ohe = pd.get_dummies(dataset.City)# could remove one of these, since it is redundent. if it isn't city 1 or 2, its always 3, so 3 can be removed.
    dataset = pd.concat([dataset[['Area', 'Distance', 'Crime Rate', 'Price']],ohe], axis=1)
    X = dataset[['Area', 'Distance', 'Crime Rate']].values # can add these back in, just need them gone for gui: 'Interlaken', 'Jeneva', 'Zurich'
    y = dataset[['Price']].values
    X1 = dataset['Area'] # remove multicollinearity
    return X, y, X1

def split_data(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=0)
    return X_train, X_test, y_train, y_test

def model_train(X_train, X_test, y_train, y_test):
    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train, y_train)
    y_pred = lin_regressor.predict(X_test)

    r_square = r2_score(y_test, y_pred)
    print('The R Square explanation power is :', r_square)

    return X_train, X_test, y_train, y_test, y_pred, lin_regressor

def model_train1(X_train, X_test, y_train, y_test):
    X_train = np.reshape(X_train, (-1,1))
    X_test = np.reshape(X_test, (-1, 1))

    y_train = np.reshape(y_train, (-1, 1))
    y_test = np.reshape(y_test, (-1, 1))

    lin_regressor = LinearRegression()
    lin_regressor.fit(X_train, y_train)
    y_pred = lin_regressor.predict(X_test)

    r_square = r2_score(y_test, y_pred)
    print('The R Square explanation power wihtout colliniearity is :', r_square)

    area_test = np.reshape(37, (1,1))
    pred_price = lin_regressor.predict(area_test)
    print('pred_price:', pred_price, lin_regressor.coef_)

    return X_train, X_test, y_train, y_test, y_pred, lin_regressor

def visualize_result(X_train, X_test, y_train, y_test, y_pred, lin_regressor):
    plt.scatter(X_train[:, -1], y_train, color='blue')
    plt.scatter(X_test[:, -1], y_test, color='red')
    plt.scatter(X_test[:, -1], y_pred, color='green')
    #plt.plot(X_test[:, -1], lin_regressor.predict(X_test), color='yellow')
    plt.title('Area price of land')
    plt.xlabel('Area of land in thousands of SQFT')
    plt.ylabel('Price of the land in million USD')
    plt.show()
    print('viz')

def simple_pred(window, lin_regressor, area, distance, crime_rate):
    area = int(area)
    distance = int(distance)
    crime_rate = int(crime_rate)
    #pred_price = lin_regressor.predict(np.reshape(area, (1,1)))
    pred_price = lin_regressor.predict(np.array([[area, distance, crime_rate]]))
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

    label1 = Label(window, text='Area of land, 1,000 sqft', fg='red', font=('Courier', 15))
    label1.pack()
    distance = StringVar()
    distance.set("")
    entry1 = Entry(window, textvariable=distance, fg='green', width=10, font=('courier', 15))
    entry1.pack()

    label2 = Label(window, text='Area of land, 1,000 sqft', fg='red', font=('Courier', 15))
    label2.pack()
    crime_rate = StringVar()
    crime_rate.set("")
    entry2 = Entry(window, textvariable=crime_rate, fg='green', width=10, font=('courier', 15))
    entry2.pack()

    predbutton = Button(window, text='Predict', fg='red', command=lambda: simple_pred(window, lin_regressor, area.get(),
                distance.get(), crime_rate.get()), height=2, width=15)
    predbutton.pack()
    window.mainloop()

if __name__ == '__main__':
    X, y, X1= readFile(file_path)
    X_train, X_test, y_train, y_test = split_data(X, y)
    X_train1, X_test1, y_train1, y_test1 = split_data(X1, y)

    X_train, X_test, y_train, y_test, y_pred, lin_regressor = model_train(X_train, X_test, y_train, y_test)
    X_train1, X_test1, y_train1, y_test1, y_pred1, lin_regressor1 = model_train1(X_train1, X_test1, y_train1, y_test1)

    visualize_result(X_train, X_test, y_train, y_test, y_pred, lin_regressor)
    visualize_result(X_train1, X_test1, y_train1, y_test1, y_pred1, lin_regressor)
    gui_easy(lin_regressor)


