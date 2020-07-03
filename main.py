"""
-*-coding: utf-8-*- 
Author: Yann Cherdo 
Creation date: 2020-06-30 17:56:32
"""

import pandas as pd
import numpy as np
from libs.data import generate_XY_continuous_TS
from sklearn.model_selection import train_test_split
from libs.utils import normalize
from libs.rnn import RNN
import matplotlib.pyplot as plt

# -------------------------------------------
# DATA LOADING
# -------------------------------------------

print('\nData loading and pre processing.')

data_path = './data_for_interview.csv'
df = pd.read_csv(data_path, index_col=0)

# pass datetime col to datetime obj
df.Datetime = pd.to_datetime(df.Datetime)

# -------------------------------------------
# FEATURE ENGINEERING
# -------------------------------------------

# Interpolate the missing values. Ok as they are few and signals should be relatively continuous
# Interpolation of missing samples regarding an hour sample period is applied in generate_XY_continuous_TS
# for each device
df = df.interpolate()

# Add the hour of the day as a feature in order to help caption environmental factors
df['Hour'] = df['Datetime'].apply(lambda d: d.hour)

# -------------------------------------------
# DATA FORMATING FOR TRAINING/TEST/VALIDATION
# -------------------------------------------

# Generate X, Y data separating each device and considering only continuous time series e.g. 
# time series samples whithout missing point respect to an hour time unit and following a sliding window.
# X are time series of 48 points (hours) and Y are their following 6 points (hours).
# Split data into devices, all devices, and select some devices from train/test other for validation. 

# X_devices_train_test = []
# Y_devices_train_test = []
X_all_devices_train_test = []
Y_all_devices_train_test = []
X_devices_valid = []
Y_devices_valid = []

num_columns = ['WaterTemperature', 'DissolvedOxygen',
       'weather_temperature', 'dewpoint', 'humidity', 'windspeed',
       'winddirection', 'Hour']
w_x = 48
w_y = 6
validation_split = 0.3
interpolate = True

# normalise values in [0, 1] respect to each numerical column
for col in num_columns:
    df[col] = normalize(df[col])

# split the df in train/test and validation. Here validation is the entire data as we just
# use it top have a better view while ploting the prediction.
valid_split_index = int((1 - validation_split)*len(df))
df_train_test = pd.concat([df[df['device'] == device].iloc[:valid_split_index] for device in df['device'].unique()])

# gather train/test and validation arrays
train_test_devices = ['device 1', 'device 2', 'device 3', 'device 4', 'device 5'] # ['device 1', 'device 2', 'device 3', 'device 4', 'device 5']
validation_devices = ['device 1', 'device 2', 'device 3', 'device 4', 'device 5']
X_columns = num_columns
Y_columns = ['DissolvedOxygen']

for device in train_test_devices: 
    df_device = df_train_test[df_train_test['device'] == device][['Datetime'] + num_columns].copy()
    X, Y = generate_XY_continuous_TS(df_device, X_columns, Y_columns, w_x, w_y, interpolate)
    # X_devices_train_test.append(np.array(X))
    # Y_devices_train_test.append(np.array(Y))
    X_all_devices_train_test.extend(X)
    Y_all_devices_train_test.extend(Y)

for device in validation_devices: 
    df_device = df[df['device'] == device][['Datetime'] + num_columns].copy()
    X, Y = generate_XY_continuous_TS(df_device, X_columns, Y_columns, w_x, w_y, interpolate)
    X_devices_valid.append(np.array(X))
    Y_devices_valid.append(np.array(Y))

X_all_devices = np.array(X_all_devices_train_test)
Y_all_devices = np.array(Y_all_devices_train_test)
x_dim = X_all_devices.shape[-1]
y_dim = Y_all_devices.shape[-1]
Y_all_devices = Y_all_devices.reshape((Y_all_devices.shape[0], -1))

# split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_all_devices, Y_all_devices, test_size=0.3, random_state=42)

print(f'\nX_train shape: {X_train.shape}')
print(f'\nY_train shape: {Y_train.shape}')
print(f'\nX_test shape: {X_test.shape}')
print(f'\nY_test shape: {Y_test.shape}')

# -------------------------------------------
# MODEL TRAINING
# -------------------------------------------

print('\nModel training.')

# Train the model
hidden_units = [150, 50]
epochs = 200
mini_batch_size = 32
learning_rate = 0.01

model = RNN(hidden_units, input_dim=X_train.shape[-1], output_dim=Y_train.shape[-1])

train_cost, test_cost = model.fit(X_train,
                                    Y_train, 
                                    X_test, 
                                    Y_test, 
                                    epochs, 
                                    mini_batch_size, 
                                    learning_rate)

# plot training and test costs
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(train_cost, label='train cost')
ax.plot(test_cost, label='test cost')
plt.yscale('log')
plt.legend(loc=2)
fig.savefig('./plots/cost.png')

# -------------------------------------------
# MODEL TESTING
# -------------------------------------------

print('\nRunning validation.')

for test_device in validation_devices:
    i = validation_devices.index(test_device)
    X_valid = X_devices_valid[i]
    Y_valid = Y_devices_valid[i]
    pred = model.predict(X_valid)

    pred = pred.reshape((-1, w_y, y_dim))

    for col in Y_columns:
        col_index = Y_columns.index(col)
        one_pred = pred[:,::w_y,col_index].reshape((-1, 1))
        one_real = Y_valid[:,::w_y,col_index].reshape((-1, 1))

        # plot sampled prediction versus reality
        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.plot(one_real, label='real data')
        ax.plot(one_pred, label='prediction')
        plt.legend(loc=2)
        plt.axvline(x=len(one_real)*(1-validation_split))
        plt.title(f'Prediction on "{test_device}" and feature "{col}"')
        fig.savefig(f'./plots/{test_device}_{col}_pred_vs_real.png')