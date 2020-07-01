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
# DATA LOADING AND PRE PROCESSING
# -------------------------------------------

print('\nData loading and pre processing.')

data_path = './data_for_interview.csv'
df = pd.read_csv(data_path, index_col=0)

# pass datetime col to datetime obj
df.Datetime = pd.to_datetime(df.Datetime)

# Generate X, Y data separating each device and considering only continuous time series e.g. 
# time series samples whithout missing point respect to an hour time unit and following a sliding window.
# X are time series of 48 points (hours) and Y are their following 6 points (hours).

X_devices = []
Y_devices = []
X_all_devices = []
Y_all_devices = []
num_columns = ['WaterTemperature', 'DissolvedOxygen',
       'weather_temperature', 'dewpoint', 'humidity', 'windspeed',
       'winddirection']
w_x = 48
w_y = 6

# normalise values in [0, 1] respect to each numerical column
for col in num_columns:
    df[col] = normalize(df[col])

devices_to_retain = ['device 1', 'device 2', 'device 3', 'device 4', 'device 5'] # ['device 1', 'device 2', 'device 3', 'device 4', 'device 5']

for device in devices_to_retain: # df['device'].unique():
    df_device = df[df['device'] == device][['Datetime'] + num_columns].copy()
    X, Y = generate_XY_continuous_TS(df_device, w_x, w_y)
    X_devices.append(np.array(X))
    Y_devices.append(np.array(Y))
    X_all_devices.extend(X)
    Y_all_devices.extend(Y)

X_all_devices = np.array(X_all_devices)
Y_all_devices = np.array(Y_all_devices)
x_dim = X_all_devices.shape[-1]
y_dim = X_all_devices.shape[-1]
Y_all_devices = Y_all_devices.reshape((Y_all_devices.shape[0], -1))

# -------------------------------------------
# MODEL TRAINING
# -------------------------------------------

# split data into train and test
X_train, X_test, Y_train, Y_test = train_test_split(X_all_devices, Y_all_devices, test_size=0.3, random_state=42)

print(f'\nX_train shape: {X_train.shape}')
print(f'\nY_train shape: {Y_train.shape}')
print(f'\nX_test shape: {X_test.shape}')
print(f'\nY_test shape: {Y_test.shape}')

# Train the model
hidden_units = [50, 30]
epochs = 100
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
plt.legend(loc=2)
fig.savefig('cost.png')

# -------------------------------------------
# MODEL TESTING
# -------------------------------------------

test_device = 0
X = X_devices[0]
Y = Y_devices[0]
pred = model.predict(X)

pred = pred.reshape((-1, w_y, y_dim))
# Y_test_original = Y_test.reshape((-1, w_y, y_dim))

for col in num_columns:
    col_index = num_columns.index(col)
    one_pred = pred[:,::w_y,col_index].reshape((-1, 1))
    one_real = Y[:,::w_y,col_index].reshape((-1, 1))

    # plot sampled prediction versus reality
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(one_pred, label='prediction')
    ax.plot(one_real, label='real data')
    plt.legend(loc=2)
    plt.title(col)
    fig.savefig(f'./plots/{col}_pred_vs_real.png')