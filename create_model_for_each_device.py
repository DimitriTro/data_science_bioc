import pandas as pd
from datetime import datetime
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf


## extract features and labels from dataset
def multivariate_data(dataset, target, start_index, end_index, history_size,target_size, step, single_step=False):
    data = []
    labels = []

    start_index = start_index + history_size
    if end_index is None:
        end_index = len(dataset) - target_size

    for i in range(start_index, end_index):
        indices = range(i-history_size, i, step)
        data.append(dataset[indices])

        if single_step:
            labels.append(target[i+target_size])
        else:
            labels.append(target[i:i+target_size])

    return np.array(data), np.array(labels)

## extract and clean data from file
def extract_data(csv_file):
    df = pd.read_csv(csv_file,index_col=0)
    df = df.sort_values(by="Datetime").reset_index(drop=True)
    df = df[pd.to_datetime(df["Datetime"]) >"2020-06-15 18:00:00"].reset_index(drop=True)
    return df

# create graph for training and validation loss
def plot_train_history(history, title,path):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(loss))

    f = plt.figure(figsize=(20, 10))

    plt.plot(epochs, loss, 'b', label='Training loss')
    plt.plot(epochs, val_loss, 'r', label='Validation loss')
    plt.title(title)
    plt.legend()
    plt.savefig(path)


# create graph for prediction examples
def create_time_steps(length):
    return list(range(-length, 0))

def multi_step_plot_graph(history, true_future, prediction,i):

    plt.subplot(2,2,i)
    num_in = create_time_steps(len(history))
    num_out = len(true_future)

    plt.plot(num_in, np.array(history[:, 0]), label='History')
    plt.plot(np.arange(num_out), np.array(true_future), 'bo',
           label='True Future')
    if prediction.any():
        plt.plot(np.arange(num_out), np.array(prediction), 'ro',
             label='Predicted Future')
    plt.legend(loc='upper left')

def plot_example_prediction(val_data,model,title,path):
    f = plt.figure(figsize=(20, 10))
    for x, y in val_data.take(1):
        for i in range(1,5):
            multi_step_plot_graph(x[i], y[i], model.predict(x)[i],i)
    plt.title(title)
    plt.savefig(path)

## create LSTM model with keras
def create_model(x_train,input_layer,hidden_layer,output_layer,dropout,clipvalue=1.0,learning_rate=0.0001,decay=1e-6,activation='relu',loss='mae'):
    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.LSTM(input_layer,return_sequences=True,input_shape=x_train.shape[-2:]))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.LSTM(hidden_layer, activation=activation))
    model.add(tf.keras.layers.Dropout(dropout))
    model.add(tf.keras.layers.Dense(output_layer))
    model.compile(optimizer=tf.keras.optimizers.RMSprop(clipvalue=clipvalue,learning_rate=learning_rate,decay=decay), loss=loss,metrics=['mse'])
    return model

def main():
    df = extract_data("data_for_interview.csv")
    tf.random.set_seed(13)

    past_history = 48 # window of 48h
    future_target = 6 # prediction for 6h
    STEP = 1 #Step of one hour
    TRAIN_SPLIT = 130
    BUFFER_SIZE = 100

    EVALUATION_INTERVAL = 100
    EPOCHS = 15

    # create model for each device
    for device in df["device"].unique():

        features_considered = ['DissolvedOxygen','WaterTemperature','weather_temperature','dewpoint','windspeed','winddirection']
        features = df[features_considered][df["device"]==device]
        features.index = df['Datetime'][df["device"]==device]

        dataset = features.values
        dataset = tf.keras.utils.normalize(dataset, axis=0, order=2)

        # split training and validation sets
        x_train_multi, y_train_multi = multivariate_data(dataset, dataset[:, 0], 0,
                                                 TRAIN_SPLIT, past_history,
                                                 future_target, STEP)
        x_val_multi, y_val_multi = multivariate_data(dataset, dataset[:, 0],
                                             TRAIN_SPLIT, None, past_history,
                                             future_target, STEP)

        BATCH_SIZE = len(x_train_multi)

        # convert sets
        train_data_multi = tf.data.Dataset.from_tensor_slices((x_train_multi, y_train_multi))
        train_data_multi = train_data_multi.cache().shuffle(BUFFER_SIZE).repeat().batch(BATCH_SIZE)

        val_data_multi = tf.data.Dataset.from_tensor_slices((x_val_multi, y_val_multi))
        val_data_multi = val_data_multi.repeat().batch(BATCH_SIZE)

        callback = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=2)

        multi_step_model = create_model(x_train_multi,64,8,future_target,0.3)

        # train model
        multi_step_history = multi_step_model.fit(train_data_multi, epochs=EPOCHS,
                                          steps_per_epoch=EVALUATION_INTERVAL,
                                          validation_data=val_data_multi,
                                          validation_steps=50,callbacks=[callback])
        # save model
        multi_step_model.save("models/LSTM_model_device_"+device.split(" ")[1])

        # create graph
        plot_train_history(multi_step_history,"Training and validation loss "+device,"loss/loss_graph_device_"+device.split(" ")[1]+".png")
        plot_example_prediction(val_data_multi,multi_step_model,"Prediction examples "+device,"prediction_example/prediction_example_graph_device_"+device.split(" ")[1]+".png")


if __name__ == "__main__":
    main()
