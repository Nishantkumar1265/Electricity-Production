!pip install tensorflow
import tensorflow as tf

from google.colab import drive
drive.mount('/content/drive')

import pandas as pd
data=pd.read_csv("/content/drive/MyDrive/Electric_Production.csv")
data

from sklearn.metrics import accuracy_score

data.head()
data.describe()

data['DATE'] = pd.to_datetime(data['DATE'])
data.set_index(data['DATE'],inplace=True)
data.tail()

sequence=data['IPG2211A2N']
print(sequence)

import numpy as np

def split_sequence(sequence, n_steps):
    x, y = [], []
    for i in range(len(sequence)):
        # find the end of this pattern
        end_idx = i + n_steps
        # check if we are beyond the sequence
        if end_idx > len(sequence) - 1:
            break
        # gather input and output parts of the pattern
        seq_x = sequence[i:end_idx]
        seq_y = sequence[end_idx]
        x.append(seq_x)
        y.append(seq_y)
    return np.array(x), np.array(y)

n_steps=1
x,y=split_sequence(sequence,n_steps)
a=x
b=y
print(a,b)

# Finding Y_cap Finding using LSTM

import matplotlib.pyplot as plt

import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense
n_features = 1
x = x.reshape((x.shape[0], x.shape[1], n_features))
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(n_steps, n_features)))

model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
#print("value of x=",x,"value of y=",y)
# Fit the model
#print(type(x),type(y))
loss_values = []  # Create an empty list to store loss values

model.fit(x, y, epochs=200, verbose=0, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: loss_values.append(logs['loss']))])
# Demonstrate prediction
y_cap_lstm=[]
for i in x:
  x_input=np.array(i)
  x_input=x_input.reshape((1,n_steps,n_features))
  y_cap_lstm.append(model.predict(x_input,verbose=0))

y_cap_lstm = [element for sublist in y_cap_lstm for subsublist in sublist for element in subsublist]
x=np.array(x).reshape(-1)

x = x.tolist()
index=x

last_10_loss = loss_values[-10:]  # Get the last 10 loss values

# Calculate RMSE for each loss value
rmse_values_lstm = [np.sqrt(loss) for loss in last_10_loss]

print("RMSE values for the last 10 losses:")
print(rmse_values_lstm)

import matplotlib.pyplot as plt

# Scatter plot of y and y_cap
plt.figure(figsize=(18, 10))
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y_cap_lstm)), y_cap_lstm, label='LSTM', color='red')
plt.title('Actual vs. LSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plotting the grah of Y and Y_cap
plt.figure(figsize=(18,10))
plt.plot(y, label='Actual')
plt.plot(y_cap_lstm, label='LSTM')
plt.title('LSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Finding Y_cap using GRU

x_gru=a
y_gru=b

import numpy as np
from keras.models import Sequential
from keras.layers import GRU, Dense

# Define the model
model = Sequential()
model.add(GRU(10, input_shape=(x_gru.shape[1], 1)))
model.add(Dense(1))

n_features = 1
x_gru = x_gru.reshape((x_gru.shape[0], x_gru.shape[1], n_features))

# Compile the model
model.compile(optimizer='adam', loss='mse')

#print(type(x),type(y))
loss_values = []  # Create an empty list to store loss values
# Fit the model
model.fit(x_gru, y_gru, epochs=200, verbose=0, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: loss_values.append(logs['loss']))])

y_cap_gru=[]
for i in x_gru:
  x_input=np.array(i)
  x_input=x_input.reshape((1,n_steps,n_features))
  y_cap_gru.append(model.predict(x_input,verbose=0))


y_gru=y_gru.tolist()

y_cap_gru = [element for sublist in y_cap_gru for subsublist in sublist for element in subsublist]

last_10_loss = loss_values[-10:]  # Get the last 10 loss values

# Calculate RMSE for each loss value
rmse_values_gru = [np.sqrt(loss) for loss in last_10_loss]

print("RMSE values for the last 10 losses:")
print(rmse_values_gru)


import matplotlib.pyplot as plt

# Scatter plot of y and y_cap
plt.figure(figsize=(18, 10))
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y_cap_gru)), y_cap_gru, label='GRU', color='red')
plt.title('Actual vs. GRU')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plotting the grah of Y and Y_cap
plt.figure(figsize=(18,10))
plt.plot(y,color='red' ,label='Actual')
plt.plot(y_cap_gru, label='GRU')
plt.title('GRU')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Finding Y_cap using BILSTM

x_bilstm=a
y_bilstm=b
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Bidirectional

n_features = 1
x_bilstm = x_bilstm.reshape((x_bilstm.shape[0], x_bilstm.shape[1], n_features))


# Define model
model = Sequential()
model.add(Bidirectional(LSTM(50, activation='relu'), input_shape=(n_steps, n_features)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')

# print(type(x),type(y))
loss_values = []  # Create an empty list to store loss values
# Fit the model
model.fit(x_bilstm, y_bilstm, epochs=200, verbose=0, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: loss_values.append(logs['loss']))])

# Demonstrate prediction
y_cap_bilstm=[]
for i in x_bilstm:
  x_input=np.array(i)
  x_input=x_input.reshape((1,n_steps,n_features))
  y_cap_bilstm.append(model.predict(x_input,verbose=0))


y_bilstm=y_bilstm.tolist()

y_cap_bilstm = [element for sublist in y_cap_bilstm for subsublist in sublist for element in subsublist]

last_10_loss = loss_values[-10:]  # Get the last 10 loss values

# Calculate RMSE for each loss value
rmse_values_bilstm = [np.sqrt(loss) for loss in last_10_loss]

print("RMSE values for the last 10 losses:")
print(rmse_values_bilstm)


import matplotlib.pyplot as plt

# Scatter plot of y and y_cap
plt.figure(figsize=(18, 10))
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y_cap_bilstm)), y_cap_bilstm, label='BiLSTM', color='red')
plt.title('Actual vs. BiLSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()


# Plotting the grah of Y and Y_cap
plt.figure(figsize=(18,10))
plt.plot(y, label='Actual')
plt.plot(y_cap_bilstm, label='BiLSTM')
plt.title('BiLSTM')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Finding Y_cap using CNN

x_cnn=a
y_cnn=b
print(len(x),len(y))

# univariate cnn lstm example
from numpy import array
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import TimeDistributed
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D

# reshape from [samples, timesteps] into [samples, subsequences, timesteps, features]
n_features = 1
n_seq = 1
n_steps = 1

x_cnn = x_cnn.reshape((x_cnn.shape[0], n_seq, n_steps, n_features))

# define model
model = Sequential()
model.add(TimeDistributed(Conv1D(filters=64, kernel_size=1, activation='relu'), input_shape=(None, n_steps, n_features)))
model.add(TimeDistributed(MaxPooling1D(pool_size=1)))
model.add(TimeDistributed(Flatten()))
model.add(LSTM(50, activation='relu'))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mse')
print(type(x),type(y))
loss_values = []  # Create an empty list to store loss values
# Fit the model
model.fit(x_cnn, y_cnn, epochs=200, verbose=0, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: loss_values.append(logs['loss']))])

# demonstrate prediction
y_cap_cnn=[]
for i in x_cnn:
  x_input=np.array(i)
  x_input=x_input.reshape((1,n_steps,n_features))
  y_cap_cnn.append(model.predict(x_input,verbose=0))



y_cnn=y_cnn.tolist()

y_cap_cnn = [element for sublist in y_cap_cnn for subsublist in sublist for element in subsublist]

last_10_loss = loss_values[-10:]  # Get the last 10 loss values

# Calculate RMSE for each loss value
rmse_values_cnn = [np.sqrt(loss) for loss in last_10_loss]

print("RMSE values for the last 10 losses:")
print(rmse_values_cnn)


import matplotlib.pyplot as plt

# Scatter plot of y and y_cap
plt.figure(figsize=(18, 10))
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y_cap_cnn)), y_cap_cnn, label='CNN', color='red')
plt.title('Actual vs. CNN')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plotting the grah of Y and Y_cap
plt.figure(figsize=(18,10))
plt.plot(y, label='Actual')
plt.plot(y_cap_cnn, label='CNN')
plt.title('CNN')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Finding Y_cap using RNN

x_rnn=a
y_rnn=b

import numpy as np
from keras.models import Sequential
from keras.layers import SimpleRNN, Dense

# Reshape input to match the expected format of RNN
x_rnn = x_rnn.reshape((x_rnn.shape[0], x_rnn.shape[1], n_features))

# Define the model
model = Sequential()
model.add(SimpleRNN(10, input_shape=(x_rnn.shape[1], 1)))
model.add(Dense(1))

# Compile the model
model.compile(optimizer='adam', loss='mse')

#print(type(x),type(y))
loss_values = []  # Create an empty list to store loss values
# Fit the model
model.fit(x_rnn, y_rnn, epochs=200, verbose=0, callbacks=[tf.keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: loss_values.append(logs['loss']))])

y_cap_rnn=[]
for i in x_rnn:
  x_input=np.array(i)
  x_input=x_input.reshape((1,n_steps,n_features))
  y_cap_rnn.append(model.predict(x_input,verbose=0))



y_rnn=y_rnn.tolist()

y_cap_rnn = [element for sublist in y_cap_rnn for subsublist in sublist for element in subsublist]
plt.scatter(range(len(y)), y, color='yellow', label='y')
plt.scatter(range(len(y_cap_rnn)), y_cap_rnn, color='red', label='y_cap_rnn')

last_10_loss = loss_values[-10:]  # Get the last 10 loss values

# Calculate RMSE for each loss value
rmse_values_rnn = [np.sqrt(loss) for loss in last_10_loss]

print("RMSE values for the last 10 losses:")
print(rmse_values_rnn)


import matplotlib.pyplot as plt

# Scatter plot of y and y_cap
plt.figure(figsize=(18, 10))
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y_cap_rnn)), y_cap_rnn, label='RNN', color='red')
plt.title('Actual vs. RNN')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plotting the grah of Y and Y_cap
plt.figure(figsize=(18,10))
plt.plot(y, label='Actual')
plt.plot(y_cap_rnn, label='RNN')
plt.title('RNN')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

print(rmse_values_lstm)
print(rmse_values_gru)
print(rmse_values_bilstm)
print(rmse_values_cnn)
print(rmse_values_rnn)

x_hybrid=a
y_hybrid=b
import numpy as np
from keras.models import Model
from keras.layers import Input, LSTM, GRU, Conv1D, MaxPooling1D, Flatten, Dense, concatenate, Bidirectional

n_features = 1
n_steps = x_hybrid.shape[1]

# Reshape the input data
x_hybrid = x_hybrid.reshape((x_hybrid.shape[0], x_hybrid.shape[1], n_features))

# LSTM model
lstm_input = Input(shape=(n_steps, n_features))
lstm = LSTM(50, activation='relu')(lstm_input)



# Dense model
dense_input = Input(shape=(n_steps, n_features))
dense = Dense(50, activation='relu')(dense_input)
dense = Flatten()(dense)

# Bidirectional LSTM model
bidirectional_input = Input(shape=(n_steps, n_features))
bidirectional = Bidirectional(LSTM(50, activation='relu'))(bidirectional_input)

# Concatenate the outputs of all models
concatenated = concatenate([lstm, dense, bidirectional])

# Final output layer
output = Dense(1)(concatenated)

# Create the model
model = Model(inputs=[lstm_input, dense_input, bidirectional_input], outputs=output)
model.compile(optimizer='adam', loss='mse')

res = []

# Fit the model
history = model.fit([x_hybrid, x_hybrid, x_hybrid], y, epochs=200, verbose=0)
loss_values = history.history['loss']  # Get the loss values from the history object

for i in x:
    # Demonstrate prediction
    x_input = np.array(i)
    x_input = x_input.reshape((1, n_steps, n_features))
    y_cap_hybrid = model.predict([x_input, x_input, x_input], verbose=0)
    res.append(y_cap_hybrid)

y_cap_hybrid = res
y_cap_hybrid = [element for sublist in y_cap_hybrid for subsublist in sublist for element in subsublist]

plt.scatter(range(len(y)), y, color='yellow', label='y')
plt.scatter(range(len(y_cap_hybrid)), y_cap_hybrid, color='red', label='y_cap_hybrid')

last_10_loss = loss_values[-10:]  # Get the last 10 loss values

# Calculate RMSE for each loss value
rmse_values_hybrid = [np.sqrt(loss) for loss in last_10_loss]

print("RMSE values for the last 10 losses:")
print(rmse_values_hybrid)

import matplotlib.pyplot as plt

# Scatter plot of y and y_cap
plt.figure(figsize=(18, 10))
plt.scatter(range(len(y)), y, label='Actual', color='blue')
plt.scatter(range(len(y_cap_hybrid)), y_cap_hybrid, label='Hybrid', color='red')
plt.title('Actual vs. Hybrid')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()
plt.show()

# Plotting the grah of Y and Y_cap
plt.figure(figsize=(18,10))
plt.plot(y, label='Actual')
plt.plot(y_cap_hybrid, label='Hybrid')
plt.title('Hybrid')
plt.xlabel('Time')
plt.ylabel('Value')
plt.legend()

# Data labels
labels = ['LSTM', 'BiLSTM', 'CNN', 'Hybrid']

# Box plot
plt.figure(figsize=(10, 6))
plt.boxplot([rmse_values_lstm, rmse_values_bilstm, rmse_values_cnn, rmse_values_hybrid], labels=labels)
plt.title('Model Comparision (w.r.t. RMSE)')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.show()

# Data labels
labels = ['GRU', 'RNN']

# Box plot
plt.figure(figsize=(10, 6))
plt.boxplot([rmse_values_gru, rmse_values_rnn], labels=labels)
plt.title('Model Comparision (w.r.t. RMSE)')
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.show()

import matplotlib.pyplot as plt
import seaborn as sns

models = ['theta', 'snaive', 'gbr_cds_dt', 'ets', 'exp_smooth', 'lstm', 'bilstm', 'cnn', 'hybrid']
RMSE = [4.4107, 4.4777, 4.7190, 4.7904, 4.8001]
a1 = rmse_values_lstm[-1]
b1 = rmse_values_gru[-1]
c1 = rmse_values_bilstm[-1]
d1 = rmse_values_cnn[-1]
e1 = rmse_values_rnn[-1]
f1 = rmse_values_hybrid[-1]
a1=round(a1, 6)
c1=round(c1, 6)
d1=round(d1, 6)
f1=round(f1, 6)
RMSE1 = [a1, c1, d1, f1]
RMSE2 = RMSE + RMSE1

plt.figure(figsize=(12,5))
plt.bar(models, RMSE2)
plt.xlabel('Models')
plt.ylabel('RMSE')
plt.title('RMSE for Different Models')

# Add values to the bars
for i in range(len(models)):
    plt.text(models[i], RMSE2[i], str(RMSE2[i]), ha='center', va='bottom')

plt.show()
