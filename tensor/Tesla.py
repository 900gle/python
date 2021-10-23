import tensorflow as tf
# import tensorflow.compat.v1 as tf
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt

tf.random.set_seed(777)
# tf.set_random_seed(777)

def data_standardization(x):
    x_np = np.asarray(x)
    return (x_np - x_np.mean()) / x_np.std()


def min_max_scaling(x):
    x_np = np.asarray(x)
    return (x_np - x_np.min()) / (x_np.max() - x_np.min() + 1e-7)


def reverse_min_max_scaling(org_x, x):
    org_x_np = np.asarray(org_x)
    x_np = np.asarray(x)
    return (x_np * (org_x_np.max() - org_x_np.min() + 1 - 7)) + org_x_np.min()


input_data_column_cnt = 6
output_data_column_cnt = 1

seq_length = 3
rnn_cell_hidden_dim = 20

forget_bias = 1.0
num_stacked_layers = 1
keep_prob = 0.8

epoch_num = 1000
learning_rate = 0.01

stock_file_name = '../TSLA.csv'
encoding = 'euc-kr'
names = ['Date', 'Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']

raw_dateframe = pd.read_csv(stock_file_name, names=names, encoding=encoding)
raw_dateframe.info()

del raw_dateframe['Date']

stock_info = raw_dateframe.values[1:].astype(np.float)
print("stock_info.shape:", stock_info.shape)
print("stock_info[0] : ", stock_info[0])

price = stock_info[:, :-1]
norm_price = min_max_scaling(price)
print("price.shape: ", price.shape)
print("price[0]: ", price[0])
print("norm_price[0]: ", norm_price[0])
print("=" * 100)

volume = stock_info[:, :-1]
norm_volume = min_max_scaling(volume)
print("volume.shape: ", volume.shape)
print("volume[0]: ", volume[0])
print("norm_volume[0]: ", norm_volume[0])
print("=" * 100)

x = np.concatenate((norm_price, norm_volume), axis=1)
print("x.shape: ", x.shape)
print("x[0]: ", x[0])
print("x[-1]: ", x[-1])
print("=" * 100)

y = x[:, [-2]]
print("y[0]: ", y[0])
print("y[-1]", y[-1])

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
    _x = x[i: i + seq_length]
    _y = y[i: i + seq_length]
    if i is 0:
        print(_x, "->", _y)
    dataX.append(_x)
    dataY.append(_y)

train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size

trainX = np.array(dataX[0:train_size])
trainY = np.array(dataY[0:train_size])

testX = np.array(dataX[train_size:len(dataX)])
testY = np.array(dataY[train_size:len(dataY)])

# X = tf.placeholder(tf.float32, [None, seq_length, input_data_column_cnt])
X = tf.Variable(tf.ones(shape=[ seq_length, input_data_column_cnt]), dtype=tf.float32)
print("X: ", X)

# Y = tf.placeholder(tf.float32, [None, 1])
Y = tf.Variable(tf.ones(shape=[ seq_length, input_data_column_cnt]), dtype=tf.float32)

print("Y: ", Y)

targets = tf.placeholder(tf.float32, [None, 1])
print("targets: ", targets)

predictions = tf.placeholder(tf.float32, [None, 1])
print("predictions: ", predictions)


def lstm_cell():
    cell = tf.contrib.rnn.BasicLSTCell(num_units=rnn_cell_hidden_dim, forget_bias=forget_bias, state_is_tuple=True,
                                       activation=tf.nn.softsign)
    if keep_prob < 1.0:
        cell = tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)
    return cell


stackedRNNs = [lstm_cell() for _ in range(num_stacked_layers)]
multi_cells = tf.contrib.rnn.MultiRNNCell(stackedRNNs, state_is_tuple=True) if num_stacked_layers > 1 else lstm_cell()

hypothesis, _states = tf.nn.dynamic_rnn(multi_cells, X, dtype=tf.float32)
print("hypothesis: ", hypothesis)

hypothesis = tf.contrib.layers.fully_connected(hypothesis[:, -1], output_data_column_cnt, activation_fn=tf.identity)

loss = tf.reduce_sum(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptmizer(learning_rate)

train = optimizer.minimize(loss)
rmse = tf.sqrt(tf.reduce_mean(tf.squared_difference(targets, predictions)))

train_error_summary = []
test_error_summary = []
test_predict = ''

sess = tf.Session()
sess.run(tf.global_variables_initializer())

start_time = datetime.datetime.now()
print('학습을 시작합니다...')

for epoch in range(epoch_num):
    _, _loss = sess.run([train, loss], feed_dict={X: trainX, Y: trainY})
    if ((epoch + 1) % 100 == 0) or (epoch == epoch_num - 1):
        train_predict = sess.run(hypothesis, feed_dict={X: trainX})
        train_error = sess.run(rmse, feed_dict={targets: trainY, predictions: train_predict})
        train_error_summary.append(train_error)

        test_predict = sess.run(hypothesis, feed_dict={X: testX})
        test_error = sess.run(rmse, feed_dict={targets: testY, predictions: test_predict})
        test_error_summary.append(test_error)

        print("epoch: {}, train_error(A): {}, test_error(B): {}, B-A: {}".format(epoch+1, train_error, test_error, test_error-train_error))

        end_time = datetime.datetime.now()
        elapsed_time = end_time - start_time
        print('elapsed_time: ', elapsed_time)
        print('elapsed_time per epoch: ', elapsed_time/epoch_num)

print('input_date_column_cnt: ', input_data_column_cnt, end='')
print('output_date_column_cnt: ', output_data_column_cnt, end='')

plt.figure(1)
plt.plot(train_error_summary, 'gold')
plt.plot(test_error_summary, 'b')
plt.xlabel('Epoch(x100)')
plt.ylabel('Root Mean Square Error')

plt.figure(2)
plt.plot(testY, 'r')
plt.plot(test_predict, 'b')
plt.xlabel('Time Period')
plt.ylabel('Stock Price')
plt.show()

recent_data =  np.array([x[len(x)-seq_length : ]])
print("recent_data.shape:", recent_data.shape)
print("recent_data:", recent_data)

test_predict = sess.run(hypothesis, feed_dic={X:recent_data})
print("test_predict", test_predict[0])
test_predict = reverse_min_max_scaling(price, test_predict)
print("Tomorrow's stock price", test_predict[0])