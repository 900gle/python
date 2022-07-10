# input_shape=(x, )  ->  실제 input(None, x)
# input_shape=(x, y) -> 실제 input(None, x, y)

from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6], [5,6,7],
           [6,7,8], [7,8,9], [8,9,10], [9,10,11], [10,11,12],
           [20,30,40], [30,40,50], [40,50,60]])
y = array([4,5,6,7,8,9,10,11,12,13,50,60,70])

print(x.shape) # (13,3)
print(y.shape) # (13,)
x = x.reshape((x.shape[0], x.shape[1], 1))
print(x.shape) # (13,3,1)

model = Sequential()
# model.add(LSTM(20, activation = 'relu', input_shape=(3,1))) # (None, 3, 1)
# model.add(LSTM(3)) # (None, 3) -> Value error
model.add(LSTM(10, activation = 'relu', input_shape=(3,1), return_sequences=True))
model.add(LSTM(10, activation = 'relu', return_sequences=True)) # (None, 3, 10)을 받는다
model.add(LSTM(10, activation = 'relu', return_sequences=True))
model.add(LSTM(10, activation = 'relu', return_sequences=True))
model.add(LSTM(10, activation = 'relu', return_sequences=True))
model.add(LSTM(10, activation = 'relu', return_sequences=True))
model.add(LSTM(10, activation = 'relu', return_sequences=True))
model.add(LSTM(10, activation = 'relu', return_sequences=True))
model.add(LSTM(3)) # 마지막은 return_sequence X
# return_sequence를 쓰면 dimension이 한개 추가 되므로 다음 Dense Layer의 인풋에 3 dim이 들어가게 되므로 안씀
# LSTM 두개를 엮을 때
model.add(Dense(5))
model.add(Dense(1))

model.summary()

model.compile(optimizer='adam', loss='mse')

from keras.callbacks import EarlyStopping
early_stopping = EarlyStopping(monitor='loss', patience=100, mode='auto')
model.fit(x, y, epochs=1, batch_size=1, verbose=2, callbacks=[early_stopping])

x_input = array([25,35,45]) # predict용
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)

# LSTM을 2개 이상 많이 엮으면 좋지 않다.


