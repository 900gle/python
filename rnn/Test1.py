from numpy import array
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 데이터
x = array([[1,2,3], [2,3,4], [3,4,5], [4,5,6]])
y = array([4,5,6,7])

print('x shape : ', x.shape) # (4,3)
print('y shape : ', y.shape) # (4,)
#  x  y
# 123 4
# 234 5
# 345 6
# 456 7

print(x)
print('-------x reshape-----------')
x = x.reshape((x.shape[0], x.shape[1], 1)) # (4,3,1) reshape 전체 곱 수 같아야 4*3=4*3*1
print('x shape : ', x.shape)
print(x)
#  x        y
# [1][2][3] 4
# .....

# 2. 모델 구성
model = Sequential()
model.add(LSTM(10, activation = 'relu', input_shape=(3,1)))
# DENSE와 사용법 동일하나 input_shape=(열, 몇개씩잘라작업)
model.add(Dense(5))
model.add(Dense(1))

model.summary()

# 3. 실행
model.compile(optimizer='adam', loss='mse')
model.fit(x, y, epochs=100, batch_size=1)

x_input = array([6,7,8])
x_input = x_input.reshape((1,3,1))

yhat = model.predict(x_input)
print(yhat)


