import tensorflow as tf 
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,  Dropout,LSTM

mnist = tf.keras.datasets.mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

#print(x_train.shape)
#print(x_train[0].shape)

x_train = x_train/255.0
x_test = x_test/255.0


model = Sequential()

model.add(LSTM(128,input_shape=(x_train.shape[1:]),activation='relu',return_sequences=True)) #relu here is rectified linear, for dense layer it wont undersatnd return sequences
#model.add(CuDNNLSTM(128,input_shape=(x_train.shape[1:]),return_sequences=True)) #relu here is rectified linear, for dense layer it wont undersatnd return sequences
#when using CuDNNLSTM do not use activation functionas tanh is the default and is required.
model.add(Dropout(0.2)) #what does Dropout do here though?

model.add(LSTM(128,activation='relu'))
#model.add(CuDNNLSTM(128))
model.add(Dropout(0.2))

model.add(Dense(32,activation="relu"))
model.add(Dropout(0.2))

model.add(Dense(10,activation="softmax")) #softmax for last dense layer 


#now do the compile
# we use an optimizer
opt = tf.keras.optimizers.Adam(lr=1e-3,decay=1e-5) 

model.compile(loss='sparse_categorical_crossentropy',optimizer=opt,metrics=['accuracy'])

model.fit(x_train,y_train,epochs=3,validation_data=(x_test,y_test))
