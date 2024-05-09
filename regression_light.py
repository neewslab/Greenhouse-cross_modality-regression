# -*- coding: utf-8 -*-
"""
Created on Wed Jan 31 20:17:44 2024

@author: duttahr1
"""



from numpy import loadtxt
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from keras.callbacks import EarlyStopping
from matplotlib import pyplot as plt



# dataset = loadtxt("Regression_Model_Greenhouse\Set_2.csv", dtype = str, delimiter=',')

dataset = loadtxt("Regression_Model_Greenhouse\greenhouse_all_data_CSV.csv", dtype = str, delimiter=',')


num_epochs = 2000
# scaler = MinMaxScaler()

data_light = np.reshape(dataset[1:,4],(-1,1)).astype(np.float)
data_PAR = np.reshape(dataset[1:,5],(-1,1)).astype(np.float)
data_temp = np.reshape(dataset[1:,7],(-1,1)).astype(np.float)
data_humidity = np.reshape(dataset[1:,8],(-1,1)).astype(np.float)
data_voltage = np.reshape(dataset[1:,9],(-1,1)).astype(np.float)


data_light = data_light/27000
data_PAR = data_PAR*0.1934*5
data_temp = data_temp*175.72/65536-46.85
data_humidity = data_humidity*125/65536-6
data_voltage = data_voltage*0.0148+0.0356

data_con = np.concatenate((data_light,data_voltage,data_PAR),1)

# scaler.fit(data_con)
# data_con = scaler.transform(data_con)
# X, y = data_con[0:60000, 1:2], data_con[0:60000, -1]
# X, y = data_con[0:60000, 1:2], data_con[0:60000, 0:1]



X, y = data_con[:, 1:2], data_con[:, 0:1]
n_features = X.shape[1]



X_tr, X_, y_tr, y_ = train_test_split(X, y, test_size=0.10, random_state=1)

X_train, X_test, y_train, y_test = train_test_split(X_tr, y_tr, test_size=0.10, random_state=1)



# define the keras model
model = Sequential()
model.add(Dense(20, input_dim=n_features, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(100, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(50, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(10, activation='relu', kernel_initializer='he_normal'))
model.add(Dense(1, activation='linear'))

es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=500, restore_best_weights=True)

# compile the keras model
model.compile(loss='mae', optimizer='adam')

# fit the keras model on the dataset
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=64, verbose=2, callbacks=[es])
# history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=num_epochs, batch_size=64, verbose=2)

# X_,y_ = data_con[60000:, 1:2], data_con[60000:, -1]
# X_,y_ =data_con[60000:, 1:2], data_con[60000:, 0:1]


# evaluate on test set
yhat = model.predict(X_)
error = mean_absolute_error(y_, yhat)
norm_error = error/(np.max(y_)-np.min(y_))

print('MAE: %.3f' % error)
print('NMAE: %.3f' % norm_error)

plt.figure()
plt.plot(history.history['loss'][2:],'blue')
plt.title('model loss',fontsize = 20)
plt.ylabel('loss',fontsize = 20)
plt.xlabel('Epoch',fontsize = 20)
plt.tick_params(labelsize = 18)
plt.ylim(0.09, 0.12)
plt.xlim(2,750)
plt.show()




# plt.figure()
# plt.plot(y_,'blue')
# plt.plot(yhat,'red')
# plt.title('Test Performance')
# plt.ylabel('PAR')
# plt.xlabel('Sample ID')
# plt.legend(['True','Predicted'])
# plt.show()


X__,y__ =data_con[60000:70000, 1:2], data_con[60000:70000, 0:1]
yhat_ = model.predict(X__)
plt.figure(figsize=(15,6))
plt.plot(y__,'blue')
plt.plot(yhat_,'red')
# plt.xlim(0,10000)
# plt.title('Test Performance')
plt.ylabel('Light',fontsize = 20)
plt.xlabel('Sample ID',fontsize = 20)
plt.tick_params(labelsize = 18)
plt.legend(['True','Predicted'],fontsize = 20)
plt.grid()
plt.show()



# serialize model to JSON
# model_json = model.to_json()
# with open("model_early_stopping.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")


# from tensorflow.keras.models import Sequential, model_from_json
# json_file = open('model_early_stopping.json', 'r')
# loaded_model_json = json_file.read()
# json_file.close()
# loaded_model = model_from_json(loaded_model_json)
# # load weights into new model
# loaded_model.load_weights("model.h5")
# print("Loaded model from disk")


pred_light = model.predict(data_voltage)

dataset_new = np.append(dataset[1:,],pred_light, axis =1)
model.save("Regression_Model_Greenhouse\Light_pred_model.h5")