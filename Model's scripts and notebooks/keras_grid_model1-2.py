import pandas
import numpy
seed = 42
numpy.random.seed(seed)
import tensorflow as tf
tf.set_random_seed(seed)
import csv
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from keras.layers.convolutional import Conv1D
from keras.layers.normalization import BatchNormalization
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
from keras.utils import np_utils
import os
import tensorflow as tf
from keras import backend as K
from keras import metrics

sess = tf.Session(
    config=tf.ConfigProto(inter_op_parallelism_threads=4,
                   intra_op_parallelism_threads=4))
K.set_session(sess)


dataframe = pandas.read_csv("1trn")
dataframe_tst = pandas.read_csv("1tst")

to_int16 = ['radiotap.present.reserved', 'wlan.fc.type_subtype', 'wlan.fc.ds', 'wlan_mgt.fixed.capabilities.cfpoll.ap', 'wlan_mgt.fixed.listen_ival', 'wlan_mgt.fixed.status_code', 'wlan_mgt.fixed.timestamp', 
'wlan_mgt.fixed.aid', 'wlan_mgt.fixed.reason_code', 'wlan_mgt.fixed.auth_seq', 'wlan_mgt.fixed.htact', 'wlan_mgt.fixed.chanwidth', 'wlan_mgt.tim.bmapctl.offset', 'wlan_mgt.country_info.environment', 
'wlan_mgt.rsn.capabilities.ptksa_replay_counter', 'wlan_mgt.rsn.capabilities.gtksa_replay_counter', 'wlan.wep.iv', 'wlan.wep.icv', 'wlan.qos.ack' ]
to_drop = ['frame.interface_id', 'frame.dlt', 'wlan.ra', 'wlan.da', 'wlan.ta', 'wlan.sa', 'wlan.bssid', 'wlan.ba.bm', 'wlan_mgt.fixed.current_ap', 
'wlan_mgt.ssid', 'wlan.tkip.extiv', 'wlan.ccmp.extiv', 'radiotap.dbm_antsignal' ]

float_col = ['frame.time_epoch', 'frame.time_delta', 'frame.time_delta_displayed', 'frame.time_relative']

dataframe.drop(to_drop, axis=1, inplace=True)
dataframe_tst.drop(to_drop, axis=1, inplace=True)


encoder = LabelEncoder()
encoder.classes_ = numpy.load('classesAWID.npy')
for column in dataframe:
	#print(column)
	#print(dataframe[column].dtype, column)	
	if column in to_int16:
		dataframe[column] = dataframe[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
		dataframe[column] = dataframe_tst[column].apply(lambda x: int(str(x), base=16) if x != '?' else x)
	if column == 'class':
		dataframe_tst[column] = encoder.transform(dataframe_tst[column])
		dataframe[column] = encoder.transform(dataframe[column])
		


dataframe = dataframe.replace(['?'], [-1])
dataframe_tst = dataframe_tst.replace(['?'], [-1])

dataframe = dataframe.convert_objects(convert_numeric=True)
dataframe_tst = dataframe_tst.convert_objects(convert_numeric=True)
for column in dataframe:
	print(dataframe[column].dtype, column)	
		
dataset = dataframe.values
X = dataset[:,0:-1]
Y = dataset[:,-1]

def create_model(optimizer='adam', activation='sigmoid', init = 'zero'):
	# create model
	print('model')
	model = Sequential()
	model.add(Dense(141, input_dim=141, kernel_initializer=init))
	model.add(BatchNormalization())
	model.add(Activation(activation='relu'))
	model.add(Dense(141, activation='relu'))
	model.add(Dense(1, activation=activation))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=[metrics.categorical_accuracy, 'accuracy'])
	return model


model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 100, 1000]
epochs = [5,10,20]
learn_rate = [0.001, 0.005, 0.01, 0.1, 0.2, 0.5]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
optimizer = ['SGD', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
#activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
#activation2 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
activation3 = ['sigmoid', 'softmax']

f = open('myfileNew', 'a')

param_grid = dict(batch_size=batch_size, epochs=epochs, optimizer=optimizer, init=init_mode, activation = activation3 )
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X,Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
f.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    f.write("%f (%f) with: %r \n" % (mean, stdev, param))
