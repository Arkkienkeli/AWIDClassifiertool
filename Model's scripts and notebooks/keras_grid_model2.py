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

sess = tf.Session(
    config=tf.ConfigProto(inter_op_parallelism_threads=1,
                   intra_op_parallelism_threads=1))
K.set_session(sess)


dataframe = pandas.read_csv("1")


for column in dataframe:
	#print(column)
	encoder = LabelEncoder()
	dataframe[column] = dataframe[column].factorize()[0]
	dataframe[column] = encoder.fit_transform(dataframe[column])
	#dataframe[column] = np_utils.to_categorical(dataframe[column])

#dataframe.drop(dataframe.columns[[76, 77, 78, 79]], axis=1)
dataset = dataframe.values
X = dataset[:,0:-1]
Y = dataset[:,-1]



dataframe_tst = pandas.read_csv("1tst")


for column in dataframe_tst:
	#print(column)
	encoder = LabelEncoder()
	dataframe_tst[column] = dataframe_tst[column].factorize()[0]
	dataframe_tst[column] = encoder.fit_transform(dataframe_tst[column])
	#dataframe[column] = np_utils.to_categorical(dataframe[column])

#dataframe.drop(dataframe.columns[[76, 77, 78, 79]], axis=1)

dataset_tst = dataframe_tst.values
X_tst = dataset[:,0:-1]
Y_tst = dataset[:,-1]



def create_model(optimizer='adam'):
	# create model
	model = Sequential()
	model.add(Dense(154, input_dim=154, kernel_initializer='zero'))
	model.add(BatchNormalization())
	model.add(Activation(activation='relu'))
	model.add(Dense(154, activation='tanh'))
	model.add(Dense(1, activation='sigmoid'))

	# Compile model
	model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['metrics.categorical_accuracy'])
	return model


model = KerasClassifier(build_fn=create_model, verbose=0)
# define the grid search parameters
batch_size = [10, 40, 100, 200, 500, 1000]
epochs = [5,10,15]
learn_rate = [0.001, 0.005, 0.01, 0.1, 0.2, 0.5]
momentum = [0.0, 0.2, 0.4, 0.6, 0.8, 0.9]
optimizer = ['SGD', 'RMSprop', 'Adagrad', 'Adadelta', 'Adam', 'Adamax', 'Nadam']
init_mode = ['uniform', 'lecun_uniform', 'normal', 'zero', 'glorot_normal', 'glorot_uniform', 'he_normal', 'he_uniform']
activation = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']
activation2 = ['softmax', 'softplus', 'softsign', 'relu', 'tanh', 'sigmoid', 'hard_sigmoid', 'linear']

f = open('myfile1', 'a')

param_grid = dict(batch_size=1000, epochs=epochs)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1)
grid_result = grid.fit(X, Y)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
f.write("Best: %f using %s \n" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))
    f.write("%f (%f) with: %r \n" % (mean, stdev, param))
