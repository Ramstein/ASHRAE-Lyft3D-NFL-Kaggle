# import h2o
# from h2o.estimators.xgboost import H2OXGBoostEstimator
# from h2o.estimators.stackedensemble import H2OStackedEnsembleEstimator
#
# from h2o.grid.grid_search import H2OGridSearch
#
#
# import lightgbm as lgb
# from sklearn.model_selection import GridSearchCV
# from sklearn.model_selection import RandomizedSearchCV
# from sklearn.metrics.regression import mean_absolute_error
#
#
#
# lgbm = lgb.LGBMModel()
#
#
# grid_result = RandomizedSearchCV(model, param_distributions=param_grid,
#                            n_jobs=-1, verbose=2,
#                            scoring='mean_absolute_error')
#
# grid_result.fit(score='mean_absolute_error')
#
# score = grid_result.best_score_
# best_lgbm = grid_result.best_estimator_
# best_param = grid_result.best_params_
#
#
#
# # Regression
# # ‘explained_variance’	metrics.explained_variance_score
# # ‘max_error’	metrics.max_error
# # ‘neg_mean_absolute_error’	metrics.mean_absolute_error
# # ‘neg_mean_squared_error’	metrics.mean_squared_error
# # ‘neg_mean_squared_log_error’	metrics.mean_squared_log_error
# # ‘neg_median_absolute_error’	metrics.median_absolute_error
# # ‘r2’	metrics.r2_score
# import h2o
# h2o.init()
#
# model = h2o.load_model('')
# # h2o.shutdown(prompt=False)

# import numpy as np
# def orientation_to_cat(x):
#     x = np.clip(x, 0, 359.999)
#     try:
#         return str(int(x/15))
#     except:
#         return "nan"
# print(orientation_to_cat(358))

# x = [True, False, True]
# y = x.apply(lambda x: 1 if x is True else 0)
# print(y)
# MLP for Pima Indians Dataset with grid search via sklearn
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import GridSearchCV
import numpy
from keras.optimizers import Adam
from keras.activations import linear, relu, tanh
# Function to create model, required for KerasClassifier
def create_model(lr=0.01, layer=12, activation=Adam, dropout_rate=0.2):
	# create model
	model = Sequential()
	model.add(Dense(layer, input_dim=8, kernel_initializer='uniform', activation=activation))

    model.add(Dropout(dropout_rate))
	model.add(Dense(8, kernel_initializer='uniform', activation=activation))
	model.add(Dense(1, kernel_initializer='uniform', activation='sigmoid'))
	# Compile model
	model.compile(loss='binary_crossentropy',
                  optimizer=Adam(lr=lr, amsgrad=True, epsilon=0.0009), metrics=['accuracy'])
	return model

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load pima indians dataset
dataset = numpy.loadtxt("pima-indians-diabetes.csv", delimiter=",")
# split into input (X) and output (Y) variables
X = dataset[:,0:8]
Y = dataset[:,8]
# create model
model = KerasClassifier(build_fn=create_model, verbose=0)

# grid search epochs, batch size and optimizer1
# optimizers = ['adam']

layer = [64, 128, 256, 512]
learn_rate = [0.0001, 0.001, 0.01]
epochs = [100]
batch = [1024, 1536, 2048]
activation = ['softmax', 'relu', 'tanh', 'sigmoid', 'linear']
dropout_rate = [0.2, 0.3, 0.4, 0.5]
param_grid = dict(lr=learn_rate, act=activation,
                  epochs=epochs,
                  batch_size=batch,
                  dropout=dropout_rate, layer=layer)

grid = GridSearchCV(estimator=model, param_grid=param_grid,
                    n_jobs=-1, cv=5)
grid_result = grid.fit(X, Y,
                       callbacks=[CRPSCallback(validation = (x_val,y_val)),es])



# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
	print("%f (%f) with: %r" % (mean, stdev, param))


from keras.layers import Activation
model = create_model()
mdoel = Activation('relu')

import pandas as pd


df = pd.read_csv()
df['hllo'].fillna(380)