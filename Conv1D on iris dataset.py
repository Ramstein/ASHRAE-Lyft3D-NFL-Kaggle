# Read the training data
import pandas as pd

data = pd.read_csv(r'C:\Datasets\kuc-hackathon-winter-2018\drugsComTrain_raw.csv')
data.head()


# Create labels based on the original article: Grässer et al. (2018)
r = data['rating']
labels = -1*(r <= 4) + 1*(r >= 7)
# Add the label column to the data
data['label'] = labels
# Check the new data
data.head()


# Check ratings to labels conversion
import matplotlib.pyplot as plt
data.plot(x = 'rating', y = 'label', kind = 'scatter')
# plt.show()

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

# Common settings for all models
WORDS = 1000
LENGTH = 100
N = 10000
N_HIDDEN = 64

# Read a part of the reviews and create training sequences (x_train)
samples = data['review'].iloc[:N]
tokenizer = Tokenizer(num_words = WORDS)
tokenizer.fit_on_texts(samples)
sequences = tokenizer.texts_to_sequences(samples)
print(sequences)
x_train = pad_sequences(sequences, maxlen = LENGTH)


from keras.utils import to_categorical

# Convert the labels to one_hot_category values
one_hot_labels = to_categorical(labels[:N], num_classes = 3)
print(x_train.shape, one_hot_labels.shape)


# We use the same plotting commands several times, so create a function for that purpose
def plot_history(history):
    f, ax = plt.subplots(1, 2, figsize=(16, 7))

    acc = history.history['acc']
    val_acc = history.history['val_acc']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(acc) + 1)

    plt.sca(ax[0])
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()

    plt.sca(ax[1])
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()

    plt.show()

def train_model(model, x, y, e = 10, bs = 32, v = 2, vs = 0.25):
    h = model.fit(x, y, epochs = e, batch_size = bs, verbose = v, validation_split = vs)
    return h



'''-----------1 - Embedding and Flatten'''
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense
#
# # First model: Embedding layer -> Flatten -> Dense classifier
# m0 = Sequential()
# m0.add(Embedding(WORDS, N_HIDDEN, input_length = LENGTH))
# m0.add(Flatten())
# m0.add(Dense(32, activation = 'relu'))
# m0.add(Dense(3, activation = 'softmax'))
# m0.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# m0.summary()
#
# # Train the first model and plot the history
# h0 = train_model(m0, x_train, one_hot_labels)
# plot_history(h0)

'''-------2 - Embedding and LSTM'''
from keras.layers import LSTM

# Second model: Embedding -> LSTM -> Dense classifier
# m1 = Sequential()
# m1.add(Embedding(WORDS, N_HIDDEN, input_length = LENGTH))
# m1.add(LSTM(N_HIDDEN))
# m1.add(Dense(3, activation = 'softmax'))
# m1.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# m1.summary()
#
# # Train the second model and plot the history
# h1 = train_model(m1, x_train, one_hot_labels)
# plot_history(h1)



'''------------3 - Embedding and GRU'''
from keras.layers import GRU
#
# # Third model: Embedding -> GRU -> Dense classifier
# m2 = Sequential()
# m2.add(Embedding(WORDS, N_HIDDEN, input_length = LENGTH))
# m2.add(GRU(LENGTH))
# m2.add(Dense(3, activation = 'softmax'))
# m2.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# m2.summary()
#
# # Train the third model and plot the history
# h2 = train_model(m2, x_train, one_hot_labels)
# plot_history(h2)


'''----------------4 - Embedding and GRU with dropout'''
# m3 = Sequential()
# m3.add(Embedding(WORDS, N_HIDDEN, input_length = LENGTH))
# m3.add(GRU(N_HIDDEN, dropout = 0.2, recurrent_dropout = 0.2))
# m3.add(Dense(3, activation = 'softmax'))
# m3.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# m3.summary()
#
# h3 = train_model(m3, x_train, one_hot_labels)
# plot_history(h3)


'''-------------5 - Embedding and stack of GRUs'''
# m4 = Sequential()
# m4.add(Embedding(WORDS, N_HIDDEN, input_length = LENGTH))
# m4.add(GRU(N_HIDDEN, dropout = 0.1, recurrent_dropout = 0.5, return_sequences = True))
# m4.add(GRU(N_HIDDEN, activation = 'relu', dropout = 0.1, recurrent_dropout = 0.5))
# m4.add(Dense(3, activation = 'softmax'))
# m4.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# m4.summary()
#
# h4 = train_model(m4, x_train, one_hot_labels)
# plot_history(h4)

'''-----------6 - Embedding and Conv1D'''
from keras.layers import Conv1D,Reshape, MaxPooling1D, GlobalMaxPooling1D, Dropout

# Sixth model: Embedding -> Conv1D & MaxPooling1D -> Dense classifier
# m5 = Sequential()
# m5.add(Embedding(WORDS, N_HIDDEN, input_length = LENGTH))
# m5.add(Conv1D(N_HIDDEN, 7, activation = 'relu'))
# m5.add(MaxPooling1D(5))
# m5.add(Dropout(0.2))
# m5.add(Conv1D(N_HIDDEN, 7, activation = 'relu'))
# m5.add(GlobalMaxPooling1D())
# m5.add(Dropout(0.2))
# m5.add(Dense(3, activation = 'softmax'))
# m5.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['acc'])
# m5.summary()
#
# h5 = train_model(m5, x_train, one_hot_labels)
# plot_history(h5)

'''--------------------7 - Embedding and mixed Conv1D and GRU'''
m6 = Sequential()

m6.add(Reshape((3, 4), input_shape=(12,)))

m6.add(Embedding(WORDS, N_HIDDEN, input_length = LENGTH, ))
m6.add(Conv1D(N_HIDDEN, 5, activation = 'relu'))
m6.add(MaxPooling1D(5))
m6.add(Conv1D(N_HIDDEN, 7, activation = 'relu'))
m6.add(GRU(N_HIDDEN, dropout = 0.1, recurrent_dropout = 0.5))
m6.add(Dense(3, activation = 'softmax'))
m6.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['mae'])
m6.summary()

h6 = train_model(m6, x_train, one_hot_labels)

plot_history(h6)



from keras.layers import BatchNormalization, PReLU, SpatialDropout1D, GlobalAveragePooling1D, GaussianNoise
from keras.layers import GlobalAveragePooling1D, AveragePooling1D

def get_model_v2(size=X.shape[1]):
    model = Sequential()
    model.add(Dense(256, input_dim=size, activation=None, use_bias=True))
    model.add(BatchNormalization(axis=-1))
    model.add(PReLU())
    model.add(SpatialDropout1D(rate=0.5))  # dropout is a type of regularisation. Regularisation helps to control overfitting
    model.add(GaussianNoise(stddev=0.1))

    model.add(Dense(256, activation=None, use_bias=True))
    model.add(BatchNormalization())
    model.add(PReLU())
    model.add(GlobalAveragePooling1D())
    model.add(AveragePooling1D(pool_size=2, strides=2, padding='same'))
    model.add(SpatialDropout1D(rate=0.2))
    model.add(BatchNormalization(axis=-1))
    # dropout is a type of regularisation. Regularisation helps to control overfitting
    model.add(Dense(199, activation='softmax'))
    model.compile(optimizer=RAdam(lr=0.001), loss='categorical_crossentropy', metrics=[])




    return model


from keras.wrappers.scikit_learn import KerasClassifier
from keras.constraints import maxnorm
from sklearn.model_selection import KFold, GroupKFold, GridSearchCV, train_test_split
from keras.optimizers import Adam

h =Adam()

# define the grid search parameters
neurons = [32, 64, 128, 256]
dropout_rate = [0.1, 0.2, 0.3, 0.4, 0.5]
learn_rate = [0.0001, 0.001, 0.01, 0.1]
momentum = [0.9, 0.99, 0.999]
batch_size = [256, 512, 1024]
epochs = [10, 20, 30]
activation = ['PRelu', 'relu', 'tanh', 'sigmoid']
optimizer = ['Adam', 'RAdam']

param_grid = dict(neurons=neurons, dropout_rate=dropout_rate, learn_rate=learn_rate,
                  momentum=momentum, batch_size=batch_size, epochs=epochs,
                  activation=activation, optimizer=optimizer,)
# create model
model= get_model_v2()
kcl_model = KerasClassifier(build_fn=model, epochs=50, batch_size=64, verbose=2)
grid = GridSearchCV(estimator=kcl_model,param_grid=param_grid,
                    scoring='roc_auc',n_jobs=-1, cv=3) #-1 means using all processors
grid_result = grid.fit(X, y, inputs=)
# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))