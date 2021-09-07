



import wget
wget.download(bar=any())


import efficientnet.keras as efn
from keras.layers import Dropout, Dense, SpatialDropout2D
from keras.models import Model
import tensorflow as tf


import efficientnet.tfkeras
from keras.models import load_model

model = efn.EfficientNetB7(include_top=False,
                   weights=None,
                   input_shape=(256, 256, 3),
                   pooling='avg')
x = model.output
x = Dropout(0.125)(x)
y_pred = Dense(6, activation = 'sigmoid')(x)

model = Model(inputs = model.input, outputs = y_pred)

model = load_model(r'C:\Users\Ramstein\Downloads\efficientnet-b7\efficientnet-b7\model.ckpt.pth',
                   custom_objects={'AUC': tf.keras.metrics.AUC()})
model.summary()





def predictions(test_df, model):
    test_preds = model.predict_generator(TestDataGenerator(test_df, None, 5, SHAPE, TEST_IMAGES_DIR), verbose = 1)
    return test_preds[:test_df.iloc[range(test_df.shape[0])].shape[0]]

def ModelCheckpointFull(model_name):
    return ModelCheckpoint(model_name,
                           monitor = 'val_loss',
                           verbose = 1,
                           save_best_only = False,
                           save_weights_only = True,
                           mode = 'min',
                           period = 1)

# Create Model
def create_model():
    K.clear_session()

    base_model =  efn.EfficientNetB2(weights = 'imagenet',
      include_top = False, pooling = 'avg', input_shape = SHAPE)
    x = base_model.output
    x = Dropout(0.125)(x)
    y_pred = Dense(6, activation = 'sigmoid')(x)

    return Model(inputs = base_model.input, outputs = y_pred)


from keras.models import load_model

model = load_model('nfjgn', )



keras.losses.cus

metric = getmetric




import wget
output_directory = r'C//'
url = 'http://www.futurecrew.com/skaven/song_files/mp3/razorback.mp3'
filename = wget.download(url, out=output_directory)






















