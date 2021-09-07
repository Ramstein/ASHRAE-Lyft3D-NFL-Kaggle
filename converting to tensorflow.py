import os
import os.path as osp
import argparse

import tensorflow as tf

from keras.models import load_model
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Dense,Activation
from keras import optimizers
from sklearn.utils import class_weight
from keras import backend as K

# Import Custom Modules
import efficientnet.keras as efn
# from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit

def swish(x):
    return K.sigmoid(x) * x


def create_model():
    K.clear_session()

    base_model = efn.EfficientNetB3(weights=None, include_top=False,
                                    pooling='avg', input_shape=(256, 256, 3))
    x = base_model.output
    y_pred = Dense(6, activation='sigmoid')(x)

    model = Model(inputs=base_model.input, outputs=y_pred)
    model.load_weights('')
    #     model.summary()
    return model

def convertGraph(modelPath, outdir, numoutputs, prefix, name):
    '''
    Converts an HD5F file to a .pb file for use with Tensorflow.
    Args:
        modelPath (str): path to the .h5 file
           outdir (str): path to the output directory
       numoutputs (int):
           prefix (str): the prefix of the output aliasing
             name (str):
    Returns:
        None
    '''

    # NOTE: If using Python > 3.2, this could be replaced with os.makedirs( name, exist_ok=True )
    if not os.path.isdir(outdir):
        os.mkdir(outdir)

    K.set_learning_phase(0)
    net_model = create_model().load_weights(modelPath)

    model = Sequential()
    # net_model = load_model(modelPath, custom_objects = {"swish":swish})

    # Alias the outputs in the model - this sometimes makes them easier to access in TF
    pred = [None] * numoutputs
    pred_node_names = [None] * numoutputs
    for i in range(numoutputs):
        pred_node_names[i] = prefix + '_' + str(i)
        pred[i] = tf.identity(net_model.output[i], name=pred_node_names[i])
    print('Output nodes names are: ', pred_node_names)

    sess = K.get_session()

    # Write the graph in human readable
    f = 'graph_def_for_reference.pb.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), outdir, f, as_text=True)
    print('Saved the graph definition in ascii format at: ', osp.join(outdir, f))

    # Write the graph in binary .pb file
    from tensorflow.python.framework import graph_util
    from tensorflow.python.framework import graph_io
    constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
    graph_io.write_graph(constant_graph, outdir, name, as_text=False)
    print('Saved the constant graph (ready for inference) at: ', osp.join(outdir, name))


if __name__ == '__main__':
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--model', '-m', dest='model', required=True,
    #                     help='REQUIRED: The HDF5 Keras model you wish to convert to .pb')
    # parser.add_argument('--numout', '-n', type=int, dest='num_out', required=True,
    #                     help='REQUIRED: The number of outputs in the model.')
    # parser.add_argument('--outdir', '-o', dest='outdir', required=False, default='./',
    #                     help='The directory to place the output files - default("./")')
    # parser.add_argument('--prefix', '-p', dest='prefix', required=False, default='k2tfout',
    #                     help='The prefix for the output aliasing - default("k2tfout")')
    # parser.add_argument('--name', dest='name', required=False, default='output_graph.pb',
    #                     help='The name of the resulting output graph - default("output_graph.pb")')
    # args = parser.parse_args()
    model_path= r'C:\Users\Ramstein\Downloads\RSNA_efficientNetB3_01-val_acc-0.9013.h5'
    convertGraph(modelPath=model_path, outdir=r'C:\Datasets',
                 numoutputs=1, prefix='model', name='tftokeras')
