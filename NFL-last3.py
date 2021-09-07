TRAIN_ABLE_FALSE=True
# if TRAIN_ABLE_FALSE:
#     os.environ['CUDA_VISIBLE_DEVICES'] = "1"
import numpy as np
import pandas as pd
import sklearn.metrics as mtr
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from keras.layers import Dense,Input,Flatten,concatenate,Dropout,Lambda,BatchNormalization
from keras.models import Model
import keras.backend as K
from keras.callbacks import Callback
from  keras.callbacks import EarlyStopping,ModelCheckpoint
import datetime

TRAIN_OFFLINE = False


pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 150)



if TRAIN_OFFLINE:
    train = pd.read_csv('../input/train.csv', dtype={'WindSpeed': 'object'})
else:
    train = pd.read_csv('/kaggle/input/nfl-big-data-bowl-2020/train.csv', dtype={'WindSpeed': 'object'})



x = pd.DataFrame().drop_duplicates()



outcomes = train[['GameId','PlayId','Yards']].drop_duplicates()