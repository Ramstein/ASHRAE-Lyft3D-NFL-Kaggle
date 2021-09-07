# import numpy as np
# import pandas as pd
#
# # some random data frames
# df1 = pd.DataFrame(dict(x=np.random.randn(100),
#                         y=np.random.randint(0, 5, 100)))
# df2 = pd.DataFrame(dict(x=np.random.randn(100),
#                         y=np.random.randint(0, 5, 100)))
#
# print(df1, '\n', df2)
#
# # concatenate them
# df_concat = pd.concat(df1, df2)
#
# print(df_concat.mean())
#
# print(df_concat.median())
#
# print(df_concat.mode())
#
# by_row_index = df_concat.groupby(df_concat.index)
# df_means = by_row_index.mean()
#
# print(df_means)
#
#
#
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.linear_model import LinearRegression, Perceptron, SGDClassifier
# from sklearn.svm import SVC, LinearSVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.naive_bayes import GaussianNB
# from sklearn.linear_model import perceptron
# from sklearn.tree import  DecisionTreeClassifier
#
#
#
# # !pip install efficientnet_pytorch
# from efficientnet_pytorch import EfficientNet
# # model = EfficientNet.from_pretrained(model_name='efficientnet-b4',
# #                                      num_classes=len(classes)+1
# #                                      )
#
# # model = EfficientNet.from_name(model_name='efficientnet-b4')
#
#
#
# # Name	# Params	Top-1 Acc.	Pretrained?
# # efficientnet-b0	5.3M	76.3	✓
# # efficientnet-b1	7.8M	78.8	✓
# # efficientnet-b2	9.2M	79.8	✓
# # efficientnet-b3	12M	    81.1	✓
# # efficientnet-b4	19M	    82.6	✓
#
#
# import torch
#
#
# model = torch.hub.load('pytorch/vision', 'deeplabv3_resnet101', pretrained=True,
#                        in_channels=3, out_channels=1)
# model.eval()
#
# # RSNA intercranial heamorraghe detection
import torch
model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
    in_channels=3, out_channels=1, init_features=32, pretrained=True)


from efficientnet_pytorch.model import EfficientNet
from efficientnet_pytorch.utils import efficientnet
classes = 10

block_args, global_params = efficientnet(width_coefficient=0.5, depth_coefficient=0.2,
                                         image_size=[256, 256], num_classes=len(classes)+1)

model = EfficientNet(blocks_args=block_args, global_params=global_params)


model = model.from_pretrained(model_name='efficientnet-b4', num_classes=len(classes)+1,
                              )

from keras.models import Model, load_model


import numpy as np
from sklearn import metrics
y = np.array([1, 1, 2, 2])
pred = np.array([0.1, 0.4, 0.35, 0.8])
fpr, tpr, thresholds = metrics.roc_curve(y, pred, pos_label=2)
metrics.auc(fpr, tpr)


import segmentation_models_pytorch as smp

eff = smp.encoders.efficientnet.EfficientNet()


# import logging
# import sys
#
# logging.basicConfig(format='%(asctime)s | %(levelname)s : %(message)s',
#                      level=logging.INFO, stream=sys.stdout)
#
# logging.info(filename)