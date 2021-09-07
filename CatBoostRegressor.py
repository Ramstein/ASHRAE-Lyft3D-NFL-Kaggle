

from keras.optimizers import Adam

import catboost as cb
from sklearn.preprocessing import LabelEncoder
import pandas as pd, numpy as np
cat_features_index = [0,1,2,3,4,5,6]

def auc(m, train, test):
    return (metrics.roc_auc_score(y_train, m.predict_proba(train)[:,1]),
                            metrics.roc_auc_score(y_test, m.predict_proba(test)[:,1]))

params = {'depth': [4, 7, 10],
          'learning_rate' : [0.01, 0.001, 0.001],
         'l2_leaf_reg': [1,4,9],
         'iterations': [300, 200, 100]}

x = np.array(params)
x = np.expand_dims(x, axis=2)
print(x.shape)
x = np.reshape(x, (-1, 29, 199))



df = pd.DataFrame(params)
print(df)

label_encoder = LabelEncoder()
df_encoded  = df.apply(label_encoder.fit_transform)
print(df_encoded)

print(label_encoder.classes_)

cbr = cb.CatBoostRegressor()

cbr.save_model()
cbr.load_model()
cbr.predict()
f= cbr.best_score_


clf = cb.CatBoostClassifier(eval_metric='AUC', iterations=500, learning_rate=0.001,
                            depth=10, l2_leaf_reg=9, verbose=None,
                            used_ram_limit='11gb', gpu_ram_part=0.97,
                            silent=None, logging_level=None)


cb_grid_search = GridSearchCV(estimator=cb, param_grid=params,
                              scoring='roc_auc',
                              cv=3, verbose=1)
cb_grid_search.fit(X=X, y=y)s

# without Categorical features



clf.fit(X=X, y=y, verbose=1, early_stopping_rounds=20, )
auc(clf, train=x_vl, test=y_vl)
clf.save_model('DSB_01-fold_id-{}.cbm'.format(fold_id))
clf.load_model()
clf.predict()

# With Categorical features
clf = cb.CatBoostClassifier(eval_metric="AUC",one_hot_max_size=31, \
                            depth=10, iterations= 500, l2_leaf_reg= 9, learning_rate= 0.15)
clf.fit(train,y_train, cat_features= cat_features_index)
auc(clf, train, test)

#
#
#
#
#
#
#
#
#
import lightgbm as lgb
from sklearn import metrics
#
def auc2(m, train, test):
    return (metrics.roc_auc_score(y_train,m.predict(train)),
                            metrics.roc_auc_score(y_test,m.predict(test)))

lg = lgb.LGBMClassifier(silent=False)
lgbmr = lgb.LGBMRegressor()
param_dist = {"max_depth": [25,50, 75],
              "learning_rate" : [0.01,0.05,0.1],
              "num_leaves": [300,900,1200],
              "n_estimators": [200]
             }

d_train = lgb.Dataset(train, label=y_train)
params = {"max_depth": 50, "learning_rate" : 0.1, "num_leaves": 900,  "n_estimators": 300}

# Without Categorical Features
model2 = lgb.train(params, d_train)

model2.save_model('')

model2 = lgb.Booster(model_file='model.txt')

auc2(model2, train, test)

#With Catgeorical Features
cate_features_name = ["MONTH","DAY","DAY_OF_WEEK","AIRLINE","DESTINATION_AIRPORT",
                 "ORIGIN_AIRPORT"]
model2 = lgb.train(params, d_train,
                   categorical_feature = cate_features_name,
model2.save_model()

                   )
auc2(model2, train, test)


import lightgbm as lgb
import os
models = []
path = r"C:\Datasets\Ashrae_Great_Energy_predictor_3\ashrae_lgbm_models"
model_files = os.listdir(path)
for file in model_files:
    lgmodel = lgb.Booster(model_file=os.path.join(path, file))
    models.append(lgmodel)
    print(file ,'loaded.')

print(models[1])
































