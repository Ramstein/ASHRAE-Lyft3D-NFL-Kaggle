# #
# # from sklearn.preprocessing import StandardScaler
# #
# # scaler = StandardScaler()
# #
# # import numpy as np
# #
# #
# # # if not (len(basetable)==22 and len(basetable.columns)==77): continue
# #
# # import pandas as pd
# # train = pd.read_csv()
# # col = len(train.columns)
# #
# #
# #
# # y = np.clip(np.cumsum(y_pred, ), a_min=0, a_max=1).tobytes()[0]
# # basetable=y
#
import pandas as pd, numpy as np
# df1 = pd.DataFrame({'A': ['A0', 'A1', 'A2', 'A3'],
#                      'B': ['B0', 'B1', 'B2', 'B3'],
#                      'C': ['C0', 'C1', 'C2', 'C3'],
#                      'D': ['D0', 'D1', 'D2', 'D3']},
#                     index=[0, 1, 2, 3])
#
#
# df2 = pd.DataFrame({'A': ['A4', 'A5', 'A6', 'A7'],
#                      'B': ['B4', 'B5', 'B6', 'B7'],
#                      'C': ['C4', 'C5', 'C6', 'C7'],
#                      'D': ['D4', 'D5', 'D6', 'D7']},
#                     index=[4, 5, 6, 7])
#
#
# df3 = pd.DataFrame({'A': ['A8', 'A9', 'A10', 'A11'],
#                      'B': ['B8', 'B9', '', 'B11'],
#                      'C': ['C8', '', 'nan', 'C11'],
#                      'D': ['D8', 'D9', 'D10', 'D11']},
#                     index=[8, 9, 10, 11])
# df = pd.DataFrame([[np.nan, 2, np.nan, 0],
#                     [3, 4, np.nan, 1],
#                     [np.nan, np.nan, np.nan, 5],
#                     [np.nan, 3, np.nan, 4]],
#                    columns=list('ABCD'))
# df
#
# # frames = [df1, df2, df3]
# #
# # print(pd.concat(frames))
#
# final = df.fillna(8, axis=1)
# fia = df3['B'].fillna(value=4)
# print(final, fia )
# b = df.fillna(0)
# print(df.isnull,b )


df = pd.read_csv(r'C:\Datasets\Ashrae_Great_Energy_predictor_3\building_metadata.csv')

x = len(df)
col = df.columns
print(x, col[0])
