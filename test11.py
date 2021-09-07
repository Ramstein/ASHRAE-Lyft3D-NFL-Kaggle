# import numpy as np
# import pandas as pd
# #
# # # students = [ ('jack', 34, 'Sydeny') ,
# # #     ('Riti', 30, 'Delhi' ) ,
# # #     ('Aadi', 16, 'New York') ]
# # #
# # # # Create a DataFrame object
# # # df = pd.DataFrame(students, columns = ['Name' , 'Age', 'City'],
# # #                      index=['a', 'b', 'c'])
# # # print(df)
# # #
# # # for index, rows in zip(df.iterrows()):
# # #     # final = pd.concat([row,row])
# # #     print(index, rows)
# # # print(final)
# #
# # # # getting the whole of the column
# # # for i in range(len(df)):
# # #     if i == 0:     temp = df.iloc[[i]];continue
# # #     temp1 = df.iloc[[i]];print(temp1)
# # #     temp = pd.concat([temp, temp1])
# # #
# # #
# # # print(temp)
# # # rowC = df.loc[ ['c'] , : ]
# #
# #
# # temp, first =0, 0
# # for i in range(len(train)/22):
# #     for j in range(22):
# #         if first==0:
# #             df[j] = train.iloc[[temp]]
# #             first+=1
# #             temp+=1
# #         else:
# #             df[j] = pd.concat([df[j-1], train.iloc[[temp]]])
# #             temp+=1
# #
# # for i in range(21):
# #     train_f = pd.concat([df[i], df[i+1]], ignore_index=True)
# #
# # # df = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# # #
# # # temp, first =0, 0
# # # df = [0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21]
# # # for i in range(int(509762/22)):
# # #     for j in range(22):
# # #         if first==0:
# # #             first+=1
# # #             temp+=1
# # #         else:
# # #             print(temp)
# # #             temp+=1
# import datetime
# # print(datetime.utcnow().strftime('%H:%M:%S.%f')[:-3])
# print(datetime.datetime.now())
#
# #da
# # nArr2D = np.array([[21,22,23]])
# # narr1 = np.array([[11,22,33], [43,77,89]], dtype=np.float16)
# # # print(nArr2D)
# #
# # # n = nArr2D+narr1
# #
# # n = np.concatenate([nArr2D, narr1], axis=0)
# # print(n)
# #
# # from sklearn.preprocessing import StandardScaler
# # scaler = StandardScaler()
# # X = scaler.fit_transform(X)
#
#
#
#
#
# # Creating a 2 dimensional numpy array
# # data = np.array([[5.8, 2.8], [6.0, 2.2]])
# # print(data)
# # index = np.array([i for i in range(2)])
# # # Creating pandas dataframe from numpy array
# # dataset = pd.DataFrame({'Column1': data[:, 0], 'Column2': data[:, 1]}, index=index)
# # print(dataset)
#
# # temp=0
# # for i in range(23170, -1, -1):
# #     print(i)
# #     temp+=1
# # print(temp)
#
#
#
#
#
#
#
#
#
# # alist = [[1,2,3,4], [11,22,33], [4], [435,456]]
# # idlist = [[i+1]*len(x) for i,x in enumerate(alist)]
# # x = np.concatenate(alist)
# # y = np.concatenate(idlist)
# #
# # print(idlist)
#
#
#
#
#
# alist = [[1,2,3,4], [11,22,33], [4], [435,456]]
# alist2 = [[1,2,3,5], [14515,22,1653], [87], [4115,456]]
# #
# # my_arrays = [np.array(x) for x in alist]
# # print(my_arrays)
# # index_arrays = [np.ones((x.shape), dtype=int)*i for i, x in enumerate(my_arrays)]
# #
# # x= np.concatenate(my_arrays)
# # y = np.concatenate(index_arrays)
# #
# # print(index_arrays)
# # print(x)
# # print(y)
#
# # df = pd.DataFrame([('bird', 389.0),
# #                     ('bird', 24.0),
# #                    ('mammal', 80.5),
# #                     ('mammal', np.nan)],
# #                    index=['falcon', 'parrot', 'lion', 'monkey'],
# #                   columns=('class', 'max_speed'))
# # my_arrays = [np.array(x) for x in alist]
# # my_arrays2  = [np.array(x) for x in alist2]
# #
# # data = np.concatenate([my_arrays, my_arrays2], axis=0)
# #
# # dataset = pd.DataFrame(data, columns=['a'])
# # dataset.reset_index(drop=True)
# # print(dataset)
# # import datetime
# # time1 = '2017-09-08T00:44:06.000Z'
# # time2 = '2017-09-08T00:44:05.000Z'
# # r = datetime.datetime.strptime(time1, "%Y-%m-%dT%H:%M:%S.%fZ")
# # x = datetime.datetime.strptime(time2, "%Y-%m-%dT%H:%M:%S.%fZ")
# #
# #
# # z = (r-x)
# # # print(r, x)
# # print(int(str(z).split(':')[-1]))
#
#
# train = pd.read_csv(r'C:\Users\Ramstein\Chromium_Downloads\nfltraining.csv')
# #
# # print(train.columns)
# train = train.sort_values()
#
#
# import time
# print(time.strftime('%H:%M:%S'))
#
#
# h = pd.concat([first, last], axis=0)
#
#
# h = pd.DataFrame.drop()
#
# h.sort_values(by='PlayId', axis=0,inplace=True)
#
#
#
#
# import os
# os.chdir()


import numpy as np
bias = [ 0.12093217,  0.26705384,  0.19426322,  0.05820028,  0.05112031,
     0.2587961,   0.06879828, -0.03979903,  0.00650737,  0.22950806,
     0.18568335,  0.09700873,  0.22154604, 0.16963722, 0.07723659,
     0.01608587, -0.15127839,  0.01302223,  0.12636952, -0.05113579,
     0.2545413,   0.05643194,  0.38390982, 0.09412267,  0.00443826,
     0.7398891,   0.07763126, -0.15044677,  0.09285322, -0.21749116,
     0.1118157,   0.56120837, -0.25816086,  0.08835327,  0.14333047,
     0.11272971,  0.39173624,  0.15946516, 0.14671084, -0.6130876,
     0.11584569,  0.17850026,  0.03994513, 0.0610672,   0.13213407,
     0.0759801,   0.05931563,  0.12084971,  0.02450513,  0.31131038,
     0.31453267,  0.13205571, -0.30115315,  0.02376384,  0.11087047,
     0.14482947,  0.03460854,  0.06482973,  0.09113415, -0.08124948,
     0.10306761, -0.30297118,  0.09408436,  0.11774008]


# a = np.array(bias)
# a = a.flatten()
# print(a)
# for i, bi in enumerate(bias)

# np.expand_dims( bi /256 -0.5 ,0)

a = np.reshape(bias, (-1,)).astype(dtype=float)
print(a.shape)

# import cv2
#
# scale = 0.2
# image = cv2.resize(image_orig, dsize=(-1, 3), fx=scale, fy=scale)  # N, C, W, H
