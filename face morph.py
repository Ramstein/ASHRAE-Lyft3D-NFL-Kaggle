import cv2
import numpy as np
# import dlib
import matplotlib.pyplot as plt

# cap = cv2.VideoCapture('/content/drive/My Drive/test_data/sample_video.mp4')
#
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
#
# while True:
#     _, frame = cap.read()
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
#     faces = detector(gray)
#     x, y = [], []
#
#     for face in faces:
#         x1 = face.left()
#         y1 = face.top()
#         x2 = face.right()
#         y2 = face.bottom()
#         #cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
#
#         landmarks = predictor(gray, face)
#         for n in range(0, 68):
#             x.append(landmarks.part(n).x)
#             y.append(landmarks.part(n).y)
#     implot = plt.imshow(frame)
#
#     plt.scatter(x, y, c='b', s=20)
#     plt.show()
#
#     cv2.imshow("Frame", frame)
#
#     key = cv2.waitKey(1)
#     if key == 27:
#         break


# cap = cv2.VideoCapture('sample_video.mp4')
# _, frame = cap.read()
# gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#
# implot = plt.imshow(frame)
#
# import random
# x, y = [], []
# for n in range(2, 50):
#     n1 = random.randint(n, n+100)
#     n2 = random.randint(n, n+100)
#     x.append(n1)
#     y.append(n2)
# plt.scatter(x, y, c='b', s=2)
#
# plt.show()

import pandas as pd
# pd.set_option(num_columns=100)

df = pd.read_csv('take.csv')


col = df.columns
for i in col:
    print(i)