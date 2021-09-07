# #
# # import librosa
# # import librosa.display
# # import matplotlib.pyplot as plt, numpy as np, pandas as pd
# #
# # filename = r'C:\Users\Ramstein\Downloads\hmpback2.wav'
# #
# # y, sr = librosa.load(filename)
# #
# # whale_song, _ = librosa.effects.trim(y)
# # librosa.display.waveplot(whale_song, sr=sr)
# #
# # n_fft = 2048
# # hop_length = 512
# # n_mels = 128
# #
# #
# # S = librosa.feature.melspectrogram(whale_song, sr=sr, n_fft=n_fft,
# #                                    hop_length=hop_length,
# #                                    n_mels=n_mels)
# # print(S)
# # S = np.array(S, dtype=float)
# # print(S.shape)
# # S_DB = librosa.power_to_db(S, ref=np.max)
# # librosa.display.specshow(S_DB, sr=sr, hop_length=hop_length, x_axis='time', y_axis='mel')
# #
# # plt.colorbar(format='%+2.0f dB')
#
# import glob
# import os, cv2, numpy as np
# from tqdm import tqdm, tqdm_notebook
#
#
#
#
#
#
#
# class BEVImageDataset():
#     def __init__(self, sample_token, train_images, train_maps):
#         self.sample_token = sample_token
#         self.train_images = train_images
#         self.train_maps = train_maps
#
#     def __len__(self):
#         return len(self.sample_token)
#
#     def __getitem__(self, idx):
#         sample_token = self.sample_token[idx]
#
#         #         sample_token = input_filepath.split("/")[-1].replace("_input.png","")
#
#         input_filepath = sorted(glob.glob(os.path.join(train_images, "*.jpeg")))
#
#         map_filepath = sorted(glob.glob(os.path.join(train_maps, "*.png")))
#         print(input_filepath, map_filepath)
#
#         im = cv2.imread(input_filepath, cv2.IMREAD_UNCHANGED)
#
#         map_im = cv2.imread(map_filepath, cv2.IMREAD_UNCHANGED)
#         print(im, map_im)
#         im = np.concatenate((im, map_im), axis=2)
#
#         im = im.astype(np.float32) / 255
#         im = np.transpose(im, (2, 0, 1))
#
#         return im, sample_token
#
# train_images = os.listdir(r'C:\Datasets\Lyft_3D_Object_Detection_for_Autonomous_Vehicles\train_images')
# train_map = r'C:\Datasets\Lyft_3D_Object_Detection_for_Autonomous_Vehicles\train_maps\map_raster_palo_alto.png'
# all_sample_tokens = []
#
#
# map_im = cv2.imread(train_map, cv2.IMREAD_UNCHANGED)
# for i, image in enumerate(train_images):
#     im = cv2.imread(image, cv2.IMREAD_UNCHANGED)
#     print(im, map_im)
#     im = np.concatenate((im, map_im), axis=2)
#     im = np.array(im).astype(np.float32)/255
#     im = np.transpose(im, (2, 0, 1))
#
#
#
#     if i==10:break
#
# train_dataset = BEVImageDataset.__init__(all_sample_tokens, train_images, train_maps)
# train_dataset[0]
#
# from keras.models import Sequential
# from keras.models import load_model
#
# model = Sequential()
# model.load_weights(filepath)


import os
test_imgs_folder = ['a', 's', 'f', 'g', 't', 'e', ]
y_pred_mean = [2,2,5 ,6,9,8]

class_names = ['m', 'n', 'b', 'v', 'c', 'x', 'z']

for i, (img, predictions) in enumerate(zip(test_imgs_folder, y_pred_mean)):

    for class_i, class_name in enumerate(class_names):

        print(img, predictions, class_i, class_name)


        # if predictions[class_i] < recall_thresholds[class_name]:
        #     image_labels_empty.add(f'{img}_{class_name}')