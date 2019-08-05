import keras
# from keras import models, layers
from keras import backend
from keras import datasets

from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
from own_test.create_model import DATA
from own_test.sfile import what_is_newest_folder

import tensorflow as tf
# tensorflow gpu configuration
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)


def main():

    folder_name = what_is_newest_folder()

    model = load_model(folder_name + '/' + 'create_model.h5')
    model.summary()

    ###############################
    # 데이터 정확도 측정을 보고 싶을때 #
    ###############################
    # data = DATA()
    #
    # score = model.evaluate(data.x_test, data.y_test)
    # print()
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])
    # 데이터 정확도 측정 끝 #

    img_path = './2.jpg'

    img = image.load_img(img_path, color_mode='grayscale', target_size=(28, 28))
    img_tensor = image.img_to_array(img)
    img_tensor = np.expand_dims(img_tensor, axis=0)
    img_tensor /= 255

    img_tensor_to_show = img_tensor.squeeze(axis=3)
    print(img_tensor.shape)

    plt.imshow(img_tensor_to_show[0], cmap='gray')
    plt.show()
    prediction_1 = model.predict([img_tensor])

    selected_index = 0
    selected_value = prediction_1[0][0]

    for i in range(len(prediction_1[0])):
        # print(prediction_1[0][i])
        if prediction_1[0][i] > selected_value:
            selected_value = prediction_1[0][i]
            selected_index = i
            print(prediction_1[0][i], " percent sure")

    print("Number is: ",  selected_index)


if __name__ == '__main__':
    main()


