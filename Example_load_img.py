
from urllib.request import urlretrieve
from PIL import Image
from skimage import img_as_uint
import os
import numpy as np
import tensorflow as tf
import skimage.io
import skimage.transform
import matplotlib.pyplot as plt


def load_img(imgDir):
    imgs = os.listdir(imgDir)
    imgNum = len(imgs)
    data = np.empty((imgNum, 1920, 2560, 3), dtype="uint8")
    # label = np.empty((imgNum,), dtype="uint8")
    for i in range (imgNum):
        img = Image.open(imgDir+"/"+imgs[i])
        arr = np.asarray(img, dtype="uint8")
        data[i, :, :, :] = arr
        # label[i] = int(imgs[i].split('.')[0])
    return data


if __name__ == '__main__':
    x = load_img("D:\Research_Stuffs/Final Design/IMAGESET/100低分化/")
    #dst = img_as_uint(x)
    plt.imshow(x[0])
    plt.show()
