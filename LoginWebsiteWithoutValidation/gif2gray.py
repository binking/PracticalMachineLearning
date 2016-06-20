#!/usr/bin.env python
# Through this file, I wanna decompose a validation picture into 4 parts
#

import matplotlib.pyplot as plt
import matplotlib.image as mg
import numpy as np
import os

def decompose(img):
    rate = [1, 14, 27, 40, 53]
    subimg1 = img[:, rate[0]:rate[1]]
    subimg2 = img[:, rate[1]:rate[2]]
    subimg3 = img[:, rate[2]:rate[3]]
    subimg4 = img[:, rate[3]:rate[4]]
    return subimg1, subimg2, subimg3, subimg4

def gif2gray(gif_img):
    return np.dot(gif_img[...,:3], [0.299, 0.587, 0.114])

def img2file(img_name, grey_img):
    plt.imsave(img_name, grey_img) # # Supported formats: eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
#  To be continued ....  AttributeError: 'str' object has no attribute 'shape'
# ______Failed_______
def main():
    # there are many completed images stored in Validation_Codes_dir
    img_list = os.listdir('./Validation_Codes_dir')[:10] # WindowsError: [Error 3] : '/Validation_Codes_dir/*.*'
    for i, img_name in enumerate(img_list):
        print(i)
        img = mg.imread('./Validation_Codes_dir' + img_name)
        grey = gif2gray(img)
        img2file('./GreyImgesSet/%d.png' % (i), grey)


if __name__=='__main__':
    main()