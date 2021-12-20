#! /usr/bin/env python
# coding=utf-8


import cv2
import numpy as np
from PIL import Image
import math
ID = 60
label_txt = ""
image_info = open(label_txt).readlines()[ID].split()

image_path = image_info[0]
image = cv2.imread(image_path)

for bbox in image_info[1:]:
    bbox = bbox.split(",")
    image = cv2.rectangle(image, (int(float(bbox[0])),
                                 int(float(bbox[1]))),
                                (int(float(bbox[2])),
                                 int(float(bbox[3]))), (255,0,0), 2)

image = Image.fromarray(np.uint8(image))
image.show()




import time
from numba import jit
@jit()
def AddHaze1(img):
    img_f = img / 255.0
    (row, col, chs) = img.shape

    A = 0.5  # 亮度
    beta = 0.01 * 8  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    t1 = time.time()
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)
    t2 = time.time()
    print('time:',t2-t1)
    img_f = img_f * 255
    img_f = np.clip(img_f, 0, 255)
    img_f = img_f.astype(np.uint8)
    return img_f
def AddHaze2(img):
    img_f = img / 255.0

    A = np.random.uniform(0.8, 0.95)
    t = np.random.uniform(0.3, 0.6)
    img_f = img_f * t + A * (1 - t)

    img_f = img_f * 255
    img_f = np.clip(img_f, 0, 255)
    img_f = img_f.astype(np.uint8)

    return img_f
def Gammafilter(img, gamma = 0.5):
    # img_f = img / 255.0
    img_f = np.power(img, gamma)

    # img_f = img_f * 255
    # img_f = np.clip(img_f, 0, 255)
    # img_f = img_f.astype(np.uint8)

    return img_f
# def DarkChannel(im,sz):
#     b,g,r = cv2.split(im)
#     dc = cv2.min(cv2.min(r,g),b);
#     kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
#     dark = cv2.erode(dc, kernel)
#     # dark = dc
#     return dark
#
# def AtmLight(im,dark):
#     [h,w] = im.shape[:2]
#     imsz = h*w
#     numpx = int(max(math.floor(imsz/1000),1))
#     darkvec = dark.reshape(imsz,1)
#     imvec = im.reshape(imsz,3)
#
#     indices = darkvec.argsort()
#     indices = indices[imsz-numpx::]
#
#     atmsum = np.zeros([1,3])
#     for ind in range(1,numpx):
#        atmsum = atmsum + imvec[indices[ind]]
#
#     A = atmsum / numpx
#     return A
#
# def TransmissionEstimate(im,A,sz):
#     omega = 0.95;
#     im3 = np.empty(im.shape,im.dtype);
#
#     for ind in range(0,3):
#         im3[:,:,ind] = im[:,:,ind]/A[0,ind]
#
#     transmission = 1 - omega*DarkChannel(im3,sz);
#     return transmission
#
# def Guidedfilter(im,p,r,eps):
#     mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r));
#     mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r));
#     mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r));
#     cov_Ip = mean_Ip - mean_I*mean_p;
#
#     mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r));
#     var_I   = mean_II - mean_I*mean_I;
#
#     a = cov_Ip/(var_I + eps);
#     b = mean_p - a*mean_I;
#
#     mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r));
#     mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r));
#
#     q = mean_a*im + mean_b;
#     return q;
#
# def TransmissionRefine(im,et):
#     gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY);
#     gray = np.float64(gray)/255;
#     r = 60;
#     eps = 0.0001;
#     t = Guidedfilter(gray,et,r,eps);
#
#     return t;
#
# def Recover(im,t,A,tx = 0.1):
#     res = np.empty(im.shape,im.dtype);
#     t = cv2.max(t,tx);
#
#     for ind in range(0,3):
#         res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
#
#     return res


def DarkChannel(im):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b);
    return dc


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort(0)
    indices = indices[(imsz - numpx):imsz]

    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def DarkIcA(im, A):
    im3 = np.empty(im.shape, im.dtype)
    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]
    return DarkChannel(im3)
'''if __name__ == '__main__':
    img = cv2.imread('/home/lwy/work/code/defog_yolov3/scripts/AM_Bing_274.png')
    cv2.imwrite('org.png', img)

    I = img.astype('float64') / 255
    I = Gammafilter(I, 0.5)

    dark_i = DarkChannel(I)
    defog_A_i = AtmLight(I, dark_i)
    IcA_i = DarkIcA(I, defog_A_i)
    tx = 1 - 0.59*IcA_i
    tx[tx < 0.01] = 0.01

    res = np.empty(I.shape,I.dtype);
    for ind in range(0,3):
        res[:,:,ind] = (I[:,:,ind]-defog_A_i[:,ind])/tx[:,:] + defog_A_i[:,ind]
    # J = (I - defog_A_i) / tf_1 + defog_A_i
    # tx_1 = tf.tile(tx, [1, 1, 1, 3])
    # return (img - defog_A[:, None, None, :])/tf.maximum(tx_1, 0.01) + defog_A[:, None, None, :]

    # dark = DarkChannel(I, 15)
    # A = AtmLight(I, dark)
    # t = TransmissionEstimate(I, A, 15)
    # # t = TransmissionRefine(img, t)
    # J = Recover(I, t, A, 0.1)

    img_f = res * 255


    # img_f = Gammafilter(img)

    img_name = 'lwyGamma' + '.png'
    cv2.imwrite(img_name, img_f)
'''
#
# cv2.imshow("src", img)
# cv2.imshow("dst", img_f)
# cv2.waitKey()


