import skimage
from skimage.segmentation import slic,mark_boundaries
from skimage import io
import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance
import cv2
import math
import numpy as np
import cv2 as cv
#
def DarkChannel(im,sz):
    b,g,r = cv2.split(im)
    dc = cv2.min(cv2.min(r,g),b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(sz,sz))
    dark = cv2.erode(dc,kernel)
    return dark

def AtmLight(im,dark):
    [h,w] = im.shape[:2]
    imsz = h*w
    numpx = int(max(math.floor(imsz/1000),1))
    darkvec = dark.reshape(imsz)
    imvec = im.reshape(imsz,3)

    indices = darkvec.argsort()
    indices = indices[imsz-numpx::]

    atmsum = np.zeros([1,3])
    for ind in range(1,numpx):
       atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A

def TransmissionEstimate(im,A,sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega*DarkChannel(im3,sz)
    return transmission

def TransmissionEstimateX(im,A,sz):
    b,g,r = cv2.split(im)
    x = (b+r+g)/3
    im3 = np.empty(im.shape, im.dtype)

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - x*DarkChannel(im3,sz)
    return transmission

def Guidedfilter(im,p,r,eps):
    mean_I = cv2.boxFilter(im,cv2.CV_64F,(r,r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F,(r,r))
    mean_Ip = cv2.boxFilter(im*p,cv2.CV_64F,(r,r))
    cov_Ip = mean_Ip - mean_I*mean_p

    mean_II = cv2.boxFilter(im*im,cv2.CV_64F,(r,r))
    var_I   = mean_II - mean_I*mean_I

    a = cov_Ip/(var_I + eps)
    b = mean_p - a*mean_I

    mean_a = cv2.boxFilter(a,cv2.CV_64F,(r,r))
    mean_b = cv2.boxFilter(b,cv2.CV_64F,(r,r))

    q = mean_a*im + mean_b
    return q

def TransmissionRefine(im,et):
    gray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray)/255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray,et,r,eps)

    return t

def Recover(w,h,im,t,A,tx = 0.1):
    res = np.empty(im.shape,im.dtype)
    t = cv2.max(t,tx)

    # for ind in range(0,3):
    #     res[:,:,ind] = (im[:,:,ind]-A[0,ind])/t + A[0,ind]
    length = len(w)
    for ind in range(0, 3):
        res[min(w):max(w), min(h):max(h), ind] = (im[min(w):max(w), min(h):max(h), ind]-A[0,ind])/t[min(w):max(w), min(h):max(h)] + A[0,ind]

    return res

sum = 0
per_image_Bmean = []
per_image_Gmean = []
per_image_Rmean = []
src = cv2.imread('canon3.bmp')

I = src.astype('float64') / 255

dark = DarkChannel(I, 15)
A = AtmLight(I, dark)

segments = slic(I, n_segments=10, compactness=15,start_label = 1)#进行SLIC分割
out=mark_boundaries(I,segments)
maxn = max(segments.reshape(int(segments.shape[0] * segments.shape[1]), ))
rrcover = np.empty(I.shape,I.dtype)

for i in range(1, maxn+1):
    a = np.array(segments == i)
    a = a.reshape(a.shape[0], a.shape[1], 1)
    a1 = np.concatenate((a, a), axis=2)
    a = np.concatenate((a1, a), axis=2)
    b = src * a
    w, h = [], []
    for x in range(b.shape[0]):
        for y in range(b.shape[1]):
            if b[x][y][0] != 0:
                w.append(x)
                h.append(y)
    c = b[min(w):max(w),min(h):max(h)]
    e = c.reshape(c.shape[0],c.shape[1],3)
    per_image_Bmean.append(np.mean(e[:,:,0]))
    per_image_Gmean.append(np.mean(e[:,:,1]))
    per_image_Rmean.append(np.mean(e[:,:,2]))
    R_mean = np.mean(per_image_Rmean)
    G_mean = np.mean(per_image_Gmean)
    B_mean = np.mean(per_image_Bmean)
    x2 = (R_mean+G_mean+B_mean)/3
    if x2 < 50:
        te = TransmissionEstimateX(I, A, 15)
    else:
        te = TransmissionEstimate(I, A * 1.5, 15)

    t = TransmissionRefine(src, te)

    J = Recover(w, h, I, t, A, 0.1)
    for ind in range(0, 3):
        rrcover[min(w):max(w), min(h):max(h), ind] = J[min(w):max(w), min(h):max(h), ind]

# cv2.imshow("dark", dark)
cv2.imshow("t", t)
# cv2.imshow('I', src)
# cv2.imshow('J', rrcover)
# cv2.imwrite("chaofenbianlv.png", rrcover* 255)
# cv2.imwrite("tiananmen_fenge_jiawu_fenge.png", rrcover * 255)
# cv2.imwrite("fog_364_fenge.jpg",rrcover*255)
cv2.waitKey()







