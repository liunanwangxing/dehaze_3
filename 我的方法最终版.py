import cv2
import numpy as np
import sys

#线性数据值
a = 0.24352651197090014
b = -0.6721574794969143
c = 248.84730937137172

#计算方差图
def calculate_rgb_variance(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算每个像素点的RGB均值
    mean_rgb = np.mean(image, axis=2, keepdims=True)

    # 计算每个像素点的RGB方差
    variance_rgb = np.mean((image - mean_rgb)**2, axis=2)

    # 将方差图像归一化到0到255范围，以便显示
    normalized_variance = cv2.normalize( variance_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    #三维图像变一维图像
    normalized_variance = cv2.cvtColor( normalized_variance, cv2.COLOR_BGR2GRAY)

    return  normalized_variance

#计算亮度
def calculate_rgb_mean(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算每个像素点的RGB平均值
    mean_rgb = np.mean(image, axis=2)

    # 将方差图像归一化到0到255范围，以便显示
    normalized_mean_rgb = cv2.normalize(mean_rgb, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    # 三维图像变一维图像
    normalized__mean_rgb = cv2.cvtColor(normalized_mean_rgb, cv2.COLOR_BGR2GRAY)

    return normalized__mean_rgb


def DarkChannel(im, sz):
    b, g, r = cv2.split(im)
    dc = cv2.min(cv2.min(r, g), b)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (sz, sz))
    dark = cv2.erode(dc, kernel)
    return dark


def AtmLight(im, dark):
    [h, w] = im.shape[:2]
    imsz = h * w
    numpx = int(max(math.floor(imsz / 1000), 1))
    darkvec = dark.reshape(imsz, 1)
    imvec = im.reshape(imsz, 3)

    indices = darkvec.argsort()
    indices = indices[imsz - numpx::]
    atmsum = np.zeros([1, 3])
    for ind in range(1, numpx):
        atmsum = atmsum + imvec[indices[ind]]

    A = atmsum / numpx
    return A


def TransmissionEstimate(im, A, sz):
    omega = 0.95
    im3 = np.empty(im.shape, im.dtype)
    # wodehi

    for ind in range(0, 3):
        im3[:, :, ind] = im[:, :, ind] / A[0, ind]

    transmission = 1 - omega * DarkChannel(im3, sz)
    return transmission


def Guidedfilter(im, p, r, eps):
    mean_I = cv2.boxFilter(im, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(im * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(im * im, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * im + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    eps = 0.0001
    t = Guidedfilter(gray, et, r, eps)

    return t

def Recover(im, t, A, tx=0.1):
    res = np.empty(im.shape, im.dtype)
    t = cv2.max(t, tx)

    for ind in range(0, 3):
        res[:, :, ind] = (im[:, :, ind] - A[0, ind]) / t + A[0, ind]

    return res

if __name__ == '__main__':
    try:
        fn = sys.argv[1]
    except:
        fn = 'foggyHouse.jpg'
    def nothing(*argv):
        pass
    imgA2 = calculate_rgb_variance(fn)
    imgB2 = calculate_rgb_mean(fn)
    # 使用之前求解得到的线性方程计算C2
    imgC2 = a * imgA2 + b * imgB2 + c
    normalized_imgC2 = cv2.normalize(imgC2, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    src = cv2.imread(fn)
    I = src.astype('float64') / 255
    dark = DarkChannel(I, 15)
    A = AtmLight(I, dark)
    t = imgC2
    J = Recover(I, t, A, 0.1)
    cv2.namedWindow('t', cv2.WINDOW_NORMAL)
    cv2.imshow("t", t)
    cv2.imshow('J', J)
    # cv2.imwrite("foggyHouse_zuizhong2.jpg",J*255)
    cv2.waitKey()
