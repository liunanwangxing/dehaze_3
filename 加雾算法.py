import cv2, math
import numpy as np


def demo():
    img_path = 'tiananmen_fenge.png'

    img = cv2.imread(img_path)
    img_f = img / 255.0
    (row, col, chs) = img.shape

    A = 0.9  # 亮度
    beta = 0.03  # 雾的浓度
    size = math.sqrt(max(row, col))  # 雾化尺寸
    center = (row // 2, col // 2)  # 雾化中心
    for j in range(row):
        for l in range(col):
            d = -0.04 * math.sqrt((j - center[0]) ** 2 + (l - center[1]) ** 2) + size
            td = math.exp(-beta * d)
            img_f[j][l][:] = img_f[j][l][:] * td + A * (1 - td)

    # cv2.imshow("src", img)
    cv2.imshow("dst", img_f)
    # cv2.imwrite('./image/w_new1.jpg',img_f*255)
    cv2.imwrite('tiananmen_fenge_jiawu.png', img_f * 255)
    cv2.waitKey()


if __name__ == '__main__':
    demo()