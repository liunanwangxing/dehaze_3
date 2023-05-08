import cv2
import numpy as np

def calculate_rgb_variance(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算每个像素点的RGB均值
    mean_rgb = np.mean(image, axis=2, keepdims=True)

    # 计算每个像素点的RGB方差
    variance_rgb = np.mean((image - mean_rgb)**2, axis=2)

    return variance_rgb

def display_image(image, title="Image"):
    cv2.imshow(title, image)

    cv2.destroyAllWindows()

input_image_path = "foggyHouse.jpg"

image = cv2.imread(input_image_path)
rgb_variance = calculate_rgb_variance(input_image_path)

# 将方差图像归一化到0到255范围，以便显示
normalized_variance = cv2.normalize(rgb_variance, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 显示原始图像和方差图像
cv2.imshow( "Original Image",image)
cv2.imshow( "RGB Variance Image",normalized_variance)
cv2.imwrite("foggyHouse_fangcha.jpg",normalized_variance)
cv2.waitKey(0)