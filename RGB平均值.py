import cv2
import numpy as np

def calculate_rgb_mean(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算每个像素点的RGB平均值
    mean_rgb = np.mean(image, axis=2)

    return mean_rgb

def display_image(image, title="Image"):
    cv2.imshow(title, image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_image_path = "foggyHouse.jpg"

image = cv2.imread(input_image_path)
rgb_mean = calculate_rgb_mean(input_image_path)

# 将平均值图像归一化到0到255范围，以便显示
normalized_mean = cv2.normalize(rgb_mean, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 显示原始图像和平均值图像
# display_image(image, "Original Image")
# display_image(normalized_mean, "RGB Mean Image")
cv2.imwrite('foggyHouse_pingjun.jpg',normalized_mean)
