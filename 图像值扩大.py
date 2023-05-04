import cv2
import numpy as np

def multiply_image(input_image_path, multiplier, output_image_path):
    # 读取图像
    image = cv2.imread(input_image_path)

    # 将图像转换为浮点数，以避免整数溢出
    image = image.astype(np.float32)

    # 对图像的所有像素点值乘以给定的数
    multiplied_image = cv2.multiply(image, multiplier)

    # 将结果图像的数据类型转换回uint8
    multiplied_image = np.clip(multiplied_image, 0, 255).astype(np.uint8)

    # 保存结果图像
    # cv2.imwrite(output_image_path, multiplied_image)

    # 显示结果图像
    cv2.imshow("Multiplied Image", multiplied_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

input_image_path = ""
# output_image_path = ""
multiplier = 1.5

multiply_image(input_image_path, multiplier, output_image_path)
