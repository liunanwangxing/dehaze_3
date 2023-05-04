import cv2
import numpy as np

def subtract_images(image1_path, image2_path):
    # 读取两个图像
    img1 = cv2.imread(image1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image2_path, cv2.IMREAD_GRAYSCALE)

    # 确保图像具有相同的尺寸
    if img1.shape != img2.shape:
        print("Error: Images have different dimensions.")
        return

    # 计算图像差
    difference = cv2.absdiff(img1, img2)

    # 显示图像差
    # cv2.imshow("Difference", difference)
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return difference

def multiply_image(input_image_path, multiplier):
    # 读取图像
    image = input_image_path

    # 将图像转换为浮点数，以避免整数溢出
    image = image.astype(np.float32)

    # 对图像的所有像素点值乘以给定的数
    multiplied_image = cv2.multiply(image, multiplier)

    # 将结果图像的数据类型转换回uint8
    multiplied_image = np.clip(multiplied_image, 0, 255).astype(np.uint8)

    cv2.imshow("Multiplied Image", multiplied_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# 设置输入图像路径
input_image1_path = "tiananmen_fenge.png"
input_image2_path = "tiananmen_fenge_jiawu_quwu.png"

# 调用subtract_images函数
difference = subtract_images(input_image1_path, input_image2_path)
multiply_image(difference,3)
