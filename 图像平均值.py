import cv2
import os

def calculate_channel_means(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算每个通道的平均值
    mean_channels = cv2.mean(image)[:3]  # 舍去alpha通道（如果存在）

    # 计算所有通道的平均值
    mean_all_channels = sum(mean_channels) / len(mean_channels)

    # 计算平均值
    mean = (mean_channels[0] + mean_channels[1] + mean_channels[2]) / 3

    # 计算方差
    biaozhuncha = ((mean_channels[0] - mean) ** 2 + (mean_channels[1] - mean) ** 2 + (mean_channels[2] - mean) ** 2) / 3



    return mean_channels, mean_all_channels, biaozhuncha

def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            mean_channels, mean_all_channels,biaozhuncha = calculate_channel_means(image_path)
            print(f"图片名称: {filename}")
            print(f"三通道值 (B, G, R): {mean_channels}")
            print(f"所有通道平均值: {mean_all_channels}")
            print(f"所有通道方差:{biaozhuncha}\n")

input_folder_path = "3"

process_images_in_folder(input_folder_path)
