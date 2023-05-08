import cv2
import os
import numpy as np

def calculate_channel_means_stddev(image_path):
    # 读取图像
    image = cv2.imread(image_path)

    # 计算每个通道的平均值和标准差
    mean_channels, stddev_channels = cv2.meanStdDev(image)

    # 去除alpha通道（如果存在），并将结果转为一维数组
    mean_channels = mean_channels.ravel()[:3]
    stddev_channels = stddev_channels.ravel()[:3]

    # 计算所有通道的平均值
    mean_all_channels = np.mean(mean_channels)

    # 计算平均值
    mean = (stddev_channels[0] + stddev_channels[1] + stddev_channels[2]) / 3

    # 计算方差
    biaozhuncha = ((stddev_channels[0] - mean) ** 2 + (stddev_channels[1] - mean) ** 2 + (stddev_channels[2] - mean) ** 2) / 3


    return mean_channels, mean_all_channels, stddev_channels, biaozhuncha

def process_images_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.gif')):
            image_path = os.path.join(folder_path, filename)
            mean_channels, mean_all_channels, stddev_channels, biaozhuncha = calculate_channel_means_stddev(image_path)
            print(f"Image: {filename}")
            print(f"Mean values for each channel (B, G, R): {mean_channels}")
            print(f"Mean value for all channels: {mean_all_channels}")
            print(f"Standard deviation for each channel (B, G, R): {stddev_channels}")
            print(f"标准差的标准差: {biaozhuncha}\n")

input_folder_path = "3"

process_images_in_folder(input_folder_path)
