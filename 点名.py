import cv2
import numpy as np
from skimage.segmentation import slic
from skimage.color import label2rgb

def superpixel_segmentation(image, num_segments=300):
    segments = slic(image, n_segments=num_segments, sigma=5)
    return label2rgb(segments, image, kind='avg')

def dark_channel_prior(image, window_size=15):
    min_channel = np.min(image, axis=2)
    dark_channel = cv2.erode(min_channel, np.ones((window_size, window_size)))
    return dark_channel

def estimate_atmospheric_light(image, dark_channel, top_percent=0.001):
    num_pixels = image.shape[0] * image.shape[1]
    num_top_pixels = int(num_pixels * top_percent)
    indices = np.argpartition(dark_channel.ravel(), -num_top_pixels)[-num_top_pixels:]
    top_pixels = image.reshape(-1, 3)[indices]
    return np.mean(top_pixels, axis=0)

def estimate_transmission(image, atmospheric_light, omega=0.95, window_size=15):
    transmission = 1 - omega * np.min((image / atmospheric_light), axis=2)
    transmission = cv2.erode(transmission, np.ones((window_size, window_size)))
    return transmission

def dehaze(image, transmission, atmospheric_light, t0=0.1):
    return np.clip((image - atmospheric_light) / np.maximum(transmission, t0) + atmospheric_light, 0, 255)

input_image_path = "tiananmen.png"
output_image_path = "path/to/your/dehazed_image.jpg"

image = cv2.imread(input_image_path)

# 应用超像素分割方法
superpixel_image = superpixel_segmentation(image)
superpixel_image = (superpixel_image * 255).astype(np.uint8)

# 应用暗通道去雾法
dark_channel = dark_channel_prior(superpixel_image)
atmospheric_light = estimate_atmospheric_light(superpixel_image, dark_channel)
transmission = estimate_transmission(superpixel_image, atmospheric_light)
dehazed_image = dehaze(superpixel_image, transmission, atmospheric_light)

# 保存去雾后的图像
cv2.imshow('1',dehazed_image)
# cv2.imwrite(output_image_path, dehazed_image)
