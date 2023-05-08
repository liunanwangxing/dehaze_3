import cv2
import numpy as np

def load_image(image_path):
    return cv2.imread(image_path,cv2.IMREAD_GRAYSCALE).astype(np.float32)

def find_linear_relationship(A1, B1, C1):
    A1_flat = A1.reshape(-1, 3)
    B1_flat = B1.reshape(-1, 3)
    C1_flat = C1.reshape(-1, 3)

    X = np.column_stack((A1_flat, B1_flat))
    Y = C1_flat

    coefficients = np.linalg.lstsq(X, Y, rcond=None)[0]
    a, b = coefficients[:3], coefficients[3:]

    return a, b

def apply_linear_relationship(A, B, a, b):
    return a * A + b * B

image_path_A1 = "tiananmen_fangcha.png"
image_path_B1 = "tiananmen_pingjun.png"
image_path_C1 = "tiananmen_t.png"

A1 = load_image(image_path_A1)
B1 = load_image(image_path_B1)
C1 = load_image(image_path_C1)

a, b = find_linear_relationship(A1, B1, C1)

# 现在可以将线性关系应用到其他图像上，例如A2和B2
image_path_A2 = "tiananmen_fangcha.png"
image_path_B2 = "tiananmen_pingjun.png"

A2 = load_image(image_path_A2)
B2 = load_image(image_path_B2)

C2 = apply_linear_relationship(A2, B2, a, b)

cv2.imshow("Generated C2", C2.astype(np.uint8))
cv2.waitKey(0)
cv2.destroyAllWindows()

# 输出线性关系系数
print("Linear relationship coefficients a and b:")
print(a)
print(b)
