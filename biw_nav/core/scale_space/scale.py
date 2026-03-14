import cv2
import numpy as np
import matplotlib.pyplot as plt

def gaussian_scale_space(image, num_scales=5, initial_sigma=1.0, scale_factor=2.0):
    scales = []
    sigma = initial_sigma
    
    for i in range(num_scales):
        # 高斯平滑
        smoothed = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma, sigmaY=sigma)
        scales.append(smoothed)
        sigma *= scale_factor  # 增大尺度

    return scales

# 加载图像并转换为灰度
image = cv2.imread('/home/zhen/cogMap/big_map/it_en.jpg', cv2.IMREAD_GRAYSCALE)

# 构建高斯尺度空间
scales = gaussian_scale_space(image)

# 可视化
plt.figure(figsize=(10, 5))
for i, scale in enumerate(scales):
    plt.subplot(1, len(scales), i + 1)
    plt.imshow(scale, cmap='gray')
    plt.title(f'Scale {i + 1}')
    plt.axis('off')
plt.tight_layout()
plt.show()
