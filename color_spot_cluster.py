import cv2
import numpy as np

# 定义聚类数量和阈值
K = 2
threshold = 1000

# 读入图像
img = cv2.imread('input_image.jpg')

# 将图像从BGR色彩空间转换为Lab色彩空间
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# 将Lab图像转换为一维数组
pixels = np.float32(lab.reshape(-1, 3))

# 运行K-Means聚类算法
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
_, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 将聚类中心转换为色斑颜色
blemish_color = np.uint8(centers[1])

# 创建色斑掩膜图像
blemish_mask = cv2.inRange(lab, blemish_color, blemish_color)

# 将输入图像与色斑掩膜图像进行按位与操作
blemish_only = cv2.bitwise_and(img, img, mask=blemish_mask)

# 查找色斑轮廓
gray = cv2.cvtColor(blemish_only, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# 绘制色斑轮廓
for contour in contours:
    # 计算轮廓面积
    area = cv2.contourArea(contour)
    
    # 如果面积超过阈值，则将其视为色斑，并在原图像上绘制
    # 如果面积超过阈值，则将其视为色斑，并在原图像上绘制
    if area > threshold:
        cv2.drawContours(img, contour, -1, (0, 0, 255), 3)

# 显示并保存图像
cv2.imshow('image', img)
cv2.imwrite('output_image.jpg', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
