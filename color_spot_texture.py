import cv2
import numpy as np

# 读取图像并转换为灰度图像
img = cv2.imread('image.jpg')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

from skimage.feature import greycomatrix, greycoprops

# 计算灰度共生矩阵
glcm = greycomatrix(gray, [5], [0], 256, symmetric=True, normed=True)

# 提取能量作为纹理特征
energy = greycoprops(glcm, 'energy')

from sklearn.cluster import KMeans

# 使用K-means对能量特征进行聚类
kmeans = KMeans(n_clusters=2, random_state=0).fit(energy.reshape(-1, 1))

# 获取聚类标签并将其重新调整为图像形状
labels = kmeans.labels_.reshape(gray.shape)

# 根据标签图像分割图像并绘制色斑轮廓
for i in range(2):
    mask = np.uint8(labels == i)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 绘制色斑轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < 50: # 忽略面积较小的轮廓
            continue
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)

# 显示结果
cv2.imshow('result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
