from tkinter import E
import numpy as np
import cv2
import sympy

Vt = np.array(
    [[-.42, -.57, -.70],
     [.81, .11, -.58],
     [.41, -.82, .41]]
)
print(Vt)
print(np.linalg.norm(Vt[2]))
print(np.transpose(Vt))
print(np.linalg.inv(Vt))

# 求导
t, v0 = sympy.symbols("t,v0")
y = v0*t-5*t**2
print(y)
y_d = sympy.diff(y, t)
print(y_d)


# 练习题2
# 二维高斯函数求偏导
x, y, x0, y0, theta = sympy.symbols("x,y,x0,y0,theta")
g = sympy.E**((-(x-x0)**2-(y-y0)**2)/(2*theta**2))
print(g)
g_dx = sympy.diff(g, x)
print(f"一阶导数：{g_dx}")
g_ddx = sympy.diff(g_dx, x)
print(f"二阶导数：{g_ddx}")
print(f"二阶导数值:{g_ddx.evalf(subs={x:x0})}")  # 二阶导数值<0，函数取得最大值
x_ret = sympy.solve(g_dx, x)
print(f"高斯函数在x={x_ret}时取得最大值")

# 代入x0
g1 = sympy.E**(-(y-y0)**2/(2*theta**2))
g1_dy = sympy.diff(g1, y)
print(f"一阶偏y导数:{g1_dy}")
y_ret = sympy.solve(g1_dy, y)
print(f"高斯函数在y={y_ret}时取得最大值")

# 练习题3
# 读取图片
dark = cv2.imread("/Users/liudongxu/Documents/cs131/HW0_out/u2dark.png", 0)
# mean
mu = dark.mean()
# 输出平均值
print(f"dark的平均值mu:{mu}")
# 最大值
max_val = dark.max()
print(f"dark的最大像素值max_val:{max_val}")
# 最小值
min_val = dark.min()
print(f"dark的最小像素值min_val:{min_val}")

# offset and scaling
dark = dark.astype(float)
dark = (dark-min_val)/(max_val-min_val)*255.0
cv2.imwrite("/Users/liudongxu/Documents/cs131/HW0_out/u2dark_1_res.png",
            dark.astype(np.uint8))

# contrast stretching
dark = (2*(dark-128)+128)
dark[dark > 255] = 255
dark[dark < 0] = 0
cv2.imwrite("/Users/liudongxu/Documents/cs131/HW0_out/u2dark_contrast_res.png",
            dark.astype(np.int8))
# 这种方法增强对比度会使像素值溢出；原图转为Float之后，只有现取值0到255所有float值都是有效的转换结果。

# edge detection
src = cv2.imread("/Users/liudongxu/Documents/cs131/HW0_out/buoys.jpg", 0)
src = src.astype(float)
edges = np.zeros(src.shape, np.float32)
# detect vertically edge
for i in range(src.shape[0]):
    for j in range(1, src.shape[1]):
        edges[i, j] = src[i, j]-src[i, j-1]
edges_to_show = abs(edges)
edges_to_show = edges_to_show.astype(np.uint8)
edges_to_show = cv2.normalize(
    edges_to_show, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imwrite(
    "/Users/liudongxu/Documents/cs131/HW0_out/vertical_edges_buoys.png", edges_to_show)
# 水中的细小边缘可能是由于水的反光，以及相机噪声导致的

# BoxBlur


def BoxBlur(edges: np.ndarray, ksize: int):
    # ksize must be odd
    assert ksize % 2 == 1
    # padding
    pad = ksize//2
    edges_pad = np.pad(edges, pad, mode="constant", constant_values=0)
    # blur
    edges_blur = np.zeros(edges.shape, np.float32)
    for i in range(edges.shape[0]):
        for j in range(edges.shape[1]):
            edges_blur[i, j] = edges_pad[i:i+ksize, j:j+ksize].mean()
    return edges_blur


large_edge = BoxBlur(edges, 3)
large_edge_to_show = abs(large_edge)
large_edge_to_show = large_edge_to_show.astype(np.uint8)
large_edge_to_show = cv2.normalize(
    large_edge_to_show, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)
cv2.imwrite("/Users/liudongxu/Documents/cs131/HW0_out/BoxBlur_vertical_edges_buoys.png",
            large_edge_to_show)
# 经过boxblur之后，水中大片的反光边被模糊了，一些细小的边或者轮廓被拓宽
