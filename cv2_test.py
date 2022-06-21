# # 寻找图像轮廓 返回修改后图像的轮廓  以及它们的层次
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 取中轴
import copy
import math

import cv2
import numpy as np
import matplotlib.pyplot as plt


def fit_line(img, lines, color=(0, 255, 0), thickness=2):
    """
    img (np.array): 原始图像
    lines (list): 霍夫直线结果线段
    thickness (int): 线宽度
    """
    line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)

    right_x = []
    right_y = []
    left_x = []
    left_y = []
    for line in lines:
        for x1, y1, x2, y2 in line:
            slope = ((y2-y1)/(x2-x1))
            if slope <= -0.2:
                left_x.extend((x1, x2))
                left_y.extend((y1, y2))

            elif slope >= 0.2:
                right_x.extend((x1, x2))
                right_y.extend((y1, y2))

    if left_x and left_y:
        left_fit = np.polyfit(left_x, left_y, 1)
        left_line = np.poly1d(left_fit)

        # if not self.x1L:
        x1L = int(img.shape[1] * 0.1)

        y1L = int(left_line(x1L))

        # if not self.x2L:
        x2L = int(img.shape[1] * 0.4)

        y2L = int(left_line(x2L))
        cv2.line(line_img, (x1L, y1L), (x2L, y2L), color, thickness)

    if right_x and right_y:
        right_fit = np.polyfit(right_x, right_y, 1)
        right_line = np.poly1d(right_fit)

        # if not self.x1R:
        x1R = int(img.shape[1] * 0.6)

        y1R = int(right_line(x1R))

        # if not self.x2R:
        x2R = int(img.shape[1] * 0.9)

        y2R = int(right_line(x2R))
        cv2.line(line_img, (x1R, y1R), (x2R, y2R), color, thickness)

    return line_img


img = cv2.imread('./2.jpg')
print(img.shape)
h, w, _ = img.shape
# # 转换为灰度图片
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# 应用直方图均衡化
# gray = cv2.equalizeHist(gray)
# _, binary = cv2.threshold(img,  127, 255, 0)
# 高斯模糊去噪
gray = cv2.GaussianBlur(gray, (9, 9), 0)
# 中值滤波
gray = cv2.medianBlur(gray, ksize=3)
# _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
# binary[binary == 255] = 1
# skel, distance = morphology.medial_axis(binary, return_distance=True)
# dist_on_skel = distance * skel
# gray = dist_on_skel.astype(np.uint8) * 255
# 边缘检测
edges = cv2.Canny(gray, 50, 100)

print(type(edges))

# minLineLength = 19
# maxLineGap = 5
minLineLength = 5
maxLineGap = 5

# 直线检测
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, np.array([]), minLineLength, maxLineGap)

# print(type(lines))
# print(lines)
# lines = fit_line(img, lines)

# print('----------------')
# print(lines)
# print('----------------')

def line(y1, x1, y2, x2):
    x = [x1, x2]
    y = [y1, y2]
    plt.plot(x, y)

result = []
for i in range(len(lines)):
    for x1, y1, x2, y2 in lines[i]:
        line(y1, x1, y2, x2)
        result.append([(x1, y1), (x2, y2)])
        cv2.line(img, (x1, y1), (x2, y2), (i * 20, 100 + i * 20, 255), 1)


print(len(result), result)
# 设置范围
plt.xlim(0, w)
plt.ylim(0, h)
plt.show()
plt.savefig('1.jpg')


def cosine_similarity(x, y, norm=False):
    """ 计算两个向量x和y的余弦相似度 """
    assert len(x) == len(y), "len(x) != len(y)"
    zero_list = [0] * len(x)
    if x == zero_list or y == zero_list:
        return float(1) if x == y else float(0)

    # method 1
    res = np.array([[x[i] * y[i], x[i] * x[i], y[i] * y[i]] for i in range(len(x))])
    print(res, 11, type(res), res[:, -1], sum(res[:, -1]), type(res[:, -1]))
    cos = sum(res[:, 0]) / (np.sqrt(sum(res[:, 1])) * np.sqrt(sum(res[:, 2])))
    return 0.5 * cos + 0.5 if norm else cos  # 归一化到[0, 1]区间内

# for x in copy.copy(result):
#     loc = np.array(x)
#     output = cv2.fitLine(loc, cv2.DIST_L2, 0, 0.01, 0.01)
#     line_angle = math.atan2(output[1], output[0])
#     line_angle_degrees = math.degrees(line_angle)
#     line_angle_degrees_data = round(line_angle_degrees, 1)
#     if line_angle_degrees_data not in (0, 90):
#         result.remove(x)




