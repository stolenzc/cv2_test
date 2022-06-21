# # 寻找图像轮廓 返回修改后图像的轮廓  以及它们的层次
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 取中轴
import copy
import math
from typing import Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt

def fit_line2(lines: np.ndarray, pos_offset=80, angle_offset=15, thickness=2):
    """
    拟合直线
    Args:
        lines (list): 霍夫直线结果线段
        pos_offset (int): 线段允许偏移量
        angle_offset (float): 线段允许偏移角度
        thickness (int): 线宽
    """
    # 已归类的线段
    line_group: List[List] = []
    # 已归类的线段偏移值
    line_offset: List[tuple] = []
    # 最终线段
    result: List = []
    # 未能归类的线段
    line_unfind: List = []

    for line in lines:
        for x1, y1, x2, y2 in line:
            print(x1, y1, x2, y2, line)
            # 水平线段
            if x1 == x2:
                angle = 90
                a = 0
                b = x1
            elif y1 == y2:
                angle = 0
                a = 0
                b = y1
            else:
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1
                angle = abs(np.arctan(a) * 57.29577)
            print(angle)
            is_find_same_line = False
            for index, offset in enumerate(line_offset):
                if abs(angle - offset[2]) < angle_offset and abs(b - offset[1]) < pos_offset:
                    line_group[index].append(line)
                    is_find_same_line = True
                    break
            if not is_find_same_line:
                line_group.append([line])
                line_offset.append((a, b, angle))
    print('--------------')
    print(line_group[0])
    print(line_offset[0])

    print('--------------')
    for index, group_item in enumerate(line_group):
        min_x, max_x = 0, 0
        if len(group_item) < 10:
            continue
        a, b, angle = line_offset[index]
        min_y, max_y = 0, 0
        for line in group_item:
            for x1, y1, x2, y2 in line:
                if angle == 0:
                    if min_x == 0:
                        min_x = min(x1, x2)
                    else:
                        min_x = min(min_x, min(x1, x2))
                    max_x = max(max_x, max(x1, x2))
                    min_y = y1
                    max_y = y1
                elif angle == 90:
                    min_x = max_x = x1
                    if min_y == 0:
                        min_y = min(y1, y2)
                    else:
                        min_y = min(min_y, min(y1, y2))
                    max_y = max(max_y, max(y1, y2))
                else:
                    if min_x == 0:
                        min_x = min(x1, x2)
                    else:
                        min_x = min(min_x, min(x1, x2))
                    max_x = max(max_x, max(x1, x2))
                    max_y = int(a * max_x + b)
                    min_y = int(a * min_x + b)
        # a, b, angle = line_offset[index]
        # max_y = int(a * max_x + b)
        # min_y = int(a * min_x + b)
        result.append([(min_x, max_x), (min_y, max_y)])
        # cv2.line(result, (min_x, min_y), (max_x, max_y), thickness)
    print(result[0])
    return result


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

# minLineLength = 19
# maxLineGap = 5
minLineLength = 5
maxLineGap = 5

# 直线检测
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 20, np.array([]), minLineLength, maxLineGap)


# cv2.imwrite('./2_result.jpg', lines)

# print(type(lines))
# print(lines)
# lines = fit_line(img, lines)

# print('----------------')
# print(lines)
# print('----------------')

# 设置范围
plt.subplot(1, 2, 1)
plt.xlim(0, w)
plt.ylim(0, h)

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


# print(len(result), result)
# plt.show()

lines = fit_line2(lines)

plt.subplot(1, 2, 2)
for line_item in lines:
    plt.plot(line_item[0], line_item[1])
plt.xlim(0, w)
plt.ylim(0, h)
plt.savefig('1.jpg')
plt.show()


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




