# # 寻找图像轮廓 返回修改后图像的轮廓  以及它们的层次
# contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 取中轴
import copy
import math
from typing import Dict, List

import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def get_line_point(lines: list, offset: tuple) -> list:
    """
    获取线段两端点坐标
    Args:
        lines (list): 霍夫直线结果线段
        offset (tuple): 线段偏移比值，线段偏移值，线段偏移角度，当角度偏移为90度时，偏移值为x坐标
    Returns:
        list: 线段两端点坐标 [point1, point2]
    """
    k, b, angle = offset
    x1, y1, x2, y2 = lines[0]
    min_x, max_x = (x1, x2) if x1 < x2 else (x2, x1)
    min_y, max_y = (y1, y2) if y1 < y2 else (y2, y1)
    for line in lines[1:]:
        x1, y1, x2, y2 = line
        min_x = min(min_x, x1, x2)
        max_x = max(max_x, x1, x2)
        min_y = min(min_y, y1, y2)
        max_y = max(max_y, y1, y2)
    if angle not in (0, 90):
        min_y = min_x * k + b
        max_y = max_y * k + b
    return [(min_x, min_y), (max_x, max_y)]


def get_pos_length(point1: tuple, point2: tuple):
    """
    获取两点之间长度
    Args:
        point1 (tuple): 点1坐标
        point2 (tuple): 点2坐标
    Returns:
        float: 两点之间长度
    """
    line_length = math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
    return line_length


def sort_group(lines: list, angle: float):
    """
    对线段组排序
    Args:
        lines (list): 线段组
        angle (float): 线段偏移角度
    Returns:
        list: 排序后的线段组
    """
    if not lines:
        return lines
    if angle == 90:
        lines = list(sorted(lines, key=lambda x: x[0][1]))
    else:
        lines = list(sorted(lines, key=lambda x: x[0][0]))
    return lines


def split_group(start_x, start_y, end_x, end_y, line, result, gap_filter):
    """
    拆分同一线段组
    """
    pass


def fit_line(lines: List[np.ndarray], pos_offset=150, angle_offset=30, gap_filter=150, line_gap=100, line_lenth=100):
    """
    拟合直线
    Args:
        lines (list): 霍夫直线结果线段
        pos_offset (int): 线段分组允许偏移量
        angle_offset (float): 线段允许偏移角度
        gap_filter (int): 缝隙大小
        point_gap (int): 线与线之间的间隙
    """
    # 已归类的线段
    line_group: List[List] = []
    # 已归类的线段偏移值
    line_offset: List[tuple] = []
    # 最终线段
    result: List = []

    # 对线段进行归类
    for line in lines:
        for x1, y1, x2, y2 in line:
            if x1 == x2:
                # 水平线段
                angle, a, b = 90, 0, x1
            elif y1 == y2:
                # 垂直线段
                angle, a, b = 0, 0, y1
            else:
                # 倾斜线段
                a = (y2 - y1) / (x2 - x1)
                b = y1 - a * x1
                angle = np.arctan(a) * 57.29577
            is_find_same_line = False
            for index, offset in enumerate(line_offset):
                if abs(angle - offset[2]) < angle_offset and abs(b - offset[1]) < pos_offset:
                    line_group[index].append(line)
                    is_find_same_line = True
                    break
            if not is_find_same_line:
                line_group.append([line])
                line_offset.append((a, b, angle))

    print(line_group)

    # 各组线段求端值
    for index, group_item in enumerate(line_group):
        min_x, max_x = 0, 0
        a, b, angle = line_offset[index]
        min_y, max_y = 0, 0

        # 仅考虑x升序的情况
        group_item = sort_group(group_item, angle)
        start_x, start_y, end_x, end_y = group_item[0][0]
        # start_x, end_x = min(start_x, end_x), max(start_x, end_x)
        for item in group_item[1:]:
            x1, y1, x2, y2 = item[0]

            if angle == 0:
                new_min_x, new_max_x = (x1, x2) if x1 < x2 else (x2, x1)
                if new_min_x > end_x and abs(new_min_x - end_x) > gap_filter:
                    if get_pos_length((start_x, start_y), (end_x, end_y)) > line_lenth:
                        result.append(((start_x, end_x), (start_y, end_y)))
                    start_x, end_x = new_min_x, new_max_x
                else:
                    start_x = min(start_x, new_min_x)
                    end_x = max(end_x, new_max_x)

            elif angle == 90:
                new_min_y, new_max_y = (y1, y2) if y1 < y2 else (y2, y1)
                if new_min_y > end_y and abs(new_min_y - end_y) > gap_filter:
                    if get_pos_length((start_x, start_y), (end_x, end_y)) > line_lenth:
                        result.append(((start_x, end_x), (start_y, end_y)))
                    start_y, end_y = new_min_y, new_max_y
                else:
                    start_y = min(start_y, new_min_y)
                    end_y = max(end_y, new_max_y)

            else:
                new_min_x, new_min_y, new_max_x, new_max_y = (x1, y1, x2, y2) if x1 < x2 else (x2, y2, x1, y1)
                if new_min_x > end_x and get_pos_length((end_x, end_y), (new_min_x, new_min_y)) > line_gap:
                    if get_pos_length((start_x, start_y), (end_x, end_y)) > line_lenth:
                        result.append(((start_x, end_x), (start_y, end_y)))
                    start_x, start_y, end_x, end_y = new_min_x, new_min_y, new_max_x, new_max_y
                else:
                    start_x, start_y = (start_x, start_y) if start_x < new_min_x else (new_min_x, new_min_y)
                    start_y = a * start_x + b
                    end_x, end_y = (end_x, end_y) if end_x > new_max_x else (new_max_x, new_max_y)
                    end_y = a * end_x + b
        if get_pos_length((start_x, start_y), (end_x, end_y)) > line_lenth:
            result.append(((start_x, end_x), (start_y, end_y)))


        # # 旧版直线拟合方法
        # for line in group_item:
        #     for x1, y1, x2, y2 in line:
        #         if angle == 0:
        #             if min_x == 0:
        #                 min_x = min(x1, x2)
        #             else:
        #                 min_x = min(min_x, min(x1, x2))
        #             max_x = max(max_x, max(x1, x2))
        #             min_y = y1
        #             max_y = y1
        #         elif angle == 90:
        #             min_x = max_x = x1
        #             if min_y == 0:
        #                 min_y = min(y1, y2)
        #             else:
        #                 min_y = min(min_y, min(y1, y2))
        #             max_y = max(max_y, max(y1, y2))
        #         else:
        #             if min_x == 0:
        #                 min_x = min(x1, x2)
        #             else:
        #                 min_x = min(min_x, min(x1, x2))
        #             max_x = max(max_x, max(x1, x2))
        #             max_y = int(a * max_x + b)
        #             min_y = int(a * min_x + b)
        # if abs(max_x-min_x) < gap_filter and abs(max_y - min_y) < gap_filter:
        #     continue
        # result.append([(min_x, max_x), (min_y, max_y)])
        # # cv2.line(result, (min_x, min_y), (max_x, max_y), thickness)
    return result

file_name = '9'

img_dir = f"./{file_name}.jpg"
save_dir = f"./{file_name}_result.jpg"

img = cv2.imread(img_dir)
# print(img.shape)
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

plt.subplot(2, 2, 1)
plt_img = imread(img_dir)
plt.imshow(img)

# 设置范围
plt.subplot(2, 2, 2)
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

lines = fit_line(lines)

# 整理返回前端数据格式
import json
return_data = []
for line in lines:
    return_data.append([{'x': float(line[0][0]), 'y': float(line[1][0])}, {'x': float(line[0][1]), 'y': float(line[1][1])}])
print(json.dumps(return_data))

plt.subplot(2, 2, 4)
for line_item in lines:
    plt.plot(line_item[0], line_item[1])
plt.xlim(0, w)
plt.ylim(0, h)
plt.savefig(save_dir)
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




