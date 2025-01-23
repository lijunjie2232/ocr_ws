import cv2
import os
from pathlib import Path
import shutil
import numpy as np
from copy import deepcopy
from traceback import print_exc
from tqdm import tqdm
from pprint import pprint


def calc_contours(gray, cvt=cv2.COLOR_BGR2GRAY):
    """
    计算图像中的轮廓。

    参数:
    gray (numpy.array): 输入的图像。

    返回:
    list: 图像中检测到的轮廓列表。
    """
    # 将输入图像转换为灰度图像
    if len(gray.shape) == 3:
        gray = cv2.cvtColor(gray, cvt)
    # _, thresh = cv2.threshold(gray, 245, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(str(TMP_DIR / "thresh.jpg"), thresh)
    # # 添加形态学闭运算以连接细长且垂直的矩形
    # kernel_x = np.ones((3, 1), np.uint8)
    # kernel_y = np.ones((1, 3), np.uint8)
    # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel_x, iterations=2)
    # closed = cv2.morphologyEx(closed, cv2.MORPH_CLOSE, kernel_y, iterations=2)
    # cv2.imwrite(str(TMP_DIR / "closed.jpg"), closed)  # 保存闭运算后的图像

    # 高斯模糊去噪
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # cv2.imwrite(str(TMP_DIR / "blurred.jpg"), blurred)

    # 二值化处理
    _, binary = cv2.threshold(blurred, 245, 255, cv2.THRESH_BINARY_INV)
    # cv2.imwrite(str(TMP_DIR / "binary.jpg"), binary)

    # 添加开运算以去除噪声
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    opening = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)
    # cv2.imwrite(str(TMP_DIR / "opening.jpg"), opening)  # 保存开运算后的图像

    # 膨胀操作
    dilation = cv2.dilate(opening, kernel, iterations=2)

    # 腐蚀操作
    erosion = cv2.erode(dilation, kernel, iterations=2)
    # cv2.imwrite(str(TMP_DIR / "new_handled.jpg"), erosion)  # 保存闭运算后的图像

    # 查找轮廓
    contours, _ = cv2.findContours(erosion, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return contours, erosion


def expand_box(box, padding=5):
    """
    扩展边界框的大小。

    参数:
    box (tuple): 原始边界框坐标 (x, y, w, h)。
    padding (int): 边界框扩展的像素值，默认为5。

    返回:
    tuple: 扩展后的边界框坐标 (x, y, w, h)。
    """
    x, y, w, h = box
    x -= padding
    y -= padding
    w += 2 * padding
    h += 2 * padding
    return (x, y, w, h)


def calculate_min_area_rect(contour):
    """
    计算轮廓的最小外接矩形。

    参数:
    contour (numpy.array): 轮廓点的数组。

    返回:
    tuple: 最小外接矩形的顶点坐标 (box)、矩形参数 (rect) 和边界框坐标 (xywh)。
    """
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    x, y, w, h = cv2.boundingRect(contour)
    xywh = (x, y, w, h)
    return box, rect, xywh


def crop_box_region(img, box):
    """
    裁剪图像中的指定边界框区域。

    参数:
    img (numpy.array): 原始图像。
    box (tuple): 边界框坐标 (x, y, w, h)。

    返回:
    numpy.array: 裁剪后的图像区域。
    """
    x, y, w, h = box
    cropped_img = img[y : y + h, x : x + w].copy()
    return cropped_img


def crop_and_extract_main_color(img, top_height=None):
    """
    剪切图片顶部指定高度的区域并提取返回主要颜色。

    参数:
    img (numpy.array): 原始图像。
    top_height (int): 剪切的顶部高度，默认为50像素。

    返回:
    tuple: 主要颜色的RGB值。
    """
    if top_height:
        image = img[:top_height].copy()
    else:
        image = img
    # 判断图像是否为灰度图像
    if len(image.shape) == 2:
        # 灰度图像
        hist = cv2.calcHist([image], [0], None, [256], [0, 256])
        main_color_gray = np.argmax(hist)
        return np.array([main_color_gray], dtype=np.uint8)
    else:
        # 彩色图像
        return np.array(
            [
                crop_and_extract_main_color(image[:, :, i])
                for i in range(image.shape[2])
            ],
            dtype=np.uint8,
        ).reshape(-1)


def compare_colors(color1, color2, threshold=30):
    """
    比较两个颜色是否相同。

    参数:
    color1 (tuple): 第一个颜色的RGB值。
    color2 (tuple): 第二个颜色的RGB值。
    threshold (int): 颜色比较的阈值，默认为30。

    返回:
    bool: 如果颜色相同返回True，否则返回False。
    """
    distance = np.linalg.norm(np.array(color1) - np.array(color2))
    return distance < threshold


def color_index(color_list, color, threshold=30):
    """
    将颜色添加到列表中，如果颜色不同则插入新颜色。

    参数:
    color_list (list): 颜色列表。
    color (tuple): 待比较的颜色。
    threshold (int): 颜色比较的阈值，默认为30。

    返回:
    int: 相同颜色的索引。
    """
    for index, existing_color in enumerate(color_list):
        if compare_colors(existing_color, color, threshold):
            return index
    return -1


def calc_rect_area(contours, draw_img=None, draw=False):
    # 筛选出符合固定高度但宽度可变的非矩形外框
    fixed_height = 80  # 假设固定高度为30像素
    tolerance = 20  # 容差范围

    filtered_contours = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if abs(h - fixed_height) <= tolerance:
            filtered_contours.append(contour)

    # 使用近似多边形来检测非矩形外框
    detected_boxes = []
    for contour in filtered_contours:
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)

        # 检查近似多边形是否为非矩形
        if len(approx) == 4:
            x, y, w, h = cv2.boundingRect(contour)
            _, _, xywh = calculate_min_area_rect(contour)
            detected_boxes.append((*xywh, approx))

    # # 绘制检测到的非矩形外框
    box_image = draw_img.copy()
    for box in detected_boxes:
        x, y, w, h, approx = box
        cv2.rectangle(box_image, (x, y), (x + w, y + h), (0, 0, 255), 2)

    return detected_boxes


def box_classification(img, x, y, w, h, box_type_threshold=150 * 3):
    crop_img = crop_box_region(img, (x, y, w, h))
    box_main_color = crop_and_extract_main_color(crop_img)
    # print(f"box主要颜色的RGB值: {box_main_color}")
    # print(box_main_color.sum() >= box_type_threshold)
    return box_main_color.sum() >= box_type_threshold * sum(box_main_color.shape)


def img_handling(image, output_dir):

    contours, handled_pic = calc_contours(image)
    detected_boxes = calc_rect_area(contours, draw_img=image, draw=True)

    #     # 调用新封装的函数
    main_color = crop_and_extract_main_color(image, 50)
    # print(f"主要颜色的RGB值: {main_color}")

    # 将颜色添加到列表中，并获取相同颜色的索引
    theme = color_index(THEMES, main_color)
    if theme == -1:
        THEMES.append(main_color)
        theme += len(THEMES)
        THEME_COUNT[str(theme)] = 0
    # print(f"颜色索引: {theme}")

    # 腐蚀操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    erosion = cv2.erode(handled_pic, kernel, iterations=2)

    # cv2.imwrite(str(output_dir / "box_eroded_image.jpg"), erosion)
    for box in detected_boxes:
        try:
            box_type = int(
                box_classification(erosion, *box[:4], box_type_threshold=150)
            )
            area_a = crop_box_region(image, box[:4])
            box_b = [*box[:4]]
            box_b_shape = BOX_SHAPE[box_type]
            box_b[1] += box_b[3] + box_b_shape[0]
            box_b[3] = box_b_shape[1]
            area_b = crop_box_region(image, box_b)
            cv2.imwrite(
                str(
                    output_dir
                    / (BOX_NAME_PATTERN % (theme, THEME_COUNT[str(theme)], "a"))
                ),
                area_a,
            )
            cv2.imwrite(
                str(
                    output_dir
                    / (BOX_NAME_PATTERN % (theme, THEME_COUNT[str(theme)], "b"))
                ),
                area_b,
            )
            THEME_COUNT[str(theme)] += 1
        except Exception as e:
            print_exc()
            raise (e)


if __name__ == "__main__":
    ROOT = Path(__file__).resolve().parent
    IN_DIR = ROOT / "zn_in"
    OUT_DIR = ROOT / "zn_out"
    TMP_DIR = ROOT / "zn_tmp"

    shutil.rmtree(TMP_DIR, ignore_errors=True)
    shutil.rmtree(OUT_DIR, ignore_errors=True)
    TMP_DIR.mkdir(exist_ok=True, parents=True)
    OUT_DIR.mkdir(exist_ok=True, parents=True)
    THEMES = []
    THEME_COUNT = {}
    EXP_COUNT = 0

    # box_type:
    # 0: offset=4, height=85
    # 1: offset=4, height=90
    BOX_SHAPE = {
        0: (4, 85),
        1: (4, 90),
    }
    BOX_NAME_PATTERN = (
        "%s_%0.4d_%s.png"  # "{THEME}_{THEME_COUNT[str(theme)]}_{[a|b]}.png"
    )

    loop = tqdm(sorted(os.listdir(IN_DIR)))
    for img in loop:
        # image = cv2.imread(str(IN_DIR / "i-034.jpg"))
        image = cv2.imread(str(IN_DIR / img))
        try:
            img_handling(image, OUT_DIR)
        except Exception as e:
            print_exc()
            cv2.imwrite(str(TMP_DIR / img), image)
            EXP_COUNT += 1
        finally:
            loop.set_postfix(
                {
                    "processed": img,
                    "错误图片数量": EXP_COUNT,
                    **THEME_COUNT,
                }
            )
    pprint("处理完成，错误图片数量: %d" % EXP_COUNT)
    pprint(THEME_COUNT)
