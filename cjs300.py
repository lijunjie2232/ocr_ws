import fitz  # PyMuPDF
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from OCREngine import OCREngine


# 第一步：将 PDF 转换为图片
def pdf_to_image(
    pdf_path,
    page=-1,
    outDir=None,
    dpi=300,
):
    pdf_document = fitz.open(pdf_path)
    pageList = {}
    namePattern = "page_%0" + str(np.ceil(np.log10(len(pdf_document)))) + "d.png"
    if page <= 0:  # return all pages
        for idx, page in enumerate(tqdm(pdf_document, desc="pdf process", leave=False)):
            pageList[idx] = page.get_pixmap(dpi=dpi)
    else:
        idx = page - 1
        assert isinstance(page, int) and page < len(
            pdf_document
        ), f"Invalid page number, specified page should between 1 and {len(pdf_document)}"
        page = pdf_document[idx]  # 获取第一页
        pageList[idx] = page.get_pixmap(dpi=dpi)
    if outDir is not None:
        for idx, page in tqdm(pageList.items(), desc="save image file", leave=False):
            image_path = outDir / (namePattern % (idx + 1))
            page.save(image_path)
    return pageList


def extract_target_region(image, templates, maxWitdh=-1, cutKana=False):
    """
    从图像中提取模板匹配后目标区域，右边界根据目标区域内的蓝色和黑色像素分隔。

    参数:
    - image_path: 目标图像的文件
    - templates: 模板图像的文件

    返回:
    - 目标区域图像（裁剪后的图像）
    """
    # 转为灰度图像
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    cropList = []

    for template in tqdm(templates, desc="matching", leave=False):

        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

        # 执行模板匹配
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)

        # 获取匹配结果的最大值和位置
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

        # 获取模板的宽度和高度
        template_height, template_width = template.shape[:2]

        # 确定目标区域的左上角坐标
        left, top = np.array(max_loc)
        left += template_width
        bottom = top + template_height
        right = left + maxWitdh if maxWitdh > 0 else image.shape[1]

        # 提取目标区域图像
        target_region = image[top:bottom, left:right]

        # 转换为HSV空间，便于分离颜色
        hsv_region = cv2.cvtColor(target_region, cv2.COLOR_BGR2HSV)

        # 设定目标区域内蓝色文字的颜色范围
        lower_blue = np.array([90, 50, 100])  # 调整下限
        upper_blue = np.array([130, 255, 255])  # 调整上限

        # 创建蓝色的掩模
        blue_mask = cv2.inRange(hsv_region, lower_blue, upper_blue)

        # 创建黑色的掩模
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([50, 50, 50])
        black_mask = cv2.inRange(hsv_region, lower_black, upper_black)

        # cv2.imwrite("blue.png", blue_mask)
        # cv2.imwrite("black.png", black_mask)
        # cv2.imwrite("target.png", target_region)

        # 查找左边最右的蓝色像素
        leftmost_blue = None
        for x in range(blue_mask.shape[1]):
            if np.sum(blue_mask[:, x]) > 0:  # 如果该列包含蓝色
                leftmost_blue = x

        # 查找右边最左的黑色像素
        rightmost_black = None
        for x in range(blue_mask.shape[1]):  # 从右往左扫描
            if np.sum(black_mask[:, x]) > 0:  # 如果该列包含黑色
                rightmost_black = x
                break

        # 如果找到左边最右的蓝色像素和右边最左的黑色像素，计算中点作为右边界
        if leftmost_blue is not None and rightmost_black is not None:
            # 计算等距平分的位置
            right_boundary = int((leftmost_blue + rightmost_black) / 2)

            # 提取目标区域图片
            cropped_image = target_region[:, :right_boundary]  # 裁剪到右边界

            cropList.append(cropped_image)  # 返回裁剪后的目标区域图像
        else:
            # whole line is blue font
            cropList.append(target_region)
    return deepcopy(cropList)


def match_template(image, template):
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(result)
    if max_val > 0.8:
        return max_loc
    return None


def get_area(image, left_top_template, right_top_template, bottom_template):
    """Summary
    divide image into 3 areas: left_top, right_top, bottom by template matching;
    left_top area is between left_top_template, width is the same as left_top_template;
    right_top area is between right_top_template and upper border of bottom_template, width is the same as right_top_template;
    bottom area is below lower border of bottom_template, width is the same as left_top_template;

    Args:
        image (_type_): image to be divided
        left_top_template (_type_): left top template
        right_top_template (_type_): right top template
        bottom_template (_type_): bottom template

    Returns:
        list of 3 areas box (x, y, w, h)
    """

    left_top_loc = match_template(image, left_top_template)
    right_top_loc = match_template(image, right_top_template)
    bottom_loc = match_template(image, bottom_template)

    left_top_x, left_top_y = left_top_loc
    right_top_x, right_top_y = right_top_loc
    bottom_x, bottom_y = bottom_loc

    left_top_w, left_top_h = left_top_template.shape[1], left_top_template.shape[0]
    right_top_w, right_top_h = right_top_template.shape[1], right_top_template.shape[0]
    bottom_w, bottom_h = bottom_template.shape[1], bottom_template.shape[0]

    left_top_area = (
        left_top_x,
        left_top_y + left_top_h,
        left_top_w,
        bottom_y - (left_top_y + left_top_h),
    )
    right_top_area = (
        right_top_x,
        right_top_y + right_top_h,
        right_top_w,
        bottom_y - (right_top_y + right_top_h),
    )
    bottom_area = (
        bottom_x,
        bottom_y + bottom_h,
        bottom_w,
        image.shape[0] - (bottom_y + bottom_h) - 130,
    )

    # return [left_top_area, right_top_area, bottom_area]

    left_top_img = image[
        left_top_area[1] : left_top_area[1] + left_top_area[3],
        left_top_area[0] : left_top_area[0] + left_top_area[2],
    ]
    right_top_img = image[
        right_top_area[1] : right_top_area[1] + right_top_area[3],
        right_top_area[0] : right_top_area[0] + right_top_area[2],
    ]
    bottom_img = image[
        bottom_area[1] : bottom_area[1] + bottom_area[3],
        bottom_area[0] : bottom_area[0] + bottom_area[2],
    ]

    return deepcopy([left_top_img, right_top_img, bottom_img])


kanatukenugu = lambda area: area[-43:]

if __name__ == "__main__":
    import shutil

    ROOT = Path(__file__).resolve().parent
    TEMPLATEDIR = ROOT / "cjs300_template"
    DPI = 300
    pdf_path = ROOT / "cjs300.pdf"

    TMPDIR = ROOT / "cjs300_tmp"
    # delete file in TMPDIR
    shutil.rmtree(TMPDIR, ignore_errors=True)
    TMPDIR.mkdir(exist_ok=True, parents=True)

    OUTDIR = ROOT / "cjs300_out"
    shutil.rmtree(OUTDIR, ignore_errors=True)
    OUTDIR.mkdir(exist_ok=True, parents=True)
    

    engine = OCREngine("japan")
    mainWordList = []
    additionalWordList = []
    WordList = []
    notSureCount = 0
    notSurePattern = "ns_%016d.png"
    notSureCount_a = 0
    notSurePattern_a = "nsa_%016d.png"

    page = -1
    image_path = pdf_to_image(
        pdf_path, page=page, outDir=TMPDIR, dpi=DPI
    )  # 将 PDF 转换为图片

    # 加载模板
    wordTemplates = []
    templateNamePattern = "order_%d.png"

    for i in tqdm(range(1, 16), desc="load template", leave=False):
        wordTemplates.append(
            cv2.imread(TEMPLATEDIR / (templateNamePattern % i))
        )  # 模板图像

    # 加载图像 -> ocr
    imgSuffix = [".png", ".jpg", ".jpeg", ".bmp"]
    for img in tqdm(list(TMPDIR.iterdir()), desc="ocr", leave=True):
        if img.suffix.lower() not in imgSuffix:
            continue
        image = cv2.imread(img)  # 目标图像

        left_top_template = cv2.imread(TEMPLATEDIR / "word.png")
        right_top_template = cv2.imread(TEMPLATEDIR / "conver.png")
        bottom_template = cv2.imread(TEMPLATEDIR / "addon.png")

        [left_top_img, right_top_img, bottom_img] = get_area(
            image, left_top_template, right_top_template, bottom_template
        )

        try:
            wordAreas = extract_target_region(
                left_top_img,
                wordTemplates,
                cutKana=False,
            )
            wordAreas = [kanatukenugu(i) for i in wordAreas]
            results = engine(wordAreas)[0]
            for idx, (result, score) in enumerate(results):
                if score < 0.8:
                    area = cv2.cvtColor(wordAreas[idx], cv2.COLOR_BGR2GRAY)
                    _, area = cv2.threshold(area, 200, 255, cv2.THRESH_BINARY)
                    area = cv2.cvtColor(area, cv2.COLOR_GRAY2BGR)

                    cv2.imwrite(OUTDIR / (notSurePattern%notSureCount), area)
                    notSureCount += 1
                else:
                    mainWordList.append(result)
                    WordList.append(result)
        except:
            import traceback
            traceback.print_exc()

        try:
            addonAreas = extract_target_region(
                bottom_img,
                wordTemplates[:10],
                maxWitdh=705,
                cutKana=False
            )
            
            addonAreas = [kanatukenugu(i) for i in addonAreas]

            results = engine(addonAreas)[0]
            for idx, (result, score) in enumerate(results):
                if score < 0.8:
                    area = cv2.cvtColor(addonAreas[idx], cv2.COLOR_BGR2GRAY)
                    _, area = cv2.threshold(area, 200, 255, cv2.THRESH_BINARY)
                    area = cv2.cvtColor(area, cv2.COLOR_GRAY2BGR)
                
                    cv2.imwrite(OUTDIR / (notSurePattern_a%notSureCount_a), area)
                    notSureCount_a += 1
                else:
                    additionalWordList.append(result)
                    WordList.append(result)
        except:
            import traceback
            traceback.print_exc()
        # pass
    
    # write world list to txt file
    with open(OUTDIR / "mainWordList.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(mainWordList))
    with open(OUTDIR / "additionalWordList.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(additionalWordList))
    with open(OUTDIR / "WordList.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(WordList))
    
    
    print(f"处理完成，结果保存为：{TMPDIR}")
