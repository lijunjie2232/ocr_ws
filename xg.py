import fitz  # PyMuPDF
import cv2
import numpy as np
from tqdm import tqdm
from pathlib import Path
from copy import deepcopy
from OCREngine import OCREngine
import traceback


class Word:
    def __init__(self, meaning: str, kanji: str, hina: str = None):
        self.meaning = meaning.strip() if meaning and isinstance(meaning, str) else ""
        self.hina = hina.strip() if hina and isinstance(hina, str) else ""
        self.kanji = kanji.strip() if kanji and isinstance(kanji, str) else ""

    def __str__(self):
        return f"(meaning: {self.meaning}, kanji: {self.kanji}, hina: {self.hina})"

    def __repr__(self):
        return self.__str__()


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


def img_crop(image, left, top, right, bottom):
    if left < 0:
        left = 0
    if top < 0:
        top = 0
    if right > image.shape[1]:
        right = image.shape[1]
    if bottom > image.shape[0]:
        bottom = image.shape[0]
    return deepcopy(image[top:bottom, left:right])


def getUnkownedName():
    with unknownLock:
        global unknownCount
        name = unknownPattern % unknownCount
        unknownCount += 1
        return name


def box_ocr(img):
    with zhEngineLock:
        return zhEngine(img, det=True, rec=False, cls=False)[0]


def ja_ocr(img):
    with jaEngineLock:
        return jaEngine(img, det=False, rec=True, cls=False)[0]


def zh_ocr(img):
    with zhEngineLock:
        return zhEngine(img, det=False, rec=True, cls=False)[0]


def cht_ocr(img):
    with chtEngineLock:
        return chtEngine(img, det=False, rec=True, cls=False)[0]


def box_parse(box, start_y):
    left, top, right, bottom = (
        int(box[0][0]),
        int(box[0][1]),
        int(box[2][0]),
        int(box[2][1]),
    )
    if top - bottom > 90:
        bottom = top + 90
    box_ = [(left, top), (right, bottom)]

    txt_top = start_y + box_[0][1]
    txt_bottom = start_y + box_[1][1]
    l_txt_left = LREIGON_LEFT + box_[0][0]
    l_txt_right = LREIGON_LEFT + box_[1][0] + 5
    return txt_top, txt_bottom, l_txt_left, l_txt_right


def find_right_boundary(image):
    """Find the right boundary of black text on a white background.

    Args:
        image (np.ndarray): Input image.

    Returns:
        int: The x-coordinate of the right boundary of the black text.
    """
    # 將圖像從 BGR 轉換到 HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # 定義黑色的 HSV 範圍
    lower_black = np.array([0, 0, 0])
    upper_black = np.array([180, 255, 50])

    # 定義橘黃色的 HSV 範圍
    lower_orange = np.array([10, 100, 100])
    upper_orange = np.array([25, 255, 255])

    # 創建黑色和橘黃色的掩碼
    mask_black = cv2.inRange(hsv, lower_black, upper_black)
    mask_orange = cv2.inRange(hsv, lower_orange, upper_orange)

    # 合併掩碼
    mask = cv2.bitwise_or(mask_black, mask_orange)

    # 找到所有的非零點（黑色字體）
    points = np.column_stack(np.where(mask > 0))

    # 找到這些點的最大 x 值，即右邊界
    right_boundary = points[:, 1].max()

    # 找到這些點的最小 x 值，即左邊界
    left_boundary = points[:, 1].min()

    return left_boundary, right_boundary


def txtboxing(img):
    try:
        left, right = find_right_boundary(img)
        left -= 5
        right += 5
        if left < 0:
            left = 0
        if right > img.shape[1]:
            right = img.shape[1]
        return img[:, left:right]
    except Exception as e:
        traceback.print_exc()
        return img


def lineProcessor(lregion, mregion, rregion):

    def txt_valid(txt, threshold=0.90):
        if len(txt) > 1:
            txt = [
                " ".join([(i[0]).strip() for i in txt]),
                min([i[1] for i in txt]),
            ]
        return txt[0][0] if txt[0][1] > threshold else None

    l_txt_region = txtboxing(lregion)
    m_txt_region = txtboxing(mregion)
    r_txt_region = txtboxing(rregion)

    l_txt_cht = cht_ocr(l_txt_region)
    l_txt_zh = zh_ocr(l_txt_region)
    l_txt_ja = ja_ocr(l_txt_region)
    l_txt = l_txt_cht
    m_txt = ja_ocr(m_txt_region)
    r_txt = ja_ocr(r_txt_region)

    try:
        l_txt = txt_valid(l_txt_cht)
        if l_txt is None:
            l_txt = txt_valid(l_txt_ja)
        if l_txt is None:
            l_txt = txt_valid(l_txt_zh)
        m_txt = txt_valid(m_txt)
        r_txt = txt_valid(r_txt)

        assert l_txt is not None
        assert m_txt is not None
        assert r_txt is not None

        wordList.append(Word(l_txt, m_txt, r_txt))

    except Exception as e:
        traceback.print_exc()
        name = getUnkownedName()
        # cv2.imwrite(OUTDIR / f"{name}_l.png", lregion)
        # cv2.imwrite(OUTDIR / f"{name}_m.png", mregion)
        # cv2.imwrite(OUTDIR / f"{name}_r.png", rregion)
        cv2.imwrite(OUTDIR / f"{name}_lt.png", l_txt_region)
        cv2.imwrite(OUTDIR / f"{name}_mt.png", m_txt_region)
        cv2.imwrite(OUTDIR / f"{name}_rt.png", r_txt_region)


def extract_area(page):
    img = cv2.imread(INDIR / page)
    assert img is not None, f"image {page} not found"
    header = match_template(img, headerTemplate)
    divider = match_template(img, dividerTemplate)
    kaiwa = match_template(img, kaiwaTemplate)
    if divider is not None:
        if header is None:
            # divider front area
            start_y = LREIGON_TOP
            end_y = divider[1]
            lregion_img = img_crop(img, LREIGON_LEFT, start_y, LREIGON_RIGHT, end_y)
            cv2.imwrite(TMPDIR / "lregine.png", lregion_img)
            boxes = box_ocr(lregion_img)
            if boxes is None:
                boxes = []
            for box in boxes:
                txt_top, txt_bottom, l_txt_left, l_txt_right = box_parse(box, start_y)
                l_txt_region = img_crop(
                    img, l_txt_left, txt_top, l_txt_right, txt_bottom
                )
                m_txt_region = img_crop(
                    img, l_txt_right, txt_top, RREIGON_LEFT, txt_bottom
                )
                r_txt_region = img_crop(
                    img, RREIGON_LEFT, txt_top, RREIGON_RIGHT, txt_bottom
                )
                lineProcessor(l_txt_region, m_txt_region, r_txt_region)

        # divider bottom area
        assert divider is not None, "divider not found"
        # get y of start
        start_y = divider[1] + dividerTemplate.shape[0] + 5
        end_y = LREIGON_BOTTOM
        lregion_img = img_crop(img, LREIGON_LEFT, start_y, LREIGON_RIGHT, end_y)
        cv2.imwrite(TMPDIR / "lregine.png", lregion_img)
        boxes = box_ocr(lregion_img)
        for box in boxes:
            txt_top, txt_bottom, l_txt_left, l_txt_right = box_parse(box, start_y)
            l_txt_region = img_crop(img, l_txt_left, txt_top, l_txt_right, txt_bottom)
            m_txt_region = img_crop(img, l_txt_right, txt_top, RREIGON_LEFT, txt_bottom)
            r_txt_region = img_crop(
                img, RREIGON_LEFT, txt_top, RREIGON_RIGHT, txt_bottom
            )
            lineProcessor(l_txt_region, m_txt_region, r_txt_region)

    elif kaiwa is not None:
        # ignore the bottom part of kaiwa
        start_y = LREIGON_TOP
        end_y = kaiwa[1]
        lregion_img = img_crop(img, LREIGON_LEFT, start_y, LREIGON_RIGHT, end_y)
        cv2.imwrite(TMPDIR / "lregine.png", lregion_img)
        boxes = box_ocr(lregion_img)
        for box in boxes:
            txt_top, txt_bottom, l_txt_left, l_txt_right = box_parse(box, start_y)
            l_txt_region = img_crop(img, l_txt_left, txt_top, l_txt_right, txt_bottom)
            m_txt_region = img_crop(img, l_txt_right, txt_top, RREIGON_LEFT, txt_bottom)
            r_txt_region = img_crop(
                img, RREIGON_LEFT, txt_top, RREIGON_RIGHT, txt_bottom
            )
            lineProcessor(l_txt_region, m_txt_region, r_txt_region)
    return page


kanatukenugu = lambda area: area[-43:]

if __name__ == "__main__":
    import shutil
    import os
    import pickle
    from multiprocessing import Lock
    from multiprocessing.pool import ThreadPool

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    ROOT = Path(__file__).resolve().parent
    TEMPLATEDIR = ROOT / "xg_template"
    INDIR = ROOT / "xg_in"

    TMPDIR = ROOT / "xg_tmp"
    # delete file in TMPDIR
    # shutil.rmtree(TMPDIR, ignore_errors=True)
    TMPDIR.mkdir(exist_ok=True, parents=True)

    OUTDIR = ROOT / "xg_out"
    shutil.rmtree(OUTDIR, ignore_errors=True)
    OUTDIR.mkdir(exist_ok=True, parents=True)

    headerTemplate = cv2.imread(TEMPLATEDIR / "header.png")  # (167, 582)
    dividerTemplate = cv2.imread(TEMPLATEDIR / "divider.png")  # (96, 59)
    kaiwaTemplate = cv2.imread(TEMPLATEDIR / "kaiwa.png")  # (124, 217)

    zhEngine = OCREngine("ch")
    zhEngineLock = Lock()
    chtEngine = OCREngine("chinese_cht")
    chtEngineLock = Lock()
    jaEngine = OCREngine("japan")
    jaEngineLock = Lock()
    unknownPattern = "unknown_%0.8d"
    unknownCount = 0
    unknownLock = Lock()
    wordList = []

    # 158 261 365 1463(77*19)
    LREIGON_LEFT = 158
    LREIGON_TOP = 261
    LREIGON_RIGHT = 158 + 365
    LREIGON_BOTTOM = 261 + 1463
    MREIGON_WIDTH = 458
    RREIGON_LEFT = 980
    RREIGON_RIGHT = 1390

    # page = "i-014.jpg"  # header test
    # page = "i-016.jpg"  # divider test
    # page = "i-017.jpg" # kaiwa test
    # extract_area(page)

    with ThreadPool(6) as pool:
        result = pool.imap(extract_area, os.listdir(INDIR))
        loop = tqdm(
            result, total=len(os.listdir(INDIR)), desc="processing", leave=False
        )
        for _ in loop:
            loop.set_postfix_str(f"{_}")


    # use pickle to store wordList
    with open(OUTDIR / "data.pickle", "wb") as f:
        pickle.dump(wordList, f)

    print(f"处理完成，结果保存为：{TMPDIR}")
