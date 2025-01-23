# anki 批量制卡
import genanki
import os
from pathlib import Path
from tqdm import tqdm

if __name__ == "__main__":
    # 创建一个模型
    model_id = 1607392319
    model = genanki.Model(
        model_id,
        "Picture Card",
        fields=[
            {"name": "FrontImage"},
            {"name": "BackImage"},
        ],
        templates=[
            {
                "name": "Card 1",
                "qfmt": "{{FrontImage}}",
                "afmt": "{{BackImage}}",
            },
        ],
    )

    # 创建一个牌组
    deck_id = 2059400110
    deck = genanki.Deck(deck_id, "JA情景短语")

    ROOT = Path(__file__).resolve().parent

    # 图片文件夹路径
    IMG_DIR = ROOT / "zn_out"  # 根据实际情况调整路径

    media_files = []

    THEME_COUNT = {"0": 621, "1": 219, "2": 228, "3": 136, "4": 146}

    # 遍历文件夹中的图片
    loop = tqdm(sorted(IMG_DIR.glob("*.png")))

    # 遍历THEME_COUNT字典
    for theme, count in tqdm(THEME_COUNT.items(), leave=False):
        for i in tqdm(range(count), leave=False):
            front_image_path = f"{theme}_{i:04d}_a.png"
            back_image_path = f"{theme}_{i:04d}_b.png"
            front_full_path = IMG_DIR / front_image_path
            back_full_path = IMG_DIR / back_image_path

            # 检查图片是否存在
            assert front_full_path.exists() and back_full_path.exists()
            media_files.append(str(front_full_path))
            media_files.append(str(back_full_path))
            note = genanki.Note(
                model=model,
                fields=[
                    """<img src="%s", alt="%s">"""
                    % (
                        front_image_path,
                        front_image_path,
                    ),
                    """<img src="%s", alt="%s"><br><img src="%s", alt="%s">"""
                    % (
                        front_image_path,
                        front_image_path,
                        back_image_path,
                        back_image_path,
                    ),
                ],
            )
            deck.add_note(note)

    # 生成APKG文件
    my_package = genanki.Package(deck)
    my_package.media_files = media_files
    my_package.write_to_file(ROOT / "zn_cards.apkg")
