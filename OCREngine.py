from paddleocr import PaddleOCR


class OCREngine:
    def __init__(self, lang="japan"):
        self.engine = PaddleOCR(lang=lang)

    def __call__(
        self,
        *args,
        det=False,
        rec=True,
        cls=False,
        **kwds,
    ):
        return self.engine.ocr(
            *args,
            **kwds,
            det=det,
            rec=rec,
            cls=cls,
        )
