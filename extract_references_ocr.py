import pathlib
import sys

import numpy as np
import pypdfium2 as pdfium
from rapidocr_onnxruntime import RapidOCR


sys.stdout.reconfigure(encoding="utf-8")

PDF_PATH = pathlib.Path(r"C:\Users\lzjia\Desktop\参考文献.pdf")
RAW_OUT = pathlib.Path(r"D:\vqa\references_ocr_raw.txt")


def main() -> None:
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    ocr = RapidOCR()
    pages = []

    for index, page in enumerate(pdf):
        image = page.render(scale=3.0).to_pil()
        result, _ = ocr(np.array(image))
        lines = [item[1] for item in (result or [])]
        pages.append(f"--- page {index + 1} ---\n" + "\n".join(lines))
        print(f"page {index + 1}/{len(pdf)} lines={len(lines)}")

    RAW_OUT.write_text("\n".join(pages), encoding="utf-8")
    print(f"wrote {RAW_OUT}")


if __name__ == "__main__":
    main()
