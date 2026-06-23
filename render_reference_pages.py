import pathlib

import pypdfium2 as pdfium


PDF_PATH = pathlib.Path(r"C:\Users\lzjia\Desktop\参考文献.pdf")
OUT_DIR = pathlib.Path(r"D:\vqa\reference_page_images")


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    pdf = pdfium.PdfDocument(str(PDF_PATH))
    for index in range(len(pdf)):
        image = pdf[index].render(scale=3.0).to_pil()
        out = OUT_DIR / f"page_{index + 1}.png"
        image.save(out)
        print(out)


if __name__ == "__main__":
    main()
