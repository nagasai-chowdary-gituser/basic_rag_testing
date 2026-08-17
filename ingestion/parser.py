from pathlib import Path

from pypdf import PdfReader
from docx import Document


def parse_pdf(file_path):
    reader = PdfReader(str(file_path))

    pages_text = []

    for page_number, page in enumerate(reader.pages, start=1):

        text = page.extract_text()

        if text:
            text = text.strip()

            if text:
                pages_text.append(text)

    return "\n\n".join(pages_text)


def parse_docx(file_path):
    document = Document(str(file_path))

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:
            paragraphs.append(text)

    return "\n\n".join(paragraphs)


def parse_txt(file_path):
    return Path(file_path).read_text(
        encoding="utf-8"
    ).strip()


def parse_file(file_path):

    path = Path(file_path)

    extension = path.suffix.lower()

    if extension == ".pdf":
        return parse_pdf(path)

    elif extension == ".docx":
        return parse_docx(path)

    elif extension == ".txt":
        return parse_txt(path)

    else:
        raise ValueError(
            f"Unsupported file type: {extension}"
        )