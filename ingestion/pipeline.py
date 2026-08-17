from ingestion.loader import validate_file
from ingestion.parser import parse_file
from ingestion.chunker import chunk_text


def ingest_file(file_path):

    path = validate_file(file_path)

    text = parse_file(path)

    if not text:
        raise ValueError(
            "Parser extracted no text from the document."
        )

    chunks = chunk_text(text)

    if not chunks:
        raise ValueError(
            "Text was extracted, but chunking produced no chunks."
        )

    return chunks