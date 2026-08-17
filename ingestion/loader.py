from pathlib import Path
SUPPORTED_EXTENSIONS={".pdf",".docx",".txt"}
def validate_file(file_path):
    path=Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"file not found {file_path}")
    if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported file type {path.suffix}")
    return path