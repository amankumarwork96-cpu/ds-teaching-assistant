import re
import os


def clean_text(text: str) -> str:
    """
    Cleans raw extracted PDF text.
    Removes excessive whitespace, newlines, and special characters.
    """
    # replace multiple newlines with single space
    text = re.sub(r'\n+', ' ', text)

    # replace multiple spaces with single space
    text = re.sub(r' +', ' ', text)

    # remove non-ASCII characters (common in PDFs)
    text = text.encode('ascii', 'ignore').decode('ascii')

    return text.strip()


def get_pdf_files(pdf_dir: str) -> list[str]:
    """
    Returns a list of all PDF file paths in a directory.
    """
    return [
        os.path.join(pdf_dir, f)
        for f in os.listdir(pdf_dir)
        if f.endswith(".pdf")
    ]


def format_sources(sources: list[str]) -> str:
    """
    Formats source citations into a clean readable string.
    Useful for displaying in terminal or logs.
    """
    if not sources:
        return "No sources found."

    return "\n".join([f"  [{i+1}] {s}" for i, s in enumerate(sources)])