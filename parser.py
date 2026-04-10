import io
from typing import Union

from PyPDF2 import PdfReader


def extract_text_from_pdf(file_obj: Union[io.BytesIO, str]) -> str:
    """
    Extract text from a PDF file.

    Parameters
    ----------
    file_obj : Union[io.BytesIO, str]
        Either a file-like object (Streamlit upload) or a file path.

    Returns
    -------
    str
        Extracted text across all pages.
    """
    if file_obj is None:
        return ""

    try:
        # ✅ Handle Streamlit UploadedFile properly
        if hasattr(file_obj, "read"):
            file_obj.seek(0)  # important for repeated reads
            reader = PdfReader(file_obj)
        else:
            reader = PdfReader(str(file_obj))

    except Exception:
        return ""

    all_text: list[str] = []

    for page in reader.pages:
        try:
            text = page.extract_text()
            if text:
                cleaned = text.strip()
                if cleaned:
                    all_text.append(cleaned)
        except Exception:
            continue

    return "\n".join(all_text).strip()