from pathlib import Path
from typing import List, Any, Iterable, Union
from langchain_community.document_loaders import PyPDFLoader, TextLoader, CSVLoader
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.document_loaders import Docx2txtLoader
from langchain_community.document_loaders.excel import UnstructuredExcelLoader
from langchain_community.document_loaders import JSONLoader
from langchain_core.documents import Document

SUPPORTED_EXTENSIONS = {".pdf", ".txt", ".csv", ".xlsx", ".docx", ".json"}


def _score_ocr_text(text: str) -> float:
    cleaned = (text or "").strip()
    if not cleaned:
        return 0.0
    total = len(cleaned)
    if total == 0:
        return 0.0
    alpha_num = sum(ch.isalnum() for ch in cleaned)
    words = len(cleaned.split())
    return (alpha_num / total) * 0.7 + min(words / 120.0, 1.0) * 0.3


def _ocr_image_with_strategies(image, pytesseract):
    from PIL import ImageFilter, ImageOps

    grayscale = ImageOps.grayscale(image)
    boosted = ImageOps.autocontrast(grayscale)
    thresholded = boosted.point(lambda pixel: 255 if pixel > 160 else 0)
    denoised = thresholded.filter(ImageFilter.MedianFilter(size=3))

    variants = [
        boosted,
        thresholded,
        denoised,
    ]
    configs = [
        "--oem 3 --psm 6",
        "--oem 3 --psm 4",
        "--oem 3 --psm 11",
    ]

    best_text = ""
    best_score = 0.0
    for variant in variants:
        for config in configs:
            candidate = pytesseract.image_to_string(variant, config=config).strip()
            score = _score_ocr_text(candidate)
            if score > best_score:
                best_text = candidate
                best_score = score

    return best_text, best_score


def _ocr_pdf_with_pytesseract(pdf_file: Path) -> List[Any]:
    try:
        import fitz
        import pytesseract
        from PIL import Image
        import io
        import shutil
    except Exception as e:
        print(f"[WARN] OCR dependencies are not available for {pdf_file}: {e}")
        return []

    if shutil.which("tesseract") is None:
        windows_tesseract = Path("C:/Program Files/Tesseract-OCR/tesseract.exe")
        if windows_tesseract.exists():
            pytesseract.pytesseract.tesseract_cmd = str(windows_tesseract)

    try:
        doc = fitz.open(str(pdf_file))
    except Exception as e:
        print(f"[WARN] OCR could not open PDF {pdf_file}: {e}")
        return []

    ocr_documents = []
    for page_idx in range(len(doc)):
        page = doc[page_idx]
        pix = page.get_pixmap(dpi=300)
        image = Image.open(io.BytesIO(pix.tobytes("png")))
        text, quality = _ocr_image_with_strategies(image, pytesseract)
        if text and quality >= 0.08:
            ocr_documents.append(
                Document(
                    page_content=text,
                    metadata={
                        "source": str(pdf_file),
                        "page": page_idx + 1,
                        "ocr": True,
                        "ocr_quality": round(quality, 4),
                    },
                )
            )

    doc.close()
    return ocr_documents


def _load_single_file(file_path: Path) -> List[Any]:
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            loaded = loader.load()
            extracted_text = "".join((doc.page_content or "") for doc in loaded).strip()
            if not extracted_text:
                print(f"[WARN] No text from PyPDFLoader, trying PyMuPDFLoader for: {file_path}")
                loaded = PyMuPDFLoader(str(file_path)).load()
                extracted_text = "".join((doc.page_content or "") for doc in loaded).strip()
            if not extracted_text:
                print(f"[WARN] No text from PyMuPDFLoader, trying OCR for: {file_path}")
                loaded = _ocr_pdf_with_pytesseract(file_path)
            print(f"[DEBUG] Loaded {len(loaded)} PDF docs from {file_path}")
            return loaded

        if suffix == ".txt":
            try:
                loaded = TextLoader(str(file_path), autodetect_encoding=True).load()
            except ModuleNotFoundError:
                loaded = TextLoader(str(file_path)).load()
            print(f"[DEBUG] Loaded {len(loaded)} TXT docs from {file_path}")
            return loaded

        if suffix == ".csv":
            loaded = CSVLoader(str(file_path)).load()
            print(f"[DEBUG] Loaded {len(loaded)} CSV docs from {file_path}")
            return loaded

        if suffix == ".xlsx":
            loaded = UnstructuredExcelLoader(str(file_path)).load()
            print(f"[DEBUG] Loaded {len(loaded)} Excel docs from {file_path}")
            return loaded

        if suffix == ".docx":
            loaded = Docx2txtLoader(str(file_path)).load()
            print(f"[DEBUG] Loaded {len(loaded)} Word docs from {file_path}")
            return loaded

        if suffix == ".json":
            loaded = JSONLoader(str(file_path)).load()
            print(f"[DEBUG] Loaded {len(loaded)} JSON docs from {file_path}")
            return loaded

        return []
    except Exception as e:
        print(f"[ERROR] Failed to load {suffix.upper()} {file_path}: {e}")
        return []


def load_documents_from_paths(file_paths: Iterable[Union[str, Path]]) -> List[Any]:
    documents = []
    normalized_paths = []
    for raw_path in file_paths:
        path = Path(raw_path).resolve()
        if not path.exists() or not path.is_file():
            continue
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        normalized_paths.append(path)

    normalized_paths = sorted(set(normalized_paths))
    print(f"[DEBUG] Loading explicit file set: {len(normalized_paths)} files")
    for file_path in normalized_paths:
        print(f"[DEBUG] Loading file: {file_path}")
        documents.extend(_load_single_file(file_path))

    print(f"[DEBUG] Total loaded documents from file set: {len(documents)}")
    return documents

def load_all_documents(data_dir: str) -> List[Any]:
    """
    Load all supported files from the data directory and convert to LangChain document structure.
    Supported: PDF, TXT, CSV, Excel, Word, JSON
    """
    # Use project root data folder
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    all_files = [
        file_path
        for file_path in data_path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_EXTENSIONS
    ]
    print(f"[DEBUG] Found {len(all_files)} supported files under {data_path}")
    return load_documents_from_paths(all_files)

# Example usage
if __name__ == "__main__":
    docs = load_all_documents("data")
    print(f"Loaded {len(docs)} documents.")
    print("Example document:", docs[0] if docs else None)