# src/pdf_ingest.py
import os, re, json, glob, logging
from pathlib import Path
from tqdm import tqdm

logging.getLogger("pdfminer").setLevel(logging.ERROR)
logging.getLogger("pdfplumber").setLevel(logging.ERROR)

RAW_DIR = "data/raw_pdfs"
PARSED_DIR = "data/parsed"
Path(PARSED_DIR).mkdir(parents=True, exist_ok=True)

# ---- 新的型号识别函数：先看文件名，再看文本 ----
def detect_model_from_string(s: str) -> str:
    # 统一大小写 & 去空格/连字符，便于匹配
    s0 = s.replace("-", " ").replace("_", " ").strip()

    # 先匹配不含数字的：Z f / Z fc
    m = re.search(r"\bZ\s*f(c)?\b", s0, re.I)
    if m:
        return "ZF" if not m.group(1) else "ZFC"

    # 再匹配常见：Z 5/6/7 + II/III 可选；也允许连写 Z6II / Z 6 II
    m = re.search(r"\bZ\s?([0-9]{1,2})(\s?I{2,3})?\b", s0, re.I)
    if m:
        num = m.group(1)
        roman = (m.group(2) or "").replace(" ", "")
        return f"Z{num}{roman}".upper()

    # D 系列（D850、D780 等）
    m = re.search(r"\bD[0-9]{3,4}\b", s0, re.I)
    if m:
        return m.group(0).upper()

    return "UNKNOWN"

def guess_model_name_from_filename(filename: str) -> str:
    base = os.path.basename(filename)
    return detect_model_from_string(base)

def guess_model_name_from_text(pages_text: list) -> str:
    # 从前几页文本里找型号（尼康手册开头通常有型号，例如 "Z f"）
    head = " ".join(pages_text[:5])  # 前5页拼一段
    return detect_model_from_string(head)

def read_with_pdfplumber(pdf_path: str):
    import pdfplumber
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, p in enumerate(pdf.pages, start=1):
            try:
                txt = p.extract_text() or ""
            except Exception:
                txt = ""
            pages.append((i, txt.strip()))
    return pages

def read_with_pymupdf(pdf_path: str):
    import fitz
    pages = []
    with fitz.open(pdf_path) as doc:
        for i, page in enumerate(doc, start=1):
            try:
                txt = page.get_text("text") or ""
            except Exception:
                txt = ""
            pages.append((i, txt.strip()))
    return pages

def parse_pdf(pdf_path: str, out_path: str):
    # 先读一遍
    pages = read_with_pdfplumber(pdf_path)
    empty_ratio = 0 if not pages else sum(1 for _,t in pages if not t)/len(pages)
    if empty_ratio > 0.6:
        pages = read_with_pymupdf(pdf_path)

    # 先用文件名猜
    model = guess_model_name_from_filename(pdf_path)

    # 再用文本兜底（文件名没识别出来时）
    if model == "UNKNOWN":
        model = guess_model_name_from_text([t for _, t in pages])

    doc_type = "specs" if "spec" in pdf_path.lower() else "manual"

    kept = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for page_no, text in tqdm(pages, desc=f"Writing {os.path.basename(pdf_path)}"):
            if not text:
                continue
            rec = {
                "model": model,
                "doc_type": doc_type,
                "page": page_no,
                "text": text
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1
    print(f"✅ Parsed {pdf_path} → {out_path}  (model={model}, pages_with_text={kept})")

def batch_parse_all():
    pdf_files = glob.glob(os.path.join(RAW_DIR, "**", "*.pdf"), recursive=True)
    if not pdf_files:
        print(f"⚠️ 没找到 PDF：{RAW_DIR}/**/*.pdf")
        return
    print(f"🔎 Found {len(pdf_files)} PDFs")
    for pdf_path in pdf_files:
        out_name = os.path.basename(pdf_path).replace(".pdf", ".jsonl")
        out_path = os.path.join(PARSED_DIR, out_name)
        parse_pdf(pdf_path, out_path)

if __name__ == "__main__":
    batch_parse_all()
