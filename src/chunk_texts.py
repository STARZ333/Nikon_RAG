import os, json
from pathlib import Path
from tqdm import tqdm

PARSED_DIR = "data/parsed"
CHUNK_DIR = "data/chunks"
Path(CHUNK_DIR).mkdir(parents=True, exist_ok=True)

def split_text(text, max_len=800, overlap=100):
    """简单滑动窗口切块"""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(len(words), start + max_len)
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end == len(words):
            break
        start = end - overlap
    return chunks

def chunk_file(in_path, out_path):
    chunks = []
    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            sub_chunks = split_text(rec["text"])
            for idx, ch in enumerate(sub_chunks):
                chunks.append({
                    "model": rec["model"],
                    "doc_type": rec["doc_type"],
                    "page": rec["page"],
                    "chunk_id": idx,
                    "text": ch
                })
    with open(out_path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"✅ Chunked {in_path} → {len(chunks)} chunks")

def batch_chunk_all():
    for fn in os.listdir(PARSED_DIR):
        if fn.endswith(".jsonl"):
            in_path = os.path.join(PARSED_DIR, fn)
            out_path = os.path.join(CHUNK_DIR, fn.replace(".jsonl", "_chunks.jsonl"))
            chunk_file(in_path, out_path)

if __name__ == "__main__":
    batch_chunk_all()
