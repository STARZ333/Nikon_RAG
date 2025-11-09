import os, json, glob
from pathlib import Path
from tqdm import tqdm
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

CHUNK_DIR = "data/chunks"
DB_DIR = "data/vector_store"       # Chroma 持久化目录
COLLECTION = "nikon_docs"
EMB_MODEL = "BAAI/bge-small-zh-v1.5"   # 中文检索首选

def load_chunks():
    files = glob.glob(os.path.join(CHUNK_DIR, "*.jsonl"))
    for fp in files:
        with open(fp, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                # 你在切块里若没写 source_file，可按文件名推断；有的话就直接用：
                meta = {
                    "model": rec.get("model", "UNKNOWN"),
                    "doc_type": rec.get("doc_type", ""),
                    "page": rec.get("page", -1),
                    "chunk_id": rec.get("chunk_id", 0),
                    "source_file": rec.get("source_file", os.path.basename(fp).replace("_chunks.jsonl",".pdf"))
                }
                yield rec["text"], meta

def main(batch_size=256):
    Path(DB_DIR).mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=DB_DIR, settings=Settings(allow_reset=True))
    # 如果想重建，先清空旧collection
    try:
        client.delete_collection(COLLECTION)
    except Exception:
        pass
    col = client.create_collection(COLLECTION, metadata={"hnsw:space": "cosine"})

    encoder = SentenceTransformer(EMB_MODEL)
    ids, docs, metas, embs = [], [], [], []
    i = 0
    for text, meta in tqdm(load_chunks(), desc="Embedding & upserting"):
        ids.append(f"{meta['model']}_{meta['source_file']}_{meta['page']}_{meta['chunk_id']}_{i}")
        docs.append(text)
        metas.append(meta)
        embs.append(encoder.encode(text, normalize_embeddings=True))
        i += 1
        if len(ids) >= batch_size:
            col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
            ids, docs, metas, embs = [], [], [], []
    if ids:
        col.add(ids=ids, documents=docs, metadatas=metas, embeddings=embs)
    print("✅ Index built:", col.count(), "chunks")

if __name__ == "__main__":
    main()
