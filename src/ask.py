# -*- coding: utf-8 -*-
"""
ask.py — Nikon RAG 问答（LLM+RAG 一体版，稳定导出）

特性：
- LLM（Ollama: qwen2.5:14b-instruct）+ 规则抽取（优先 specs）
- 证据只抄 specs/manual 里的“原句/原行”，LLM 输出强约束：value 必须是 evidence 子串
- 导出 eval.py 需要的全部符号：CORPUS、FEATURES、DEFAULT_TESTS、detect_models、detect_feature_keys、
  build_where_filter、retrieve、extract_fields_from_chunks、need_supplement、select_chunks_for_field、
  llm_extract_one、find_field_by_key
- CLI：单问；批量写入 TXT（outputs/qa_run_*.txt 或 --out 指定）
"""

import os
import re
import json
import glob
import math
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any
import sys

# 可选：LLM HTTP
try:
    import requests  # type: ignore
except Exception:
    requests = None

# ============ 配置 ============

# 语料目录（支持冒号分隔多个）优先于默认
ENV_CORPUS = os.environ.get("NIKON_CORPUS_DIR", "")
DEFAULT_BASES = ["./corpus", "./data", "/mnt/data"]

# Ollama
OLLAMA_BASE = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:14b-instruct")
LLM_LOG_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "llm.log"))

# ============ 日志 ============

def _log_llm(line: str):
    try:
        with open(LLM_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(line.rstrip("\n") + "\n")
    except Exception:
        pass

# ============ 语料加载 ============

def _iter_jsonl_files():
    bases: List[str] = []
    if ENV_CORPUS:
        for p in ENV_CORPUS.split(":"):
            p = p.strip()
            if p:
                bases.append(p)
    # 常见数据路径
    # 优先 data/parsed 或 data/chunks 下的 jsonl
    bases += ["./data/parsed", "./data/chunks"] + DEFAULT_BASES

    seen = set()
    for base in bases:
        if not os.path.isdir(base):
            continue
        for pat in ("*.jsonl", "nikon*.jsonl", "Nikon*.jsonl"):
            for p in glob.glob(os.path.join(base, "**", pat), recursive=True):
                rp = os.path.realpath(p)
                if os.path.isfile(rp) and rp not in seen:
                    seen.add(rp)
                    yield rp

def load_corpus() -> List[Dict[str, Any]]:
    docs: List[Dict[str, Any]] = []
    seen = set()
    for path in _iter_jsonl_files():
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    text = (rec.get("text") or "").strip()
                    model = (rec.get("model") or rec.get("camera") or "").strip()
                    if not text or not model:
                        continue
                    doc_type = (rec.get("doc_type") or rec.get("doctype") or "").strip().lower()
                    if doc_type not in ("specs", "manual"):
                        # chunks/parsed 里有的写法不一致，兜底判断
                        # 粗略规则：文件名/来源里带 specs 归 specs，否则 manual
                        srcnm = (rec.get("source") or "").lower()
                        if "spec" in srcnm:
                            doc_type = "specs"
                        elif "manual" in srcnm:
                            doc_type = "manual"
                        else:
                            # 未知一律按 manual 处理，避免把说明书当规格表
                            doc_type = "manual"

                    page = int(rec.get("page") or rec.get("pageno") or rec.get("page_no") or 0)
                    source = rec.get("source") or f"Nikon_{model}_{doc_type}.pdf"

                    item = {
                        "text": text,
                        "model": model,
                        "doc_type": doc_type,
                        "page": page,
                        "source": source
                    }
                    key = (item["model"], item["doc_type"], item["source"], item["page"], hash(item["text"]))
                    if key in seen:
                        continue
                    seen.add(key)
                    docs.append(item)
        except Exception:
            # 单文件异常跳过
            pass
    return docs

CORPUS: List[Dict[str, Any]] = load_corpus()
if not CORPUS:
    print("[WARN] 没有加载到任何语料 JSONL。请将 JSONL 放到 ./data/parsed 或 ./data/chunks，或设置 NIKON_CORPUS_DIR。", file=sys.stderr)

# ============ 简易检索 ============

def _tokenize(s: str) -> List[str]:
    s = re.sub(r"[^\w\u4e00-\u9fff]+", " ", s.lower())
    return s.split()

def _score(query: str, d: Dict[str, Any], model_hint: Optional[str]=None) -> float:
    text = d["text"]
    score = 0.0
    q = query.lower()

    # 粗 BM25-ish 计数
    for kw in set(_tokenize(q)):
        if len(kw) <= 1:
            continue
        cnt = text.lower().count(kw)
        if cnt:
            score += math.log1p(cnt)

    # 字段词加权
    field_terms = [
        "有效像素","总像素","每秒幅数","高速连拍","高速画面捕捉","c120","c60","c30","c15",
        "视频画面尺寸","视频分辨率","帧率","帧频","raw视频","n-raw","prores",
        "存储介质","双存储卡插槽","取景器","显示屏","快门速度","闪光同步速度",
        "尺寸（宽×高×厚）","尺寸(宽×高×厚)","尺寸（宽x高x厚）","重量",
        "帧尺寸（像素）和帧速率（raw"
    ]
    tlow = text.lower()
    for ph in field_terms:
        if ph.lower() in tlow:
            score += 0.2
        if ph.lower() in q:
            score += 1.2

    # 机型命中
    if model_hint and model_hint.lower() in tlow:
        score += 0.6

    # specs 文档优先
    if (d.get("doc_type") or "").lower() == "specs":
        score += 1.2

    return score

def build_where_filter(models: List[str], exact: bool=True):
    models = [m.upper() for m in models if m]
    def _pred(d: Dict[str, Any]) -> bool:
        if not models:
            return True
        mm = (d.get("model") or "").upper()
        if not mm:
            return False
        if exact:
            return mm in models
        return any(x in mm for x in models)
    return _pred

def retrieve(query: str, where=None, topk: int=16) -> List[Dict[str, Any]]:
    if where is None:
        def where(_): return True
    scored = []
    for d in CORPUS:
        if not where(d):
            continue
        s = _score(query, d, d.get("model"))
        if s > 0:
            scored.append((s, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [x[1] for x in scored[:topk]]

# ============ 字段定义 & 抽取器（正则） ============

@dataclass
class FieldSpec:
    key: str
    title: str
    synonyms: List[str] = field(default_factory=list)
    regex_getters: List[Any] = field(default_factory=list)
    prefer_specs: bool = False  # 对维度/重量等更信 specs

def rgx(pattern):
    reg = re.compile(pattern, re.S)
    def _get(text):
        m = reg.search(text)
        if not m:
            return None
        seg = m.group(0).strip()
        return seg, seg
    return _get

def sentence_after(label_pattern):
    reg = re.compile(label_pattern, re.S)
    def _get(text):
        m = reg.search(text)
        if not m:
            return None
        tail = text[m.end():]
        cut = re.split(r"[。；;]\s*|\n", tail, maxsplit=1)[0].strip()
        if not cut:
            return None
        sent = (text[m.start():m.end()] + cut).strip()
        return cut, sent
    return _get

def section_first_sentence(header, stop_keywords, prefer_keywords=None, maxlen=220):
    reg = re.compile(rf"{header}\s*", re.S)
    def _get(text):
        m = reg.search(text)
        if not m:
            return None
        sub = text[m.end():]
        stop_idx = len(sub)
        for s in stop_keywords:
            idx = sub.find(s)
            if idx != -1 and idx < stop_idx:
                stop_idx = idx
        block = sub[:stop_idx]
        cands = [seg.strip() for seg in re.split(r"[。；;]\s*|\n", block) if seg.strip()]
        if not cands:
            return None
        if prefer_keywords:
            for kw in prefer_keywords:
                for seg in cands:
                    if kw in seg:
                        return seg[:maxlen], seg[:maxlen]
        seg = cands[0][:maxlen]
        return seg, seg
    return _get

def burst_max_getter(text: str):
    pats = [
        r"每秒幅数[（(]?近似值[)）]?\s*[:：]?\s*[^。\n]*?(约?\d+[^。/\n]*?幅/秒)",
        r"每秒幅数\s*[:：]?\s*[^。\n]*?(约?\d+[^。/\n]*?幅/秒)",
        r"最[高]?约?(\d+[^。\n]*?幅/秒)"
    ]
    for p in pats:
        m = re.search(p, text, re.S)
        if m:
            sent = re.search(r"(每秒幅数[^。；\n]+|最[高]?[约]?\d+[^。；\n]*幅/秒)", text, re.S)
            val = m.group(1)
            if sent:
                return val, sent.group(0).strip()
            return val, m.group(0).strip()
    # C120/C60/C30/C15 提取一个
    cands = []
    for tag in ("C120","C60","C30","C15"):
        m = re.search(rf"{tag}[^。；\n]*?约?\d+[^。；\n]*?幅/秒", text)
        if m: cands.append(m.group(0))
    if cands:
        one = cands[0]
        mm = re.search(r"(约?\d+[^。；\n]*?幅/秒)", one)
        if mm:
            return mm.group(0), one
    return None

def video_res_fps_getter(text: str):
    labels = [
        r"视频画[面⾯]\s*尺[寸⼨][（(]?像素[)）]?\s*和\s*帧[率频]",
        r"视频\s*分辨率\s*和\s*帧[率频]",
        r"视频\s*记录\s*尺[寸⼨][（(]?像素[)）]?\s*和\s*帧[率频]",
    ]
    for lab in labels:
        m = re.search(lab, text, re.I | re.S)
        if not m:
            continue
        tail = text[m.end():]
        cut = re.split(r"[。；;]\s*|\n", tail, maxsplit=1)[0].strip()
        if cut:
            sent = (text[m.start():m.end()] + cut).strip()
            if re.search(r"(8K|4K|3840\s*[x×]\s*2160|7680\s*[x×]\s*4320|120p|100p|60p|50p|30p|25p|24p)", cut, re.I):
                return cut, sent
    return None

def raw_video_res_getter(text: str):
    lab = r"帧[尺]?寸[（(]?像素[)）]?\s*和\s*帧[率频速]\s*[（(]?\s*RAW\s*视频\s*[)）]?"
    m = re.search(lab, text, re.I | re.S)
    tail = None
    if m:
        tail = text[m.end():]
    else:
        m2 = re.search(r"RAW\s*视频.*?(帧[尺]?寸|分辨率).{0,30}(帧[率频速])", text, re.I | re.S)
        if not m2:
            m3 = re.search(r"(帧[尺]?寸|分辨率).{0,30}(帧[率频速]).{0,60}RAW\s*视频", text, re.I | re.S)
            if not m3:
                return None
            tail = text[m3.end():]
        else:
            tail = text[m2.end():]
    cut = re.split(r"[。；;]\s*|\n", tail, maxsplit=1)[0].strip()
    if cut:
        return cut, cut
    return None

def shutter_speed_getter_loose(text: str):
    m = re.search(r"快门速度\s*[:：]?\s*(.+)", text)
    if not m:
        return None
    line = re.split(r"[。；;]\s*|\n", m.group(1), maxsplit=1)[0].strip()
    if line:
        return line, "快门速度 " + line
    return None

def dimensions_value_only(text: str):
    m = re.search(r"约?\s*\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*[x×]\s*\d+(?:\.\d+)?\s*mm", text, re.I)
    if m:
        v = m.group(0).strip()
        return v, v
    return None

FEATURES: List[FieldSpec] = [
    FieldSpec(
        key="effective_pixels",
        title="有效像素",
        synonyms=["有效像素","像素","像素数"],
        regex_getters=[rgx(r"有效像素\s*数?\s*[:：]?\s*约?[0-9,．\.]+万")]
    ),
    FieldSpec(
        key="total_pixels",
        title="总像素",
        synonyms=["总像素","总像素数"],
        regex_getters=[rgx(r"总像素\s*数?\s*[:：]?\s*约?[0-9,．\.]+万")]
    ),
    FieldSpec(
        key="burst_max",
        title="连拍最高每秒幅数",
        synonyms=["连拍","每秒幅数","高速连拍","高速画面捕捉","C120","C60","C30","C15"],
        regex_getters=[burst_max_getter]
    ),
    FieldSpec(
        key="video_res_fps",
        title="视频分辨率与帧率",
        synonyms=["视频分辨率与帧率","视频分辨率","帧率","帧频","8K","4K","120p","100p","60p","50p","30p","25p","24p"],
        regex_getters=[video_res_fps_getter]
    ),
    FieldSpec(
        key="raw_video_res",
        title="RAW视频帧尺寸/帧率",
        synonyms=["RAW视频帧尺寸","RAW视频帧率","帧尺寸（像素）和帧速率（RAW视频）","N-RAW","ProRes RAW"],
        regex_getters=[raw_video_res_getter]
    ),
    FieldSpec(
        key="video_codecs",
        title="视频压缩/编码",
        synonyms=["视频压缩","视频编码","编解码","N-RAW","ProRes","H.265","HEVC","H.264","AVC"],
        regex_getters=[sentence_after(r"视频压缩\s*")]
    ),
    FieldSpec(
        key="video_filefmt",
        title="视频文件格式",
        synonyms=["视频文件格式","文件格式（视频）","视频文件类型","NEV","MOV","MP4"],
        regex_getters=[sentence_after(r"视频文件格式\s*"), sentence_after(r"文件格式（视频）\s*"), sentence_after(r"视频文件类型\s*")]
    ),
    FieldSpec(
        key="photo_formats",
        title="照片文件格式",
        synonyms=["文件格式（图像品质）","静止图像文件格式","照片文件格式","图像文件格式","NEF","JPEG","HEIF"],
        regex_getters=[rgx(r"(?:文件格式（图像品质）|静止图像文件格式|照片文件格式|图像文件格式)[^。\n]+")]
    ),
    FieldSpec(
        key="storage_media",
        title="存储介质",
        synonyms=["存储介质","存储卡","卡类型","CFexpress","SD","XQD","UHS"],
        regex_getters=[sentence_after(r"存储介质\s*")]
    ),
    FieldSpec(
        key="card_slots",
        title="双卡槽配置",
        synonyms=["双存储卡插槽","双卡槽","卡槽"],
        regex_getters=[sentence_after(r"双存储卡插槽\s*")]
    ),
    FieldSpec(
        key="shutter_speed",
        title="快门速度",
        synonyms=["快门速度","快门范围"],
        regex_getters=[sentence_after(r"快门速度\s*"), shutter_speed_getter_loose]
    ),
    FieldSpec(
        key="flash_sync",
        title="闪光同步速度",
        synonyms=["闪光同步速度","闪光同步"],
        regex_getters=[sentence_after(r"闪光同步速度\s*")]
    ),
    FieldSpec(
        key="evf",
        title="取景器（要点）",
        synonyms=["取景器","EVF","电子取景器"],
        regex_getters=[section_first_sentence(
            header=r"取景器",
            stop_keywords=["显示屏","快门","存储介质","双存储卡插槽","测光","视频","照片","文件格式","尺寸","重量"],
            prefer_keywords=["画点","OLED","取景器","0.8倍","0.5英"]
        )]
    ),
    FieldSpec(
        key="lcd",
        title="显示屏（要点）",
        synonyms=["显示屏","屏幕","LCD","翻折","翻转","触摸"],
        regex_getters=[section_first_sentence(
            header=r"显示屏",
            stop_keywords=["取景器","快门","存储介质","双存储卡插槽","测光","视频","照片","文件格式","尺寸","重量"],
            prefer_keywords=["英寸","LCD","可翻","触摸","分辨率","覆盖率"]
        )]
    ),
    FieldSpec(
        key="dimensions",
        title="机身尺寸",
        synonyms=["机身尺寸","尺寸（宽×高×厚）","尺寸(宽×高×厚)","尺寸（宽x高x厚）","尺寸(宽x高x厚)"],
        regex_getters=[sentence_after(r"尺寸[（(]宽[×x]?[高]?[×x]?[厚][)）]\s*"), dimensions_value_only],
        prefer_specs=True
    ),
    FieldSpec(
        key="weight",
        title="机身重量",
        synonyms=["机身重量","重量"],
        regex_getters=[sentence_after(r"重量\s*")],
        prefer_specs=True
    ),
]

DEFAULT_KEYS = ["effective_pixels", "total_pixels", "burst_max"]

# ============ 解析辅助 ============

def list_all_models() -> List[str]:
    ms = []
    for d in CORPUS:
        m = (d.get("model") or "").upper()
        if not m: 
            continue
        if m not in ms:
            ms.append(m)
    return ms

ALL_MODELS = list_all_models()

def detect_models(query: str) -> List[str]:
    q = query.upper()
    found = []
    for m in ALL_MODELS:
        if m and m in q:
            found.append(m)
    if not found:
        # 兜底识别：Z5II/Z6III/Z7II/Z8/Z9/ZF/D850
        m = re.findall(r"\b(Z[0-9]+III|Z[0-9]+II|Z[0-9]+|ZF|D850)\b", q)
        if m:
            found = list(dict.fromkeys(m))
    return found[:2] if len(found) > 2 else found

def is_compare_query(query: str) -> bool:
    return (" 与 " in query) or ("和" in query) or ("对比" in query) or ("差异" in query) or ("区别" in query)

def find_field_by_key(key: str) -> Optional[FieldSpec]:
    for f in FEATURES:
        if f.key == key:
            return f
    return None

def detect_feature_keys(query: str) -> List[str]:
    q = query.replace(" ","")
    asked: List[str] = []
    has_video = ("视频" in q)

    # 显式映射
    if "像素" in q and "effective_pixels" not in asked:
        asked.append("effective_pixels")
    if "连拍" in q and "burst_max" not in asked:
        asked.append("burst_max")

    for f in FEATURES:
        for syn in f.synonyms:
            if syn in q:
                if has_video and f.key in ("effective_pixels","dimensions","weight","photo_formats"):
                    # 视频语境下，除非被明确点名，否则不过度引入
                    if f.key == "photo_formats" and "文件格式" in q:
                        asked.append(f.key); break
                    if f.key in ("dimensions","weight") and ("机身" in q or "重量" in q or "尺寸" in q):
                        asked.append(f.key); break
                    continue
                asked.append(f.key)
                break

    if not asked:
        asked = DEFAULT_KEYS[:]

    # 去重保序
    uniq=[]
    for k in asked:
        if k not in uniq:
            uniq.append(k)
    return uniq

# ============ 正则抽取与合并 ============

def _apply_getters(text: str, fs: FieldSpec) -> Optional[Tuple[str,str]]:
    for getter in fs.regex_getters:
        try:
            out = getter(text)
        except Exception:
            out = None
        if out:
            return out
    return None

def _format_cite(model: str, source: str, page: int) -> str:
    return f"{model} | {source} | p.{page}"

def _model_sort(models: List[str]) -> List[str]:
    return models[:2]

def extract_fields_from_chunks(chunks: List[Dict[str, Any]],
                               models: List[str],
                               feature_keys: List[str]) -> Dict[str, Dict[str, Dict[str, str]]]:
    fields: Dict[str, Dict[str, Dict[str, str]]] = {m: {} for m in models}
    keyset = set(feature_keys)
    for d in chunks:
        model = (d.get("model") or "").upper()
        if model not in fields:
            continue
        text = d.get("text") or ""
        page = d.get("page") or 0
        source = d.get("source") or "Nikon_specs.pdf"
        for k in keyset:
            if k in fields[model]:
                continue
            fs = find_field_by_key(k)
            if not fs:
                continue
            # prefer_specs：若此块非 specs，先跳过，后面补检/LLM再补
            if fs.prefer_specs and (d.get("doc_type","").lower() != "specs"):
                continue
            got = _apply_getters(text, fs)
            if not got:
                continue
            value, ev = got
            fields[model][k] = {
                "value": value.strip(),
                "evidence": ev.strip(),
                "cite": _format_cite(model, source, page)
            }
    return fields

def merge_fields(base: Dict[str, Dict[str, Dict[str, str]]],
                 extra: Dict[str, Dict[str, Dict[str, str]]]):
    for m, kv in extra.items():
        base.setdefault(m, {})
        for k, v in kv.items():
            if k not in base[m]:
                base[m][k] = v

def need_supplement(fields, models, keys) -> List[Tuple[str,str]]:
    miss = []
    for m in models:
        for k in keys:
            if k not in fields.get(m, {}):
                miss.append((m,k))
    return miss

def select_chunks_for_field(models: List[str], field_key: str, topk: int = 12) -> List[Dict[str, Any]]:
    fs = find_field_by_key(field_key)
    where = build_where_filter(models, exact=True)
    hint = fs.title if fs else field_key
    q = " ".join(models) + " " + hint
    cand = retrieve(q, where=where, topk=64)

    # prefer_specs：把 specs 提前
    if fs and fs.prefer_specs:
        specs = [c for c in cand if (c.get("doc_type","").lower()=="specs")]
        others = [c for c in cand if (c.get("doc_type","").lower()!="specs")]
        cand = specs + others
    return cand[:topk]

# ============ LLM （Ollama）===========

def ollama_chat(messages: List[Dict[str,str]], model: Optional[str]=None, timeout: int=120) -> Optional[str]:
    if requests is None:
        return None
    url = OLLAMA_BASE.rstrip("/") + "/api/chat"
    payload = {
        "model": model or OLLAMA_MODEL,
        "messages": messages,
        "stream": False
    }
    try:
        _log_llm(f"[LLM] POST {url} model={payload['model']}")
        resp = requests.post(url, json=payload, timeout=timeout)
        _log_llm(f"[LLM] status={resp.status_code}")
        if resp.status_code != 200:
            return None
        data = resp.json()
        if isinstance(data, dict):
            msg = data.get("message") or {}
            out = (msg.get("content") or "").strip()
            _log_llm(f"[LLM] out.head={out[:220].replace(chr(10),' ')}")
            return out
        return None
    except Exception as e:
        _log_llm(f"[LLM][ERR] {e}")
        return None

def llm_extract_one(model: str, field_key: str, chunks: List[Dict[str, Any]]) -> Optional[Dict[str,str]]:
    fs = find_field_by_key(field_key)
    if not fs:
        return None
    chunks = chunks[:10]
    if not chunks:
        return None

    lines = []
    for idx, c in enumerate(chunks, 1):
        cite = _format_cite((c.get("model") or model).upper(), c.get("source",""), c.get("page") or 0)
        txt = (c.get("text","") or "").strip()
        lines.append(f"### CHUNK {idx} | {cite}\n{txt}")

    sys_prompt = (
        "你是一个严格的规格抽取器。只能从提供的【证据】里“逐字抄写”答案，不得推断或改写。\n"
        "任务：仅针对一个字段，从证据中复制最小必要片段作为 value，并给出包含该 value 的 evidence 原句/原行，"
        "cite 用证据头部提供的“MODEL | SOURCE | p.PAGE”。\n"
        "约束：\n"
        "1) value 必须是 evidence 的子串；\n"
        "2) 若证据没有该字段，输出空 JSON：{\"value\":\"\",\"evidence\":\"\",\"cite\":\"\"}；\n"
        "3) 仅输出 JSON，不要多余文本。"
    )
    user_prompt = (
        f"机型：{model}\n"
        f"字段：{fs.title}\n"
        "请从下方【证据】中抽取。\n\n"
        "【证据】\n" + "\n\n".join(lines) + "\n\n"
        "【输出 JSON 模板】\n"
        "{\"value\":\"\",\"evidence\":\"\",\"cite\":\"\"}"
    )

    content = ollama_chat(
        [{"role": "system", "content": sys_prompt},
         {"role": "user", "content": user_prompt}]
    )
    if not content:
        return None
    try:
        data = json.loads(content)
        val = (data.get("value") or "").strip()
        ev = (data.get("evidence") or "").strip()
        ci = (data.get("cite") or "").strip()
        if val and ev and (val in ev):
            ok_cite = False
            for c in chunks:
                if _format_cite((c.get("model") or model).upper(), c.get("source",""), c.get("page") or 0) == ci:
                    ok_cite = True
                    break
            if not ok_cite:
                # cite 不匹配则改用首条证据的 cite
                c0 = chunks[0]
                ci = _format_cite((c0.get("model") or model).upper(), c0.get("source",""), c0.get("page") or 0)
            return {"value": val, "evidence": ev, "cite": ci}
    except Exception:
        return None
    return None

# ============ 输出构建 ============

def make_table(fields: Dict[str, Dict[str, Dict[str,str]]],
               models: List[str],
               feature_keys: List[str]) -> str:
    ms = _model_sort(models)
    if len(ms) == 1:
        header = f"| 字段 | {ms[0]} |\n| - | - |"
    else:
        header = f"| 字段 | {ms[0]} | {ms[1]} |\n| - | - | - |"
    lines = [header]
    for k in feature_keys:
        fs = find_field_by_key(k)
        if not fs:
            continue
        row = [fs.title]
        for m in ms:
            cell = "—"
            val = fields.get(m, {}).get(k, {}).get("value")
            if val:
                cell = val
            row.append(cell)
        if len(ms) == 1:
            lines.append(f"| {row[0]} | {row[1]} |")
        else:
            lines.append(f"| {row[0]} | {row[1]} | {row[2]} |")
    return "\n".join(lines)

def make_context(fields: Dict[str, Dict[str, Dict[str,str]]],
                 models: List[str],
                 feature_keys: List[str]) -> str:
    bullets = []
    seen = set()
    for m in _model_sort(models):
        for k in feature_keys:
            item = fields.get(m, {}).get(k)
            if not item:
                continue
            ev = item.get("evidence","").strip()
            cite = item.get("cite","")
            line = f"- {ev} —— 引用：{cite}"
            if line not in seen:
                seen.add(line)
                bullets.append(line)
    return "\n".join(bullets) if bullets else ""

def make_conclusion(fields: Dict[str, Dict[str, Dict[str,str]]],
                    models: List[str],
                    feature_keys: List[str],
                    query: str) -> str:
    ms = _model_sort(models)
    if len(ms) == 2 and set(feature_keys).issubset({"effective_pixels","total_pixels","burst_max"}) and feature_keys:
        def v(m,k): return fields.get(m,{}).get(k,{}).get("value")
        eff_equal = v(ms[0],"effective_pixels") and (v(ms[0],"effective_pixels")==v(ms[1],"effective_pixels"))
        tot_equal = ("total_pixels" in feature_keys) and v(ms[0],"total_pixels") and (v(ms[0],"total_pixels")==v(ms[1],"total_pixels"))
        burst_equal = v(ms[0],"burst_max") and (v(ms[0],"burst_max")==v(ms[1],"burst_max"))
        msgs = []
        if eff_equal: msgs.append("有效像素相同")
        elif v(ms[0],"effective_pixels") or v(ms[1],"effective_pixels"): msgs.append("有效像素不同")
        if "total_pixels" in feature_keys:
            if tot_equal: msgs.append("总像素相同")
            elif v(ms[0],"total_pixels") or v(ms[1],"total_pixels"): msgs.append("总像素不同")
        if burst_equal: msgs.append("最高连拍相同")
        elif v(ms[0],"burst_max") or v(ms[1],"burst_max"): msgs.append("最高连拍不同")
        if msgs:
            return "结论：" + "，".join(msgs) + "。"
    return "结论：已列出所询字段。"

# ============ 主问答 ============

def ask(query: str, use_llm: bool=True) -> Tuple[str, str]:
    # 兜底：若像“hello”这种非相机问题，给提示
    if not re.search(r"(Z\d|Zf|D850|尼康|Nikon|像素|连拍|视频|机身|存储|卡槽|取景器|显示屏|快门|闪光)", query, re.I):
        answer = ("结论：这是一个非相机问题或信息不足，请补充具体机型/字段。\n"
                  "| 字段 | 示例 |\n| - | - |\n| 提示 | 未检测到相机型号/尼康关键词/规格字段。 |")
        return answer, ""

    models = detect_models(query)
    if not models:
        # 若问句没带型号，按频次取两个最常见（或回退 Z8/Z9）
        freq = {}
        for d in CORPUS:
            m = (d.get("model") or "").upper()
            if not m: continue
            freq[m] = freq.get(m,0) + 1
        models = sorted(freq.keys(), key=lambda x: -freq[x])[:2] or ["Z8","Z9"]

    feature_keys = detect_feature_keys(query)

    # 初检（正则）
    where = build_where_filter(models, exact=True)
    chunks = retrieve(query, where=where, topk=32)
    fields = extract_fields_from_chunks(chunks, models, feature_keys)

    # 逐字段补检（正则）
    missing = need_supplement(fields, models, feature_keys)
    if missing:
        for (m,k) in list(missing):
            more = select_chunks_for_field([m], k, topk=16)
            fields2 = extract_fields_from_chunks(more, [m], [k])
            merge_fields(fields, fields2)

    # LLM 填补（仅缺失字段）
    if use_llm:
        missing = need_supplement(fields, models, feature_keys)
        for (m,k) in missing:
            field_chunks = select_chunks_for_field([m], k, topk=12)
            got = llm_extract_one(m, k, field_chunks)
            if got and got.get("value"):
                fields.setdefault(m, {})[k] = got

    # 输出文本
    conclusion = make_conclusion(fields, models, feature_keys, query)
    table = make_table(fields, models, feature_keys)

    # 引用
    cites = []
    seen_cites = set()
    for m in models:
        for k in feature_keys:
            c = fields.get(m, {}).get(k, {}).get("cite")
            if c and c not in seen_cites:
                seen_cites.add(c)
                cites.append(c)
    cite_block = ""
    if cites:
        cite_block = "\n\n引用：\n" + "\n".join(cites)

    answer = f"{conclusion}\n{table}{cite_block}"
    context = make_context(fields, models, feature_keys)
    return answer, context

# ============ 批量 & CLI ============

DEFAULT_TESTS = [
    "Z8 与 Z9 的像素与连拍差异？",
    "Z5II 与 Z6III 的像素与连拍差异？",
    "Z7II 与 Z5II 的连拍规格对比？",
    "D850 与 Z7II 的像素对比？",
    "Z8 支持哪些视频编码？",
    "Z9 的视频分辨率与帧率有哪些？",
    "Z6III 的 RAW 视频帧尺寸和帧率？",
    "Z8 的存储介质与双卡槽配置？",
    "Zf 的照片文件格式是否支持 HEIF？",
    "Z8 的取景器与显示屏规格是什么？",
    "Z9 的快门速度与闪光同步速度？",
    "Z8 的机身尺寸与重量？",
    "Z8 和 Z9 有什么区别？",
    "Zf 与 Z6III 的像素与连拍对比？",
    "Z8 与 Z9 的存储卡配置有何不同？",
    "Z6III 的视频文件格式与编码？",
    "Z9 的 RAW 视频帧尺寸和帧率？",
    "Z8 与 Z9 的视频分辨率和帧频对比？",
    "Z6III 的视频压缩与文件格式各支持哪些？",
    "Z5II 与 Z6III 的存储介质与双卡槽差异？",
    "Z7II 的取景器和显示屏规格是什么？",
    "Z8 的机身尺寸（宽×高×厚）与重量是多少？",
    "Z9 的快门速度范围与闪光同步速度？",
    "Zf 的照片文件格式是否支持 HEIF？",
    "D850 与 Z7II 的有效像素与总像素有何不同？",
    "Z8 的 RAW 视频帧尺寸（像素）和帧率有哪些？",
    "Z9 的 N-RAW 与 ProRes RAW 是否支持？列出具体编码。",
    "Z6III 与 Z5II 的最高连拍每秒幅数对比？",
    "Z8 与 Z9 的存储卡类型与卡槽配置差异？",
    "Z5II 的视频画面尺寸（像素）和帧率（用×写法）有哪些？",
    "Zf 的连拍与有效像素对比 Z6III？",
    "Z8 的视频文件格式（容器）是什么？",
    "Z9 的显示屏（LCD）与取景器（EVF）要点？",
    "Z6III 的快门速度与闪光同步速度？",
    "Z7II 的存储介质与双存储卡插槽配置？",
    "D850 的视频分辨率与帧频列表？",
    "Z8 与 Z9 的像素与连拍规格对比（像素、连拍）？",
]

def run_single(q: str, use_llm: bool=True):
    ans, ctx = ask(q, use_llm=use_llm)
    print("==== Answer ====\n", ans)
    print("\n==== Context(证据句子) ====\n")
    print(ctx)

def run_batch(out_path: Optional[str]=None, tests: Optional[List[str]]=None, use_llm: bool=True):
    tests = tests or DEFAULT_TESTS
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if not out_path:
        out_fn = f"outputs/qa_run_{datetime.now().strftime('%Y%m%d-%H%M%S')}.txt"
        os.makedirs(os.path.dirname(out_fn), exist_ok=True)
        out_path = out_fn

    lines = []
    lines.append("# 尼康 RAG 批量测试")
    lines.append(f"- 时间：{now}")
    lines.append(f"- 输出文件：{out_path}")
    lines.append("- 说明：严格抄写证据；只输出被问到的字段（未点名则为默认三字段）；每个展示值均附对应证据句子。")
    lines.append("")

    for i, q in enumerate(tests, 1):
        ans, ctx = ask(q, use_llm=use_llm)
        lines.append("="*80)
        lines.append(f"Q{i}: {q}\n")
        lines.append("==== Answer ====")
        lines.append(ans)
        lines.append("\n==== Context(证据句子) ====\n")
        lines.append(ctx if ctx else "")
        lines.append("")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"[OK] 批量测试完成：{out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("question", nargs="*", help="问题（可多词）")
    parser.add_argument("--batch", action="store_true", help="运行内置批量测试")
    parser.add_argument("--out", type=str, default=None, help="批量输出文件路径")
    parser.add_argument("--no-llm", action="store_true", help="关闭 LLM，仅使用规则 RAG（调试用）")
    args = parser.parse_args()

    use_llm = not args.no_llm  # 默认启用 LLM

    if args.batch:
        run_batch(out_path=args.out, use_llm=use_llm)
        return

    if args.question:
        q = " ".join(args.question)
    else:
        q = "Z8 与 Z9 的像素与连拍差异？"

    ans, ctx = ask(q, use_llm=use_llm)
    print("==== Answer ====\n", ans)
    print("\n==== Context(证据句子) ====\n", ctx)

# ======= 导出符号（便于 eval.py 可靠 import）=======
__all__ = [
    "CORPUS",
    "FieldSpec",
    "FEATURES",
    "DEFAULT_TESTS",
    "detect_models",
    "detect_feature_keys",
    "build_where_filter",
    "retrieve",
    "extract_fields_from_chunks",
    "need_supplement",
    "select_chunks_for_field",
    "llm_extract_one",
    "find_field_by_key",
    "ask",
    "run_batch",
    "run_single",
]

if __name__ == "__main__":
    main()
