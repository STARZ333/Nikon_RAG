# -*- coding: utf-8 -*-
"""
src/eval.py — Nikon RAG 评测脚本（无检索依赖版）

特性：
1) --report：解析 outputs/qa_run.txt 的表格作为“预测”；
2) 直接评测：可选调用 ask.ask()（LLM+RAG）生成预测后打分（若 ask.ask 存在）；
3) “金标”仅从 CORPUS 中 doc_type == 'specs' 的文本用值-only 正则抽取；
4) 不依赖 ask.retrieve（你当前环境没有 retrieve），因此不会再出现 Gold=0；
5) 结果输出：
   - outputs/eval_metrics_YYYYMMDD-HHMMSS.json
   - outputs/eval_errors_YYYYMMDD-HHMMSS.csv
"""

import os
import re
import sys
import csv
import json
import time
import argparse
from datetime import datetime
from typing import List, Dict, Tuple, Optional, Any

# ---------------- 基础路径 & 导入 ask.py ----------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import ask as ASK  # 同目录 src/ 下的 ask.py
except Exception as e:
    print(f"[FATAL] 无法导入 ask.py：{e}", file=sys.stderr)
    sys.exit(2)

ASK_PATH = getattr(ASK, "__file__", "UNKNOWN")
CORPUS: List[Dict[str, Any]] = getattr(ASK, "CORPUS", [])
FEATURES = getattr(ASK, "FEATURES", [])
DEFAULT_TESTS = getattr(ASK, "DEFAULT_TESTS", [])

ask_fn = getattr(ASK, "ask", None)  # 可选的直接问答
# 注：我们不使用 ASK.retrieve（你的环境没有），后续全部用 CORPUS 直扫

# ---------------- 小工具 ----------------
def now_str(fmt="%Y%m%d-%H%M%S"):
    return datetime.now().strftime(fmt)

def ensure_dir(p: str):
    d = os.path.dirname(p)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)

def normalize_val(s: str) -> str:
    return re.sub(r"\s+", "", s or "").strip()

def most_common_models_from_corpus(n=2) -> List[str]:
    freq = {}
    for d in CORPUS:
        m = (d.get("model") or "").upper()
        if not m:
            continue
        freq[m] = freq.get(m, 0) + 1
    return sorted(freq.keys(), key=lambda x: -freq[x])[:n] or ["Z8", "Z9"]

# ---------------- 值-only 正则 getter（金标用） ----------------
# 所有 getter 必须返回 (value, evidence)，其中 value 是 evidence 的子串
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
        if not m: return None
        tail = text[m.end():]
        cut = re.split(r"[。；;]\s*|\n", tail, maxsplit=1)[0].strip()
        if not cut: return None
        sent = (text[m.start():m.end()] + cut).strip()
        # 取 cut 作为 value，sent 作为 evidence
        return cut, sent
    return _get

def section_first_sentence(header, stop_keywords, prefer_keywords=None, maxlen=220):
    reg = re.compile(rf"{header}\s*", re.S)
    def _get(text):
        m = reg.search(text)
        if not m: return None
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
                        seg = seg[:maxlen]
                        return seg, seg
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
            # value 用 group(1) 或整段中的“约...幅/秒”
            val = (m.group(1) if m.lastindex else None) or re.search(r"约?\d+[^。/\n]*?幅/秒", m.group(0)).group(0)
            sent = re.search(r"(每秒幅数[^。；\n]+|最[高]?[约]?\d+[^。；\n]*幅/秒)", text, re.S)
            if sent:
                return val.strip(), sent.group(0).strip()
            return val.strip(), m.group(0).strip()
    # 回退：C120/C60/C30/C15
    for tag in ("C120","C60","C30","C15"):
        m = re.search(rf"{tag}[^。；\n]*?约?\d+[^。；\n]*?幅/秒", text)
        if m:
            mm = re.search(r"(约?\d+[^。；\n]*?幅/秒)", m.group(0))
            if mm:
                return mm.group(1).strip(), m.group(0).strip()
    return None

def video_res_fps_getter(text: str):
    labels = [
        r"视频画[面⾯]\s*尺[寸⼨][（(]?像素[)）]?\s*和\s*帧[率频]",
        r"视频\s*分辨率\s*和\s*帧[率频]"
    ]
    for lab in labels:
        m = re.search(lab, text, re.I|re.S)
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
    m = re.search(lab, text, re.I|re.S)
    tail = None
    if m:
        tail = text[m.end():]
    else:
        m2 = re.search(r"(RAW\s*视频).*?(帧[尺]?寸|帧[率频速])", text, re.I|re.S)
        if m2:
            tail = text[m2.end():]
    if tail is None:
        return None
    cut = re.split(r"[。；;]\s*|\n", tail, maxsplit=1)[0].strip()
    if cut:
        return cut, cut
    return None

# 字段键 ↔ 标题
BUILTIN_KEY2TITLE = {
    "effective_pixels": "有效像素",
    "total_pixels": "总像素",
    "burst_max": "连拍最高每秒幅数",
    "video_res_fps": "视频分辨率与帧率",
    "raw_video_res": "RAW视频帧尺寸/帧率",
    "video_codecs": "视频压缩/编码",
    "video_filefmt": "视频文件格式",
    "photo_formats": "照片文件格式",
    "storage_media": "存储介质",
    "card_slots": "双卡槽配置",
    "shutter_speed": "快门速度",
    "flash_sync": "闪光同步速度",
    "evf": "取景器（要点）",
    "lcd": "显示屏（要点）",
    "dimensions": "机身尺寸",
    "weight": "机身重量",
}
BUILTIN_TITLE2KEY = {v: k for k, v in BUILTIN_KEY2TITLE.items()}

# 字段 → 值-only getter 列表
FALLBACK_GETTERS: Dict[str, List[Any]] = {
    "effective_pixels": [rgx(r"有效像素\s*数?\s*[:：]?\s*约?[0-9,，．\.]+万")],
    "total_pixels":     [rgx(r"总像素\s*数?\s*[:：]?\s*约?[0-9,，．\.]+万")],
    "burst_max":        [burst_max_getter],
    "video_res_fps":    [video_res_fps_getter],
    "raw_video_res":    [raw_video_res_getter],
    "video_codecs":     [sentence_after(r"视频压缩\s*")],
    "video_filefmt":    [sentence_after(r"视频文件格式\s*")],
    "photo_formats":    [rgx(r"文件格式（图像品质）[^。\n]+")],
    "storage_media":    [sentence_after(r"存储介质\s*")],
    "card_slots":       [sentence_after(r"双存储卡插槽\s*")],
    "shutter_speed":    [sentence_after(r"快门速度\s*")],
    "flash_sync":       [sentence_after(r"闪光同步速度\s*")],
    "evf": [section_first_sentence(
        header=r"取景器",
        stop_keywords=["显示屏","快门","存储介质","双存储卡插槽","测光","视频","照片","文件格式","尺寸","重量"],
        prefer_keywords=["画点","OLED","取景器","0.8倍","0.5英"]
    )],
    "lcd": [section_first_sentence(
        header=r"显示屏",
        stop_keywords=["取景器","快门","存储介质","双存储卡插槽","测光","视频","照片","文件格式","尺寸","重量"],
        prefer_keywords=["英寸","LCD","可翻","触摸","分辨率","覆盖率"]
    )],
    "dimensions": [sentence_after(r"尺寸[（(]宽[×x]?[高]?[×x]?[厚][)）]\s*")],
    "weight":     [sentence_after(r"重量\s*")],
}

# 如果 ask.FEATURES 提供了更精细的 getter，就合并进去（优先使用内置值-only，不覆盖）
try:
    for f in FEATURES or []:
        k = getattr(f, "key", None)
        getters = getattr(f, "regex_getters", None)
        if k and getters and k in FALLBACK_GETTERS:
            # 将 ask 的 getter 追加作为候选（放在后面）
            FALLBACK_GETTERS[k].extend([g for g in getters if callable(g)])
except Exception:
    pass

# ---------------- 解析报告（预测表格） ----------------
def parse_report_tables(report_path: str, debug: bool=False) -> List[Tuple[str, Dict[str, Dict[str, str]], List[str], List[str]]]:
    """
    返回 (question, fields_pred, models, feature_keys)
    fields_pred: {model: {field_key: {"value": str}}}
    """
    with open(report_path, "r", encoding="utf-8") as f:
        content = f.read()

    blocks = re.split(r"=+\s*Q\d+:\s*", content)
    results = []
    for blk in blocks[1:]:
        m_q = re.match(r"(.*?)\n", blk)
        if not m_q:
            continue
        question = m_q.group(1).strip()

        m_ans = re.search(r"==== Answer ====\s*(.*?)\n==== Context", blk, flags=re.S)
        if not m_ans:
            continue
        ans = m_ans.group(1)

        # 表头
        m_header = re.search(r"^\|\s*字段\s*\|\s*(.*?)\|\s*$", ans, flags=re.M)
        if not m_header:
            continue
        model_cells = [c.strip() for c in m_header.group(1).split("|")]
        models = [c for c in model_cells if c]
        models = [m.upper() for m in models]

        fields_pred: Dict[str, Dict[str, Dict[str, str]]] = {m: {} for m in models}
        feature_keys: List[str] = []

        for line in ans.splitlines():
            line = line.strip()
            if not line.startswith("|"):
                continue
            if "字段" in line:
                continue
            if set(line.replace("|", "").strip()) == set("-"):
                continue
            cells = [c.strip() for c in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            field_title = cells[0]
            vals = cells[1:]

            # 标题 -> key
            k = BUILTIN_TITLE2KEY.get(field_title)
            if k is None:
                # 若 ask.FEATURES 有标题映射，也兼容
                for f in FEATURES or []:
                    if getattr(f, "title", None) == field_title:
                        k = getattr(f, "key", None)
                        break
            if not k:
                # 不认识的标题，跳过该行
                continue

            if k not in feature_keys:
                feature_keys.append(k)

            for i, m in enumerate(models):
                v = vals[i] if i < len(vals) else "—"
                if v and v != "—":
                    fields_pred[m][k] = {"value": v}

        if debug:
            print(f"[DBG] Q: {question}")
            print(f"      models: {models}")
            print(f"      feature_keys: {feature_keys}")

        results.append((question, fields_pred, models, feature_keys))
    return results

# ---------------- 直接跑 LLM+RAG（可选） ----------------
def predict_structured(question: str) -> Tuple[Dict[str, Dict[str, Dict[str,str]]], List[str], List[str], float]:
    """
    若 ask.ask 存在：调用生成“表格答案”，再用与 parse_report_tables 相同逻辑解析成结构化预测。
    否则：返回空预测。
    """
    t0 = time.time()
    fields_pred: Dict[str, Dict[str, Dict[str,str]]] = {}
    models: List[str] = []
    feature_keys: List[str] = []

    if callable(ask_fn):
        ans_text, _ctx = ask_fn(question, use_llm=True)
        fake = "==== Answer ====\n" + ans_text + "\n==== Context(证据句子) ====\n"
        parsed = parse_report_tables_from_answer(fake)
        if parsed:
            _q, fields_pred, models, feature_keys = parsed[0]

    latency = time.time() - t0
    return fields_pred, models, feature_keys, latency

def parse_report_tables_from_answer(answer_block_text: str):
    m_header = re.search(r"^\|\s*字段\s*\|\s*(.*?)\|\s*$", answer_block_text, flags=re.M)
    if not m_header:
        return []
    model_cells = [c.strip() for c in m_header.group(1).split("|")]
    models = [c for c in model_cells if c]
    models = [m.upper() for m in models]

    fields_pred: Dict[str, Dict[str, Dict[str,str]]] = {m: {} for m in models}
    feature_keys: List[str] = []

    for line in answer_block_text.splitlines():
        line = line.strip()
        if not line.startswith("|"):
            continue
        if "字段" in line:
            continue
        if set(line.replace("|", "").strip()) == set("-"):
            continue
        cells = [c.strip() for c in line.strip("|").split("|")]
        if len(cells) < 2:
            continue
        field_title = cells[0]
        vals = cells[1:]
        k = BUILTIN_TITLE2KEY.get(field_title)
        if k is None:
            for f in FEATURES or []:
                if getattr(f, "title", None) == field_title:
                    k = getattr(f, "key", None); break
        if not k:
            continue
        if k not in feature_keys:
            feature_keys.append(k)
        for i, m in enumerate(models):
            v = vals[i] if i < len(vals) else "—"
            if v and v != "—":
                fields_pred[m][k] = {"value": v}
    return [("INLINE", fields_pred, models, feature_keys)]

# ---------------- 金标（仅扫描 CORPUS 中 specs） ----------------
def gold_from_specs(models: List[str], feature_keys: List[str], debug: bool=False) -> Dict[str, Dict[str, Dict[str,str]]]:
    """
    稳健版金标：
      - 直接遍历 CORPUS，过滤出指定 model 的 specs 文档；
      - 用 FALLBACK_GETTERS（值-only）抽取 value/evidence/cite。
    """
    fields_gold: Dict[str, Dict[str, Dict[str,str]]] = {m: {} for m in models}
    models_set = set([m.upper() for m in models])

    specs_pool = []
    for d in CORPUS:
        if (d.get("doc_type","").lower() == "specs") and ((d.get("model") or "").upper() in models_set):
            specs_pool.append(d)

    for k in feature_keys:
        getters = FALLBACK_GETTERS.get(k, [])
        # 若 ask.FEATURES 里有附加 getter，就追加（已在上面合并，这里只是防御）
        # 遍历所有 specs 段，先到先得
        for c in specs_pool:
            model = (c.get("model") or "").upper()
            if k in fields_gold.get(model, {}):
                continue
            text = c.get("text") or ""
            page = c.get("page") or 0
            source = c.get("source") or "Nikon_specs.pdf"
            ok = None
            for getter in getters:
                try:
                    out = getter(text)
                except Exception:
                    out = None
                if out:
                    ok = out
                    break
            if ok:
                vv, ev = ok
                vv = (vv or "").strip(); ev = (ev or "").strip()
                if vv:
                    fields_gold.setdefault(model, {})[k] = {
                        "value": vv,
                        "evidence": ev,
                        "cite": f"{model} | {source} | p.{page}"
                    }

    # 对缺失字段再扫一遍（按型号定向）
    for m in models:
        mm = m.upper()
        for k in feature_keys:
            if fields_gold.get(mm, {}).get(k):
                continue
            getters = FALLBACK_GETTERS.get(k, [])
            for c in specs_pool:
                if (c.get("model","").upper() != mm):
                    continue
                text = c.get("text") or ""
                page = c.get("page") or 0
                source = c.get("source") or "Nikon_specs.pdf"
                ok = None
                for getter in getters:
                    try:
                        out = getter(text)
                    except Exception:
                        out = None
                    if out:
                        ok = out
                        break
                if ok:
                    vv, ev = ok
                    vv = (vv or "").strip(); ev = (ev or "").strip()
                    if vv:
                        fields_gold.setdefault(mm, {})[k] = {
                            "value": vv,
                            "evidence": ev,
                            "cite": f"{mm} | {source} | p.{page}"
                        }
                        break

    if debug:
        for m in models:
            mm = m.upper()
            miss = [k for k in feature_keys if not fields_gold.get(mm, {}).get(k)]
            if miss:
                print(f"[DBG][gold] {mm} 缺失字段: {miss}")
            else:
                print(f"[DBG][gold] {mm} 金标齐全")

    return fields_gold

# ---------------- 指标计算 ----------------
def metrics_for_batch(items: List[Tuple[str, Dict[str, Dict[str, Dict[str,str]]], List[str], List[str], float]],
                      debug: bool=False) -> Tuple[Dict[str,Any], List[Dict[str,Any]]]:
    total_gold = 0
    total_pred_nonempty = 0
    exact_hit = 0
    loose_hit = 0
    per_field_stats: Dict[str, Dict[str,int]] = {}
    errors = []

    for (q, pred_fields, models, feature_keys, latency) in items:
        # 统一大写
        models = [m.upper() for m in models]
        gold = gold_from_specs(models, feature_keys, debug=debug)

        for m in models:
            for k in feature_keys:
                g = (gold.get(m, {}) or {}).get(k, {})
                gval = g.get("value", "")
                if not gval:
                    # 没有金标不计分母
                    continue
                total_gold += 1

                pval = (pred_fields.get(m, {}).get(k, {}) or {}).get("value", "")
                if pval:
                    total_pred_nonempty += 1

                norm_g = normalize_val(gval)
                norm_p = normalize_val(pval)

                ok_exact = (norm_p == norm_g) and bool(norm_p)
                ok_loose = (bool(norm_p) and (norm_p in norm_g or norm_g in norm_p))

                if ok_exact:
                    exact_hit += 1
                if ok_loose:
                    loose_hit += 1

                stat = per_field_stats.setdefault(k, {"gold":0,"pred_nonempty":0,"em":0,"loose":0})
                stat["gold"] += 1
                if pval:
                    stat["pred_nonempty"] += 1
                if ok_exact:
                    stat["em"] += 1
                if ok_loose:
                    stat["loose"] += 1

                if not ok_exact:
                    errors.append({
                        "question": q,
                        "model": m,
                        "field_key": k,
                        "field_title": BUILTIN_KEY2TITLE.get(k, k),
                        "pred_value": pval,
                        "gold_value": gval
                    })

    def pct(x): return round(100.0 * x, 3)
    cov = (total_pred_nonempty/total_gold) if total_gold else 0.0
    em  = (exact_hit/total_gold) if total_gold else 0.0
    loose = (loose_hit/total_gold) if total_gold else 0.0

    metrics = {
        "summary": {
            "total_gold_fields": total_gold,
            "predicted_nonempty": total_pred_nonempty,
            "coverage_pct": pct(cov),
            "EM_pct": pct(em),
            "loose_match_pct": pct(loose),
        },
        "per_field": {
            BUILTIN_KEY2TITLE.get(k, k): {
                "gold": v["gold"],
                "pred_nonempty": v["pred_nonempty"],
                "coverage_pct": pct(v["pred_nonempty"]/v["gold"] if v["gold"] else 0.0),
                "EM_pct": pct(v["em"]/v["gold"] if v["gold"] else 0.0),
                "loose_match_pct": pct(v["loose"]/v["gold"] if v["gold"] else 0.0),
            }
            for k, v in per_field_stats.items()
        }
    }
    return metrics, errors

# ---------------- 主程序 ----------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=str, default=None, help="问题列表文件（每行一问）")
    parser.add_argument("--report", type=str, default=None, help="从批量报告解析评测，不重新调用 LLM")
    parser.add_argument("--debug", action="store_true", help="打印调试信息")
    args = parser.parse_args()

    if args.debug:
        print(f"[DBG] ask.py = {ASK_PATH}")
        print(f"[DBG] CORPUS size = {len(CORPUS)}")
        try:
            from collections import Counter
            print(f"[DBG] doc_type 分布 = {Counter([(d.get('doc_type','') or '').lower() for d in CORPUS])}")
        except Exception:
            pass
        print(f"[DBG] FEATURES size = {len(FEATURES) if FEATURES else 0}")

    run_id = now_str()
    metrics_path = os.path.join(ROOT, "outputs", f"eval_metrics_{run_id}.json")
    errors_path  = os.path.join(ROOT, "outputs", f"eval_errors_{run_id}.csv")
    ensure_dir(metrics_path); ensure_dir(errors_path)

    items: List[Tuple[str, Dict[str, Dict[str, Dict[str,str]]], List[str], List[str], float]] = []

    if args.report:
        report_path = args.report
        if not os.path.exists(report_path):
            print(f"[FATAL] 报告不存在：{report_path}", file=sys.stderr)
            sys.exit(2)
        parsed = parse_report_tables(report_path, debug=args.debug)
        if not parsed:
            print(f"[ERR] 未从报告解析到表格：{report_path}", file=sys.stderr)
            sys.exit(2)
        print(f"[OK] 从报告解析到 {len(parsed)} 个问题。")
        for (q, fields_pred, models, feature_keys) in parsed:
            items.append((q, fields_pred, models, feature_keys, 0.0))
    else:
        # 直接 LLM+RAG（只有在 ask.ask 存在时可用）
        if args.questions:
            with open(args.questions, "r", encoding="utf-8") as f:
                questions = [ln.strip() for ln in f if ln.strip()]
        else:
            questions = list(DEFAULT_TESTS) or ["Z8 与 Z9 的像素与连拍差异？"]

        print(f"[RUN] 直接评测 {len(questions)} 个问题（调用 ask.ask / LLM+RAG）...")
        for i, q in enumerate(questions, 1):
            fields_pred, models, feature_keys, latency = predict_structured(q)
            items.append((q, fields_pred, models, feature_keys, latency))
            print(f"[{i}/{len(questions)}] OK  {latency:.2f}s  {q}")

    # 计算指标
    metrics, errors = metrics_for_batch(items, debug=args.debug)

    # 写文件
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(errors_path, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["question","model","field_key","field_title","pred_value","gold_value"])
        w.writeheader()
        for row in errors:
            w.writerow(row)

    print(f"[OK] 指标写入: {metrics_path}")
    print(f"[OK] 错误明细写入: {errors_path}")
    s = metrics["summary"]
    print("\n=== Summary ===")
    print(f"Gold 字段数: {s['total_gold_fields']}")
    print(f"有预测值字段数: {s['predicted_nonempty']}")
    print(f"Coverage: {s['coverage_pct']}%")
    print(f"Exact Match: {s['EM_pct']}%")
    print(f"Loose Match: {s['loose_match_pct']}%")

if __name__ == "__main__":
    main()
