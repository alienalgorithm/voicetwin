# -*- coding: utf-8 -*-
"""V5 文字处理实测：老师的母本（voicetwin/data/lexicon/core_corpus.tsv）每一句，用官方 V5 整合包里真的文字处理程序
变成音素，看训练时、生成时各变成了什么，哪里不一样。

量的是这几条路（全部用整合包自己的代码、词典和模型：G2PW、chinese-roberta、nltk 数据、fast_langdetect）：
1. 训练，官方的方法：GPT_SoVITS/prepare_datasets/1-get-text.py（整句 clean_text(文字, "zh")），真的跑一遍；
2. 训练，VoiceTwin 的「中英文一起」：voicetwin/backends/gsv_scripts/get_text_mixed.py，真的跑一遍（连同 BERT 特征文件）；
   另外在这个进程里用同样的函数（LangSegmenter + gsv_mixed_text.merge_segments + clean_text）再算一遍，拿到每一段、
   每个字、每个英文词的结果，和上面跑出来的核对；
3. 生成：VoiceTwin 怎么发（voicetwin.synth.script.parse_script 切句 → gsv_english.engine_text → send_lang，
   gptsovits.py 的 _payload：text_split_method = cut0），api_v2 收到以后 TTS.run 做的文字处理就是
   TextPreprocessor.preprocess(文字, text_lang, "cut0", 版本)（pre_seg_text → get_phones_and_bert），这里直接调用它，
   BERT 用真的 chinese-roberta；text_lang 除了 VoiceTwin 发的 zh，也量 all_zh 和 auto 做比较；
4. 英文读音预检（gsv_english.engine_text / precheck / find）：预检说会读错的，引擎实际是不是这样读；自动改的那几处，
   改之前、改之后真实的音素是不是按想的那样变了（母本 + 几句造的例句）。

每个英文词怎么读的：照 english.py 的 en_G2p.__call__ 的分支一步一步记下来（读音都用它自己的 cmu / qryword / predict），
接起来以后必须和 clean_text 真正的结果一模一样，不一样就记成「对不上」，不拿猜的结果充数。

用法：
  python run_text_check.py --gsv <整合包目录> --repo <voicetwin 仓库> --out <结果文件夹> --work <临时文件夹> [--limit N]
  python run_text_check.py --selftest     # 只检查这个脚本自己的比较、统计、写报告的函数（不需要整合包）
结果：<out>/结果.md（中文小结）、<out>/结果.json、<out>/逐句.csv。
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import difflib
import hashlib
import importlib
import importlib.util
import io
import json
import os
import platform
import re
import subprocess
import sys
import time
import traceback
import types
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

# ============================================================================ 常量
#: 音素里表示停顿的符号（text/symbols.py 的 punctuation + pu_symbols）
PAUSES = frozenset([",", ".", "!", "?", "…", "-", "SP", "SP2", "SP3"])
LATIN_RE = re.compile(r"[A-Za-z]")
CJK_RE = re.compile(r"[㐀-䶿一-鿿豈-﫿]")
#: 和 gsv_english._WORD_RE 一样（预检按这个找词，这里也按这个，才能对上）
WORD_RE = re.compile(r"[A-Za-z][A-Za-z'’]*")
DIGIT_RE = re.compile(r"\d")
CLOSERS = "\"'”’)）」』"
#: 输入里的标点按停顿的种类归类（chinese2.rep_map 和英文规范化以后都是这几种）
PCLASS: Dict[str, str] = {}
for _chars, _cls in ((("，", ",", "、", "；", ";", "：", ":", "·", "/"), ","), (("。", "."), "."), (("！", "!"), "!"),
                     (("？", "?"), "?"), (("…", "~", "～"), "…"), (("-", "—", "–"), "-"),
                     (("￥", "^"), "SP")):
    for _c in _chars:
        PCLASS[_c] = _cls
#: english.g2p 去掉的符号（" " 是词和词之间的分隔，不算去掉）
EN_DROP = (" ", "<pad>", "UW", "</s>", "<s>")
SEV_ORDER = ["same", "edge_punct", "mid_punct", "reading", "content"]
SEV_ZH = {"same": "一样", "edge_punct": "只差开头或结尾的停顿符号", "mid_punct": "句子中间的停顿不一样",
          "reading": "同样的字、词读音不一样", "content": "字、词本身不一样"}
STATUS_ZH = {
    "ok": "正常（词典里有）", "homograph": "多音词（按词性选读音）", "name": "人名词典里有", "letter": "单个字母",
    "letter_A": "单独的大写 A（读成字母 A）", "spelled_caps": "全大写，一个字母一个字母地读",
    "spelled_oov": "词典里没有、不超过 3 个字母，按字母读", "split": "大小写混着，拆开读",
    "segmented": "词典里没有，拆成几个词读", "guessed": "词典里没有，神经网络猜读音",
    "changed": "文字被改了（例如弯引号被删掉）", "missing": "在引擎的结果里找不到（被丢掉，或者和别的词合并了）",
    "removed": "被英文的规范化整个删掉", "dropped_in_zh": "被分到中文段里，中文的处理把它删掉了",
    "kept_in_zh": "被分到中文段里（中文的处理没删）", "lost_in_segmenter": "中英分段时丢了", "error": "处理出错",
}
#: 「读得可能不对」的情况（统计、列表、和预检核对时用）
BAD_STATUS = ("spelled_caps", "spelled_oov", "split", "segmented", "guessed", "changed", "missing", "removed",
              "dropped_in_zh", "lost_in_segmenter", "error")
#: 预检的 kind → 实测应该是哪种情况
EXPECT = {"caps": {"spelled_caps"}, "acronym": {"spelled_caps"}, "split": {"split"}, "letter_a": {"letter_A"},
          "short": {"spelled_oov"}, "guess": {"guessed", "segmented"}, "fixed_reading": {"ok"},
          "swapped": {"homograph"}, "read": {"homograph"}}
#: 自动改的写法，核对用的造句（来自 tests/test_english_precheck.py 的「必须改」那一组，再加几句预检只提醒的）
PROBES = [
    "However, the data show otherwise; we must act.", "Note: this is important.", "翻译成 the answer：答案",
    "the sentence is as follows:", "You can call him Dr.Banner or Hulk.", "叫他Dr. Banner",
    "Mr. Smith and Mrs. Lee met Ms. Wang.", "He is No. 1 in the class.", "the U.S. government", "He moved to the U.S.",
    "at 9 a.m. every day", "Tom vs. Jerry", "in the 1990s", "the 1900s and the 2000s", "the 2010s",
    "back in the ’90s", "I’m sure it’s fine.", "don’t go", "AND的第二种连接方式", "live本身属于是不及物动词",
    "A good student works hard.", "I object to this plan.", "Please read it.", "Wait 10s please.",
]


# ============================================================================ 小工具（不需要整合包）
@contextlib.contextmanager
def quiet():
    """官方代码打印很多东西（每句话的切分、进度条），收起来，日志只留这个脚本自己的。"""
    buf_out, buf_err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        yield buf_out


def has_cjk(text: str) -> bool:
    return bool(CJK_RE.search(text or ""))


def short(text: Any, n: int = 80) -> str:
    s = str(text)
    return s if len(s) <= n else s[: n - 1] + "…"


def find_seq(hay: Sequence[str], needle: Sequence[str], start: int = 0) -> Optional[int]:
    n = len(needle)
    if n == 0:
        return None
    for i in range(max(0, start), len(hay) - n + 1):
        if list(hay[i:i + n]) == list(needle):
            return i
    return None


def contains_seq(hay: Sequence[str], needle: Sequence[str]) -> bool:
    return bool(needle) and find_seq(list(hay), list(needle)) is not None


def pclass(ch: str) -> str:
    return PCLASS.get(ch, ch)


def input_marks(text: str) -> List[Tuple[int, str]]:
    """一段文字里的标点（位置, 字符）：不算字母、数字、汉字、空格、%（会读成「百分之」/ percent），
    也不算英文词中间的 ' ’ -（I'm、well-known）。"""
    out = []
    for i, ch in enumerate(text or ""):
        if ch.isalnum() or ch.isspace() or ch == "%":
            continue
        if (ch in "'’-" and 0 < i < len(text) - 1 and LATIN_RE.match(text[i - 1])
                and LATIN_RE.match(text[i + 1])):
            continue
        out.append((i, ch))
    return out


def is_pause_unit(u: Dict[str, Any]) -> bool:
    return bool(u.get("ph")) and all(p in PAUSES for p in u["ph"])


def mark_of(u: Dict[str, Any]) -> str:
    p = u["ph"][0]
    return "SP" if p.startswith("SP") else p


def align_marks(in_chars: Sequence[str], out_syms: Sequence[str]) -> Tuple[List[str], List[str]]:
    """输入的标点和音素里的停顿符号按顺序对齐：每个输入标点是 kept / dropped / changed:<换成的>；另外返回多出来的。"""
    a = [pclass(c) for c in in_chars]
    b = list(out_syms)
    status = ["dropped"] * len(a)
    added: List[str] = []
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            for i in range(i1, i2):
                status[i] = "kept"
        elif tag == "replace":
            n = min(i2 - i1, j2 - j1)
            for k in range(n):
                status[i1 + k] = "changed:" + b[j1 + k]
            added += b[j1 + n:j2]
        elif tag == "insert":
            added += b[j1:j2]
    return status, added


def after_english(text: str, i: int, prev: str = "") -> bool:
    """text[i] 前面（跳过空格、右引号、右括号；到这一段开头时接着看上一段）是不是英文字母。"""
    s = (prev or "") + (text or "")[:i]
    j = len(s) - 1
    while j >= 0 and (s[j].isspace() or s[j] in CLOSERS):
        j -= 1
    return j >= 0 and bool(LATIN_RE.match(s[j]))


def zh_units(norm: str, phones: Sequence[str], word2ph: Optional[Sequence[int]], lang: str = "zh"
             ) -> Tuple[List[Dict[str, Any]], bool]:
    """中文段：规范化以后的每个字（包括标点）和它的音素（word2ph 是每个字几个音素）。"""
    phones = list(phones)
    if word2ph is None or len(word2ph) != len(norm) or sum(int(x) for x in word2ph) != len(phones):
        return [{"t": norm, "ph": phones, "lang": lang, "path": "whole"}], False
    units, k = [], 0
    for ch, n in zip(norm, word2ph):
        units.append({"t": ch, "ph": phones[k:k + int(n)], "lang": lang})
        k += int(n)
    return units, True


def qry_path(o_word: str, g: Any, segment: Callable[[str], List[str]]) -> str:
    """en_G2p.qryword 会走哪条路（照 english.py 的判断顺序，不改任何东西）。"""
    word = o_word.lower()
    if len(word) > 1 and word in g.cmu:
        return "dict"
    if o_word.istitle() and word in g.namedict:
        return "name"
    if len(word) <= 3:
        return "letters"
    if re.match(r"^([a-z]+)('s)$", word):
        return "possessive(" + qry_path(word[:-2], g, segment) + ")"
    comps = segment(word.lower())
    if len(comps) == 1:
        return "predict"
    return "segment(" + "+".join(f"{c}:{qry_path(c, g, segment)}" for c in comps) + ")"


def trace_english(tagged: Sequence[Tuple[str, str]], g: Any, segment: Callable[[str], List[str]],
                  symbols: Any) -> List[Dict[str, Any]]:
    """english.py 的 en_G2p.__call__ + english.g2p 对每个词做了什么（分支照抄，读音用 g 自己的 cmu / qryword）。
    tagged = pos_tag(word_tokenize(规范化以后的文字))。"""
    units = []
    for o_word, pos in tagged:
        word = o_word.lower()
        alt = None
        if re.search("[a-z]", word) is None:
            pron, path = [word], "symbol"
        elif len(word) == 1:
            pron, path = (["EY1"], "letter_A") if o_word == "A" else (g.cmu[word][0], "letter")
        elif word in g.homograph2features:
            pron1, pron2, pos1 = g.homograph2features[word]
            first = pos.startswith(pos1) or (len(pos) < len(pos1) and pos == pos1[: len(pos)])
            pron, path = (pron1, "homograph1") if first else (pron2, "homograph2")
            alt = pron2 if first else pron1
        else:
            pron, path = g.qryword(o_word), qry_path(o_word, g, segment)
        kept = [("UNK" if p == "<unk>" else p) for p in pron if p not in EN_DROP]
        removed = [p for p in pron if p in EN_DROP and p != " "]
        final = []
        for p in kept:  # english.replace_phs
            if p in symbols:
                final.append(p)
            elif p == "'":
                final.append("-")
            else:
                removed.append(p)
        u = {"t": o_word, "pos": pos, "path": path, "ph": final, "lang": "en"}
        if removed:
            u["removed"] = removed
        if alt is not None:
            u["alt"] = list(alt)
        units.append(u)
    return units


def finish_english(units: List[Dict[str, Any]], real: Sequence[str], symbols: Any, norm: str
                   ) -> Tuple[List[Dict[str, Any]], bool]:
    """cleaner.clean_text 对英文段的最后两步（不到 4 个音素前面加「,」；不认识的符号变成 UNK），然后和真正的结果核对。"""
    out = [dict(u) for u in units]
    if sum(len(u["ph"]) for u in out) < 4:
        out = [{"t": "", "ph": [","], "lang": "en", "path": "short_prefix", "added": True}] + out
    for u in out:
        u["ph"] = ["UNK" if p not in symbols else p for p in u["ph"]]
    if [p for u in out for p in u["ph"]] != list(real):
        return [{"t": norm, "ph": list(real), "lang": "en", "path": "whole"}], False
    return out, True


def classify_word(w: str, iso: Sequence[str], hit: Sequence[Dict[str, Any]]) -> str:
    if len(iso) > 1:
        return "spelled_caps" if all(len(t) == 1 for t in iso) else "split"
    if iso[0].lower() != w.lower():
        return "changed"
    p = str(hit[0].get("path", "")) if hit else ""
    if p == "letters":
        return "spelled_oov"
    if p == "predict":
        return "guessed"
    if p.startswith("segment"):
        return "segmented"
    if p.startswith("possessive"):
        inner = p[len("possessive("):-1]
        return "ok" if inner in ("dict", "name") else ("spelled_oov" if inner == "letters" else "guessed")
    if p == "name":
        return "name"
    if p.startswith("homograph"):
        return "homograph"
    if p in ("letter_A",):
        return "letter_A"
    if p == "letter":
        return "letter"
    return "ok"


def word_reports(seg_text: str, units: Sequence[Dict[str, Any]], iso_tokens: Callable[[str], List[str]]
                 ) -> List[Dict[str, Any]]:
    """英文段里原来的每个词，在引擎的结果里变成了什么。iso_tokens(词)：这个词单独经过英文规范化 + 分词以后的样子。"""
    toks = [u for u in units if LATIN_RE.search(str(u.get("t") or "")) and u.get("path") != "short_prefix"]
    # 分词器把 well-known、neither-nor 留成一个词，原文按 - 拆成两个：都按 - 拆开来对（记下属于哪个词）
    subs: List[Tuple[str, int]] = [(piece.lower(), k) for k, u in enumerate(toks)
                                   for piece in str(u["t"]).split("-") if LATIN_RE.search(piece)]
    low = [s for s, _k in subs]
    cursor = 0
    out = []
    for m in WORD_RE.finditer(seg_text or ""):
        w = m.group(0).rstrip("'’")
        iso = [p for t in iso_tokens(w) for p in t.split("-") if LATIN_RE.search(p)]
        rec: Dict[str, Any] = {"w": w}
        if not iso:
            rec["status"] = "removed"
            out.append(rec)
            continue
        pos = find_seq(low, [t.lower() for t in iso], cursor)
        if pos is None:
            rec.update(status="missing", iso=list(iso))
            out.append(rec)
            continue
        idx = list(dict.fromkeys(k for _s, k in subs[pos:pos + len(iso)]))
        hit = [toks[k] for k in idx]
        cursor = pos + len(iso)
        rec["tokens"] = [str(u["t"]) for u in hit]
        rec["paths"] = [str(u.get("path", "")) for u in hit]
        rec["ph"] = " ".join(p for u in hit for p in u["ph"])
        if any(u.get("alt") for u in hit):
            rec["alt"] = " ".join(" ".join(u.get("alt") or []) for u in hit)
        if len(hit) == 1 and len(iso) == 1 and str(hit[0]["t"]).lower() != iso[0].lower():
            rec["part_of"] = str(hit[0]["t"])
        rec["status"] = classify_word(w, iso, hit)
        out.append(rec)
    return out


def segment_loss(text: str, seg_texts: Sequence[str]) -> List[str]:
    """中英分段以后接起来和原文不一样：丢掉的字。"""
    joined = "".join(seg_texts)
    if joined == text:
        return []
    lost = []
    for tag, i1, i2, _j1, _j2 in difflib.SequenceMatcher(None, text, joined, autojunk=False).get_opcodes():
        if tag in ("delete", "replace"):
            lost.append(text[i1:i2])
    return lost


def unit_key(u: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    return ("" if is_pause_unit(u) else str(u.get("t") or "").lower(), tuple(u.get("ph") or ()))


def show_units(units: Sequence[Dict[str, Any]], n: int = 120) -> str:
    return short(" ".join(f"{u.get('t') or '∅'}[{' '.join(u.get('ph') or [])}]" for u in units), n) or "（没有）"


def units_text(units: Sequence[Dict[str, Any]]) -> str:
    out = ""
    for u in units:
        t = str(u.get("t") or "")
        if out and LATIN_RE.match(t[:1]) and (LATIN_RE.match(out[-1:]) or out[-1:].isdigit()):
            out += " "
        out += t
    return out


def diff_units(a: Sequence[Dict[str, Any]], b: Sequence[Dict[str, Any]]) -> Tuple[str, List[Dict[str, str]]]:
    """两串单位（字、英文词、停顿）哪里不一样。返回 (最严重的一种, 每一处)。"""
    ka = [unit_key(u) for u in a]
    kb = [unit_key(u) for u in b]
    kinds, details = [], []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ka, kb, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        # 一样长的替换一对一对地看（「行」读音不一样、句末「，」换成「。」是两处不同的差别）
        spans = ([(i1 + k, i1 + k + 1, j1 + k, j1 + k + 1) for k in range(i2 - i1)]
                 if tag == "replace" and i2 - i1 == j2 - j1 else [(i1, i2, j1, j2)])
        for s1, s2, t1, t2 in spans:
            ua, ub = list(a[s1:s2]), list(b[t1:t2])
            if all(is_pause_unit(u) or not u.get("ph") for u in ua + ub):
                edge = (s1 == 0 and t1 == 0) or (s2 == len(a) and t2 == len(b))
                kind = "edge_punct" if edge else "mid_punct"
            elif (len(ua) == len(ub)
                  and all(str(x.get("t") or "").lower() == str(y.get("t") or "").lower() for x, y in zip(ua, ub))):
                kind = "reading"
            else:
                kind = "content"
            kinds.append(kind)
            details.append({"kind": kind, "a": show_units(ua), "b": show_units(ub),
                            "ctx": short(units_text(a[max(0, s1 - 6):s2 + 6]), 60)})
    sev = max(kinds, key=SEV_ORDER.index) if kinds else "same"
    return sev, details


def seg_punct(segs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """每一段输入里的标点到了音素里变成了什么。"""
    rows: List[Dict[str, Any]] = []
    prev = ""
    for seg in segs:
        t = str(seg.get("text") or "")
        marks = input_marks(t)
        if seg.get("error") or not seg.get("units"):
            status = ["error"] * len(marks)
        else:
            outs = [mark_of(u) for u in seg["units"] if is_pause_unit(u) and not u.get("added")]
            status, _added = align_marks([c for _i, c in marks], outs)
        for (i, ch), st in zip(marks, status):
            full = prev + t
            k = len(prev) + i
            rows.append({"ch": ch, "lang": seg.get("lang", ""), "after_en": after_english(t, i, prev), "status": st,
                         "ctx": full[max(0, k - 12):k + 2]})
        prev = t
    return rows


def numbers_of(segs: Sequence[Dict[str, Any]]) -> List[Dict[str, str]]:
    return [{"lang": s.get("lang", ""), "text": str(s.get("text") or ""), "norm": str(s.get("norm") or "")}
            for s in segs if DIGIT_RE.search(str(s.get("text") or ""))]


def seg_words(segs: Sequence[Dict[str, Any]], full_text: str, iso_tokens: Callable[[str], List[str]]
              ) -> List[Dict[str, Any]]:
    """一句话（几段）里每个英文词的情况：英文段按 word_reports；中文段里的英文字母会被 chinese2 删掉；分段时丢的另算。"""
    out: List[Dict[str, Any]] = []
    for s in segs:
        t = str(s.get("text") or "")
        if s.get("error"):
            out += [{"w": m.group(0).rstrip("'’"), "status": "error"} for m in WORD_RE.finditer(t)]
        elif s.get("lang") == "en":
            out += word_reports(t, s.get("units") or [], iso_tokens)
        else:
            kept = LATIN_RE.search(str(s.get("norm") or ""))
            out += [{"w": m.group(0).rstrip("'’"), "status": "kept_in_zh" if kept else "dropped_in_zh"}
                    for m in WORD_RE.finditer(t)]
    have = Counter(r["w"] for r in out)
    for m in WORD_RE.finditer(full_text or ""):
        w = m.group(0).rstrip("'’")
        if have[w] > 0:
            have[w] -= 1
        else:
            out.append({"w": w, "status": "lost_in_segmenter"})
    return out


def raw_punct_dropped(raw_segs: Sequence[Dict[str, Any]]) -> int:
    """改之前：英文后面的 ; : ；：有几个在音素里没有停顿（被删掉了）。"""
    return sum(1 for p in seg_punct(raw_segs) if p["ch"] in ";:；：" and p["after_en"] and p["status"] == "dropped")


def check_fix_request(fixes: Sequence[Dict[str, str]], raw_units: Sequence[Dict[str, Any]],
                      fix_units: Sequence[Dict[str, Any]], cmu: Dict[str, Any], raw_dropped: Optional[int] = None
                      ) -> List[Dict[str, Any]]:
    """自动改的每一处，真实的音素是不是按想的那样变了。raw_dropped：改之前英文后面的 ; : 有几个被删掉了
    （None = 不知道，按每一处都被删掉算）。"""
    raw_ph = [p for u in raw_units for p in u["ph"]]
    fix_ph = [p for u in fix_units for p in u["ph"]]

    def pron(w: str) -> List[str]:
        v = cmu.get(w)
        return list(v[0]) if v else []

    def n_sym(ph: Sequence[str], syms: Iterable[str]) -> int:
        s = set(syms)
        return sum(1 for p in ph if p in s)

    out = []
    n_punct = sum(1 for f in fixes if f["kind"] == "punct")
    if n_punct:
        delta = n_sym(fix_ph, (",", ".")) - n_sym(raw_ph, (",", "."))
        need = n_punct if raw_dropped is None else min(n_punct, raw_dropped)
        expect = (f"改之前有 {need} 个被删掉、没有停顿，改了以后多出 {need} 个停顿符号" if need
                  else "改之前这几个标点已经有停顿（分到了中文段里，中文的处理换成逗号），改了以后停顿不能少")
        out.append({"fix": f"英文里的 ; :（{n_punct} 处）", "expect": expect,
                    "ok": delta >= need, "measured": f"停顿符号多了 {delta} 个"})
    for f in fixes:
        kind, before, after = f["kind"], f["before"], f["after"]
        if kind == "punct":
            continue
        head = (re.match(r"[A-Za-z]+", before) or re.match("", "")).group(0)
        if kind == "abbr" and head == "Dr":
            bad, good = pron("dr"), pron("doctor")
            ok = contains_seq(raw_ph, bad) and contains_seq(fix_ph, good) and not contains_seq(fix_ph, bad)
            expect = f"改之前有 dr 的读音 {' '.join(bad)}，改了以后是 doctor {' '.join(good)}"
        elif kind == "abbr" and head in ("Mr", "Mrs", "Ms"):
            ok = n_sym(fix_ph, ".") < n_sym(raw_ph, ".") and contains_seq(fix_ph, pron(head.lower()))
            expect = "缩写后面的点不再变成句号（句号少了），读音不变"
        elif kind == "abbr" and head.lower() == "no":
            ok = contains_seq(fix_ph, pron("number")) and not contains_seq(fix_ph, pron("no") + ["."])
            expect = f"读成 number {' '.join(pron('number'))}，不再是 no + 句号"
        elif kind == "abbr" and head.lower() == "vs":
            ok = contains_seq(fix_ph, pron("versus"))
            expect = f"读成 versus {' '.join(pron('versus'))}"
        elif kind == "abbr":
            ok = n_sym(fix_ph, ".") < n_sym(raw_ph, ".")
            expect = "字母中间的点不再变成句号（句号少了）"
        elif kind == "decade":
            sec = pron("seconds")
            ok = contains_seq(raw_ph, sec) and not contains_seq(fix_ph, sec)
            expect = f"改之前有 seconds {' '.join(sec)}，改了以后没有"
        elif kind == "apostrophe":
            ok = fix_ph != raw_ph and any("'" in str(u.get("t") or "") for u in fix_units)
            expect = "词中间的 ' 留下来（读音变了）"
        else:
            ok, expect = fix_ph != raw_ph, "读音变了"
        out.append({"fix": f"{before} → {after}", "expect": expect, "ok": bool(ok)})
    return out


def verify_findings(findings: Sequence[Any], words: Sequence[Dict[str, Any]], cmu: Dict[str, Any]
                    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """预检（find）说会读错的每一处，实测是不是这样；实测读得不对劲、预检没提到的词。"""
    by_word: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for r in words:
        by_word[r["w"]].append(r)
    checks = []
    named = set()
    for f in findings:
        named.add(f.word)
        recs = by_word.get(f.word) or []
        if f.kind not in EXPECT:
            checks.append({"kind": f.kind, "word": f.word, "detail": f.detail, "confirmed": None,
                           "real": "；".join(f"{r['status']} {r.get('ph', '')}" for r in recs) or "不是单个词，不核对"})
            continue
        if not recs:
            checks.append({"kind": f.kind, "word": f.word, "detail": f.detail, "confirmed": False, "real": "实测里找不到这个词"})
            continue
        r = recs[0]
        ok = r["status"] in EXPECT[f.kind]
        if ok and f.kind == "fixed_reading":
            first = cmu.get(f.word.lower())
            ok = bool(first) and r.get("ph") == " ".join(first[0])
        checks.append({"kind": f.kind, "word": f.word, "detail": f.detail, "confirmed": bool(ok),
                       "real": f"{STATUS_ZH.get(r['status'], r['status'])}：{r.get('ph', '')}"})
    missed = [r for r in words if r["status"] in BAD_STATUS and r["w"] not in named]
    return checks, missed


# ============================================================================ 官方代码（整合包里的）
class Official:
    """整合包里真的文字处理程序：text.cleaner、LangSegmenter、english、TextPreprocessor（不改它们，只记录它们做了什么）。"""

    def __init__(self, gsv: Path, version: str, bert_dir: Path, mixed: Any, with_tp: bool = True):
        self.gsv, self.version = gsv, version
        os.chdir(gsv)  # chinese2 用相对路径找 G2PW（GPT_SoVITS/text/G2PWModel），和 api_v2 一样在整合包根目录运行
        for p in (str(gsv / "GPT_SoVITS"), str(gsv)):
            if p in sys.path:
                sys.path.remove(p)
            sys.path.insert(0, p)
        os.environ["version"] = version
        os.environ["is_half"] = "False"
        os.environ["bert_path"] = str(bert_dir)
        self.mixed = mixed
        self.notes: List[str] = []
        with quiet():
            from text import symbols2  # noqa: WPS433
            from text.cleaner import clean_text
            from text.LangSegmenter import LangSegmenter
            import text.english as E
            import wordsegment
        self.clean_text, self.LangSegmenter, self.E, self.wordsegment = clean_text, LangSegmenter, E, wordsegment
        self.id2sym = list(symbols2.symbols)
        self.sym_set = set(symbols2.symbols)
        self._iso: Dict[str, List[str]] = {}
        self.tp = None
        if with_tp:
            self._load_tp(bert_dir)

    # ------------------------------------------------------------------ 生成时的文字处理（TextPreprocessor）
    def _load_tp(self, bert_dir: Path) -> None:
        from transformers import AutoModelForMaskedLM, AutoTokenizer
        try:
            with quiet():
                from TTS_infer_pack.TextPreprocessor import TextPreprocessor
            self.notes.append("TTS_infer_pack（连同 TTS.py）照常 import，TextPreprocessor 用的就是 api_v2 用的那一份")
        except Exception as exc:  # noqa: BLE001 - TTS.py 缺东西也不影响文字处理：不经过包的 __init__ 直接加载
            for k in [k for k in sys.modules if k == "TTS_infer_pack" or k.startswith("TTS_infer_pack.")]:
                del sys.modules[k]
            pkg = types.ModuleType("TTS_infer_pack")
            pkg.__path__ = [str(self.gsv / "GPT_SoVITS" / "TTS_infer_pack")]
            sys.modules["TTS_infer_pack"] = pkg
            with quiet():
                TextPreprocessor = importlib.import_module("TTS_infer_pack.TextPreprocessor").TextPreprocessor
            self.notes.append(f"TTS_infer_pack 的 __init__（TTS.py）没能 import（{type(exc).__name__}: {short(exc, 200)}），"
                              "TextPreprocessor.py 单独加载（文字处理的代码一样）")
        tok = AutoTokenizer.from_pretrained(str(bert_dir))
        model = AutoModelForMaskedLM.from_pretrained(str(bert_dir))
        model = model.eval() if hasattr(model, "eval") else model
        tp = TextPreprocessor(model, tok, "cpu")
        self.tp = tp
        self._pieces: List[Dict[str, Any]] = []
        self._preseg: Optional[List[str]] = None
        orig_seg, orig_gpb = tp.segment_and_extract_feature_for_text, tp.get_phones_and_bert
        orig_cti, orig_pre = tp.clean_text_inf, tp.pre_seg_text

        def seg_feat(text, language, version="v1"):
            self._pieces.append({"text": text, "lang": language, "groups": []})
            return orig_seg(text, language, version)

        def gpb(text, language, version, final=False):
            if not self._pieces:
                self._pieces.append({"text": text, "lang": language, "groups": []})
            self._pieces[-1]["groups"].append({"text": text, "final": final, "calls": []})
            return orig_gpb(text, language, version, final)

        def cti(text, language, version="v2"):
            phones, word2ph, norm = orig_cti(text, language, version)
            self._pieces[-1]["groups"][-1]["calls"].append((text, language, list(phones), word2ph, norm))
            return phones, word2ph, norm

        def pre(text, lang, method):
            out = orig_pre(text, lang, method)
            self._preseg = list(out)
            return out

        # 实例上的属性盖过类里的方法：官方代码里 self.xxx(...) 的调用都会经过这里（只记录，不改参数和结果）
        tp.segment_and_extract_feature_for_text = seg_feat
        tp.get_phones_and_bert = gpb
        tp.clean_text_inf = cti
        tp.pre_seg_text = pre

    def iso_tokens(self, w: str) -> List[str]:
        if w not in self._iso:
            with quiet():
                self._iso[w] = list(self.E.word_tokenize(self.E.text_normalize(w)))
        return self._iso[w]

    def units(self, lang: str, norm: str, phones: List[str], word2ph: Any) -> Tuple[List[Dict[str, Any]], bool]:
        if lang in ("zh", "yue"):
            return zh_units(norm, phones, word2ph, lang)
        if lang == "en":
            with quiet():
                tagged = self.E.pos_tag(self.E.word_tokenize(norm))
                raw = trace_english(tagged, self.E._g2p, self.wordsegment.segment, self.E.symbols)
            return finish_english(raw, phones, self.sym_set, norm)
        return [{"t": norm, "ph": list(phones), "lang": lang, "path": "whole"}], True

    def clean(self, text: str, lang: str) -> Dict[str, Any]:
        seg: Dict[str, Any] = {"lang": lang, "text": text}
        try:
            with quiet():
                phones, word2ph, norm = self.clean_text(text, lang, self.version)
        except Exception as exc:  # noqa: BLE001
            seg.update(error=f"{type(exc).__name__}: {short(exc, 200)}", units=[], phones=[], norm="")
            return seg
        seg.update(phones=list(phones), norm=norm, word2ph=list(word2ph) if word2ph is not None else None)
        seg["units"], seg["trace_ok"] = self.units(lang, norm, list(phones), word2ph)
        return seg

    def mixed_train(self, text: str, lang: str) -> Tuple[List[Dict[str, Any]], str]:
        """get_text_mixed.py 对一行训练素材做的（同样的函数、同样的顺序），每一段的结果。"""
        text = re.sub(" {2,}", " ", text)
        if hasattr(self.mixed, "english_commas"):   # 和 get_text_mixed.py 一样：英文后面的「，」「、」→「,」
            text = self.mixed.english_commas(text)
        if lang == "zh" or (lang == "en" and has_cjk(text)):
            with quiet():
                raw = self.LangSegmenter.getTexts(text)
            segs = []
            for seg_text, seg_lang in self.mixed.merge_segments(raw):
                s = self.clean(seg_text.replace("%", "-").replace("￥", ","), seg_lang)
                s["orig"] = seg_text
                segs.append(s)
            return segs, text
        s = self.clean(text.replace("%", "-").replace("￥", ","), lang)
        s["orig"] = text
        return [s], text

    def infer(self, text: str, lang: str) -> Dict[str, Any]:
        """api_v2 → TTS.run → TextPreprocessor.preprocess(text, text_lang, "cut0", 版本)：每一块、每一段的结果。"""
        assert self.tp is not None
        self._pieces, self._preseg = [], None
        res: Dict[str, Any] = {"text": text, "lang": lang}
        try:
            with quiet():
                out = self.tp.preprocess(text, lang, "cut0", self.version)
        except Exception as exc:  # noqa: BLE001
            res.update(error=f"{type(exc).__name__}: {short(exc, 200)}", phones=[], pieces=[], units=[])
            return res
        phones = [self.id2sym[int(i)] for item in out for i in list(item["phones"])]
        pieces = []
        for pc in self._pieces:
            g = pc["groups"][-1] if pc["groups"] else None
            segs = []
            for t, lng, ids, w2p, norm in (g["calls"] if g else []):
                ph = [self.id2sym[int(i)] for i in ids]
                s = {"lang": lng.replace("all_", ""), "text": t, "norm": norm, "phones": ph}
                s["units"], s["trace_ok"] = self.units(s["lang"], norm, ph, w2p)
                segs.append(s)
            pieces.append({"text": pc["text"], "group_text": g["text"] if g else "", "retried": len(pc["groups"]) > 1,
                           "segs": segs, "norm": "".join(s["norm"] for s in segs)})
        kept = [p for p in pieces if p["segs"] and p["norm"] != ""]  # preprocess 跳过 norm_text 是空的
        flat = [p for pc in kept for s in pc["segs"] for p in s["phones"]]
        res.update(phones=phones, pieces=pieces, preseg=self._preseg, consistent=flat == phones,
                   units=[u for pc in kept for s in pc["segs"] for u in s["units"]],
                   segs=[s for pc in kept for s in pc["segs"]])
        return res


# ============================================================================ 训练脚本（子进程，真的跑）
def read_name2text(path: Path) -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8").split("\n"):
        if not line.strip():
            continue
        parts = line.split("\t")
        while len(parts) < 4:
            parts.append("")
        name, phones, w2p, norm = parts[0], parts[1], parts[2], "\t".join(parts[3:])
        out[name] = {"phones": phones.split(" ") if phones else [], "word2ph": w2p, "norm": norm}
    return out


def run_script(cmd: List[str], env: Dict[str, str], cwd: Path, log: Path, timeout: int = 7200) -> Dict[str, Any]:
    t0 = time.time()
    with open(log, "w", encoding="utf-8") as f:
        try:
            r = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout)
            code = r.returncode
        except subprocess.TimeoutExpired:
            code = -9
    text = log.read_text(encoding="utf-8", errors="replace")
    vt = {k: int(v) for k, v in re.findall(r"^VT_(EN_PHONES|EN_LINES)\s+(\d+)", text, re.M)}
    fail = re.findall(r"^VT_FAIL (.*)$", text, re.M)
    tracebacks = text.count("Traceback (most recent call last)")
    return {"cmd": Path(cmd[-1]).name, "returncode": code,
            "seconds": round(time.time() - t0, 1), "vt": vt, "fail": fail, "tracebacks": tracebacks,
            "tail": text.strip().splitlines()[-8:]}


def bert_check(bert_dir: Path, names: Sequence[str], phones_of: Dict[str, Dict[str, Any]],
               spans_of: Dict[str, List[Tuple[str, int]]]) -> Dict[str, Any]:
    """get_text_mixed.py 写的 3-bert/<名字>.pt：长度 == 音素个数；英文段是 0、中文段不是 0。"""
    stat = Counter()
    bad: List[str] = []
    try:
        import torch
    except Exception as exc:  # noqa: BLE001
        return {"error": f"torch 用不了：{exc}"}
    for name in names:
        path = bert_dir / f"{name}.pt"
        spans = spans_of.get(name) or []
        need = any(lang == "zh" for lang, _n in spans)
        if not path.exists():
            stat["missing" if need else "none_needed"] += 1
            if need:
                bad.append(f"{name}：没有 BERT 文件")
            continue
        try:
            feat = torch.load(str(path), map_location="cpu")
            shape = tuple(feat.shape)
            n = len((phones_of.get(name) or {}).get("phones") or [])
            if shape[0] != 1024 or shape[-1] != n:
                stat["shape_bad"] += 1
                bad.append(f"{name}：形状 {shape}，音素 {n} 个")
                continue
            k, zero_ok = 0, True
            for lang, m in spans:
                if m:
                    part = feat[:, k:k + m]
                    is_zero = bool((part == 0).all())
                    if (lang == "en") != is_zero:
                        zero_ok = False
                k += m
            if k != n:
                stat["span_len_bad"] += 1
                bad.append(f"{name}：分段的音素加起来 {k} 个，文件 {n} 个")
            elif zero_ok:
                stat["ok"] += 1
            else:
                stat["zero_bad"] += 1
                bad.append(f"{name}：英文段不是 0，或者中文段是 0")
        except Exception as exc:  # noqa: BLE001
            stat["load_error"] += 1
            bad.append(f"{name}：读不了（{type(exc).__name__}: {short(exc, 100)}）")
    return {"stat": dict(stat), "bad": bad[:20]}


# ============================================================================ 一句一句地量
def load_voicetwin(repo: Path) -> Dict[str, Any]:
    if str(repo) not in sys.path:
        sys.path.append(str(repo))  # 放在最后：不盖过整合包的模块
    from voicetwin.backends import gsv_english as G
    from voicetwin.backends import gsv_mixed_text as M
    from voicetwin.synth.script import parse_script
    from voicetwin.utils.textutil import detect_lang, ensure_final_punct_train, send_lang
    return {"G": G, "M": M, "parse_script": parse_script, "detect_lang": detect_lang, "send_lang": send_lang,
            "ensure_final_punct_train": ensure_final_punct_train}


def read_corpus(path: Path) -> List[Tuple[str, str]]:
    rows = []
    for ln in path.read_text(encoding="utf-8").splitlines():
        if not ln or ln.startswith("#"):
            continue
        parts = ln.split("\t", 1)
        if len(parts) == 2 and parts[1].strip():
            rows.append((parts[0].strip(), parts[1]))
    return rows


def requests_of(vt: Dict[str, Any], text: str) -> List[Dict[str, Any]]:
    """VoiceTwin 生成这句话时一个一个发给 api_v2 的请求（讲稿里就这一句；读音词典是空的）。"""
    G = vt["G"]
    out = []
    for s in vt["parse_script"](text, lexicon=()):
        new, fixes = G.rewrite(s.text)
        out.append({"display": s.display, "raw": s.text, "sent": new, "lang": vt["send_lang"](new, s.lang),
                    "raw_lang": vt["send_lang"](s.text, s.lang),
                    "fixes": [{"kind": f.kind, "before": f.before, "after": f.after} for f in fixes]})
    return out


def compact_segs(segs: Sequence[Dict[str, Any]]) -> List[List[str]]:
    return [[s.get("lang", ""), str(s.get("text") or ""), str(s.get("norm") or ""), s.get("error", "")] for s in segs]


def analyse_sentence(off: Official, vt: Dict[str, Any], sid: str, text: str, sub: Dict[str, Dict[str, Any]],
                     dic: Any) -> Dict[str, Any]:
    G = vt["G"]
    lang0 = vt["detect_lang"](text)
    train_lang = vt["send_lang"](text, lang0)
    train_text = vt["ensure_final_punct_train"](text.replace("|", " "), lang0)
    name = f"{sid}.wav"
    rec: Dict[str, Any] = {"id": sid, "text": text, "train_lang": train_lang, "train_text": train_text,
                           "n_words": len(WORD_RE.findall(text))}
    # ---- 训练：官方
    o = sub["official"].get(name)
    rec["official"] = {"phones": " ".join(o["phones"]) if o else None, "norm": o["norm"] if o else None,
                       "latin_left": len(LATIN_RE.findall(o["norm"])) if o else None}
    # ---- 训练：中英文一起（子进程的结果 + 这里重算的每一段）
    segs, seg_input = off.mixed_train(train_text, train_lang)
    mine = [p for s in segs for p in s.get("phones", [])]
    m = sub["mixed"].get(name)
    rec["mixed"] = {"phones": " ".join(m["phones"]) if m else None, "norm": m["norm"] if m else None,
                    "same_as_here": (m["phones"] == mine) if m else None,
                    "trace_ok": all(s.get("trace_ok", True) for s in segs), "errors": [s["error"] for s in segs if s.get("error")]}
    rec["train_segs"] = compact_segs(segs)
    rec["train_spans"] = [("zh" if s.get("lang") == "zh" else "en", len(s.get("phones", []))) for s in segs]
    rec["train_seg_lost"] = segment_loss(seg_input, [s.get("orig", s.get("text", "")) for s in segs])
    train_units = [u for s in segs for u in s.get("units", [])]
    rec["train_words"] = seg_words(segs, seg_input, off.iso_tokens)
    rec["train_punct"] = seg_punct(segs)
    rec["train_numbers"] = numbers_of(segs)
    rec["train_short_prefix"] = sum(1 for u in train_units if u.get("path") == "short_prefix")
    rec["train_find"] = [(f.kind, f.word) for f in G.find(train_text, dic)]
    checks, missed = verify_findings(G.find(train_text, dic), rec["train_words"], off.E._g2p.cmu)
    rec["train_find_checks"], rec["train_find_missed"] = checks, missed
    # ---- 生成
    reqs = requests_of(vt, text)
    infer_units: List[Dict[str, Any]] = []
    rec["requests"] = []
    for r in reqs:
        inf = off.infer(r["sent"], r["lang"])
        rr: Dict[str, Any] = {"display": r["display"], "raw": r["raw"], "sent": r["sent"], "lang": r["lang"],
                              "fixes": r["fixes"], "error": inf.get("error", ""), "phones": " ".join(inf["phones"]),
                              "consistent": inf.get("consistent"), "preseg": inf.get("preseg"),
                              "segs": compact_segs(inf.get("segs", [])),
                              "retried": any(p["retried"] for p in inf.get("pieces", []))}
        segs_i = inf.get("segs", [])
        rr["trace_ok"] = all(s.get("trace_ok", True) for s in segs_i)
        rr["seg_lost"] = segment_loss(re.sub(" {2,}", " ", (inf.get("preseg") or [""])[0]) if inf.get("preseg") else "",
                                      [s["text"] for s in segs_i]) if len(inf.get("pieces", [])) == 1 else []
        rr["words"] = ([{"w": mm.group(0).rstrip("'’"), "status": "error"} for mm in WORD_RE.finditer(r["sent"])]
                       if inf.get("error") else seg_words(segs_i, " ".join(inf.get("preseg") or [r["sent"]]), off.iso_tokens))
        rr["punct"] = seg_punct(segs_i)
        rr["numbers"] = numbers_of(segs_i)
        rr["short_prefix"] = sum(1 for u in inf.get("units", []) if u.get("path") == "short_prefix")
        pre = inf.get("preseg") or []
        rr["lead_added"] = bool(pre) and pre[0][:1] in "。." and r["sent"][:1] not in "。."
        rr["tail_added"] = bool(pre) and pre[-1][-1:] in "。." and r["sent"].rstrip()[-1:] not in "。."
        finds = G.find(r["sent"], dic)
        rr["find"] = [(f.kind, f.word) for f in finds]
        rr["find_checks"], rr["find_missed"] = verify_findings(finds, rr["words"], off.E._g2p.cmu)
        if r["raw"] != r["sent"]:  # 预检自动改过：改之前的也量，核对改的效果
            raw = off.infer(r["raw"], r["raw_lang"])
            rr["raw_phones"] = " ".join(raw["phones"])
            rr["fix_checks"] = check_fix_request(r["fixes"], raw.get("units", []), inf.get("units", []), off.E._g2p.cmu,
                                                 raw_punct_dropped(raw.get("segs", [])))
        if r["lang"] == "zh":
            rr["modes"] = {}
            for mode in ("all_zh", "auto"):
                alt = off.infer(r["sent"], mode)
                if alt.get("error"):
                    rr["modes"][mode] = {"sev": "error", "error": alt["error"]}
                    continue
                sev, det = diff_units(inf.get("units", []), alt.get("units", []))
                rr["modes"][mode] = {"sev": sev, "details": det[:6], "phones": " ".join(alt["phones"]) if sev != "same" else ""}
        infer_units += inf.get("units", [])
        rec["requests"].append(rr)
    sev, det = diff_units(train_units, infer_units)
    rec["diff"] = {"sev": sev, "details": det[:10]}
    return rec


# ============================================================================ 汇总
def _ex(rows: List[Any], n: int = 8) -> List[Any]:
    return rows[:n]


def summarize(recs: Sequence[Dict[str, Any]], meta: Dict[str, Any], dic_info: Dict[str, Any],
              precheck_json: Dict[str, Any], probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    S: Dict[str, Any] = {"meta": meta, "dictionary": dic_info, "precheck": precheck_json}
    n = len(recs)
    en_recs = [r for r in recs if r["n_words"]]
    S["corpus"] = {"sentences": n, "with_english": len(en_recs), "english_words": sum(r["n_words"] for r in recs),
                   "with_digits": sum(1 for r in recs if DIGIT_RE.search(r["text"]))}
    # ---- 官方 1-get-text.py
    off_out = [r for r in recs if r["official"]["phones"] is not None]
    lost_words = sum(r["n_words"] for r in en_recs if r["train_lang"] == "zh")
    S["official"] = {"sentences_out": len(off_out),
                     "missing": [r["id"] for r in recs if r["official"]["phones"] is None][:30],
                     "en_sentences_zh": sum(1 for r in en_recs if r["train_lang"] == "zh"),
                     "en_words_in_zh_sentences": lost_words,
                     "latin_left_in_norm": sum(r["official"]["latin_left"] or 0 for r in off_out),
                     "no_en_same_as_mixed": sum(1 for r in recs if not r["n_words"] and r["official"]["phones"] is not None
                                                and r["official"]["phones"] == r["mixed"]["phones"]),
                     "no_en_total": sum(1 for r in recs if not r["n_words"]),
                     "no_en_diff": [r["id"] for r in recs if not r["n_words"] and r["official"]["phones"] != r["mixed"]["phones"]][:20]}
    # ---- 中英文一起
    S["mixed"] = {"sentences_out": sum(1 for r in recs if r["mixed"]["phones"] is not None),
                  "same_as_here": sum(1 for r in recs if r["mixed"]["same_as_here"]),
                  "not_same": [r["id"] for r in recs if r["mixed"]["same_as_here"] is False][:20],
                  "trace_bad": [r["id"] for r in recs if not r["mixed"]["trace_ok"]][:20],
                  "seg_errors": [(r["id"], r["mixed"]["errors"]) for r in recs if r["mixed"]["errors"]][:20],
                  "seg_lost": [(r["id"], r["train_seg_lost"]) for r in recs if r["train_seg_lost"]][:20],
                  "short_prefix": sum(r["train_short_prefix"] for r in recs)}
    reqs = [(r, q) for r in recs for q in r["requests"]]
    S["infer"] = {"requests": len(reqs), "errors": [(r["id"], q["sent"], q["error"]) for r, q in reqs if q["error"]][:20],
                  "inconsistent": [(r["id"], q["sent"]) for r, q in reqs if q.get("consistent") is False][:20],
                  "trace_bad": [(r["id"], q["sent"]) for r, q in reqs if not q["trace_ok"]][:20],
                  "seg_lost": [(r["id"], q["seg_lost"]) for r, q in reqs if q["seg_lost"]][:20],
                  "sentences_split": sum(1 for r in recs if len(r["requests"]) > 1),
                  "lead_added": sum(1 for _r, q in reqs if q["lead_added"]),
                  "lead_added_ex": _ex([q["preseg"][0] for _r, q in reqs if q["lead_added"] and q["preseg"]]),
                  "tail_added": sum(1 for _r, q in reqs if q["tail_added"]),
                  "retried": [(r["id"], q["sent"]) for r, q in reqs if q["retried"]][:20],
                  "short_prefix": sum(q["short_prefix"] for _r, q in reqs)}
    # ---- 英文词
    for stage in ("train", "infer"):
        words = ([(r["id"], w) for r in recs for w in r["train_words"]] if stage == "train"
                 else [(r["id"], w) for r, q in reqs for w in q["words"]])
        cnt = Counter(w["status"] for _i, w in words)
        uniq: Dict[str, Dict[str, Dict[str, Any]]] = defaultdict(dict)
        for sid, w in words:
            if w["status"] in ("ok", "letter"):
                continue
            key = f"{w['w']}"
            e = uniq[w["status"]].setdefault(key, {"w": w["w"], "count": 0, "ids": [], "ph": w.get("ph", ""),
                                                  "tokens": w.get("tokens", w.get("iso", [])), "paths": w.get("paths", []),
                                                  "alt": w.get("alt", "")})
            e["count"] += 1
            if sid not in e["ids"] and len(e["ids"]) < 5:
                e["ids"].append(sid)
        S[f"words_{stage}"] = {"count": dict(cnt), "total": len(words),
                               "by_status": {k: sorted(v.values(), key=lambda e: -e["count"]) for k, v in uniq.items()}}
    # ---- 标点
    for stage in ("train", "infer"):
        rows = ([(r["id"], p) for r in recs for p in r["train_punct"]] if stage == "train"
                else [(r["id"], p) for r, q in reqs for p in q["punct"]])
        after: Dict[str, Counter] = defaultdict(Counter)
        other: Dict[str, Counter] = defaultdict(Counter)
        ex: Dict[str, List[str]] = defaultdict(list)
        for sid, p in rows:
            st = "kept" if p["status"] == "kept" else ("changed" if p["status"].startswith("changed") else p["status"])
            book = after if p["after_en"] else other
            key = f"{p['ch']}|{p['lang']}"
            book[key][st] += 1
            book[key]["total"] += 1
            if st != "kept" and len(ex[key + ("|en" if p["after_en"] else "")]) < 4:
                ex[key + ("|en" if p["after_en"] else "")].append(f"{sid}「{p['ctx']}」→ {p['status']}")
        S[f"punct_{stage}"] = {"after_en": {k: dict(v) for k, v in after.items()},
                               "other": {k: dict(v) for k, v in other.items()}, "examples": dict(ex)}
    # ---- 数字
    S["numbers"] = [{"id": r["id"], "text": r["text"], "train": r["train_numbers"],
                     "infer": [x for q in r["requests"] for x in q["numbers"]],
                     "modes": {m: q.get("modes", {}).get(m, {}).get("sev") for q in r["requests"] if q["numbers"]
                               for m in ("all_zh", "auto")}}
                    for r in recs if DIGIT_RE.search(r["text"])]
    # ---- 训练和生成
    sev = Counter(r["diff"]["sev"] for r in recs)
    S["diff"] = {"count": dict(sev), "examples": {k: [{"id": r["id"], "text": short(r["text"], 70), "details": r["diff"]["details"][:3]}
                                                      for r in recs if r["diff"]["sev"] == k][:10]
                                                  for k in SEV_ORDER if k != "same"},
                 "kinds": dict(Counter(d["kind"] for r in recs for d in r["diff"]["details"]))}
    # ---- zh / all_zh / auto
    S["modes"] = {}
    for mode in ("all_zh", "auto"):
        got = [(r, q, q["modes"][mode]) for r, q in reqs if mode in (q.get("modes") or {})]
        S["modes"][mode] = {"requests": len(got), "count": dict(Counter(x["sev"] for _r, _q, x in got)),
                            "examples": [{"id": r["id"], "sent": short(q["sent"], 70), "sev": x["sev"],
                                          "details": x.get("details", [])[:2], "error": x.get("error", "")}
                                         for r, q, x in got if x["sev"] != "same"][:12]}
    # ---- 预检
    checks = [(r["id"], c) for r, q in reqs for c in q["find_checks"]]
    tchecks = [(r["id"], c) for r in recs for c in r["train_find_checks"]]
    S["find_infer"] = {"total": len(checks), "confirmed": sum(1 for _i, c in checks if c["confirmed"]),
                       "not_confirmed": [(i, c) for i, c in checks if c["confirmed"] is False][:30],
                       "not_checked": [(i, c) for i, c in checks if c["confirmed"] is None][:30],
                       "by_kind": dict(Counter(c["kind"] for _i, c in checks)),
                       "examples": _ex([(i, c) for i, c in checks if c["confirmed"]], 30)}
    S["find_train"] = {"total": len(tchecks), "confirmed": sum(1 for _i, c in tchecks if c["confirmed"]),
                       "not_confirmed": [(i, c) for i, c in tchecks if c["confirmed"] is False][:30],
                       "by_kind": dict(Counter(c["kind"] for _i, c in tchecks))}
    missed = Counter()
    missed_ex: Dict[str, Dict[str, Any]] = {}
    for r, q in reqs:
        for w in q["find_missed"]:
            k = f"{w['w']}|{w['status']}"
            missed[k] += 1
            missed_ex.setdefault(k, {"w": w["w"], "status": w["status"], "ph": w.get("ph", ""), "id": r["id"]})
    S["find_missed_infer"] = [dict(missed_ex[k], count=c) for k, c in missed.most_common()]
    fixes = [(r["id"], q) for r, q in reqs if q.get("fix_checks") is not None]
    S["fixes_corpus"] = [{"id": i, "raw": q["raw"], "sent": q["sent"], "raw_phones": q["raw_phones"],
                          "phones": q["phones"], "checks": q["fix_checks"]} for i, q in fixes]
    S["fixes_probe"] = probes
    S["engine_text_changed"] = len(fixes)
    return S


# ============================================================================ 写结果
def _table(head: Sequence[str], rows: Iterable[Sequence[Any]]) -> List[str]:
    out = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        out.append("| " + " | ".join(str(x).replace("|", "｜").replace("\n", " ") for x in r) + " |")
    return out


def write_md(S: Dict[str, Any], path: Path) -> None:
    L: List[str] = []
    m, c = S["meta"], S["corpus"]
    L += ["# V5 文字处理实测结果（run_text_check.py 生成，不要手改）", "",
          f"运行：{m.get('finished', '')}，用时 {m.get('seconds', '?')} 秒；整合包 {m.get('package', '?')}；版本参数 version={m.get('version')}；"
          f"Python {m.get('python', '?')}。", ""]
    if m.get("limit"):
        L += [f"**注意：这次只量了前 {m['limit']} 句（--limit），不是全部母本。**", ""]
    if m.get("failed"):
        L += [f"**注意：{len(m['failed'])} 句分析时出错，没有算进下面的数（第 10 节）。**", ""]
    # ---- 0. 结论
    o, mx, inf = S["official"], S["mixed"], S["infer"]
    wt, wi = S["words_train"], S["words_infer"]
    bad_i = sum(v for k, v in wi["count"].items() if k in BAD_STATUS)
    bad_t = sum(v for k, v in wt["count"].items() if k in BAD_STATUS)
    d = S["diff"]["count"]
    L += ["## 0. 小结（全是这次实测出来的数）", "",
          f"- 母本 {c['sentences']} 句，{c['with_english']} 句夹着英文，一共 {c['english_words']} 个英文词；{c['with_digits']} 句有阿拉伯数字。",
          f"- 训练，官方的方法（1-get-text.py）：{o['sentences_out']} 句有结果；夹着英文、按中文处理的 {o['en_sentences_zh']} 句里的 "
          f"{o['en_words_in_zh_sentences']} 个英文词，规范化以后的文字里还剩 {o['latin_left_in_norm']} 个英文字母（0 就是英文全被丢掉）。",
          f"- 训练，VoiceTwin 的「中英文一起」（get_text_mixed.py）：{mx['sentences_out']} 句有结果，其中 {mx['same_as_here']} 句和这里用同样的函数重算的一模一样；"
          f"英文词 {wt['total']} 个，其中 {bad_t} 个读得可能不对（下面第 4 节）。",
          f"- 生成（VoiceTwin 发给 api_v2 的样子）：{inf['requests']} 个请求，英文词 {wi['total']} 个，其中 {bad_i} 个读得可能不对。",
          "- 训练和生成的音素：" + "；".join(f"{SEV_ZH[k]} {d.get(k, 0)} 句" for k in SEV_ORDER) + "。",
          f"- 英文段不到 4 个音素时，引擎（cleaner.clean_text）在它前面自己加一个「,」停顿：训练 {mx['short_prefix']} 处、"
          f"生成 {inf['short_prefix']} 处；生成时引擎在请求开头自己加「。」（第一个标点前不到 4 个字）{inf['lead_added']} 个请求，训练时没有。",
          f"- 英文读音预检：生成时预检说会读错的 {S['find_infer']['total']} 处，实测核对上 {S['find_infer']['confirmed']} 处；"
          f"发给引擎的文字被自动改了的请求 {S['engine_text_changed']} 个；造的例句 {len(S['fixes_probe'])} 句，"
          f"改的效果实测对的 {sum(1 for p in S['fixes_probe'] for x in p.get('checks', []) if x['ok'])}/"
          f"{sum(len(p.get('checks', [])) for p in S['fixes_probe'])} 处。", ""]
    # ---- 1. 怎么测的
    L += ["## 1. 怎么测的", "",
          "- 用的全是整合包自己的代码、词典和模型（下面是文件大小和 SHA-256 前 12 位）：", ""]
    L += _table(["文件", "字节", "SHA-256"], [(k, v.get("size"), v.get("sha")) for k, v in m.get("files", {}).items()])
    L += ["", "- 各个库的版本：" + "，".join(f"{k} {v}" for k, v in m.get("libs", {}).items()), ""]
    L += [f"- {x}" for x in m.get("notes", [])]
    L += ["- 训练用的文字 = 母本这一句按 exporters.gptsovits_list_text 的规则（末尾没有标点补「，」，语言按 send_lang），"
          "真的写成训练列表，让 1-get-text.py 和 get_text_mixed.py 各跑一遍（和老师电脑上一样的环境变量，CPU、不用半精度）。",
          "- 生成用的文字 = 把这一句当成讲稿（读音词典是空的）：parse_script 切句 → engine_text → send_lang，"
          "每个请求调用 TextPreprocessor.preprocess(文字, text_lang, \"cut0\", 版本)（api_v2 收到请求以后 TTS.run 做的就是这一步）。",
          "- 每个英文词：照 english.py 的分支记下每个词走的路（词典 / 人名词典 / 按字母 / 拆词 / 神经网络猜 / 多音词），"
          "接起来和真正的音素核对，对不上的单独列出来（第 10 节）。", ""]
    # ---- 2. 官方
    L += ["## 2. 训练：官方的方法（1-get-text.py）", "",
          f"- 子进程：{m.get('scripts', {}).get('official', {}).get('returncode')}（退出码），"
          f"{m.get('scripts', {}).get('official', {}).get('seconds')} 秒，日志里 Traceback {m.get('scripts', {}).get('official', {}).get('tracebacks')} 个。",
          f"- 没有结果的句子：{len(o['missing'])} 句 {' '.join(o['missing'][:10])}",
          f"- 没有英文的 {o['no_en_total']} 句里，{o['no_en_same_as_mixed']} 句和「中英文一起」的音素一模一样；不一样的：{' '.join(o['no_en_diff'][:10]) or '没有'}。", ""]
    # ---- 3. 中英文一起
    ms = m.get("scripts", {}).get("mixed", {})
    L += ["## 3. 训练：VoiceTwin 的「中英文一起」（get_text_mixed.py）", "",
          f"- 子进程：退出码 {ms.get('returncode')}，{ms.get('seconds')} 秒，VT_EN_PHONES {ms.get('vt', {}).get('EN_PHONES')}，"
          f"VT_EN_LINES {ms.get('vt', {}).get('EN_LINES')}，VT_FAIL {ms.get('fail') or '没有'}，日志里 Traceback {ms.get('tracebacks')} 个。",
          f"- BERT 特征文件（3-bert）：{json.dumps(m.get('bert', {}).get('stat', {}), ensure_ascii=False)}"
          + (f"；有问题的：{'；'.join(m.get('bert', {}).get('bad', [])[:5])}" if m.get('bert', {}).get('bad') else ""),
          f"- 和这里重算的不一样的句子：{' '.join(mx['not_same']) or '没有'}；每个英文词记录对不上的：{' '.join(mx['trace_bad']) or '没有'}。",
          f"- 中英分段时丢了字的：{mx['seg_lost'] or '没有'}；处理出错的段：{mx['seg_errors'] or '没有'}。",
          f"- 英文段不到 4 个音素、引擎在前面自己加了一个「,」（停顿）：训练 {mx['short_prefix']} 处，生成 {inf['short_prefix']} 处。", ""]
    # ---- 4. 英文词
    L += ["## 4. 英文词", ""]
    L += _table(["情况", "训练（中英文一起）", "生成"],
                [(STATUS_ZH.get(k, k), wt["count"].get(k, 0), wi["count"].get(k, 0))
                 for k in sorted(set(wt["count"]) | set(wi["count"]), key=lambda k: list(STATUS_ZH).index(k) if k in STATUS_ZH else 99)])
    L.append("")
    for k in BAD_STATUS + ("letter_A", "name", "homograph"):
        for stage, W in (("生成", wi), ("训练", wt)):
            items = W["by_status"].get(k) or []
            if not items:
                continue
            L.append(f"### {STATUS_ZH.get(k, k)}（{stage}，{len(items)} 个不同的词）")
            L.append("")
            L += _table(["词", "几处", "引擎里变成", "怎么处理的", "音素", "句子"],
                        [(e["w"], e["count"], " ".join(e.get("tokens") or []),
                          " ".join(e.get("paths") or []) + (f"（另一个读音 {e['alt']}）" if e.get("alt") else ""),
                          e.get("ph", ""), " ".join(e["ids"][:3])) for e in items[:25]])
            L.append("")
    # ---- 5. 标点
    L += ["## 5. 标点", "", "「留下」= 音素里有对应的停顿符号；「丢掉」= 没有（包括和旁边的标点合并成一个的）；「换成」= 换成了别的停顿符号。", ""]
    for stage, title in (("infer", "生成"), ("train", "训练")):
        P = S[f"punct_{stage}"]
        L.append(f"### 紧跟在英文后面的标点（{title}）")
        L.append("")
        L += _table(["标点", "在哪种段", "几处", "留下", "丢掉", "换成别的"],
                    [(k.split("|")[0], k.split("|")[1], v.get("total", 0), v.get("kept", 0), v.get("dropped", 0), v.get("changed", 0))
                     for k, v in sorted(P["after_en"].items(), key=lambda kv: -kv[1].get("total", 0))])
        L.append("")
        drops = [(k, v) for k, v in P["other"].items() if v.get("dropped") or v.get("changed")]
        if drops:
            L.append(f"别的地方丢掉或换掉的标点（{title}）：" + "；".join(
                f"{k.split('|')[0]}（{k.split('|')[1]} 段）丢 {v.get('dropped', 0)}、换 {v.get('changed', 0)} / {v.get('total', 0)}"
                for k, v in sorted(drops, key=lambda kv: -kv[1].get('dropped', 0))))
            L.append("")
        exs = [f"{x}" for k, v in P["examples"].items() for x in v]
        if exs:
            L.append(f"例子（{title}）：")
            L += [f"- {x}" for x in exs[:16]]
            L.append("")
    semi = {st: sum(v.get("total", 0) for k, v in S[f"punct_{st}"]["after_en"].items() if k.split("|")[0] in ";:；：")
            for st in ("train", "infer")}
    L += [f"英文后面紧跟的 ; : ；：训练 {semi['train']} 处、生成 {semi['infer']} 处（母本里没有的话，造的例句上的实测见第 9 节）。", ""]
    L += [f"生成时引擎在句子前面自己加「。」（第一个标点前不到 4 个字）：{inf['lead_added']} 个请求，例如 "
          + "、".join(f"「{short(x, 20)}」" for x in inf["lead_added_ex"][:4]),
          f"生成时句子最后引擎自己补「。」：{inf['tail_added']} 个请求；音素不到 6 个、引擎在前面加「.」重来一次：{len(inf['retried'])} 个请求。", ""]
    # ---- 6. 数字
    L += ["## 6. 数字", ""]
    for x in S["numbers"]:
        L.append(f"- {x['id']}「{short(x['text'], 60)}」")
        for s in x["train"]:
            L.append(f"  - 训练：{s['lang']} 段「{short(s['text'], 40)}」→「{short(s['norm'], 60)}」")
        for s in x["infer"]:
            L.append(f"  - 生成：{s['lang']} 段「{short(s['text'], 40)}」→「{short(s['norm'], 60)}」")
        if x["modes"]:
            L.append(f"  - 生成时换成 all_zh / auto：{x['modes']}")
    L.append("")
    # ---- 7. 训练和生成
    L += ["## 7. 训练和生成的音素哪里不一样", "",
          "训练 = 「中英文一起」处理整行素材；生成 = VoiceTwin 把同一句话当讲稿发给 api_v2（可能切成几个请求、文字经过 tts_normalize 和 engine_text）。", ""]
    L += _table(["情况", "几句"], [(SEV_ZH[k], S["diff"]["count"].get(k, 0)) for k in SEV_ORDER])
    L += ["", "不一样的地方一共（一句里可能有几处）：" + "；".join(f"{SEV_ZH.get(k, k)} {v} 处" for k, v in S["diff"]["kinds"].items()), ""]
    for k, exs in S["diff"]["examples"].items():
        if not exs:
            continue
        L.append(f"### {SEV_ZH[k]}（例子）")
        for e in exs[:8]:
            L.append(f"- {e['id']}「{e['text']}」")
            for dd in e["details"]:
                L.append(f"  - {SEV_ZH.get(dd['kind'], dd['kind'])}：训练 {dd['a']} ／ 生成 {dd['b']}（附近：{dd['ctx']}）")
        L.append("")
    # ---- 8. modes
    L += ["## 8. 生成时 text_lang 换成 all_zh、auto 会怎样（VoiceTwin 发的是 zh）", ""]
    for mode, X in S["modes"].items():
        L.append(f"- {mode}：{X['requests']} 个有汉字的请求，" + "；".join(f"{SEV_ZH.get(k, k)} {v}" for k, v in X["count"].items()))
        for e in X["examples"][:6]:
            det = "；".join(f"{SEV_ZH.get(dd['kind'], dd['kind'])}：zh {dd['a']} ／ {mode} {dd['b']}" for dd in e["details"]) or e["error"]
            L.append(f"  - {e['id']}「{e['sent']}」：{det}")
    L.append("")
    # ---- 9. 预检
    P = S["precheck"]
    L += ["## 9. 英文读音预检（gsv_english.py）", "",
          f"- 整合包词典：VoiceTwin 读到 {S['dictionary'].get('vt_words')} 个词、{S['dictionary'].get('vt_names')} 个人名；"
          f"english.py 自己的 _g2p.cmu 有 {S['dictionary'].get('engine_words')} 个词；两边一样：{S['dictionary'].get('same')}。",
          "- 生成前给老师看的话（母本当成一篇讲稿）：", f"  - {P.get('risk_line') or '（没有）'}", f"  - {P.get('fix_line') or '（没有）'}",
          "- 训练前给老师看的话：", f"  - {P.get('training_line') or '（没有）'}", ""]
    fi = S["find_infer"]
    L += [f"### 预检说会读错的，实测是不是这样（生成，{fi['total']} 处，核对上 {fi['confirmed']} 处）", ""]
    L += _table(["句子", "预检", "词", "预检说", "实测"],
                [(i, ch["kind"], ch["word"], short(ch["detail"], 40), short(ch["real"], 60))
                 for i, ch in (fi["not_confirmed"] + fi["examples"])[:30]])
    L.append("")
    if fi["not_confirmed"]:
        L.append(f"**核对不上的 {len(fi['not_confirmed'])} 处**：" + "；".join(f"{i} {ch['word']}（{ch['real']}）" for i, ch in fi["not_confirmed"][:10]))
        L.append("")
    ft = S["find_train"]
    L += [f"训练用的文字上预检报的 {ft['total']} 处，实测核对上 {ft['confirmed']} 处；核对不上的："
          + ("；".join(f"{i} {ch['word']}（{ch['real']}）" for i, ch in ft["not_confirmed"][:10]) or "没有"), ""]
    L += ["### 实测读得可能不对、预检没提到的词（生成）", "",
          "多音词按词性猜读音（预检只管 read、object、rebel 和词典里只有一个读音的那几个），引擎这次选的读音见第 4 节「多音词」表："
          + ("、".join(f"{e['w']}（{e.get('ph', '')}，另一个读音 {e.get('alt', '')}）"
                      for e in S["words_infer"]["by_status"].get("homograph", [])[:12]) or "没有") + "。", ""]
    L += _table(["词", "情况", "几处", "音素", "例如"],
                [(x["w"], STATUS_ZH.get(x["status"], x["status"]), x["count"], x["ph"], x["id"]) for x in S["find_missed_infer"][:40]])
    L.append("")
    L += ["### 自动改的那几处：改之前、改之后的真实音素", ""]
    for title, rows in (("母本", S["fixes_corpus"]), ("造的例句", S["fixes_probe"])):
        L.append(f"**{title}**（{len(rows)} 句）")
        for x in rows:
            L.append(f"- 「{x['raw']}」→「{x['sent']}」")
            L.append(f"  - 改之前：{x.get('raw_phones', '')}")
            L.append(f"  - 改之后：{x.get('phones', '')}")
            for ch in x.get("checks", []):
                L.append(f"  - {'对' if ch['ok'] else '**不对**'}：{ch['fix']}（想要：{ch['expect']}{'；' + ch['measured'] if ch.get('measured') else ''}）")
        L.append("")
    # ---- 10. 出错
    L += ["## 10. 出错、对不上的地方", "",
          f"- 生成时出错的请求：{inf['errors'] or '没有'}",
          f"- 生成时每一段接起来和最后的音素对不上：{inf['inconsistent'] or '没有'}",
          f"- 生成时每个英文词的记录接起来对不上：{inf['trace_bad'] or '没有'}",
          f"- 生成时中英分段丢了字：{inf['seg_lost'] or '没有'}",
          f"- 分析时出错的句子：{m.get('failed') or '没有'}",
          f"- 子进程日志最后几行：官方 {m.get('scripts', {}).get('official', {}).get('tail', [])[-3:]}；中英文一起 {ms.get('tail', [])[-3:]}", ""]
    path.write_text("\n".join(L) + "\n", encoding="utf-8")


CSV_HEAD = ["编号", "原文", "训练用的文字", "训练语言", "官方_音素", "官方_规范化文字", "中英文一起_音素", "中英文一起_和重算一样",
            "生成_发给引擎的文字", "生成_语言", "生成_音素", "训练和生成", "差别", "英文词数", "读得可能不对的英文词（训练）",
            "读得可能不对的英文词（生成）", "丢掉的标点（训练）", "丢掉的标点（生成）", "数字（生成）", "预检（生成）",
            "预检自动改的", "改之前的音素", "all_zh", "auto"]


def csv_row(r: Dict[str, Any]) -> List[Any]:
    def bad_words(ws: Iterable[Dict[str, Any]]) -> str:
        return "；".join(f"{w['w']}:{w['status']}:{w.get('ph', '')}" for w in ws if w["status"] in BAD_STATUS)

    def drops(ps: Iterable[Dict[str, Any]]) -> str:
        return "；".join(f"{p['ch']}({p['lang']}{',英文后' if p['after_en'] else ''}):{p['status']}" for p in ps if p["status"] != "kept")

    q = r["requests"]
    return [r["id"], r["text"], r["train_text"], r["train_lang"], r["official"]["phones"] or "", r["official"]["norm"] or "",
            r["mixed"]["phones"] or "", r["mixed"]["same_as_here"], " ｜ ".join(x["sent"] for x in q), " ｜ ".join(x["lang"] for x in q),
            " ｜ ".join(x["phones"] for x in q), SEV_ZH.get(r["diff"]["sev"], r["diff"]["sev"]),
            "；".join(f"{SEV_ZH.get(d['kind'], d['kind'])}:{d['a']}→{d['b']}" for d in r["diff"]["details"][:4]),
            r["n_words"], bad_words(r["train_words"]), bad_words(w for x in q for w in x["words"]),
            drops(r["train_punct"]), drops(p for x in q for p in x["punct"]),
            "；".join(f"{n['lang']}:{n['text']}→{n['norm']}" for x in q for n in x["numbers"]),
            "；".join(f"{k}:{w}" for x in q for k, w in x["find"]),
            "；".join(f"{f['before']}→{f['after']}" for x in q for f in x["fixes"]),
            " ｜ ".join(x.get("raw_phones", "") for x in q if x.get("raw_phones")),
            " ｜ ".join(SEV_ZH.get(x.get("modes", {}).get("all_zh", {}).get("sev", ""), "") for x in q),
            " ｜ ".join(SEV_ZH.get(x.get("modes", {}).get("auto", {}).get("sev", ""), "") for x in q)]


def write_outputs(out: Path, S: Dict[str, Any], recs: Sequence[Dict[str, Any]]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    slim = []
    for r in recs:
        # JSON 里不重复存（逐句.csv 里都有）：官方的音素只在和「中英文一起」不一样时存；规范化的文字在 train_segs 里
        x = {k: v for k, v in r.items() if k not in ("train_find_checks", "train_spans")}
        x["official"] = dict(r["official"], norm=None,
                             phones=None if r["official"]["phones"] == r["mixed"]["phones"] else r["official"]["phones"],
                             same_as_mixed=r["official"]["phones"] == r["mixed"]["phones"])
        x["mixed"] = dict(r["mixed"], norm=None)
        x["train_words"] = [w for w in r["train_words"] if w["status"] not in ("ok", "letter")]
        x["train_punct"] = [p for p in r["train_punct"] if p["status"] != "kept"]
        x["requests"] = []
        for q in r["requests"]:
            y = {k: v for k, v in q.items() if k not in ("display",) and not (k == "raw" and v == q["sent"])}
            y["words"] = [w for w in q["words"] if w["status"] not in ("ok", "letter")]
            y["punct"] = [p for p in q["punct"] if p["status"] != "kept"]
            if q.get("modes"):
                y["modes"] = {m: {k: v for k, v in mv.items() if k != "phones"} for m, mv in q["modes"].items()}
            x["requests"].append(y)
        slim.append(x)
    # 每句一行（git 里好比较）；整个文件还是一个合法的 JSON：{"summary": …, "sentences": [ … ]}
    with open(out / "结果.json", "w", encoding="utf-8") as f:
        f.write('{"summary": ' + json.dumps(S, ensure_ascii=False, separators=(",", ":")) + ',\n "sentences": [\n')
        f.write(",\n".join(json.dumps(x, ensure_ascii=False, separators=(",", ":")) for x in slim))
        f.write("\n]}\n")
    with open(out / "逐句.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(CSV_HEAD)
        for r in recs:
            w.writerow(csv_row(r))
    write_md(S, out / "结果.md")


# ============================================================================ 主程序
def file_info(gsv: Path) -> Dict[str, Dict[str, Any]]:
    out = {}
    for rel in ("GPT_SoVITS/text/english.py", "GPT_SoVITS/text/en_normalization/expend.py", "GPT_SoVITS/text/chinese2.py",
                "GPT_SoVITS/text/cleaner.py", "GPT_SoVITS/text/LangSegmenter/langsegmenter.py",
                "GPT_SoVITS/text/zh_normalization/text_normlization.py", "GPT_SoVITS/text/cmudict.rep",
                "GPT_SoVITS/text/engdict-hot.rep", "GPT_SoVITS/TTS_infer_pack/TextPreprocessor.py",
                "GPT_SoVITS/TTS_infer_pack/text_segmentation_method.py", "GPT_SoVITS/prepare_datasets/1-get-text.py",
                "GPT_SoVITS/text/G2PWModel/g2pW.onnx", "api_v2.py"):
        p = gsv / rel
        if not p.exists():
            out[rel] = {"size": "没有", "sha": ""}
            continue
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        out[rel] = {"size": p.stat().st_size, "sha": h.hexdigest()[:12]}
    return out


def lib_versions() -> Dict[str, str]:
    from importlib import metadata
    out = {}
    for name in ("torch", "transformers", "g2p_en", "nltk", "wordsegment", "inflect", "split-lang", "fast-langdetect",
                 "jieba", "jieba_fast", "pypinyin", "cn2an", "opencc", "onnxruntime"):
        try:
            out[name] = metadata.version(name)
        except Exception:  # noqa: BLE001
            out[name] = "没装"
    return out


def run_probes(off: Official, vt: Dict[str, Any]) -> List[Dict[str, Any]]:
    out = []
    for text in PROBES:
        for r in requests_of(vt, text):
            inf = off.infer(r["sent"], r["lang"])
            x = {"raw": r["raw"], "sent": r["sent"], "lang": r["lang"], "phones": " ".join(inf["phones"]),
                 "error": inf.get("error", ""), "finds": [(f.kind, f.word, f.detail) for f in vt["G"].find(r["sent"], None)]}
            if r["raw"] != r["sent"]:
                raw = off.infer(r["raw"], r["raw_lang"])
                x["raw_phones"] = " ".join(raw["phones"])
                x["checks"] = check_fix_request(r["fixes"], raw.get("units", []), inf.get("units", []), off.E._g2p.cmu,
                                                raw_punct_dropped(raw.get("segs", [])))
            else:
                x["raw_phones"] = x["phones"]
            out.append(x)
    return out


def main_real(args: argparse.Namespace) -> int:
    t0 = time.time()
    gsv, repo, out, work = (Path(p).resolve() for p in (args.gsv, args.repo, args.out, args.work))
    work.mkdir(parents=True, exist_ok=True)
    bert_dir = gsv / "GPT_SoVITS" / "pretrained_models" / "chinese-roberta-wwm-ext-large"
    vt = load_voicetwin(repo)
    rows = read_corpus(repo / "voicetwin" / "data" / "lexicon" / "core_corpus.tsv")
    if args.limit:
        rows = rows[: args.limit]
    meta: Dict[str, Any] = {"package": gsv.name, "version": args.version, "python": platform.python_version(),
                            "limit": args.limit or 0, "files": file_info(gsv), "libs": lib_versions(), "notes": []}
    # ---- 训练列表（和 exporters.gptsovits_list_text 一样的规则）
    lst = work / "train.list"
    lines = []
    for sid, text in rows:
        lang0 = vt["detect_lang"](text)
        lines.append(f"{sid}.wav|vt|{vt['send_lang'](text, lang0)}|{vt['ensure_final_punct_train'](text.replace('|', ' '), lang0)}")
    lst.write_text("\n".join(lines) + "\n", encoding="utf-8")
    env = dict(os.environ)
    paths = [str(gsv), str(gsv / "GPT_SoVITS"), str(gsv / "GPT_SoVITS" / "BigVGAN"), str(gsv / "tools")]
    env.update({"PYTHONPATH": os.pathsep.join(paths + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])),
                "inp_text": str(lst), "inp_wav_dir": str(work / "wav"), "exp_name": "textcheck", "i_part": "0",
                "all_parts": "1", "is_half": "False", "version": args.version, "bert_pretrained_dir": str(bert_dir),
                "bert_path": str(bert_dir)})
    scripts: Dict[str, Any] = {}
    sub: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for key, cmd, extra in (
            ("official", [sys.executable, "-s", "GPT_SoVITS/prepare_datasets/1-get-text.py"], {}),
            ("mixed", [sys.executable, "-s", str(repo / "voicetwin" / "backends" / "gsv_scripts" / "get_text_mixed.py")],
             {"VOICETWIN_MIXED_TEXT": str(repo / "voicetwin" / "backends" / "gsv_mixed_text.py")})):
        opt = work / key
        if opt.exists():
            import shutil
            shutil.rmtree(opt)
        e = dict(env, opt_dir=str(opt), **extra)
        print(f"[{time.strftime('%H:%M:%S')}] 训练的处理文字：{key} …", flush=True)
        scripts[key] = run_script(cmd, e, gsv, work / f"{key}.log")
        print(f"  退出码 {scripts[key]['returncode']}，{scripts[key]['seconds']} 秒", flush=True)
        sub[key] = read_name2text(opt / "2-name2text-0.txt")
    meta["scripts"] = scripts
    # ---- 这个进程里用官方代码一句一句地量
    print(f"[{time.strftime('%H:%M:%S')}] 加载官方的文字处理程序和 BERT …", flush=True)
    off = Official(gsv, args.version, bert_dir, vt["M"])
    meta["notes"] += off.notes
    G = vt["G"]
    dic = G.load_dictionary(gsv / "GPT_SoVITS" / "text")
    eng = set(off.E._g2p.cmu)
    dic_info = {"vt_words": len(dic.words) if dic else None, "vt_names": len(dic.names) if dic else None,
                "engine_words": len(eng), "same": (set(dic.words) == eng) if dic else None}
    recs = []
    for i, (sid, text) in enumerate(rows, 1):
        try:
            recs.append(analyse_sentence(off, vt, sid, text, sub, dic))
        except Exception as exc:  # noqa: BLE001 - 一句出错不影响别的句子：记下来
            traceback.print_exc()
            meta.setdefault("failed", []).append(f"{sid}：{type(exc).__name__}: {short(exc, 200)}")
        if i % 100 == 0 or i == len(rows):
            print(f"  {i}/{len(rows)} 句，{round(time.time() - t0)} 秒", flush=True)
    meta["bert"] = bert_check(work / "mixed" / "3-bert", [f"{r['id']}.wav" for r in recs], sub["mixed"],
                              {f"{r['id']}.wav": r["train_spans"] for r in recs})
    # ---- 预检
    seg_texts = [q["raw"] for r in recs for q in r["requests"]]
    rep_i = G.precheck(seg_texts, dic, auto_fix=True)
    rep_t = G.precheck([r["train_text"] for r in recs], dic, auto_fix=False)
    pj = {"infer": rep_i.to_json(), "train": rep_t.to_json(), "risk_line": rep_i.risk_line(), "fix_line": rep_i.fix_line(),
          "training_line": rep_t.training_line()}
    probes = run_probes(off, vt)
    meta["seconds"] = round(time.time() - t0)
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    S = summarize(recs, meta, dic_info, pj, probes)
    write_outputs(out, S, recs)
    print(f"[{time.strftime('%H:%M:%S')}] 写好了：{out}", flush=True)
    return 0 if recs else 1


# ============================================================================ 自检（不需要整合包）
def selftest() -> int:
    fails: List[str] = []

    def check(cond: bool, what: str) -> None:
        if not cond:
            fails.append(what)

    # 标点
    check([c for _i, c in input_marks("as,这个 I'm well-known。“好”")] == [",", "。", "“", "”"], "input_marks")
    st, added = align_marks(["，", "：", "。"], [",", "."])
    check(st.count("dropped") == 1 and st[2] == "kept" and added == [], f"align_marks 丢一个 {st}")
    st, added = align_marks(["；"], ["."])
    check(st == ["changed:."], f"align_marks 换 {st}")
    st, added = align_marks([], [","])
    check(added == [","], "align_marks 多出")
    check(after_english("as，这个", 2) and after_english("，这个", 0, "the answer ") and not after_english("好，", 1), "after_english")
    # 中文单位
    u, ok = zh_units("你好,", ["n", "i3", "h", "ao3", ","], [2, 2, 1])
    check(ok and [x["t"] for x in u] == ["你", "好", ","] and u[2]["ph"] == [","], "zh_units")
    _u, ok = zh_units("你好", ["n", "i3"], [2, 2])
    check(not ok, "zh_units 对不上")

    # 英文：照 english.py 的分支
    class FakeG:
        cmu = {"as": [["AE1", "Z"]], "a": [["AH0"]], "n": [["EH1", "N"]], "d": [["D", "IY1"]], "live": [["L", "AY1", "V"]],
               "dr": [["D", "R", "AY1", "V"]], "doctor": [["D", "AA1", "K", "T", "ER0"]], "seconds": [["S", "EH1", "K", "AH0", "N", "D", "Z"]],
               "b": [["B", "IY1"]], "x": [["EH1", "K", "S"]], "number": [["N", "AH1", "M", "B", "ER0"]], "no": [["N", "OW1"]]}
        namedict = {"banner": [["B", "AE1", "N", "ER0"]]}
        homograph2features = {"read": (["R", "IY1", "D"], ["R", "EH1", "D"], "VBP")}

        def qryword(self, o_word):
            w = o_word.lower()
            if len(w) > 1 and w in self.cmu:
                return self.cmu[w][0]
            if o_word.istitle() and w in self.namedict:
                return self.namedict[w][0]
            if len(w) <= 3:
                return [p for ch in w for p in (["EY1"] if ch == "a" else self.cmu[ch][0])]
            return ["UW", "Q"]  # 「神经网络」：UW 会被 english.g2p 去掉，Q 不在符号表里

    seg = lambda w: [w] if w != "jimwho" else ["jim", "who"]  # noqa: E731
    syms = {"AE1", "Z", "AH0", "EH1", "N", "D", "IY1", "L", "AY1", "V", "EY1", "R", "B", "K", "S", ",", ".", "-", "UNK",
            "AA1", "T", "ER0", "EH0", "AH1", "M", "OW1"}
    g = FakeG()
    t = trace_english([("as", "IN"), (",", ","), ("A", "DT"), ("N", "NNP"), ("D", "NNP"), ("read", "VB"), ("bx", "NN"),
                       ("zork", "NN"), ("Banner", "NNP")], g, seg, syms)
    check([x["path"] for x in t] == ["dict", "symbol", "letter_A", "letter", "letter", "homograph1", "letters", "predict", "name"],
          f"trace_english 路径 {[x['path'] for x in t]}")
    check(t[7]["ph"] == [] and t[7]["removed"] == ["UW", "Q"], f"trace_english 去掉 {t[7]}")
    check(t[6]["ph"] == ["B", "IY1", "EH1", "K", "S"], "trace_english 按字母")
    real = [p for x in t for p in x["ph"]]
    fu, ok = finish_english(t, real, syms, "norm")
    check(ok and fu[0]["path"] == "dict", "finish_english 对上")
    fu, ok = finish_english(t[:1], [",", "AE1", "Z"], syms, "as")
    check(ok and fu[0]["path"] == "short_prefix" and fu[0]["added"], "finish_english 不到 4 个音素加逗号")
    _fu, ok = finish_english(t[:1], ["AE1", "Z"], syms, "as")
    check(not ok, "finish_english 对不上")
    check(qry_path("JimWho", g, seg).startswith("segment("), "qry_path 拆词")
    check(qry_path("Banner's", g, seg) == "possessive(letters)" or qry_path("Banner's", g, seg).startswith("possessive"), "qry_path 所有格")
    # 每个词
    iso = {"AND": ["A", "N", "D"], "McDonald's": ["Mc", "Donald's"], "I’m": ["Im"]}.get
    units = [{"t": "A", "ph": ["EY1"], "path": "letter_A"}, {"t": "N", "ph": ["EH1", "N"], "path": "letter"},
             {"t": "D", "ph": ["D", "IY1"], "path": "letter"}, {"t": "live", "ph": ["L", "AY1", "V"], "path": "dict"},
             {"t": "Im", "ph": ["IH1", "M"], "path": "dict"}, {"t": "zork", "ph": ["Z"], "path": "predict"}]
    ws = word_reports("AND live I’m zork gone", units, lambda w: (iso(w) or [w]))
    check([w["status"] for w in ws] == ["spelled_caps", "ok", "changed", "guessed", "missing"], f"word_reports {[w['status'] for w in ws]}")
    check(ws[0]["ph"] == "EY1 EH1 N D IY1", "word_reports 音素")
    sw = seg_words([{"lang": "zh", "text": "好A", "norm": "好"}, {"lang": "en", "text": "live", "units": units[3:4]}],
                   "好A live Bob", lambda w: [w])
    check([w["status"] for w in sw] == ["dropped_in_zh", "ok", "lost_in_segmenter"], f"seg_words {[w['status'] for w in sw]}")
    check(segment_loss("ab,cd", ["ab", "cd"]) == [","], "segment_loss")
    hy = word_reports("a well-known fact", [{"t": "a", "ph": ["AH0"], "path": "dict"},
                                            {"t": "well-known", "ph": ["W", "EH1", "L", "N", "OW1", "N"], "path": "segment(well:dict+known:dict)"},
                                            {"t": "fact", "ph": ["F"], "path": "dict"}], lambda w: [w])
    check([(w["w"], w["status"], w.get("part_of")) for w in hy] == [("a", "ok", None), ("well", "segmented", "well-known"),
                                                                    ("known", "segmented", "well-known"), ("fact", "ok", None)],
          f"word_reports 连字符 {[(w['w'], w['status']) for w in hy]}")
    # 训练和生成比较
    A = [{"t": "这", "ph": ["zh", "e4"]}, {"t": ",", "ph": [","]}, {"t": "as", "ph": ["AE1", "Z"]}, {"t": "行", "ph": ["x", "ing2"]},
         {"t": ",", "ph": [","]}]
    B = [{"t": ".", "ph": ["."]}, {"t": "这", "ph": ["zh", "e4"]}, {"t": "as", "ph": ["AE1", "Z"]}, {"t": "行", "ph": ["h", "ang2"]},
         {"t": ".", "ph": ["."]}]
    sev, det = diff_units(A, B)
    check(sev == "reading" and [d["kind"] for d in det] == ["edge_punct", "mid_punct", "reading", "edge_punct"],
          f"diff_units {sev} {[d['kind'] for d in det]}")
    check(diff_units(A, A)[0] == "same", "diff_units 一样")
    check(diff_units(A, A[:2] + [{"t": "is", "ph": ["IH1", "Z"]}] + A[3:])[0] == "content", "diff_units 文字")
    # 标点（每段）
    rows = seg_punct([{"lang": "en", "text": "the answer：", "units": [{"t": "the", "ph": ["DH"]}, {"t": "answer", "ph": ["AE1"]}]},
                      {"lang": "zh", "text": "，答案。", "units": [{"t": ",", "ph": [","]}, {"t": "答", "ph": ["d", "a2"]},
                                                                {"t": "案", "ph": ["an4"]}, {"t": ".", "ph": ["."]}]}])
    check([(r["ch"], r["after_en"], r["status"]) for r in rows] == [("：", True, "dropped"), ("，", False, "kept"), ("。", False, "kept")],
          f"seg_punct {[(r['ch'], r['after_en'], r['status']) for r in rows]}")
    # 自动改的效果
    cmu = FakeG.cmu
    raw = [{"t": "Dr", "ph": ["D", "R", "AY1", "V"]}, {"t": ".", "ph": ["."]}, {"t": "Banner", "ph": ["B", "AE1", "N", "ER0"]}]
    fix = [{"t": "Doctor", "ph": ["D", "AA1", "K", "T", "ER0"]}, {"t": "Banner", "ph": ["B", "AE1", "N", "ER0"]}]
    ch = check_fix_request([{"kind": "abbr", "before": "Dr.", "after": "Doctor"}], raw, fix, cmu)
    check(len(ch) == 1 and ch[0]["ok"], f"check_fix Dr {ch}")
    ch = check_fix_request([{"kind": "abbr", "before": "Dr.", "after": "Doctor"}], raw, raw, cmu)
    check(not ch[0]["ok"], "check_fix Dr 没变要报不对")
    ch = check_fix_request([{"kind": "punct", "before": ";", "after": ","}], [{"t": "a", "ph": ["AH0"]}],
                           [{"t": "a", "ph": ["AH0"]}, {"t": ",", "ph": [","]}], cmu)
    check(ch[0]["ok"], "check_fix 标点")
    ch0 = check_fix_request([{"kind": "punct", "before": "：", "after": ","}], [{"t": ",", "ph": [","]}], [{"t": ",", "ph": [","]}],
                            cmu, raw_dropped=0)
    check(ch0[0]["ok"], "check_fix 标点本来就有停顿")
    ch1 = check_fix_request([{"kind": "punct", "before": ";", "after": ","}], [{"t": "a", "ph": ["AH0"]}], [{"t": "a", "ph": ["AH0"]}],
                            cmu, raw_dropped=1)
    check(not ch1[0]["ok"], "check_fix 标点没加上要报不对")
    ch = check_fix_request([{"kind": "decade", "before": "1990s", "after": "nineteen nineties"}],
                           [{"t": "seconds", "ph": cmu["seconds"][0]}], [{"t": "nineties", "ph": ["N"]}], cmu)
    check(ch[0]["ok"], "check_fix 年代")
    # 预检核对

    class F:
        def __init__(self, kind, word, detail=""):
            self.kind, self.word, self.detail = kind, word, detail

    checks, missed = verify_findings([F("caps", "AND"), F("fixed_reading", "live"), F("guess", "zork"), F("street", "St.")],
                                     ws, cmu)
    check([c["confirmed"] for c in checks] == [True, True, True, None], f"verify_findings {[c['confirmed'] for c in checks]}")
    check([w["w"] for w in missed] == ["I’m", "gone"], f"verify_findings 没提到 {[w['w'] for w in missed]}")
    # 汇总和写报告（假数据走一遍，不能出错）
    rec = {"id": "x1", "text": "这个as,行。", "train_lang": "zh", "train_text": "这个as,行。", "n_words": 1,
           "official": {"phones": "zh e4", "norm": "这个,行.", "latin_left": 0},
           "mixed": {"phones": "zh e4 AE1 Z", "norm": "这个as,行.", "same_as_here": True, "trace_ok": True, "errors": []},
           "train_segs": [["zh", "这个", "这个", ""]], "train_spans": [("zh", 4), ("en", 2)], "train_seg_lost": [],
           "train_words": ws, "train_punct": rows, "train_numbers": [], "train_short_prefix": 1, "train_find": [("caps", "AND")],
           "train_find_checks": checks, "train_find_missed": missed,
           "requests": [{"display": "这个as,行。", "raw": "这个as,行。", "sent": "这个as,行。", "lang": "zh", "fixes": [],
                         "error": "", "phones": ". zh e4", "consistent": True, "preseg": ["。这个as,行。"], "segs": [],
                         "retried": False, "trace_ok": True, "seg_lost": [], "words": ws, "punct": rows,
                         "numbers": [{"lang": "zh", "text": "50%", "norm": "百分之五十"}], "short_prefix": 1,
                         "lead_added": True, "tail_added": False, "find": [("caps", "AND")], "find_checks": checks,
                         "find_missed": missed, "raw_phones": "x", "fix_checks": ch,
                         "modes": {"all_zh": {"sev": "same", "details": []}, "auto": {"sev": "content", "details": det[:1]}}}],
           "diff": {"sev": sev, "details": det}}
    meta = {"package": "假的", "version": "v5dev", "files": {"a.py": {"size": 1, "sha": "abc"}}, "libs": {"g2p_en": "2.1.0"},
            "notes": ["自检"], "scripts": {"official": {"returncode": 0}, "mixed": {"returncode": 0, "vt": {"EN_PHONES": 2}}},
            "bert": {"stat": {"ok": 1}, "bad": []}}
    S = summarize([rec], meta, {"vt_words": 1, "vt_names": 0, "engine_words": 1, "same": True},
                  {"risk_line": "r", "fix_line": "", "training_line": "t"},
                  [{"raw": "Dr.Banner", "sent": "Doctor Banner", "raw_phones": "D R", "phones": "D AA1", "checks": ch}])
    check(S["diff"]["count"] == {sev: 1} and S["words_infer"]["count"].get("spelled_caps") == 1, "summarize")
    check(S["punct_infer"]["after_en"].get("：|en", {}).get("dropped") == 1, f"summarize 标点 {S['punct_infer']['after_en']}")
    tmp = Path(os.environ.get("TMPDIR", "/tmp")) / f"run_text_check_selftest_{os.getpid()}"
    try:
        write_outputs(tmp, S, [rec])
        md = (tmp / "结果.md").read_text(encoding="utf-8")
        check("## 9. 英文读音预检" in md and "spelled" not in md.split("## 4.")[0], "write_md")
        with open(tmp / "逐句.csv", encoding="utf-8-sig") as f:
            got = list(csv.reader(f))
        check(len(got) == 2 and len(got[1]) == len(CSV_HEAD), "write_csv")
        json.loads((tmp / "结果.json").read_text(encoding="utf-8"))
    finally:
        for p in sorted(tmp.glob("*")):
            p.unlink()
        if tmp.exists():
            tmp.rmdir()
    if fails:
        print("自检没通过：\n- " + "\n- ".join(fails))
        return 1
    print("自检通过")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--gsv")
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[3]))
    ap.add_argument("--out", default=str(Path(__file__).resolve().parent / "结果"))
    ap.add_argument("--work", default="/tmp/v5_text_check")
    ap.add_argument("--version", default="v5dev")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)
    if args.selftest:
        return selftest()
    if not args.gsv:
        ap.error("要 --gsv <整合包目录>")
    return main_real(args)


if __name__ == "__main__":
    sys.exit(main())
