"""把核对好的母本放进程序自带的母本标准库（2026-11-24）：重新生成
voicetwin/data/lexicon/core_corpus.tsv 和 core_fixes.tsv。

1. core_corpus.tsv：原来的每一句（按 id）换成 母本_录音核对后.csv 里的文字（11-24 对着录音核对改的 27 句），
   后面接着 11-24 新录音逐句定稿的 204 句（research/新录音转录/核对结果/逐句定稿.json，id 和句子都不重复）。
2. core_fixes.tsv（按句子 id 的修缮记录：老师原来的句子里这几个字改成那几个字）：录音核对改了的句子，
   让「识别的原文（母本_原文.csv）按这一句的记录改完」正好是录音核对后的文字——
   10 月份的记录和录音核对矛盾的去掉（she → he：录音里就是 she；「从一种 → 从句中」：录音里是「从句中的」），
   还对的留着（借词 → 介词），录音核对多改的地方加上新的记录（原因写「对着录音核对（11-24）」）。
   每一条的「原来」在原文里只出现一次、各条不重叠，所以按顺序改（上传的 transcripts.csv）和一起改（一键校正）结果一样。

运行：python research/文字校正/老师的母本/更新程序自带母本.py（不用装别的）；再跑一次结果不变。
"""

import csv
import difflib
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
CORPUS = ROOT / "voicetwin" / "data" / "lexicon" / "core_corpus.tsv"
FIXES = ROOT / "voicetwin" / "data" / "lexicon" / "core_fixes.tsv"
FINAL = ROOT / "research" / "新录音转录" / "核对结果" / "逐句定稿.json"
WHY = "对着录音核对（11-24）"
CORPUS_HEAD = ("# 核心母本：老师自己的讲课文字（老师 10-03 上传，逐句修缮过；2026-11-24 又对着录音逐句核对过，改了 {n} 句，"
               "见 research/文字校正/老师的母本/录音核对改动_2026-11-24.tsv；删除的句子不在里面）。老师同意公开。id\\t文字")
CORPUS_NEW = ("# 2026-11-24 新录音（001.1.1.1.-.wav、1.wav，26 分钟）的 {n} 句：两路独立校对一致的定稿，"
              "定不了的 9 句老师听过确认（research/新录音转录/核对结果/逐句定稿.json）。")
FIXES_NEW = "# 2026-11-24 对着录音核对改的（research/文字校正/老师的母本/录音核对改动_2026-11-24.tsv）："


def _csv(name):
    return {r["id"]: r["text"] for r in csv.DictReader(open(HERE / name, encoding="utf-8-sig"))}


def _word(c):
    return c.isascii() and (c.isalnum() or c == "'")


def _blocks(a, b):
    """a → b 不一样的地方 [(a 里开始, 结束, b 里开始, 结束)]：按字比，英文不切开单词，挨得很近的合成一处。"""
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes():
        if tag == "equal":
            continue
        while (i1 > 0 and _word(a[i1 - 1]) and (i1 < len(a) and _word(a[i1]) or j1 < len(b) and _word(b[j1]))) \
                and j1 > 0 and a[i1 - 1] == b[j1 - 1]:
            i1, j1 = i1 - 1, j1 - 1  # 往前扩到单词开头
        while (i2 < len(a) and _word(a[i2]) and (i2 > 0 and _word(a[i2 - 1]) or j2 > 0 and _word(b[j2 - 1]))) \
                and j2 < len(b) and a[i2] == b[j2]:
            i2, j2 = i2 + 1, j2 + 1  # 往后扩到单词结尾
        if out and i1 - out[-1][1] < 4:
            out[-1] = (out[-1][0], max(i2, out[-1][1]), out[-1][2], max(j2, out[-1][3]))
        else:
            out.append((i1, i2, j1, j2))
    return out


def _entries(raw, want):
    """raw → want 的记录 [(原来, 改成)]：每一处往两边加字，直到「原来」在 raw 里只出现一次（不跨到别的地方）。"""
    blocks = _blocks(raw, want)
    out = []
    lo = 0  # 前一条用到哪儿（各条不重叠）
    for k, (i1, i2, j1, j2) in enumerate(blocks):
        hi = blocks[k + 1][0] if k + 1 < len(blocks) else len(raw)
        pad = 2
        while True:
            a, b = max(lo, i1 - pad), min(hi, i2 + pad)
            while a > lo and _word(raw[a]) and _word(raw[a - 1]):
                a -= 1
            while b < hi and _word(raw[b - 1]) and _word(raw[b]):
                b += 1
            if raw.count(raw[a:b]) == 1 or (a == lo and b == hi):
                break
            pad += 1
        out.append((raw[a:b], want[j1 - (i1 - a):j2 + (b - i2)]))
        lo = b
    return out


def _apply_each(raw, entries):
    """一键校正的用法（_row_fixes）：每一条都在原来的文字里找，一起改。"""
    spans = []
    for w, r in entries:
        k = raw.find(w)
        if k < 0:
            return None
        spans.append((k, k + len(w), r))
    spans.sort()
    if any(spans[i][1] > spans[i + 1][0] for i in range(len(spans) - 1)):
        return None
    out, pos = [], 0
    for s, e, r in spans:
        out += [raw[pos:s], r]
        pos = e
    return "".join(out) + raw[pos:]


def _apply_seq(raw, entries):
    """上传的 transcripts.csv 的用法（_clean_uploaded）：按顺序 replace。"""
    for w, r in entries:
        raw = raw.replace(w, r)
    return raw


def main():
    orig, october, checked = _csv("母本_原文.csv"), _csv("母本_修缮后.csv"), _csv("母本_录音核对后.csv")
    lines = CORPUS.read_text(encoding="utf-8").splitlines()
    head = lines[0]
    ids = []
    for ln in lines[1:]:
        if ln.startswith("#"):
            break  # 新录音那一节（再跑一次时）
        ids.append(ln.split("\t", 1)[0])
    changed = [i for i in ids if checked[i] != october[i]]
    final = json.loads(FINAL.read_text(encoding="utf-8"))
    seen_ids, seen_text = set(ids), {checked[i] for i in ids}
    new = []
    for x in final:
        t = x["定稿"].strip()
        assert t and "\t" not in t and "\n" not in t, x["id"]
        if x["id"] in seen_ids or t in seen_text:
            print("重复，没有加：", x["id"])
            continue
        seen_ids.add(x["id"])
        seen_text.add(t)
        new.append((x["id"], t))
    assert head.startswith("# 核心母本")
    out = [CORPUS_HEAD.format(n=len(changed))] + [f"{i}\t{checked[i]}" for i in ids]
    out += [CORPUS_NEW.format(n=len(new))] + [f"{i}\t{t}" for i, t in new]
    CORPUS.write_text("\n".join(out) + "\n", encoding="utf-8")

    # ---------------------------------------------------------------- core_fixes.tsv
    # 10 月份的记录从 修缮记录.tsv 读（core_fixes.tsv 原来就是它去掉「行」那一列），所以再跑一次结果不变
    head_lines = [ln for ln in FIXES.read_text(encoding="utf-8").splitlines() if ln.startswith("#")]
    head_lines = head_lines[:head_lines.index(FIXES_NEW)] if FIXES_NEW in head_lines else head_lines
    rows = [r[1:5] for r in csv.reader(open(HERE / "修缮记录.tsv", encoding="utf-8"), delimiter="\t")][1:]
    keep_rows, added, dropped = [], [], []
    plan = {}
    for rid in changed:
        raw, want = orig[rid], checked[rid]
        olds = [(r[1], r[2]) for r in rows if r[0] == rid]
        kept = [(w, r) for w, r in olds if raw.count(w) == 1 and r in want]
        for _ in range(len(olds) + 1):
            mid = _apply_seq(raw, kept)
            more = _entries(mid, want) if mid != want else []
            allx = kept + more
            ok = all(raw.count(w) == 1 for w, _r in allx) and _apply_each(raw, allx) == want \
                and _apply_seq(raw, allx) == want
            if ok or not kept:
                break
            kept = kept[:-1]  # 和新的记录重叠：少留一条 10 月份的再试
        if not ok:
            kept, more = [], [(raw, want)]  # 拆不开：整句一条
            assert _apply_each(raw, more) == want
        plan[rid] = (kept, more)
        dropped += [(rid, w, r) for w, r in olds if (w, r) not in kept]
        added += [(rid, w, r) for w, r in more]
    for r in rows:
        if r[0] in plan and (r[1], r[2]) not in plan[r[0]][0]:
            continue
        keep_rows.append(r)
    out = head_lines + ["\t".join(r) for r in keep_rows] + [FIXES_NEW]
    out += [f"{rid}\t{w}\t{r}\t{WHY}" for rid, w, r in added]
    FIXES.write_text("\n".join(out) + "\n", encoding="utf-8")

    # 核对：原文按每一句的记录改完，正好是程序自带母本里的这一句（录音核对改了的句子）
    table = {}
    for ln in FIXES.read_text(encoding="utf-8").splitlines():
        if ln and not ln.startswith("#"):
            p = ln.split("\t")
            table.setdefault(p[0], []).append((p[1], p[2]))
    for rid in changed:
        assert _apply_seq(orig[rid], table.get(rid, [])) == checked[rid], rid
        assert _apply_each(orig[rid], table.get(rid, [])) == checked[rid], rid
    print(f"core_corpus.tsv：原来的 {len(ids)} 句里改了 {len(changed)} 句，加了新录音 {len(new)} 句，一共 {len(ids) + len(new)} 句")
    print(f"core_fixes.tsv：去掉 10 月份的 {len(dropped)} 条，加了 {len(added)} 条")
    for rid, w, r in dropped:
        print("  去掉", rid, w, "→", r)
    for rid, w, r in added:
        print("  加上", rid, w, "→", r)


if __name__ == "__main__":
    sys.exit(main())
