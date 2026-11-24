"""每个切片（按 transcribe.py / transcribe-old.yml 的切法，id 一样）的最终文字：
- 新录的两段：research/新录音转录/核对结果/逐句定稿.json（两路核对 + 老师确认，id 就是切片 id）；
- 以前的 15 个录音：母本_录音核对后.csv 是按 10 月份的切法分行的，这次切的位置不一样，
  所以把母本的字（带标点）按三个识别引擎听到的字对齐到每个切片上，切片的文字 = 母本里对应的那一段（带标点）。
输出 research/训练实测/逐句文字.json：[{id, 录音, 开始秒, 秒, 文字, 来源, 对齐质量}]，对齐质量差的标出来（再用对齐检查核实）。
用法：python build_texts.py（在仓库根目录）
"""
import csv
import difflib
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OLD = ROOT / "research/新录音转录/以前的课/结果"
MOTHER = ROOT / "research/文字校正/老师的母本/母本_录音核对后.csv"
NEW = ROOT / "research/新录音转录/核对结果/逐句定稿.json"
OUT = ROOT / "research/训练实测/逐句文字.json"
#: 母本里漏掉、录音里确实有的话（两个以上引擎一致听到，按上下文核过）：切片 id → 文字
OVERRIDES = {
    "0006_769708_0058": ("比如说这个例句，it's the same doctor as my father consulted.",
                         "母本漏了这句英文例句；whisper 和 sensevoice 都听到 It's the same doctor as my father consulted"),
}
TOK = re.compile(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[一-鿿]|\d")


def toks(t):
    return [m.group(0).lower() for m in TOK.finditer(t or "")]


def nospace(t):
    return re.sub(r"(?<=[一-鿿]) (?=[一-鿿])", "", t or "")


def ed(a, b):
    p = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        c = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            c[j] = min(p[j] + 1, c[j - 1] + 1, p[j - 1] + (x != y))
        p = c
    return p[-1]


def main():
    mother = list(csv.DictReader(open(MOTHER, encoding="utf-8-sig")))
    by_src = {}
    for r in mother:
        by_src.setdefault(r["id"].rsplit("_", 2)[0], []).append(r)
    out = []
    for d in sorted(OLD.iterdir()):
        f = d / "逐句对照.json"
        if not f.exists():
            continue
        clips = json.loads(f.read_text(encoding="utf-8"))
        src = clips[0]["id"].rsplit("_", 2)[0]
        lines = by_src[src]
        # 母本全文（行之间不加东西），记下每个词元在全文里的位置
        text = "".join(r["text"] for r in lines)
        mt, mspan = [], []
        for m in TOK.finditer(text):
            mt.append(m.group(0).lower())
            mspan.append((m.start(), m.end()))
        # 切片的词元：三个引擎里和母本最像的那个（中文多半是 funasr / sensevoice，英文多半是 whisper），按切片顺序连起来
        ct, cid = [], []
        best = []
        for i, c in enumerate(clips):
            cands = [toks(c["SenseVoice"]), toks(nospace(c["第二个引擎"])), toks(c["whisper识别"])]
            best.append(cands)
            t = cands[0] or cands[1] or cands[2]
            ct += t
            cid += [i] * len(t)
        owner = [None] * len(mt)
        for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, ct, mt, autojunk=False).get_opcodes():
            if tag in ("equal", "replace") and i2 > i1:
                for k in range(j1, j2):
                    owner[k] = cid[i1 + min(i2 - i1 - 1, (k - j1) * (i2 - i1) // max(1, j2 - j1))]
        # 母本多出来的词元（insert）：归到前后最近的切片（前后都有时归前一个）
        last = None
        for k in range(len(mt)):
            if owner[k] is None:
                owner[k] = last
            last = owner[k]
        nxt = None
        for k in range(len(mt) - 1, -1, -1):
            if owner[k] is None:
                owner[k] = nxt
            nxt = owner[k]
        # 每个切片拿连续的一段（取最早到最晚的词元；中间被别的切片占了的说明对齐有问题，标出来）
        for i, c in enumerate(clips):
            ks = [k for k in range(len(mt)) if owner[k] == i]
            item = {"id": c["id"], "录音": d.name + ".wav", "开始秒": c["开始秒"], "秒": c["秒"], "来源": "母本_录音核对后"}
            if not ks:
                item.update(文字="", 对齐质量="没有对上母本的字（这一段可能不是讲课内容）")
                out.append(item)
                continue
            k0, k1 = min(ks), max(ks)
            c0 = mspan[k0][0]
            c1 = mspan[k1][1]
            nxt_start = mspan[k1 + 1][0] if k1 + 1 < len(mspan) else len(text)
            tail = re.match(r"[\s，。、？！：；,.?!:;\"”’）)]*", text[c1:nxt_start]).group(0).rstrip()
            seg = text[c0:c1] + tail
            contiguous = len(ks) == k1 - k0 + 1
            mtoks = mt[k0:k1 + 1]
            cer = min(ed(cand, mtoks) / max(1, len(mtoks)) for cand in best[i] if cand) if any(best[i]) else 1.0
            item.update(文字=seg, 引擎和母本最小差别=round(cer, 3),
                        对齐质量="好" if contiguous and cer <= 0.25 else ("不连续" if not contiguous else f"和引擎差别大（{cer:.0%}）"))
            out.append(item)
    for o in out:
        if o["id"] in OVERRIDES:
            o["文字"], why = OVERRIDES[o["id"]]
            o.update(来源="母本_录音核对后 + 补上漏掉的话", 对齐质量="好（人工核对）", 说明=why)
    for r in json.loads(NEW.read_text(encoding="utf-8")):
        out.append({"id": r["id"], "录音": r["录音"], "开始秒": r["开始秒"], "秒": r["秒"], "文字": r["定稿"],
                    "来源": "新录音_逐句定稿", "对齐质量": "好"})
    for i, o in enumerate(out, 1):
        o["行"] = i
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    from collections import Counter
    print(len(out), Counter(o["对齐质量"] if o["对齐质量"] == "好" else o["对齐质量"][:4] for o in out))


if __name__ == "__main__":
    main()
