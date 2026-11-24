"""对齐检查：把核对好的每一句文字（每个汉字、每个英文单词）对到录音的时间轴上，看对不对得上。

用 torchaudio 的 MMS_FA（Meta 的多语言强制对齐模型，中文先转成拼音）。每一句输出：
- 每个字 / 词的对齐分数（0~1，越低越可疑：多半是字写错了、多写了，或者那里其实没说）；
- 句子开头、结尾有没有对不上字的人声（可能少写了字）；
- 每个标点处实际停了多久（标点和真实的停顿对不对得上）。
可疑的标准不写死：按全部字词分数的分布，最低的 2% 和明显的离群值列出来，再逐个人工看。

用法：python align_check.py --list 要转录.txt --rows 核对结果/逐句定稿.json --out 核对结果/对齐检查
"""
import argparse
import csv
import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from pypinyin import Style, lazy_pinyin

PUNCT = "，。？！：；、,.?!:;"


def words_of(text: str):
    """[(显示的字/词, 拼音或小写英文字母, 后面紧跟的标点)]。数字、符号跳过。"""
    out = []
    for m in re.finditer(r"[A-Za-z]+(?:['’][A-Za-z]+)?|[一-鿿]|[%s]" % re.escape(PUNCT), text):
        t = m.group(0)
        if t in PUNCT:
            if out:
                out[-1][2] += t
            continue
        if re.match(r"[一-鿿]", t):
            py = lazy_pinyin(t, style=Style.NORMAL)[0]
            rom = re.sub(r"[^a-z]", "", py.replace("ü", "u").replace("v", "u"))
        else:
            rom = re.sub(r"[^a-z']", "", t.lower().replace("’", "'"))
        if rom:
            out.append([t, rom, ""])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--rows", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/align")
    a = ap.parse_args()
    out, work = Path(a.out), Path(a.work)
    out.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    audio = {}
    for line in Path(a.list).read_text(encoding="utf-8").splitlines():
        u = line.strip()
        if not u or u.startswith("#"):
            continue
        name = urllib.parse.unquote(u.rsplit("/", 1)[-1])
        dst = work / name
        if not dst.exists():
            urllib.request.urlretrieve(u, dst)
        x, sr = sf.read(str(dst), dtype="float32", always_2d=True)
        x = torch.from_numpy(x.mean(axis=1))
        audio[name] = torchaudio.functional.resample(x, sr, 16000)
        print("读入", name, round(len(audio[name]) / 16000, 1), "秒", flush=True)

    bundle = torchaudio.pipelines.MMS_FA
    model = bundle.get_model(with_star=False).eval()
    tokenizer, aligner = bundle.get_tokenizer(), bundle.get_aligner()
    rows = json.loads(Path(a.rows).read_text(encoding="utf-8"))
    per_word, per_row = [], []
    for r in rows:
        wav = audio[r["录音"]]
        s0, dur = float(r["开始秒"]), float(r["秒"])
        seg = wav[int(s0 * 16000):int((s0 + dur) * 16000)]
        ws = words_of(r["定稿"])
        if not ws or len(seg) < 1600:
            per_row.append({"行": r["行"], "问题": "没有字或录音太短"})
            continue
        with torch.inference_mode():
            emission, _ = model(seg.unsqueeze(0))
        try:
            spans = aligner(emission[0], tokenizer([w[1] for w in ws]))
        except Exception as exc:  # noqa: BLE001  对不齐（字比声音多太多等）
            per_row.append({"行": r["行"], "问题": f"对不齐：{exc}"})
            continue
        ratio = seg.shape[0] / emission.shape[1] / 16000   # 每一帧多少秒
        times = []
        for w, sp in zip(ws, spans):
            score = float(sum(t.score * len(t) for t in sp) / sum(len(t) for t in sp))
            t0, t1 = sp[0].start * ratio, sp[-1].end * ratio
            times.append((t0, t1))
            per_word.append({"行": r["行"], "字词": w[0], "分数": round(score, 4), "开始": round(s0 + t0, 2)})
        # 标点处的停顿：这个字结束到下一个字开始
        pauses = []
        for i, w in enumerate(ws[:-1]):
            gap = times[i + 1][0] - times[i][1]
            pauses.append({"后面": w[2] or "（无标点）", "停顿秒": round(gap, 3)})
        # 开头 / 结尾没有字对上的人声（用音量判断，比最大音量低 35 dB 以下算安静）
        x = seg.numpy()
        hop = 160
        rms = np.sqrt(np.convolve(x ** 2, np.ones(hop) / hop, mode="same") + 1e-12)
        loud = rms > rms.max() * (10 ** (-35 / 20))
        def speech_sec(a0, a1):
            i0, i1 = int(a0 * 16000), int(a1 * 16000)
            return float(loud[i0:i1].sum()) / 16000 if i1 > i0 else 0.0
        head = speech_sec(0, times[0][0])
        tail = speech_sec(times[-1][1], len(x) / 16000)
        scores = [p["分数"] for p in per_word if p["行"] == r["行"]]
        per_row.append({"行": r["行"], "平均分": round(float(np.mean(scores)), 4), "最低分": round(min(scores), 4),
                        "最低的字词": ws[int(np.argmin(scores))][0], "开头没对上的人声秒": round(head, 2),
                        "结尾没对上的人声秒": round(tail, 2), "停顿": pauses})
        if r["行"] % 20 == 0:
            print("对齐到第", r["行"], "句", flush=True)

    allscores = np.array([p["分数"] for p in per_word])
    q02 = float(np.quantile(allscores, 0.02))
    flagged = [p for p in per_word if p["分数"] <= q02]
    edge = [r for r in per_row if r.get("开头没对上的人声秒", 0) > 0.3 or r.get("结尾没对上的人声秒", 0) > 0.3]
    bad = [r for r in per_row if "问题" in r]
    # 标点和停顿：句号 / 问号处的停顿 vs 逗号处 vs 没标点处
    kinds = {"句号问号": [], "逗号": [], "没标点": []}
    for r in per_row:
        for p in r.get("停顿", []):
            k = "句号问号" if any(c in p["后面"] for c in "。？！?!.") else ("逗号" if any(c in p["后面"] for c in "，,、；;：:") else "没标点")
            kinds[k].append(p["停顿秒"])
    summary = {"字词": len(per_word), "分数分布": {q: round(float(np.quantile(allscores, q)), 4) for q in (0.01, 0.02, 0.05, 0.25, 0.5)},
               "最低2%的字词": len(flagged), "开头或结尾有没对上的人声的句子": len(edge), "对不齐的句子": len(bad),
               "停顿秒（中位数 / 第90百分位）": {k: [round(float(np.median(v)), 3), round(float(np.quantile(v, 0.9)), 3)] if v else None
                                       for k, v in kinds.items()}}
    (out / "汇总.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    (out / "每句.json").write_text(json.dumps(per_row, ensure_ascii=False, indent=1), encoding="utf-8")
    with open(out / "可疑的字词.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["行", "字词", "分数", "开始"])
        w.writeheader()
        w.writerows(sorted(flagged, key=lambda p: p["分数"]))
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
