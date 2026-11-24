"""三个识别引擎投票（voicetwin/data/asr_vote.py）的实测：老师新录的 26 分钟课，204 句，两路独立核对的定稿当标准答案。

数据：research/新录音转录/核对结果/逐句定稿.json（每句有 whisper / funasr / sensevoice 三个引擎各自听到的文字和「定稿」），
      research/新录音转录/结果/逐句对照.json（真的跑程序时 FunASR 原样的输出、程序当时「一键全部文字校正」以后的文字）。

字错率怎么算（和 核对结果/统计.json 一样）：NFKC、英文转小写，只留字母、数字、汉字（标点、空格不算），
按字算编辑距离，所有句子加起来除以定稿的字数；「有错的句子」= 编辑距离不是 0 的句子。

两部分：
  1. 纯文字（不用模型、不联网，tests/test_asr_vote.py 也跑这一部分）：每个引擎单独的、
     程序原来的结果（当时真跑的 whisper + 一键校正）、三个引擎投票以后只改有把握的（= 一键校正会直接采用的）；
  2. 走一遍程序（要 pypinyin、jieba，整合包里都有）：假的识别引擎按上面记下的文字回答，跑「查找可能的错字」+
     「📝 一键全部文字校正」，量校对表里的文字。投票关掉（prepare.proofcheck_vote: off）和打开各一次。

运行（仓库根目录）：PYTHONPATH=. python research/文字校正/三引擎投票/实测.py  → 结果写进同一个文件夹的 实测结果.txt
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
TRUTH = ROOT / "research" / "新录音转录" / "核对结果" / "逐句定稿.json"
RUN = ROOT / "research" / "新录音转录" / "结果" / "逐句对照.json"
ENGINES = ("whisper", "funasr", "sensevoice")


def chars(text: str) -> List[str]:
    t = unicodedata.normalize("NFKC", str(text or "")).lower()
    return [c for c in t if c.isalnum()]


def edit_distance(a: Sequence[str], b: Sequence[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        cur = [i] + [0] * len(b)
        for j, y in enumerate(b, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (x != y))
        prev = cur
    return prev[-1]


def score(hyps: Sequence[str], truths: Sequence[str]) -> Tuple[float, int, int]:
    """(字错率 %, 有错的句子, 错的字数)。"""
    errs = total = bad = 0
    for h, t in zip(hyps, truths):
        tc = chars(t)
        e = edit_distance(chars(h), tc)
        errs += e
        total += len(tc)
        bad += e > 0
    return round(100.0 * errs / max(1, total), 2), bad, errs


def load() -> List[Dict[str, Any]]:
    return json.loads(TRUTH.read_text(encoding="utf-8"))


def vote_text(row: Dict[str, Any]) -> str:
    """现在的文字（whisper）+ 三个引擎投票有把握的改动（「一键全部文字校正」会直接采用的）。"""
    from voicetwin.data import asr_vote

    edits = asr_vote.vote(row["whisper"], {"funasr": row["funasr"], "sensevoice": row["sensevoice"]})
    return asr_vote.apply(row["whisper"], edits, sure_only=True)


def pure_text(rows: Sequence[Dict[str, Any]] = ()) -> Dict[str, Tuple[float, int, int]]:
    """第 1 部分：每个引擎、程序原来的结果、投票。"""
    rows = list(rows) or load()
    truths = [r["定稿"] for r in rows]
    out = {e: score([r[e] for r in rows], truths) for e in ENGINES}
    if RUN.exists():
        before = {r["id"]: r["一键校正以后"] for r in json.loads(RUN.read_text(encoding="utf-8"))}
        if all(r["id"] in before for r in rows):
            out["程序原来（whisper + 一键校正，真跑的）"] = score([before[r["id"]] for r in rows], truths)
    out["三个引擎投票（只改有把握的）"] = score([vote_text(r) for r in rows], truths)
    return out


# ---------------------------------------------------------------------------- 第 2 部分：走一遍程序
class _Fake:
    """假的识别引擎：按记下的文字回答（片段 id → 文字）。"""

    def __init__(self, name: str, answers: Dict[str, str]):
        self.name, self.answers = name, answers
        self.label = f"{name}（记下的文字）"
        self.diff = True
        self.model_id = "recorded"

    def applies(self, rec: Dict[str, Any]) -> bool:
        return True

    def load(self) -> None:
        pass

    def recognize(self, wav: Any, lang: str):
        return self.answers.get(wav, ""), None

    def close(self) -> None:
        pass


def app_flow(vote: bool) -> Dict[str, Any]:
    import numpy as np
    import soundfile as sf

    from voicetwin import workflows as wf
    from voicetwin.config import load_config
    from voicetwin.data import proofcheck as pc
    from voicetwin.data import review

    rows = load()
    raw = {r["id"]: r for r in json.loads(RUN.read_text(encoding="utf-8"))} if RUN.exists() else {}
    answers = {
        pc.ENGINE_FUNASR: {r["id"]: (raw.get(r["id"]) or {}).get("第二个引擎") or r["funasr"] for r in rows},
        pc.ENGINE_SENSEVOICE: {r["id"]: r["sensevoice"] for r in rows},
    }
    saved = (pc._has, pc._make_checker, pc._load_wav16)
    with tempfile.TemporaryDirectory() as tmp:
        cfg = load_config(overrides={
            "workspace": str(Path(tmp) / "ws"), "backend": "dummy",
            "prepare": {"asr": {"engine": "faster-whisper"}, "proofcheck_vote": "auto" if vote else "off"},
        }, user_config=False)
        project = wf.Project(cfg, "投票实测").ensure()
        recs = []
        for r in rows:
            rel = f"clips/{r['id']}.wav"
            sf.write(str(project.root / rel), np.zeros(1600, dtype=np.float32), 16000)
            recs.append({"id": r["id"], "path": rel, "text": r["whisper"], "lang": "zh", "keep": True,
                         "duration": r.get("秒") or 3.0, "split": "train"})
        project.save_manifest(recs)
        pc._has = lambda m: m in ("funasr", "modelscope", "torch")
        pc._make_checker = lambda name, c: _Fake(name, answers.get(name, {}))
        pc._load_wav16 = lambda p, rec: rec["id"]
        try:
            t0 = time.time()
            found = pc.find_suspects(project, cfg)
            fix = wf.run_transcript_fix(cfg, "投票实测", adopt_all=True)
            secs = time.time() - t0
        finally:
            pc._has, pc._make_checker, pc._load_wav16 = saved
        draft = review.load_draft(project)
        texts, red = {}, 0
        for rec in project.load_manifest():
            cur = review.current_values(rec, draft.get(rec["id"]))["text"]
            texts[rec["id"]] = cur
            red += bool(review.analyze(rec, cur)["active"])
    truths = [r["定稿"] for r in rows]
    return {"score": score([texts[r["id"]] for r in rows], truths), "red": red, "seconds": round(secs, 1),
            "note": found.get("note", ""), "adopted": (fix.get("adopted") or {}).get("changes")}


def main() -> int:
    lines = ["三个识别引擎投票的实测（老师新录的 26 分钟课，204 句，两路独立核对的定稿当标准答案）", ""]
    lines.append("1. 纯文字（字错率 %，有错的句子 / 204，错的字数）")
    for name, (cer, bad, errs) in pure_text().items():
        lines.append(f"   {name}：{cer}%，{bad} 句，{errs} 个字")
    lines += ["", "2. 走一遍程序（查找可能的错字 → 📝 一键全部文字校正；假的识别引擎按记下的文字回答）"]
    try:
        for vote in (False, True):
            res = app_flow(vote)
            cer, bad, errs = res["score"]
            lines.append(f"   投票{'打开' if vote else '关掉'}：{cer}%，{bad} 句，{errs} 个字；一键校正直接改了 "
                         f"{res['adopted']} 处，还在「可能有错」的 {res['red']} 句；用时 {res['seconds']} 秒")
            lines.append(f"     查错字的说明：{res['note']}")
    except ImportError as exc:
        lines.append(f"   没做（缺少 {exc.name}：要用有 pypinyin、jieba 的环境）")
    text = "\n".join(lines) + "\n"
    (HERE / "实测结果.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))
    sys.exit(main())
