"""用官方 V5 整合包里真的 GPT-SoVITS 代码和真的底模，把声音分身从头跑一遍（GitHub Actions 上，只有处理器）。

做什么（每一步的结果、用时、出错原因都写进 <out>/结果.json 和 结果.md，出错也接着做下一步能做的）：
  1. 用 espeak-ng 合成一段「讲课录音」（中英文都有）+ 字幕：不是老师的声音，只用来检查整套流程能不能跑通；
  2. 准备素材 → 确认训练素材 → 环境检查；
  3. 训练：和老师用的默认设置一样，只练 V5（v5dev 的 LoRA，官方 s2_train_v3_lora.py）+ GPT，练完自动挑；
  4. 生成一段讲稿（快速档）；
  5. 「一次请求同时生成好几个版本」在 V5 上能不能用：同一句话同时生成 3 个，能不能按数字静音切开、
     每个版本和单独生成的长度差多少、用时多少（程序现在只对 v2 / v2Pro / v2ProPlus 这么做）。

用法：python run_real_v5.py --gsv <整合包解压的文件夹> --out <结果文件夹> [--work <工作文件夹>]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
import wave
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

ZH = [
    "今天我们来学习定语从句，这是英语语法里非常重要的一部分。",
    "首先我们看一个最简单的例子，大家注意关系代词的位置。",
    "这个句子里面，which 指代前面的名词，后面是一个完整的从句。",
    "大家想一想，这里为什么不能用 what 呢？",
    "好，我们再来看第二个例子，这次的先行词是一个人。",
    "当先行词是人的时候，我们一般用 who 或者 that。",
    "注意，介词后面不能直接用 that，要用 which 或者 whom。",
    "接下来我们做几道练习题，大家先自己试一试。",
    "第一题的答案是 B，因为后面的从句缺少主语。",
    "第二题稍微难一点，很多同学容易选错。",
    "我们回到原文，看看作者是怎么用这个结构的。",
    "这篇文章讲的是城市交通的问题，第一段提出了观点。",
    "第二段作者举了两个例子来支持自己的看法。",
    "最后一段是总结，同时也提出了一些建议。",
    "好，今天的内容就到这里，大家课后把单词再复习一遍。",
    "下节课我们讲非限制性定语从句，大家提前预习一下。",
    "记住，逗号后面不能用 that，这是一个常考点。",
    "我们再把刚才的句子读一遍，注意语调和停顿。",
]
EN = [
    "This is a very important sentence in the passage.",
    "The author uses a relative clause to describe the city.",
    "Let's read the second paragraph together.",
    "Which word does the pronoun refer to here?",
    "Remember, we cannot use that after a comma.",
    "The answer to the first question is B.",
]
NARRATE = ("大家好，今天我们继续学习定语从句。上节课我们讲了关系代词 which 和 who 的用法，"
           "今天我们来看介词加关系代词的结构。First, let's look at an example from the passage.")


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def espeak(text: str, voice: str, out: Path) -> None:
    subprocess.run(["espeak-ng", "-v", voice, "-s", "150", "-w", str(out), text], check=True,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def make_lecture(folder: Path, repeats: int = 3) -> Path:
    """espeak-ng 合成「讲课录音」：每句之间 0.5~0.9 秒静音，配一份字幕。返回 wav 路径。"""
    import numpy as np
    import soundfile as sf

    from voicetwin.data.subtitles import Cue, write_srt

    folder.mkdir(parents=True, exist_ok=True)
    tmp = folder / "_tmp.wav"
    parts, cues, t, sr = [], [], 0.5, 22050
    rng = np.random.default_rng(1)
    parts.append(np.zeros(int(0.5 * sr), np.float32))
    for rep in range(repeats):
        for i, s in enumerate(ZH + EN):
            espeak(s, "en-us" if s in EN else "cmn", tmp)
            w, sr = sf.read(str(tmp), dtype="float32")
            w = w * (0.9 + 0.05 * rep)
            parts.append(w)
            cues.append(Cue(t + 0.02, t + len(w) / sr - 0.02, s))
            t += len(w) / sr
            gap = float(rng.uniform(0.5, 0.9))
            parts.append(np.zeros(int(gap * sr), np.float32))
            t += gap
    tmp.unlink(missing_ok=True)
    wav = folder / "第1课.wav"
    sf.write(str(wav), np.concatenate(parts), sr)
    write_srt(cues, wav.with_suffix(".srt"))
    return wav


def wav_info(path: Path) -> dict:
    with wave.open(str(path), "rb") as f:
        return {"sr": f.getframerate(), "seconds": round(f.getnframes() / f.getframerate(), 3), "channels": f.getnchannels()}


class Rec:
    def __init__(self):
        self.items = []
        self.last = ""

    def __call__(self, frac, msg=""):
        if msg and msg != self.last:
            self.items.append([round(float(frac), 4), msg, time.strftime("%H:%M:%S")])
            self.last = msg
            log(f"  进度 {frac:.0%} {msg}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gsv", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--work", default="/tmp/vt_real")
    ap.add_argument("--lora-epochs", type=int, default=2)
    ap.add_argument("--sovits-epochs", type=int, default=2)
    ap.add_argument("--gpt-epochs", type=int, default=2)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    res: dict = {"steps": {}, "started": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save():
        (out / "结果.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")

    def step(name, fn):
        log(f"▶ {name}")
        t0 = time.time()
        try:
            got = fn()
            res["steps"][name] = {"ok": True, "seconds": round(time.time() - t0, 1), "result": got}
            log(f"✅ {name}（{time.time() - t0:.0f} 秒）")
            return got
        except Exception as exc:  # noqa: BLE001
            res["steps"][name] = {"ok": False, "seconds": round(time.time() - t0, 1),
                                  "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
            log(f"❌ {name}：{exc}")
            return None
        finally:
            save()

    import torch

    from voicetwin import workflows as wf
    from voicetwin.backends import gptsovits as g
    from voicetwin.backends.base import SynthRequest, get_backend
    from voicetwin.config import load_config

    res["env"] = {"python": sys.version.split()[0], "torch": torch.__version__, "cuda": torch.cuda.is_available(),
                  "gsv": str(args.gsv), "voicetwin_default_version": g.DEFAULT_VERSION}
    lecture = step("合成讲课录音（espeak-ng）", lambda: str(make_lecture(work / "素材")))
    if not lecture:
        return 1
    res["lecture"] = wav_info(Path(lecture))
    cfg = load_config(overrides={
        "workspace": str(work / "workspace"), "backend": "gptsovits", "prepare": {"asr": {"engine": "none"}},
        "backends": {"gptsovits": {"root": str(args.gsv), "python": sys.executable, "port": 9880, "is_half": False,
                                   "startup_timeout": 900, "version": "v5dev",
                                   "train": {"lora_epochs": args.lora_epochs, "sovits_epochs": args.sovits_epochs,
                                             "gpt_epochs": args.gpt_epochs}}},
    }, user_config=False)
    voice = "实测声音"

    def prepare():
        s = wf.run_prepare(cfg, voice, [str(Path(lecture).parent)])
        wf.review_confirm(cfg, voice)
        return {k: s.get(k) for k in ("clips", "kept", "seconds", "references") if isinstance(s, dict)} or str(s)[:500]

    step("准备素材 + 确认训练素材", prepare)
    step("环境检查", lambda: [{k: r.get(k) for k in ("item", "status", "detail")} for r in wf.doctor(cfg)])

    rec = Rec()

    def train():
        info = wf.run_train(cfg, voice, "gptsovits", select=True, progress=rec, mode="standard")
        project = wf.open_project(cfg, voice, must_exist=True)
        m = project.load_models().get("gptsovits") or {}
        sel = m.get("selection") or {}
        return {"versions": info.get("versions"), "timing": {k: v for k, v in info.items() if k.endswith("_s")},
                "sovits": m.get("sovits"), "gpt": m.get("gpt"), "selected": m.get("selected"),
                "params_lora": (m.get("params") or {}).get("lora"),
                "results": [{k: r.get(k) for k in ("id", "version", "score", "sim", "speaker_pct", "cer")}
                            for r in sel.get("results") or []]}

    trained = step("训练（V5：v5dev 的 LoRA + GPT）并自动挑选", train)
    res["progress_messages"] = rec.items[-60:]
    save()
    if trained:
        for p in trained.get("sovits") or []:
            res.setdefault("detected", {})[Path(p).name] = g.detect_sovits_version(p).get("version")
        save()

    def narrate():
        r = wf.run_narrate(cfg, voice, NARRATE, out=str(out / "生成_快速档.wav"), backend_name="gptsovits",
                           quality="fast")
        return {"file": Path(r.audio_path).name, "duration": r.duration, **wav_info(Path(r.audio_path))}

    step("生成一段讲稿（快速档）", narrate)

    def batch_test():
        project = wf.open_project(cfg, voice, must_exist=True)
        b = get_backend("gptsovits", cfg, project)
        ref = project.load_references()[0]
        req = SynthRequest(text="我们来看第二段，作者举了两个例子来支持自己的看法。", lang="zh",
                           ref_audio=project.abspath(ref["path"]), ref_text=ref["text"], ref_lang=ref.get("lang", "zh"),
                           seed=7, speed=1.0)
        b.start()
        try:
            ver = b._batch_version()
            report = {"model_version": ver, "batch_versions_now": list(g.BATCH_VERSIONS),
                      "why_one_by_one_before": b._single_reason(req, 3)}
            t0 = time.time()
            single = b.synthesize_many(req, 3, work / "一个一个")
            report["one_by_one"] = {"seconds": round(time.time() - t0, 1), "count": len(single),
                                    "durations": [wav_info(p)["seconds"] for p, _ in single], "sr": wav_info(single[0][0])["sr"]}
            g.BATCH_VERSIONS = tuple(g.BATCH_VERSIONS) + (ver,)  # 只在这次实测里把 V5 也当成能同时生成的
            t0 = time.time()
            many = b.synthesize_many(req, 3, work / "同时生成")
            report["batched"] = {"seconds": round(time.time() - t0, 1), "count": len(many),
                                 "durations": [wav_info(p)["seconds"] for p, _ in many]}
            report["why_not_batched"] = b._single_reason(req, 3)
            report["batch_stats"] = {k: v for k, v in (b.batch_stats or {}).items() if k != "last_many"}
            for p, k in many:
                (out / f"同时生成_{k}.wav").write_bytes(Path(p).read_bytes())
            return report
        finally:
            b.stop()

    step("同时生成好几个版本（V5）", batch_test)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save()
    lines = ["# 真实 V5 实测结果（GitHub Actions，只有处理器）", "", f"开始 {res['started']}，结束 {res['finished']}", "",
             f"环境：{json.dumps(res['env'], ensure_ascii=False)}", ""]
    for name, s in res["steps"].items():
        lines.append(f"- {'✅' if s['ok'] else '❌'} {name}（{s['seconds']} 秒）" + ("" if s["ok"] else f"：{s['error']}"))
    (out / "结果.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0 if all(s["ok"] for s in res["steps"].values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
