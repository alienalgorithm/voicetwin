import sys
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from voicetwin.backends.workers.dummy_worker import synth_speech  # noqa: E402
from voicetwin.config import load_config  # noqa: E402
from voicetwin.data.subtitles import Cue, write_srt  # noqa: E402

SENTENCES = [
    "今天我们来学习Python里面的列表推导式。",
    "它可以让代码变得更加简洁，也更容易阅读。",
    "首先我们看一个最简单的例子。",
    "大家想一想，这段代码输出的结果是什么？",
    "Next, let's look at a slightly more complex example.",
    "We can add a condition at the end of the expression.",
    "好，这就是今天的全部内容，我们下节课再见！",
    "接下来我们讲字典推导式，它和列表推导式非常相似。",
    "区别在于我们需要同时给出键和值。",
    "This makes it very convenient to build lookup tables.",
]


def make_lecture(path: Path, repeats: int = 3, sr: int = 44100, f0: float = 150.0, with_srt: bool = True, seed: int = 1):
    rng = np.random.default_rng(seed)
    parts = [np.zeros(int(0.6 * sr))]
    cues = []
    t = 0.6
    for rep in range(repeats):
        for i, s in enumerate(SENTENCES):
            w = synth_speech(s, sr=sr, seed=rep * 100 + i + seed * 1000, f0=f0)
            parts.append(w)
            cues.append(Cue(t + 0.05, t + len(w) / sr - 0.05, s))
            t += len(w) / sr
            gap = rng.uniform(0.4, 1.0)
            parts.append(np.zeros(int(gap * sr)))
            t += gap
    wav = np.concatenate(parts).astype(np.float32)
    wav += rng.normal(0, 0.0015, len(wav)).astype(np.float32)
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)
    if with_srt:
        write_srt(cues, path.with_suffix(".srt"))
    return path


@pytest.fixture(scope="session")
def lecture_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("lectures")
    make_lecture(d / "第1课.wav")
    return d


@pytest.fixture(autouse=True)
def _mixed_text_dry_run(monkeypatch):
    """测试里没有真的 torch / BERT：VoiceTwin 自己的「处理文字」（gsv_scripts/get_text_mixed.py）按测试用的办法跑
    （VOICETWIN_TEXT_DRYRUN=1，用仿真 GPT-SoVITS 目录里的替身模块）。不设的话它在测试里总是失败：以前悄悄退回官方的方法，
    现在「一模一样」档和练 V5 时会停下（官方的方法会丢掉英文）。要测它失败的情况，测试自己弄坏替身。"""
    monkeypatch.setenv("VOICETWIN_TEXT_DRYRUN", "1")


@pytest.fixture(autouse=True)
def _no_sv_download(monkeypatch):
    """测试不联网下载声纹模型（要测下载本身的测试自己换回真的 download，并换掉网络请求）。"""
    from voicetwin.eval import sv_models

    monkeypatch.setattr(sv_models, "download", lambda cfg, progress=None, keys=None: [])


#: 老师的母本（research/文字校正/老师的母本/）：母本_原文.csv 是识别引擎写的句子（10 月份修缮以前）；程序自带的母本
#: （core_corpus.tsv）的正确答案是 母本_录音核对后.csv（10 月份只看文字修缮的 母本_修缮后.csv，11-24 又对着录音逐句核对、
#: 改了 27 句；core_corpus.tsv 和它一模一样，见 test_mother_first.py::test_builtin_corpus_is_the_checked_master）。
MOTHER_DIR = ROOT / "research" / "文字校正" / "老师的母本"
CHECKED_CSV = MOTHER_DIR / "母本_录音核对后.csv"
#: 识别的时候整句漏掉的英文例句（录音里有，这一句识别的文字里一个字都没有；11-24 录音核对时补进母本）。一键校正不会把
#: 母本里多出来的一整句补进来：句首句尾漏掉的字只在同一个 id 的那一句补，而且最多 3 个（mother_first.EDGE_MAX），
#: 再长的可能是切开的后半句、在下一个片段里（见 test_mother_first.py::test_split_piece_is_not_padded_...）。
#: 所以一键校正以后这一句还是识别的样子，正确答案按「母本去掉这一句」算。
MISSED_BY_ASR = {"0006_9498bb_0053": "it's the same doctor as my father consulted."}


def teacher_answer(rid: str, checked: str) -> str:
    """一键校正以后这一句应该是的样子：录音核对后的母本 checked，识别时整句漏掉的英文例句不算（MISSED_BY_ASR）。"""
    tail = MISSED_BY_ASR.get(rid)
    if tail:
        assert checked.endswith(tail), (rid, checked)
        return checked[:-len(tail)]
    return checked


def make_cfg(workspace: Path, **extra):
    overrides = {"workspace": str(workspace), "backend": "dummy", "prepare": {"asr": {"engine": "none"}},
                 "speaker_encoder": "mfcc", "similarity": {"model_dir": str(Path(workspace) / "_sv_models")}}
    for k, v in extra.items():
        overrides[k] = dict(overrides[k], **v) if k == "similarity" else v
    # 以前的测试都按 v2ProPlus 写（单一版本的流程）；V5 的测试自己写 version: v5dev（见 test_gsv_v5.py）
    gsv = dict((overrides.get("backends") or {}).get("gptsovits") or {})
    gsv.setdefault("version", "v2ProPlus")
    overrides["backends"] = dict(overrides.get("backends") or {}, gptsovits=gsv)
    return load_config(overrides=overrides, user_config=False)


@pytest.fixture(scope="session")
def prepared(tmp_path_factory, lecture_dir):
    """素材准备 + 风格分析，session 级共享（较慢）。"""
    from voicetwin import workflows as wf

    ws = tmp_path_factory.mktemp("ws")
    cfg = make_cfg(ws)
    summary = wf.run_prepare(cfg, "测试声音", [str(lecture_dir)])
    wf.review_confirm(cfg, "测试声音")  # 和老师一样：先「✅ 确认训练素材」才能训练（复制这个声音时确认记录跟着走）
    project = wf.open_project(cfg, "测试声音", must_exist=True)
    return cfg, project, summary


def confirm_material(cfg, voice: str) -> None:
    """测试里训练前「✅ 确认训练素材」（只记下现在的素材，不重新统计）：和老师一样，没确认不能训练。"""
    from voicetwin import workflows as wf
    from voicetwin.data import review

    p = wf.open_project(cfg, voice, must_exist=True)
    review.save_confirmed(p, p.load_manifest())
