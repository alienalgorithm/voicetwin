"""参考录音按 V5 的真实用法过滤（research/V5极致/统一计划.md A6）。

V5 只拿参考录音最后 500 帧梅尔谱（每秒 100 帧 = 5.0 秒）当声学提示，GPT 接着参考录音的结尾往下说。所以量每条参考录音的
最后 5 秒：里面有超过你句中停顿 p97 的长停顿、或者结尾不是说话声（停了一下以后的呼吸声、杂音），就只在别的都不合适时才用。
这里用停顿位置已知的合成片段测：只算最后 5 秒、跨进来的停顿只算一截、呼吸 / 杂音 / 真的字分得开、门槛来自你自己的停顿统计、
references.json 和参考录音库（「一模一样」挑参考）都是干净的排在前面、干净的不够时照样用、库重建时不重新读音频。只用处理器。"""

import json

import numpy as np
import pytest
import soundfile as sf
from scipy import signal

from voicetwin import workflows as wf
from voicetwin.backends.workers.dummy_worker import synth_speech
from voicetwin.data import references as R
from voicetwin.style import twin_profile as tp
from voicetwin.synth import search as S
from voicetwin.synth.script import ScriptSegment
from voicetwin.utils.audio import trim_silence

SR = 16000


def _voice(d, sr=SR, f=180.0, a=0.3):
    t = np.arange(int(d * sr)) / sr
    return (a * np.sin(2 * np.pi * f * t) * (0.6 + 0.4 * np.sin(2 * np.pi * 4 * t))).astype(np.float32)


def _sil(d, sr=SR):
    return np.zeros(int(d * sr), np.float32)


def _breath(d, sr=SR, a=0.03, seed=5):
    """呼吸声：300~3000 Hz 的噪声（没有音高）。"""
    b, aa = signal.butter(4, [300 / (sr / 2), 3000 / (sr / 2)], btype="band")
    return signal.lfilter(b, aa, a * np.random.default_rng(seed).normal(size=int(d * sr))).astype(np.float32)


def _clip(parts, sr=SR, seed=0):
    w = np.concatenate(parts)
    return w + np.random.default_rng(seed).normal(0, 0.0005, w.size).astype(np.float32)


def _tail(wav, sr=SR, limit=0.8):
    """和挑参考时一样：去掉首尾静音（留 80 毫秒）当参考录音，停顿按量你句中停顿的方法量。"""
    _, a, b = trim_silence(wav, sr, pad_ms=80)
    return R.reference_tail(wav, sr, a, b, tp.source_gaps(wav, sr), limit), (a / sr, b / sr)


# ---------------------------------------------------------------------------- 最后 5 秒里的停顿
def test_only_pauses_in_the_last_five_seconds_count():
    # 停顿在倒数 1.5~2.7 秒：在最后 5 秒里，1.2 秒 > 0.8 → 有问题
    got, _ = _tail(_clip([_sil(.4), _voice(3.0), _sil(1.2), _voice(1.5), _sil(.4)]))
    assert got["tail_end"] == "voice" and abs(got["tail_pause"] - 1.2) < 0.06 and got["tail_bad"]
    # 同样 1.2 秒的停顿在开头（离结尾 5.5 秒以上）：V5 用不到那一段，不算
    got, _ = _tail(_clip([_sil(.4), _voice(1.0), _sil(1.2), _voice(5.5), _sil(.4)]))
    assert got == {"tail_pause": 0.0, "tail_end": "voice", "tail_bad": False}
    # 跨进最后 5 秒的停顿只算落在里面的那一截（停顿 1.2 秒，约 0.3 秒在最后 5 秒里）
    wav = _clip([_sil(.4), _voice(2.0), _sil(1.2), _voice(4.6), _sil(.4)])
    got, (s0, s1) = _tail(wav)
    gap = tp.source_gaps(wav, SR)[0]
    assert abs(got["tail_pause"] - (gap[1] - (s1 - 5.0))) < 0.002 and 0.2 < got["tail_pause"] < 0.5
    assert not got["tail_bad"]
    # 门槛是你自己的 p97：同一条录音，p97 比这个停顿长就没问题；量不出来（None）时不查这一项
    wav = _clip([_sil(.4), _voice(3.0), _sil(1.2), _voice(1.5), _sil(.4)])
    assert not _tail(wav, limit=1.5)[0]["tail_bad"] and not _tail(wav, limit=None)[0]["tail_bad"]


# ---------------------------------------------------------------------------- 结尾是不是说话声
@pytest.mark.parametrize("sr", [16000, 32000, 48000])
def test_ending_breath_or_noise_after_a_pause_is_not_voice(sr):
    body = _voice(5.0, sr)
    breath = _clip([_sil(.4, sr), body, _sil(.3, sr), _breath(.4, sr), _sil(.4, sr)], sr)
    click = _clip([_sil(.4, sr), body, _sil(.3, sr), _breath(.03, sr, a=0.3), _sil(.4, sr)], sr)
    for wav in (breath, click):
        got, (_, s1) = _tail(wav, sr)
        assert s1 > 5.6  # 呼吸 / 杂音比静音阈值响：去首尾静音时留下来了（不查就会发给引擎）
        assert got["tail_end"] == "not_voice" and got["tail_bad"] and got["tail_pause"] < 0.8
    # 停了一下以后是一个真的字（有音高；更低、更轻也一样）：结尾是说话声
    for last in (_voice(.3, sr, f=220), _voice(.25, sr, f=90, a=0.05)):
        got, _ = _tail(_clip([_sil(.4, sr), body, _sil(.3, sr), last, _sil(.4, sr)], sr), sr)
        assert got == {"tail_pause": got["tail_pause"], "tail_end": "voice", "tail_bad": False}
    # 像说话的合成声音（有共振峰、音高在变）：最后一个字是说话声，呼吸不是
    words = synth_speech("我们今天来学习一个新的句型大家注意", sr=sr, seed=1)
    word = synth_speech("好", sr=sr, seed=2)
    got, _ = _tail(_clip([_sil(.4, sr), words, _sil(.3, sr), word, _sil(.4, sr)], sr), sr)
    assert got["tail_end"] == "voice"
    got, _ = _tail(_clip([_sil(.4, sr), words, _sil(.3, sr), _breath(.4, sr, a=0.05), _sil(.4, sr)], sr), sr)
    assert got["tail_end"] == "not_voice"
    # 没有停顿：结尾就是说话声
    assert _tail(_clip([_sil(.4, sr), _voice(6.0, sr), _sil(.4, sr)], sr), sr)[0]["tail_end"] == "voice"


# ---------------------------------------------------------------------------- 门槛来自你自己的停顿
def test_pause_limit_is_her_own_p97(tmp_path):
    from conftest import make_cfg

    project = wf.open_project(make_cfg(tmp_path / "ws"), "停顿")
    recs = [{"id": f"c{i}", "keep": True, "pauses": [0.2 + 0.01 * i, 5.0, 0.05]} for i in range(10)]
    # 说话习惯还没量：用素材准备时每段量的句中停顿（只算 0.12~4 秒的）算 p97
    assert R.pause_limit(project, recs) == pytest.approx(np.percentile([0.2 + 0.01 * i for i in range(10)], 97))
    assert R.pause_limit(project, recs[:7]) is None  # 不到 8 个停顿：不查
    # 量出来了：用 twin_profile.json 里的 p97（「一模一样」打分判断异常停顿用的也是它）
    project.root.mkdir(parents=True, exist_ok=True)
    (project.root / tp.TWIN_PROFILE_FILE).write_text(json.dumps(
        {"version": tp.TWIN_PROFILE_VERSION, "inner_pause_p97": {"value": 0.734, "n": 400}}), encoding="utf-8")
    assert R.pause_limit(project, recs) == 0.734


# ---------------------------------------------------------------------------- 「一模一样」挑参考：干净的排前面
def _seg(text, kind="statement"):
    return ScriptSegment(text=text, display=text, lang="zh", kind=kind, index=0, pause_after="sentence")


def test_shortlist_puts_clean_tails_first():
    refs = [{"id": "q_bad", "path": "a.wav", "text": "大家想一想，这道题应该怎么做呢？", "lang": "zh", "kind": "question",
             "source": "v1", "score": 3.0, "tail_bad": True},
            {"id": "q_ok", "path": "b.wav", "text": "这个句子的主语到底是哪个词呢？", "lang": "zh", "kind": "question",
             "source": "v2", "score": 2.0},
            {"id": "s_bad", "path": "c.wav", "text": "我们今天先来看第一个例子，大家注意听。", "lang": "zh",
             "kind": "statement", "source": "v3", "score": 2.9, "tail_bad": True},
            {"id": "s_ok", "path": "d.wav", "text": "接下来我们再看一个稍微复杂一点的例子。", "lang": "zh",
             "kind": "statement", "source": "v4", "score": 2.5}]
    pool = S.pool_from_refs(refs)
    prior = S.prior_scores(pool)
    q = _seg("大家觉得这个例子应该怎么改呢？", kind="question")
    got = [e["id"] for e in S.shortlist_refs(q, pool, prior, 2)]
    assert got == ["q_ok", "s_ok"]  # 分数更高的问句最后 5 秒有问题：先用干净的；第二条（别的视频）也挑干净的
    # 干净的不够 R 条：有问题的照样补上（不会一条参考都没有）
    assert [e["id"] for e in S.shortlist_refs(q, pool, prior, 4)][2:] == ["q_bad", "s_bad"]
    only_bad = [e for e in pool if e["id"].endswith("_bad")]
    assert [e["id"] for e in S.shortlist_refs(q, only_bad, S.prior_scores(only_bad), 2)] == ["q_bad", "s_bad"]


# ---------------------------------------------------------------------------- references.json 和参考录音库
def _project(tmp_path, specs):
    """specs：[(id, 文字, 片段, 声纹分数)]。"""
    from conftest import make_cfg

    project = wf.open_project(make_cfg(tmp_path / "ws"), "参考")
    project.clips_dir.mkdir(parents=True, exist_ok=True)
    recs = []
    for i, (cid, text, wav, sim) in enumerate(specs):
        path = project.clips_dir / f"{cid}.wav"
        sf.write(str(path), wav, SR)
        recs.append({"id": cid, "path": project.relpath(path), "text": text, "lang": "zh", "keep": True,
                     "split": "train", "rate": 4.5, "forced_cuts": 0, "duration": round(len(wav) / SR, 3),
                     "speaker_sim": sim, "snr": 30.0, "source": f"v{i}", "start": 0.0, "end": len(wav) / SR,
                     "pauses": [0.25, 0.3, 0.28]})
    return project, recs


def _specs():
    long_pause = _clip([_sil(.4), _voice(3.0), _sil(1.5), _voice(2.0), _sil(.4)], seed=1)
    breath_end = _clip([_sil(.4), _voice(5.5), _sil(.3), _breath(.4), _sil(.4)], seed=2)
    clean = [_clip([_sil(.4), _voice(6.0 + 0.1 * k), _sil(.4)], seed=10 + k) for k in range(5)]
    return ([("q_pause", "大家想一想这段代码会输出什么呢？", long_pause, 0.99),
             ("q_clean", "这个句子的主语到底是哪个词呢？", clean[0], 0.85),
             ("s_breath", "我们今天先来看第一个例子大家注意听。", breath_end, 0.99)]
            + [(f"s_clean{k}", f"这是第{'甲乙丙丁'[k - 1]}句完整的陈述句。", clean[k], 0.85) for k in range(1, 5)])


def test_references_json_prefers_clean_tails(tmp_path):
    project, recs = _project(tmp_path, _specs())
    refs = R.select_references(project, recs, {"references": {"count": 4}})
    by_id = {r["id"]: r for r in refs}
    # 分数最高的问句（最后 5 秒停了 1.5 秒）、分数最高的陈述句（停了一下以后是呼吸声）都换成了干净的
    assert "q_pause" not in by_id and by_id["q_clean"]["kind"] == "question" and not by_id["q_clean"]["tail_bad"]
    assert "s_breath" not in by_id and sum(1 for r in refs if r["kind"] == "statement") == 4
    assert all(r["tail_end"] == "voice" and r["tail_pause"] == 0.0 for r in refs)
    saved = json.loads(project.references_path.read_text(encoding="utf-8"))
    assert [r["id"] for r in saved] == [r["id"] for r in refs]
    # 只有有问题的：照样用（不会没有问句的参考），排在同一种句子的最后
    project2, recs2 = _project(tmp_path / "2", [s for s in _specs() if s[0] != "q_clean"])
    refs2 = R.select_references(project2, recs2, {"references": {"count": 4}})
    q = [r for r in refs2 if r["kind"] == "question"]
    assert [r["id"] for r in q] == ["q_pause"] and q[0]["tail_bad"] and q[0]["tail_pause"] > 1.4


def test_reference_bank_marks_tails_and_reuses_the_measurement(tmp_path, monkeypatch):
    project, recs = _project(tmp_path, _specs())
    bank = {e["id"]: e for e in R.build_reference_bank(project, recs)}
    data = R.load_reference_bank(project)
    assert data["version"] == R.BANK_VERSION == 2 and data["tail_limit"] == pytest.approx(0.3, abs=0.01)
    assert bank["q_pause"]["tail_bad"] and bank["q_pause"]["tail_end"] == "voice"
    assert abs(bank["q_pause"]["tail_pause"] - 1.5) < 0.06
    assert bank["s_breath"]["tail_bad"] and bank["s_breath"]["tail_end"] == "not_voice"
    assert not any(bank[k]["tail_bad"] for k in bank if "clean" in k)
    # 「一模一样」挑参考：库里有问题的排在干净的后面
    pool = S.pool_from_bank(data)
    q = _seg("大家觉得这个例子应该怎么改呢？", kind="question")
    assert S.shortlist_refs(q, pool, S.prior_scores(pool), 1)[0]["id"] == "q_clean"
    # 重建（片段文件、去静音的位置、里面的停顿都没变）：结尾不再读音频；门槛变了（你的 p97 重新量了）只重新判断
    calls = []
    real = R.tail_ending
    monkeypatch.setattr(R, "tail_ending", lambda *a, **k: calls.append(1) or real(*a, **k))
    (project.root / tp.TWIN_PROFILE_FILE).write_text(json.dumps(
        {"version": tp.TWIN_PROFILE_VERSION, "inner_pause_p97": {"value": 2.0, "n": 400}}), encoding="utf-8")
    again = {e["id"]: e for e in R.build_reference_bank(project, recs)}
    assert calls == [] and R.load_reference_bank(project)["tail_limit"] == 2.0
    assert not again["q_pause"]["tail_bad"] and again["s_breath"]["tail_bad"]
    assert {k: v["tail_end"] for k, v in again.items()} == {k: v["tail_end"] for k, v in bank.items()}
    # 片段文件变了：这一条重新量
    path = project.abspath(recs[0]["path"])
    sf.write(str(path), _clip([_sil(.4), _voice(6.5), _sil(.4)], seed=99), SR)
    R.build_reference_bank(project, recs)
    assert calls == [1]
