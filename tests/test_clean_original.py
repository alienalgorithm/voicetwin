"""原声版本尽量干净、不改音色（research/V5极致/轻度美化设计.md 第 1 节 A2、A3、A4）。

测的东西（每条规则一个坏样本、一个好样本）：
- A2 真峰值：fs/4 的正弦、相位 45° 时真峰值比采样峰值高 3.01 dB（数学事实）；低频正弦两者一样；没有声音是 −inf。
  每句音量（_levels）和「去杂音」版本调回响度（_match_levels）按真峰值把最高点限在 −1 dBTP：
  采样峰值只有 −4 dBFS、真峰值高 3 dB 的句子，以前按采样峰值放过去（真峰值 +2 dBTP），现在单独调低到 −1 dBTP；
  不超过的句子系数不动（只乘系数，不改音色）。
- A3 绝对静音自检：拼好的音轨句子以外 0 个非零采样、每句首尾是 0 → 通过；停顿、开头、结尾里有一个 1e-7、
  句子首尾不是 0、句子被截断 → 都抓到。整篇生成时有一个版本不过就报错、一个文件都不写（包括已经没问题的版本）；
  通过时报告里写着查了几句、句子以外查了多少个采样。
- A4 导出格式：选 mp3 / m4a 时小结第一行如实说「停顿里不再是完全的 0」；wav / flac 不说。这句话是实测的：
  同一篇存成 flac 读回来停顿全是 0，存成 mp3 / m4a 读回来停顿里有不是 0 的采样。网页的 MP3 选项也这样写，不写「听不见」。
"""

import json
import uuid
from pathlib import Path

import numpy as np
import pytest

from test_assembly_identical import _fake_noisereduce, _gaps_are_zero, _results
from test_identical_search import _narrator
from voicetwin import workflows as wf
from voicetwin.synth import engine as eng
from voicetwin.utils.audio import fade, load_audio, silence_violations, true_peak_db

SR = 16000
CEIL = eng.PEAK_CEILING_DB


def _uniq(text: str) -> str:
    return text + "编号" + "".join(str(int(c, 16) % 10) for c in uuid.uuid4().hex[:6]) + "。"


def _sample_peak_db(w) -> float:
    return float(20.0 * np.log10(np.max(np.abs(np.asarray(w, dtype=np.float64)))))


def _intersample(seed: int, amp: float, sr: int = SR, seconds: float = 2.0) -> np.ndarray:
    """「说话」一样一段一段的 fs/4 正弦（相位 45°）：采到的点都在 ±0.707 倍处，两个采样之间还原出来的最高点高 3 dB。"""
    rng = np.random.default_rng(seed)
    t = np.arange(int(sr * seconds)) / sr
    env = np.zeros_like(t)
    pos = 0.1
    while pos < seconds - 0.3:
        dur = float(rng.uniform(0.15, 0.25))
        a, b = int(pos * sr), int((pos + dur) * sr)
        env[a:b] = np.hanning(b - a) ** 0.25
        pos += dur + float(rng.uniform(0.05, 0.12))
    w = amp * env * np.sin(2 * np.pi * (sr / 4) * t + np.pi / 4)
    return fade(w.astype(np.float32), sr, in_ms=10.0, out_ms=10.0)


# ============================================================================ A2 真峰值
def test_true_peak_is_3db_above_sample_peak_for_fs4_sine_at_45_degrees():
    sr = 48000
    t = np.arange(sr) / sr
    x = fade((0.5 * np.sin(2 * np.pi * sr / 4 * t + np.pi / 4)).astype(np.float32), sr, in_ms=10.0, out_ms=10.0)
    sp, tp = _sample_peak_db(x), true_peak_db(x)
    assert sp == pytest.approx(20 * np.log10(0.5 * np.sin(np.pi / 4)), abs=0.01)
    assert tp - sp == pytest.approx(3.0103, abs=0.05), (sp, tp)  # 坏样本：采样峰值少算了 3 dB
    low = fade((0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32), sr, in_ms=10.0, out_ms=10.0)
    assert true_peak_db(low) == pytest.approx(_sample_peak_db(low), abs=0.02)  # 好样本：低频时两者一样
    noise = np.random.default_rng(0).standard_normal(sr).astype(np.float32) * 0.1
    assert true_peak_db(noise) >= _sample_peak_db(noise)
    assert true_peak_db(np.zeros(100, np.float32)) == float("-inf")
    assert true_peak_db(np.zeros(0, np.float32)) == float("-inf")


def test_levels_cap_true_peak_not_sample_peak(tmp_path):
    n, _ = _narrator(tmp_path, tier="none")
    n._identical["twin"] = {"loudness": {"sd": 1.3, "p05": -2.3, "p95": 1.2, "n": 30}}
    wavs = [_intersample(k, 0.2) for k in range(4)]
    for w in wavs:  # 测试信号本身：真峰值比采样峰值高 3 dB（按采样峰值调到 −1 dBFS，真峰值就是 +2 dBTP）
        assert true_peak_db(w) - _sample_peak_db(w) > 2.5
    results = _results(wavs)
    layout = n._layout(results)
    # 目标很响：调上去以后采样峰值约 −2.3 dBFS（以前按采样峰值不调低），真峰值约 +0.7 dBTP，现在每句单独调到 −1 dBTP
    n.profile = {"loudness": {"source_lufs": -3.0}}
    gains, info = n._levels(results, layout, SR)
    assert info["peak_lowered"] == [1, 2, 3, 4]
    for w, g in zip(wavs, gains):
        assert true_peak_db(w * np.float32(g)) <= CEIL + 1e-3
        assert true_peak_db(w * np.float32(g)) == pytest.approx(CEIL, abs=1e-3)  # 只调到正好 −1，不多压
    # 好样本：目标不响，真峰值都在 −1 dBTP 以下，一句都不调低
    n.profile = {"loudness": {"source_lufs": -30.0}}
    gains, info = n._levels(results, layout, SR)
    assert info["peak_lowered"] == []
    assert all(true_peak_db(w * np.float32(g)) < CEIL for w, g in zip(wavs, gains))


def test_match_levels_caps_true_peak_of_the_denoised_version(tmp_path):
    n, _ = _narrator(tmp_path, tier="none")
    a0, a1 = _intersample(1, 0.05), _intersample(2, 0.95)
    assert _sample_peak_db(a1) < CEIL - 2 and true_peak_db(a1) > CEIL  # 第 2 句：采样峰值约 −3.5 dBFS 没超，真峰值超了
    b1 = a1 * np.float32(0.5)  # 去杂音让声音小了 6 dB
    gains_b, info = n._match_levels([a0, a1], [a0, b1], [1.0, 1.0], [1], SR)
    assert info["peak_lowered"] == [2]  # 以前按采样峰值：没超 −1 dBFS，不调低，真峰值超过 −1 dBTP
    assert true_peak_db(b1 * np.float32(gains_b[1])) == pytest.approx(CEIL, abs=1e-3)
    assert gains_b[0] == 1.0  # 没去杂音的句子：系数一个比特都不差
    # 好样本：调回响度以后真峰值也没超，照常调回去（+6 dB）
    c1 = _intersample(3, 0.1)
    gains_c, info = n._match_levels([a0, c1], [a0, c1 * np.float32(0.5)], [1.0, 1.0], [1], SR)
    assert info["peak_lowered"] == [] and 20 * np.log10(gains_c[1]) == pytest.approx(6.02, abs=0.05)


# ============================================================================ A3 绝对静音自检
def _rendered(tmp_path):
    n, _ = _narrator(tmp_path, tier="none")
    wavs = [_intersample(k, 0.3, seconds=1.0) for k in range(3)]
    results = _results(wavs)
    layout = n._layout(results)
    return n, layout, n._render(layout, [(w, SR) for w in wavs], SR)


def test_silence_check_passes_rendered_track_and_catches_every_kind_of_break(tmp_path):
    n, layout, audio = _rendered(tmp_path)
    regions = eng.sentence_regions(layout, SR)
    good = silence_violations(audio, regions)
    assert good["outside_nonzero"] == 0 and good["edges_bad"] == [] and good["first_at"] is None
    assert good["outside_samples"] == len(audio) - sum(m for _, m in regions)
    assert np.count_nonzero(audio) > 0
    rec = n._check_silence([("未去杂音", audio)], layout, SR)
    assert rec == {"passed": True, "sentences": 3,
                   "versions": {"未去杂音": {"outside_nonzero": 0, "outside_samples": good["outside_samples"],
                                             "edges_bad": []}}}
    (s0, m0), (s1, _) = regions[0], regions[1]
    gap = (s0 + m0 + s1) // 2
    cases = {"停顿": gap, "开头": 3, "结尾": len(audio) - 1}
    for name, at in cases.items():
        bad = audio.copy()
        bad[at] = 1e-7
        got = silence_violations(bad, regions)
        assert got["outside_nonzero"] == 1 and got["first_at"] == at, name
        with pytest.raises(RuntimeError, match="程序自检没通过：「去杂音」版本句子以外有 1 个采样不是 0"):
            n._check_silence([("未去杂音", audio), ("去杂音", bad)], layout, SR)
    for at, which in ((s1, 2), (s1 + regions[1][1] - 1, 2), (s0, 1)):  # 句子第一个 / 最后一个采样不是 0
        bad = audio.copy()
        bad[at] = 0.01
        assert silence_violations(bad, regions)["edges_bad"] == [which]
        with pytest.raises(RuntimeError, match=f"第 {which} 句的开头或结尾不是 0"):
            n._check_silence([("未去杂音", bad)], layout, SR)
    # 最后一句被截断（音轨比排好的位置短）：截断处不是 0，也抓到
    s2, m2 = regions[2]
    cut = audio[: s2 + int(np.argmax(np.abs(audio[s2:s2 + m2]))) + 1]
    assert np.abs(cut[-1]) > 0 and silence_violations(cut, regions)["edges_bad"] == [3]


def test_narrate_writes_nothing_when_the_silence_check_fails(prepared, tmp_path, monkeypatch):
    cfg, project, _ = prepared
    out = tmp_path / "坏的"
    orig = eng.Narrator._render

    def leaky(self, layout, pieces, sr, gains=None):
        audio = orig(self, layout, pieces, sr, gains)
        audio[0] = np.float32(1e-6)  # 开头的静音里多了一个很小的数
        return audio

    monkeypatch.setattr(eng.Narrator, "_render", leaky)
    with pytest.raises(RuntimeError, match="程序自检没通过：「未去杂音」版本句子以外有 1 个采样不是 0（第一个在第 0.000 秒）"):
        wf.run_narrate(cfg, project.voice, _uniq("静音自检坏样本第一句") + _uniq("第二句"),
                       out=str(out / "课.wav"), quality="fast")
    assert not out.exists() or not any(out.iterdir())


def test_identical_writes_no_version_when_one_version_fails(prepared, tmp_path, monkeypatch):
    """两个版本：只有「去杂音」版本坏了，也一个文件都不写（「未去杂音」版本也不写，报告、字幕都没有）。"""
    cfg, project, _ = prepared
    _fake_noisereduce(monkeypatch)
    monkeypatch.setattr(eng, "_vram_tier", lambda: "none")
    orig = eng.Narrator._make_variants

    def bad_b(self, *a, **k):
        variants, audio = orig(self, *a, **k)
        b = audio["去杂音"].copy()
        b[-1] = np.float32(1e-6)  # 结尾最后一个采样
        return variants, dict(audio, **{"去杂音": b})

    monkeypatch.setattr(eng.Narrator, "_make_variants", bad_b)
    out = tmp_path / "两个版本"
    with pytest.raises(RuntimeError, match="「去杂音」版本句子以外有 1 个采样不是 0"):
        wf.run_narrate(cfg, project.voice, _uniq("两个版本自检第一句，先停一下") + _uniq("第二句"),
                       out=str(out / "课.wav"), candidates=2)
    assert not out.exists() or not any(out.iterdir())


def test_identical_report_records_the_silence_check(prepared, tmp_path, monkeypatch):
    cfg, project, _ = prepared
    _fake_noisereduce(monkeypatch)
    monkeypatch.setattr(eng, "_vram_tier", lambda: "none")
    res = wf.run_narrate(cfg, project.voice, _uniq("自检通过第一句，先停一下") + _uniq("第二句也在这里"),
                         out=str(tmp_path / "好的.wav"), candidates=2)
    rep = json.loads(res.report_path.read_text(encoding="utf-8"))
    chk = rep["silence_check"]
    assert chk["passed"] is True and chk["sentences"] == len(rep["segments"])
    assert set(chk["versions"]) == {v["name"] for v in rep["variants"]} == {"未去杂音", "去杂音"}
    for v in rep["variants"]:
        row = chk["versions"][v["name"]]
        assert row["outside_nonzero"] == 0 and row["edges_bad"] == [] and row["outside_samples"] > 0
        _gaps_are_zero(v["path"], rep["segments"])  # 写出的文件也是：句子之间每个采样都是 0
    assert eng.LOSSY_NOTE not in rep["notes"]  # wav：不说「不再是完全的 0」


# ============================================================================ A4 导出格式
def _gap_nonzero(path: Path, segments, margin: float = 0.002) -> int:
    """读回来的文件里，开头和句子之间（报告里的时间只留到毫秒，两边各让 2 毫秒）有几个采样不是 0。"""
    wav, sr = load_audio(path)
    m = int(margin * sr)
    n = int(np.count_nonzero(wav[: int(round(segments[0]["start"] * sr)) - m]))
    for a, b in zip(segments, segments[1:]):
        n += int(np.count_nonzero(wav[int(round(a["end"] * sr)) + m:int(round(b["start"] * sr)) - m]))
    return n


def test_lossy_formats_say_pauses_are_not_exact_zero_and_it_is_measured(prepared, tmp_path):
    cfg, project, _ = prepared
    script = _uniq("格式测试第一句，先停一下") + _uniq("第二句在这里") + _uniq("第三句")
    got = {}
    for ext in ("wav", "flac", "mp3", "m4a"):
        res = wf.run_narrate(cfg, project.voice, script, out=str(tmp_path / ext / f"课.{ext}"), quality="fast")
        rep = json.loads(res.report_path.read_text(encoding="utf-8"))
        assert res.audio_path.suffix == f".{ext}"
        assert rep["silence_check"]["passed"] is True  # 压缩之前的那条音轨每个版本都查过
        got[ext] = (_gap_nonzero(res.audio_path, rep["segments"]), list(res.notes))
    for ext in ("wav", "flac"):  # 好样本：精确的 0，不说
        assert got[ext][0] == 0, ext
        assert eng.LOSSY_NOTE not in got[ext][1], ext
    for ext in ("mp3", "m4a"):  # 坏样本：读回来停顿里真有不是 0 的采样，小结第一行如实说
        assert got[ext][0] > 0, ext
        assert got[ext][1][0] == eng.LOSSY_NOTE, ext
    assert "wav 或 flac" in eng.LOSSY_NOTE and "不再是完全的 0" in eng.LOSSY_NOTE


def test_webui_mp3_choice_says_pauses_are_not_exact_zero():
    from voicetwin.webui import app as A

    labels = dict((v, k) for k, v in A.FORMAT_CHOICES)
    assert "不再是完全的 0" in labels["mp3"] and "要绝对静音请选 WAV" in labels["mp3"]
    assert "听不见" not in labels["mp3"]  # 没在她的声音上实测过听不听得见，不写
    assert "绝对静音" in labels["wav"]
