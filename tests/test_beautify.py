"""「轻度美化」版本 C（research/V5极致/轻度美化设计.md 第 2、3、4、7、8 节）。

每条规则一个坏样本、一个好样本（设计第 8 节）：
1. s = 0 时逐位相同；去齿音 / 压缩没触发的句子也逐位相同。
2. 合成的讲课音轨逐句走 C：停顿里非零采样 0 个、每句首尾 0、长度相同、互相关滞后 0；
   同一条链在拼好的整篇上跑（坏样本）必须被 G9（绝对静音自检）抓到。
3. FIR 实测幅度响应：80 Hz–16 kHz 以内 |T| ≤ 2.5 dB，以外是 0 dB（误差写在测试里）；最小相位：冲激响应主峰之前没有能量；
   minimum_phase 的「平方根」陷阱：不按平方设计，得到的 dB 只有一半。
4. 真峰值：fs/4 正弦、相位 45° 时真峰值 = 采样峰值 + 3.01 dB；S5 调电平按真峰值限在 −1 dBTP。
5. 压缩的增益衰减 ≤ 3 dB（s = 0.5 时 ≤ 1.5 dB），去齿音的衰减 ≤ 4 dB。
6. 关卡（假打分器）：Δ = −10 个百分点、CER 上升、+12 dB 搁架（G6）、8 kHz 以上能量超出 p95（G7）、20 dB 白噪声（G8）、
   移调一个半音（G4）都被拒；只改音量全部通过。
7. 一次性校准、整篇减半到 0、生成流程里的接线、设置默认关、网页文字（不写没实测过的说法）。
G10（美学评分模型）只在 CI 的打擂台里跑，这里不下载任何模型。
"""

import json
import math
import re
import zlib

import numpy as np
import pytest
from scipy import signal

from test_assembly_identical import _fake_noisereduce, _uniq
from test_identical_search import _narrator
from voicetwin import workflows as wf
from voicetwin.eval.metrics import Score
from voicetwin.synth import beautify as BT
from voicetwin.synth import engine as eng
from voicetwin.synth.script import ScriptSegment
from voicetwin.utils.audio import fade, silence_violations, speech_noise_floor_db, true_peak_db

SR = 48000


# ============================================================================ 合成的「讲课」声音
def _speechy(sec=3.0, seed=0, sib=0.06, f0=180.0, sr=SR, amp=0.15):
    """像说话的一句：有音高起伏的谐波（F0 附近）、一个一个「字」的包络、几处齿音（5–8 kHz 噪声）；首尾淡入淡出。"""
    rng = np.random.default_rng(seed)
    n = int(sec * sr)
    t = np.arange(n) / sr
    ff = f0 * (1 + 0.05 * np.sin(2 * np.pi * 1.5 * t))
    ph = 2 * np.pi * np.cumsum(ff) / sr
    v = sum((0.6 / k) * np.sin(k * ph) for k in range(1, int(0.45 * sr / f0 / 1.06)))
    env = np.zeros(n)
    pos = 0.05
    while pos < sec - 0.3:
        d = rng.uniform(0.12, 0.3)
        a, b = int(pos * sr), int((pos + d) * sr)
        env[a:b] = np.hanning(b - a) ** 0.5 * rng.uniform(0.3, 1.0)
        pos += d + rng.uniform(0.03, 0.1)
    x = v * env * amp
    if sib:
        sos = signal.butter(4, [5000, 8000], btype="band", fs=sr, output="sos")
        nz = signal.sosfilt(sos, rng.standard_normal(n))
        senv = np.zeros(n)
        m = int(0.08 * sr)
        for c in np.linspace(0.3, sec - 0.4, max(2, int(sec))):
            a = int(c * sr)
            senv[a:a + m] = np.hanning(m)
        x = x + sib * nz * senv
    return fade(x.astype(np.float32), sr, in_ms=10.0, out_ms=10.0)


def _her(seed=0, n=6):
    """「她本人」的统计量：几段合成录音，来自 3 个文件（σ_她 按文件分组量）。"""
    clips = [(_speechy(4.0, seed + k, sib=0.05 + 0.03 * (k % 3)), SR, f"src{k % 3}") for k in range(n)]
    return BT.her_stats(clips)


def _ltas(her):
    return np.asarray([np.nan if v is None else float(v) for v in her["ltas_db"]])


# ============================================================================ 1. s = 0 逐位相同
def test_strength_zero_is_bit_exact_and_untriggered_dynamics_return_the_input():
    x = _speechy(2.0, 1)
    her = _her()
    out, info = BT.beautify_sentence(x, SR, BT.make_plan(SR, 0.0, her))
    assert np.array_equal(out, x) and info["changed"] is False  # 好样本：s = 0 逐位相同
    assert BT.make_plan(SR, 0.0, her).fir is None
    # 去齿音：阈值比任何一帧都高 → 原样返回；压缩：阈值比最响的地方还高 → 原样返回
    x64 = x.astype(np.float64)
    y, di = BT.deess(x64, SR, threshold_db=50.0)
    assert np.array_equal(y, x64) and di["max_cut_db"] == 0.0
    y, ci = BT.compress(x64, SR, threshold_db=20.0, ratio=2.0)
    assert np.array_equal(y, x64) and ci["max_gr_db"] == 0.0
    # 坏样本：s > 0 真的改了声音（不然上面的「逐位相同」什么都没测）
    out, info = BT.beautify_sentence(x, SR, BT.make_plan(SR, 0.25, her))
    assert info["changed"] and not np.array_equal(out, x) and len(out) == len(x)


# ============================================================================ 2. 逐句处理与绝对数字静音
def _results48(wavs):
    out = []
    for i, w in enumerate(wavs):
        seg = ScriptSegment(text=f"第{i}句话在这里", display=f"第{i}句话在这里", lang="zh", kind="statement", index=i,
                            pause_after="sentence")
        out.append(eng.SegmentResult(seg, w, SR, {}, "r1", False, i))
    return out


def test_per_sentence_chain_keeps_digital_silence_length_and_alignment(tmp_path):
    n, _ = _narrator(tmp_path, tier="none")
    her = _her()
    wavs = [_speechy(2.5, 20 + k) for k in range(4)]
    gap = int(0.003 * SR)  # 第 2 句中间有 3 ms 的精确 0（≥ 1 ms）：处理后要恢复成 0
    wavs[1][SR:SR + gap] = 0.0
    c_her = np.random.default_rng(3).uniform(-1.5, 1.5, len(BT.BAND_CENTERS))
    plan = BT.make_plan(SR, 1.0, her, c_her, air=True)
    outs = []
    for w in wavs:
        y, info = BT.beautify_sentence(w, SR, plan)
        assert info["changed"] and len(y) == len(w)
        assert BT.chain_lag(w, y, plan, SR) == 0  # 和「只过静态 EQ」比：没有整体延迟
        assert BT.chain_lag(w, np.concatenate([[0.0], y[:-1]]), plan, SR) == 1  # 坏样本：晚了 1 个采样
        outs.append(y)
    assert np.all(outs[1][SR:SR + gap] == 0.0)
    results = _results48(wavs)
    layout = n._layout(results)
    regions = eng.sentence_regions(layout, SR)
    audio = n._render(layout, [(y, SR) for y in outs], SR)
    good = silence_violations(audio, regions)
    assert good["outside_nonzero"] == 0 and good["edges_bad"] == [] and good["outside_samples"] > SR // 2
    n._check_silence([("轻度美化", audio)], layout, SR)  # 好样本：A3 自检通过
    # 坏样本：同一条链在拼好的整篇上跑（不是逐句）：停顿里出现非零采样，G9 / A3 必须抓到
    plain = n._render(layout, [(w, SR) for w in wavs], SR)
    whole = BT.apply_fir(plain, plan.fir)
    whole, _ = BT.deess(whole, SR, plan.deess_thr_db, plan.deess_max_db)
    bad = silence_violations(whole.astype(np.float32), regions)
    assert bad["outside_nonzero"] > 1000
    with pytest.raises(RuntimeError, match="程序自检没通过：「轻度美化」版本句子以外有"):
        n._check_silence([("轻度美化", whole.astype(np.float32))], layout, SR)


# ============================================================================ 3. FIR
F = np.geomspace(20.0, 0.95 * SR / 2, 4000)
INSIDE = (F >= BT.EQ_LO_HZ) & (F <= BT.EQ_HI_HZ)
OUTSIDE = (F <= BT.TAPER_LO_HZ) | (F >= BT.TAPER_HI_HZ)
# 实测误差（4096 点最小相位 FIR，48 kHz；probe 见提交说明）：100 Hz 以上最大 0.07 dB；56–100 Hz 的余弦过渡区
# （频率分辨率约 6 Hz）最大 0.21 dB，加上高通最大 0.30 dB；30–60 Hz（高通已经压了 5–16 dB）最大 0.43 dB
FIR_ERR_DB = 0.1
FIR_ERR_LOW_DB = 0.35
OUTSIDE_DB = 0.05     # 实测最大 0.013 dB


@pytest.mark.parametrize("s", [1.0, 0.5, 0.25])
@pytest.mark.parametrize("c_kind", ["none", "alternating", "huge"])
def test_eq_response_is_bounded_inside_and_zero_outside(s, c_kind):
    c_her = {"none": None, "alternating": [1.5 if i % 2 else -1.5 for i in range(len(BT.BAND_CENTERS))],
             "huge": [10.0] * len(BT.BAND_CENTERS)}[c_kind]  # +10 dB 的锚点（坏样本）：先限到 ±1.5，总量再限到 ±2.5
    h = BT.design_eq(SR, s, c_her, air=True, hpf_fc=None)
    resp = BT.fir_response_db(h, SR, F)
    want = BT.target_db(F, s, c_her, air=True)
    assert np.max(np.abs(want)) <= BT.EQ_LIMIT_DB + 1e-9
    assert np.max(np.abs(resp[INSIDE])) <= BT.EQ_LIMIT_DB + FIR_ERR_DB
    assert np.max(np.abs(resp[OUTSIDE])) <= OUTSIDE_DB
    err = np.abs(resp - want)
    assert np.max(err[F >= 100.0]) <= FIR_ERR_DB and np.max(err) <= FIR_ERR_LOW_DB
    # 叠加高通：实测 = T + 2 阶 Butterworth（80 Hz 以下按高通往下走）
    h2 = BT.design_eq(SR, s, c_her, air=True, hpf_fc=75.0)
    r2 = BT.fir_response_db(h2, SR, F)
    err = np.abs(r2 - (want + BT.hpf_db(F, 75.0)))
    assert np.max(err[F >= 100.0]) <= FIR_ERR_DB
    assert np.max(err[(F >= 60.0) & (F < 100.0)]) <= FIR_ERR_LOW_DB
    assert np.max(err[(F >= 30.0) & (F < 60.0)]) <= 0.6
    at75 = float(BT.target_db(np.array([75.0]), s, c_her)[0])
    assert BT.fir_response_db(h2, SR, np.array([75.0]))[0] == pytest.approx(-3.01 + at75, abs=0.2)  # 高通在 fc 处 −3 dB


def test_tilt_matches_the_design_values_and_scales_with_strength():
    # RBJ 均衡器的恒等式：峰值在中心频率正好是增益；搁架在转折频率是一半、远处是满的
    for kind, f0, g, q in BT.TILT:
        at = BT._shelf_peak_db(np.array([f0]), kind, f0, g, q)[0]
        assert at == pytest.approx(g if kind == "peak" else g / 2, abs=1e-9), kind
    assert BT._shelf_peak_db(np.array([1.0]), "low_shelf", 250.0, 1.5, 0.7)[0] == pytest.approx(1.5, abs=1e-3)
    assert BT._shelf_peak_db(np.array([1e7]), "high_shelf", 11000.0, 1.5, 0.7)[0] == pytest.approx(1.5, abs=1e-3)
    f = np.geomspace(20, 20000, 200)
    full = BT.tilt_db(f, 1.0, air=True)
    assert np.max(full) <= 1.5 + 1.0 + 1e-6 and np.min(full) >= -1e-6  # 只有很小的提升（提升比衰减更容易被听出来）
    half = sum(BT._shelf_peak_db(f, k, f0, g / 2, q) for k, f0, g, q in BT.TILT)
    assert BT.tilt_db(f, 0.5, air=True) == pytest.approx(half, abs=1e-9)  # s 乘在每一项的 dB 增益上
    no_air = BT.tilt_db(f, 1.0, air=False)
    assert no_air == pytest.approx(full - BT._shelf_peak_db(f, *BT.TILT[2]), abs=1e-9)  # 不开空气感：没有高搁架


def test_eq_is_minimum_phase_and_avoids_the_square_root_trap():
    h = BT.design_eq(SR, 1.0, None, air=True, hpf_fc=70.0)
    assert len(h) == BT.FIR_TAPS
    e = h ** 2
    pk = int(np.argmax(np.abs(h)))
    assert pk == 0 and float(np.sum(e[:pk])) == 0.0  # 主峰之前没有能量（没有预振）
    assert np.sum(e[: int(0.002 * SR)]) / np.sum(e) > 0.99  # 能量集中在开头 2 ms
    # 「平方根」陷阱（坏样本）：按目标幅度（不平方）做线性相位、再转最小相位，实测的 dB 只有目标的一半
    grid = np.linspace(0.0, SR / 2, 8193)
    db = BT.target_db(grid, 1.0, None, air=True)
    naive = signal.minimum_phase(signal.firwin2(2 * BT.FIR_TAPS - 1, grid, 10 ** (db / 20), fs=SR, window="hann"))
    f = np.array([250.0, 3200.0, 14000.0])
    want = BT.target_db(f, 1.0, None, air=True)
    assert BT.fir_response_db(naive, SR, f) == pytest.approx(want / 2, abs=0.05)
    good = BT.design_eq(SR, 1.0, None, air=True, hpf_fc=None)
    assert BT.fir_response_db(good, SR, f) == pytest.approx(want, abs=0.05)  # 好样本：按平方设计，正好是目标


# ============================================================================ 4. 真峰值
def test_true_peak_math_and_level_back_caps_true_peak():
    t = np.arange(SR) / SR
    x = fade((0.5 * np.sin(2 * np.pi * SR / 4 * t + np.pi / 4)).astype(np.float32), SR, in_ms=10.0, out_ms=10.0)
    sp = float(20 * np.log10(np.max(np.abs(x))))
    assert true_peak_db(x) - sp == pytest.approx(3.0103, abs=0.05)  # 数学事实
    up = 7.0  # 要调高 7 dB：采样峰值 −2.0 dBFS（按采样峰值没超），真峰值 +1.0 dBTP（超了）
    ref = x * np.float32(10 ** (up / 20))
    assert sp + up < BT.PEAK_CEILING_DB < true_peak_db(x) + up
    out, g = BT.level_back(x, ref, SR)
    assert true_peak_db(out) == pytest.approx(BT.PEAK_CEILING_DB, abs=1e-3)  # 按真峰值限到 −1 dBTP
    assert up - g == pytest.approx(true_peak_db(x) + up - BT.PEAK_CEILING_DB, abs=0.01)  # 少调了约 2 dB
    quiet = x * np.float32(0.3)
    out2, g2 = BT.level_back(x, quiet, SR)  # 好样本：调回去也没超，照常调（只乘系数）
    assert g2 == pytest.approx(20 * np.log10(0.3), abs=0.01) and np.array_equal(out2 == 0, x == 0)


# ============================================================================ 5. 压缩、去齿音的上限
def test_compressor_and_deesser_never_cut_more_than_their_caps():
    x = _speechy(3.0, 5, sib=0.5).astype(np.float64) * 3
    # 压缩：阈值很低（每个地方都超过），比例 2:1 → 想压十几 dB，最多只压 3 dB
    y, ci = BT.compress(x, SR, threshold_db=-60.0, ratio=2.0, max_gr_db=BT.COMP_MAX_GR_DB)
    sel = np.abs(x) > 1e-4
    gain_db = 20 * np.log10(np.abs(y[sel]) / np.abs(x[sel]))
    assert ci["max_gr_db"] == pytest.approx(3.0, abs=1e-6) and gain_db.min() >= -3.0 - 1e-6 and gain_db.max() <= 1e-9
    her = _her()
    plan = BT.make_plan(SR, 0.5, her)
    assert plan.comp_max_db == 1.5 and plan.comp_ratio == 1.5 and plan.deess_max_db == 2.0
    assert BT.make_plan(SR, 1.0, her).comp_ratio == 2.0
    y, ci = BT.compress(x, SR, -60.0, plan.comp_ratio, plan.comp_max_db)
    assert 20 * np.log10(np.abs(y[sel]) / np.abs(x[sel])).min() >= -1.5 - 1e-6
    # 去齿音：阈值很低（齿音很重），最多压 4 dB；输出 = 其余部分 + g(t)·齿音带，g ≥ −4 dB
    y, di = BT.deess(x, SR, threshold_db=-40.0, max_cut_db=BT.DEESS_MAX_DB)
    band = BT.deess_band(x, SR)
    rest = x - band
    nz = np.abs(band) > 1e-3
    g = (y - rest)[nz] / band[nz]
    assert di["max_cut_db"] == pytest.approx(4.0, abs=0.01)  # 坏样本：真的想压很多，上限起作用
    assert g.min() >= 10 ** (-4.0 / 20) - 1e-6 and g.max() <= 1.0 + 1e-6


# ============================================================================ 6. 关卡（假打分器 / 量出来的数）
def test_null_floors_follow_the_design_numbers():
    # 设计 G2 的推算：μ = −0.1、σ = 0.9、N = 100 句时阈值约 −3.1 个百分点
    assert BT.null_each_floor(-0.1, 0.9, 100) == pytest.approx(-3.1, abs=0.05)
    assert BT.null_mean_floor(-0.1, 0.9, 25) == pytest.approx(-0.1 - 2 * 0.9 / 5)


def test_identity_gate_rejects_a_10_point_drop_and_passes_volume_only():
    rng = np.random.default_rng(7)
    k = 30
    base = list(rng.uniform(92, 101, k))
    secs = [3.0] * k
    vol = [b + d for b, d in zip(base, rng.normal(-0.1, 0.9, k))]  # 只改音量：和空白对照同一个分布
    g = BT.gate_identity(base, vol, secs, -0.1, 0.9, 85.0)
    assert g["g1"] is True and g["g2_fail"] == []  # 好样本
    one = list(base)
    one[4] -= 10.0
    g = BT.gate_identity(base, one, secs, -0.1, 0.9, 85.0)
    assert g["g2_fail"] == [4]  # 坏样本：一句低 10 个百分点
    g = BT.gate_identity(base, [b - 10 for b in base], secs, -0.1, 0.9, 85.0)
    assert g["g1"] is False and len(g["g2_fail"]) == k  # 坏样本：整篇低 10
    # 原来 ≥ 85% 的句子美化后掉到 85% 以下：不过（差得不多也不行）；原来就不到 85% 的句子只看 Δ
    g = BT.gate_identity([85.5, 80.0], [84.9, 79.5], [3.0, 3.0], -0.1, 0.9, 85.0)
    assert g["g2_fail"] == [0]
    # 人声不到 2 秒的句子不做这一关（短句子分数太不可靠）
    g = BT.gate_identity([95.0, 95.0], [95.0, 70.0], [3.0, 1.2], -0.1, 0.9, 85.0)
    assert g["g2_fail"] == [] and g["short"] == [1] and g["n"] == 1


def test_content_gate_rejects_more_errors_unless_the_null_also_jumps():
    assert BT.gate_content(0.0, 0.0) is True and BT.gate_content(0.05, 0.02) is True
    assert BT.gate_content(0.0, 0.1) is False                      # 坏样本：错字变多
    assert BT.gate_content(0.0, 0.1, [0.0, 0.0]) is False
    assert BT.gate_content(0.0, 0.1, [0.1, 0.0]) is True           # 原声改 1 dB 音量也会识别成这样：不算
    assert BT.gate_content(None, 0.1) is None                      # 没有识别模型：如实说没查


def test_timbre_gate_rejects_a_12db_shelf_and_passes_the_designed_eq():
    her = _her()
    base = [_speechy(3.0, 40 + k) for k in range(3)]
    plan = BT.make_plan(SR, 1.0, her, None, air=True)
    good = [BT.beautify_sentence(w, SR, plan)[0] for w in base]
    spec_b = BT.speech_spectrum(base, SR)
    lb = BT.band_db(spec_b)
    exp = plan.expected_change(spec_b)
    g = BT.gate_timbre(lb, BT.band_db(BT.speech_spectrum(good, SR, masks=spec_b.masks)), _ltas(her), 1.0, exp, 1.0)
    assert g["passed"] is True
    # 只有线性 EQ（不做动态处理）时：实测和设计应该带来的变化相差不到 LTAS_TOL_DB
    lin = [BT.beautify_sentence(w, SR, plan, dynamics=False)[0] for w in base]
    dev = BT.band_db(BT.speech_spectrum(lin, SR, masks=spec_b.masks)) - lb - exp
    ok = np.isfinite(dev)
    assert np.max(np.abs(dev[ok] - np.median(dev[ok]))) <= BT.LTAS_TOL_DB
    # 坏样本：+12 dB 的高搁架（处理链做了设计以外的事）
    b, a = _rbj_high_shelf(4000.0, 12.0, SR)
    shelf = [signal.lfilter(b, a, w).astype(np.float32) for w in base]
    g = BT.gate_timbre(lb, BT.band_db(BT.speech_spectrum(shelf, SR, masks=spec_b.masks)), _ltas(her), 1.0, exp, 1.0)
    assert g["passed"] is False and g["bands_ok"] is False and g["dist_ok"] is False
    # 坏样本：离她更远，超过她自己不同文件之间的差别（σ_她）
    g = BT.gate_timbre(lb, lb, _ltas(her) + np.linspace(-3, 3, len(lb)), 0.1, np.zeros(len(lb)), 1.0)
    assert g["dist_ok"] is True  # C 和 base 一样：不会更远
    g = BT.gate_timbre(_ltas(her), lb + np.linspace(-3, 3, len(lb)), _ltas(her), 0.1, np.zeros(len(lb)), 1.0)
    assert g["dist_ok"] is False


def _rbj_high_shelf(f0, gain_db, sr, q=0.7):
    a = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / sr
    alpha = np.sin(w0) / (2 * q)
    cw = np.cos(w0)
    b = np.array([a * ((a + 1) + (a - 1) * cw + 2 * np.sqrt(a) * alpha), -2 * a * ((a - 1) + (a + 1) * cw),
                  a * ((a + 1) + (a - 1) * cw - 2 * np.sqrt(a) * alpha)])
    den = np.array([(a + 1) - (a - 1) * cw + 2 * np.sqrt(a) * alpha, 2 * ((a - 1) - (a + 1) * cw),
                    (a + 1) - (a - 1) * cw - 2 * np.sqrt(a) * alpha])
    return b / den[0], den / den[0]


def test_hf_gate_rejects_energy_above_8k_beyond_her_p95():
    her = _her()
    base = [_speechy(3.0, 50 + k) for k in range(3)]
    hf_b = BT.hf_ratio_db(BT.speech_spectrum(base, SR))
    assert her["hf_p05_db"] is not None and her["hf_p95_db"] is not None
    lo, hi = hf_b - 3.0, hf_b + 3.0
    g = BT.gate_hf(hf_b, hf_b + 0.5, lo, hi, -8.0, -8.5, -3.0)
    assert g["passed"] is True  # 好样本
    sos = signal.butter(4, 8500, btype="highpass", fs=SR, output="sos")
    hiss = [w + 0.3 * signal.sosfilt(sos, np.random.default_rng(1).standard_normal(len(w))).astype(np.float32)
            * (np.abs(w) > 1e-3) for w in base]
    hf_c = BT.hf_ratio_db(BT.speech_spectrum(hiss, SR))
    assert hf_c > hi
    assert BT.gate_hf(hf_b, hf_c, lo, hi, None, None, None)["hf_ok"] is False  # 坏样本：8 kHz 以上超出她的 p95
    assert BT.gate_hf(hf_b, None, lo, hi, -8.0, -2.0, -3.0)["sib_ok"] is False  # 坏样本：齿音比她的 p95 重
    # base 本来就在范围外：C 不比 base 离得更远就算过
    assert BT.gate_hf(hi + 2.0, hi + 1.5, lo, hi, None, None, None)["passed"] is True
    assert BT.gate_hf(hi + 2.0, hi + 2.5, lo, hi, None, None, None)["passed"] is False


def test_noise_gate_rejects_20db_white_noise():
    base = _speechy(3.0, 60, sib=0.0)
    fb = speech_noise_floor_db(base, SR)
    assert fb is None or fb <= -60.0
    rms = float(np.sqrt(np.mean(base[np.abs(base) > 1e-3].astype(np.float64) ** 2)))
    white = np.random.default_rng(2).standard_normal(len(base)).astype(np.float32)
    noisy = base + white * np.float32(rms / 10)  # 信噪比 20 dB
    fc = speech_noise_floor_db(noisy, SR)
    assert fc is not None and fc > -60.0
    assert BT.gate_noise(fb, fc, true_peak_db(noisy))["passed"] is False         # 坏样本
    assert BT.gate_noise(fb, fb, true_peak_db(base))["passed"] is True            # 好样本
    assert BT.gate_noise(-50.0, -51.0, -3.0)["passed"] is True                    # base 本来就 −50：不高于它就行
    assert BT.gate_noise(-50.0, -49.0, -3.0)["floor_ok"] is False
    assert BT.gate_noise(None, None, -0.5)["peak_ok"] is False                    # 真峰值超过 −1 dBTP
    # 原声本来就有一点底噪（−54 dBFS）：只过静态 EQ（开着空气感）底噪高了 0.2 dB，按设计的 EQ 来的就不算
    # （按设计的字面「不高于 base」这样的句子永远过不了）；压缩以后调回响度、比「只过静态 EQ」还高（坏样本）：不过
    assert BT.gate_noise(-54.0, -53.8, -3.0)["passed"] is False
    assert BT.gate_noise(-54.0, -53.8, -3.0, floor_eq=-53.8)["passed"] is True
    assert BT.gate_noise(-54.0, -52.8, -3.0, floor_eq=-53.8)["passed"] is False
    assert BT.gate_noise(-70.0, -61.0, -3.0, floor_eq=-75.0)["passed"] is True   # 还在 −60 以下：照常通过


def test_pitch_gate_rejects_a_semitone_shift_and_passes_volume_and_eq():
    pytest.importorskip("librosa")
    base = _speechy(3.0, 70, sib=0.0, f0=170.0)
    f0b = eng.Narrator._f0_median(base, SR)
    assert f0b == pytest.approx(170.0, rel=0.06)
    null = eng.Narrator._f0_median(base * np.float32(1.12), SR)
    plan = BT.make_plan(SR, 1.0, _her())
    eq = BT.beautify_sentence(base, SR, plan)[0]
    g = BT.gate_pitch(len(base), len(eq), f0b, eng.Narrator._f0_median(eq, SR), [null])
    assert g["passed"] is True  # 好样本：线性 EQ 不改音高
    up = _speechy(3.0, 70, sib=0.0, f0=170.0 * 2 ** (1 / 12))
    g = BT.gate_pitch(len(base), len(up), f0b, eng.Narrator._f0_median(up, SR), [null])
    assert g["passed"] is False and g["shift_st"] > 0.8  # 坏样本：移调一个半音
    assert BT.gate_pitch(len(base), len(base) + 1, f0b, f0b)["passed"] is False  # 长度变了
    assert BT.f0_floor_st(200.0) == pytest.approx(12 * math.log2(1 + 200 / 16000))


def test_alignment_gate_ignores_the_highpass_phase_but_catches_a_delay():
    """G5：最小相位高通让很稳的基频超前，直接比波形的互相关会移十几个采样（不是延迟）；和「只过静态 EQ」比才是 0。"""
    import sys
    from pathlib import Path as _P

    sys.path.insert(0, str(_P(eng.__file__).parent.parent / "backends" / "workers"))
    from dummy_worker import synth_speech

    sr = 24000
    w = synth_speech("美化对齐测试第一句，先停一下。", sr=sr, seed=0)
    plan = BT.make_plan(sr, 0.5, dict(_her(), hpf_hz=60.0))
    y, _ = BT.beautify_sentence(w, sr, plan)
    assert BT.xcorr_lag(w, y, int(0.02 * sr)) != 0  # 设计里的量法：在测试引擎的声音上误报
    assert BT.chain_lag(w, y, plan, sr) == 0
    assert BT.chain_lag(w, np.concatenate([np.zeros(3), y[:-3]]), plan, sr) == 3  # 真的晚了 3 个采样：抓到


def test_strength_halves_down_to_zero():
    assert [BT.next_strength(s) for s in (1.0, 0.5, 0.25)] == [0.5, 0.25, 0.0]
    assert BT.strength_word(0.5) == "一半" and BT.strength_word(0.25) == "四分之一"


# ============================================================================ 一次性校准
def _calib_items(k=8):
    return [(_speechy(3.0, 100 + j, amp=0.05 + 0.02 * j), SR) for j in range(k)]


def _match(items):
    """找出一段声音是哪一段原录音：返回 (第几段, 音量系数, 波形相对差, 频谱距离 dB)。
    只改音量：波形相对差是 0；处理过：频谱距离（LTAS，对齐整体电平）随强度变大（高通的相位会改波形，但不改频谱）。"""
    refs = [w.astype(np.float64) for w, _ in items]
    ltas = [BT.ltas_db([w], SR) for w, _ in items]

    def find(y):
        y = np.asarray(y, dtype=np.float64)
        best = (-1, 1.0, 1.0)
        for j, r in enumerate(refs):
            if len(r) != len(y):
                continue
            g = float(np.dot(y, r) / np.dot(r, r))
            d = float(np.linalg.norm(y - g * r) / np.linalg.norm(g * r))
            if d < best[2]:
                best = (j, g, d)
        j = best[0]
        dist = BT.ltas_distance(BT.ltas_db([y], SR), ltas[j]) if j >= 0 else 99.0
        return best + (float(dist),)
    return find


def test_calibration_picks_the_largest_strength_indistinguishable_from_volume_only():
    items = _calib_items()
    her = _her()
    find = _match(items)
    # 先量每个强度把声音改了多少（和 calibrate 里一模一样的处理），再把假打分的扣分定在 s = 0.5 和 1 之间
    r = {}
    for s in BT.STRENGTHS:
        plan = BT.make_plan(SR, s, her)
        r[s] = float(np.mean([find(BT.level_back(BT.beautify_sentence(w, SR, plan)[0], w, SR)[0])[3]
                              for w, _ in items]))
    assert r[1.0] > r[0.5] > r[0.25] > 0  # 前提：强度越大改得越多

    def pct_fn(w, sr):
        """原录音 98%；只改音量：98 ± 0.4（空白对照的波动，第几段轮流 + / −）；处理过：98 − k·频谱距离。"""
        j, g, d, dist = find(w)
        if d < 1e-6:
            return (98.0 if abs(g - 1.0) < 1e-6 else 98.0 + (0.4 if j % 2 else -0.4)), len(w) / sr
        return 98.0 - k_pen * dist, len(w) / sr

    sigma = 0.4 * math.sqrt(len(items) / (len(items) - 1))
    floor = BT.null_mean_floor(0.0, sigma, len(items))
    k_pen = abs(floor) / math.sqrt(r[1.0] * r[0.5])  # s = 0.5 平均扣得比 |floor| 少、s = 1 扣得多
    cal = BT.calibrate(items, her, pct_fn)
    assert cal["n"] == len(items) and cal["sigma_null"] == pytest.approx(sigma, abs=1e-3)
    assert cal["mu_null"] == pytest.approx(0.0, abs=1e-9) and cal["floor"] == pytest.approx(floor, abs=1e-3)
    st = cal["strengths"]
    for s in BT.STRENGTHS:
        assert st[str(s)]["passed"] == (st[str(s)]["mean_delta"] >= cal["floor"])
    assert st["1.0"]["passed"] is False and st["0.5"]["passed"] is True and cal["s_star"] == 0.5
    # 都不过 → s* = 0，如实说原因
    k_pen = 1000.0
    cal = BT.calibrate(items, her, pct_fn)
    assert cal["s_star"] == 0.0 and "四分之一" in cal["why"]
    # 录音不够 3 段 → 不校准
    cal = BT.calibrate(items[:2], her, pct_fn)
    assert cal["s_star"] == 0.0 and "至少要 3 段" in cal["why"]


class FakeJudge:
    """假的声纹打分：fn(去掉首尾 0 的声音, 采样率) → 像你本人 %（不封顶）。"""
    available = True
    calibrated = True

    def __init__(self, fn=None, sig="fake-1"):
        self.fn = fn or (lambda w, sr: 98.0)
        self.sig = sig
        self.calls = 0

    def signature(self):
        return self.sig

    def judge(self, wav, sr):
        self.calls += 1
        w = np.trim_zeros(np.asarray(wav, dtype=np.float32))
        v = float(self.fn(w, sr))
        return {"pct": min(v, 100.0), "pct_raw": v, "seconds": len(w) / sr}


def test_calibration_file_lives_next_to_models_and_is_reused(prepared, tmp_path):
    import shutil

    from conftest import make_cfg
    from voicetwin.project import Project
    from voicetwin.synth.select import ensure_beautify_calibration

    _, project, _ = prepared
    shutil.copytree(project.root, tmp_path / "ws" / project.voice)
    cfg = make_cfg(tmp_path / "ws")
    proj = Project(cfg, project.voice)
    recs = proj.load_manifest(only_kept=True)
    for r in recs[:4]:  # 保证有几段验证集（没参加训练的真实录音）
        r["split"] = "val"
    proj.save_manifest(recs)
    judge = FakeJudge()
    cal = ensure_beautify_calibration(cfg, proj, judge)
    path = proj.root / BT.CALIB_FILE
    assert path.exists() and path.parent == proj.models_path.parent  # 和 models.json 在同一个文件夹
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["sig"] == cal["sig"] and saved["her"]["n_clips"] > 0 and "s_star" in saved
    assert cal["n"] >= 3 and cal["s_star"] == 1.0  # 分数不随处理变：和只改音量分不开，最大强度
    calls = judge.calls
    again = ensure_beautify_calibration(cfg, proj, judge)
    assert judge.calls == calls and again["sig"] == cal["sig"]  # 录音、打分标准、处理链都没变：直接读
    other = FakeJudge(sig="fake-2")
    ensure_beautify_calibration(cfg, proj, other)
    assert other.calls > 0  # 打分标准变了：重做
    # 没有精准声纹打分：不校准、不写文件，如实说
    path.unlink()
    out = ensure_beautify_calibration(cfg, proj, None)
    assert out["s_star"] == 0.0 and "不可用" in out["why"] and not path.exists()


# ============================================================================ 生成流程里的版本 C
class BeautyScorer:
    """假的「一模一样」打分器：pct_fn(声音) → 像你本人 %；cer_fn(声音) → 错字率（use_asr 时）。"""

    def __init__(self, pct_fn=None, cer_fn=None):
        self.pct_fn = pct_fn or (lambda w: 97.0)
        self.cer_fn = cer_fn or (lambda w: 0.0)
        self.asr_calls = 0

    def full(self, wav, sr, text, lang, mult=1.0, prepared=None, use_asr=True, cer=None, check_pauses=True):
        w = np.trim_zeros(np.asarray(wav, dtype=np.float32))
        p = float(self.pct_fn(w))
        c = None
        if use_asr:
            self.asr_calls += 1
            c = float(self.cer_fn(w))
        return Score(total=p / 100.0, pct=min(p, 100.0), pct_raw=p, speech_seconds=len(w) / sr, cer=c)


def _key(w):
    """和音量无关的指纹：同一段声音乘一个系数，指纹一样。"""
    w = np.asarray(w, dtype=np.float64)
    m = float(np.max(np.abs(w))) or 1.0
    return zlib.crc32(np.round(w / m, 3).astype(np.float32).tobytes())


def _beauty_setup(tmp_path, scorer=None, secs=(3.0, 3.0, 1.2, 3.0), s_star=1.0, sigma=0.9, use_asr=False):
    n, _ = _narrator(tmp_path, tier="none", scorer=scorer or BeautyScorer())
    n._judge = FakeJudge()
    wavs = [_speechy(sec, 200 + k) for k, sec in enumerate(secs)]
    results = _results48(wavs)
    layout = n._layout(results)
    her = BT.her_stats([(w, SR, f"s{k % 3}") for k, w in enumerate(_speechy(4.0, 300 + j) for j in range(6))])
    hf_b = BT.hf_ratio_db(BT.speech_spectrum(wavs, SR))
    her.update(ltas_db=list(BT.ltas_db(wavs, SR)), sigma_her=1.0, hf_p05_db=hf_b - 6, hf_p95_db=hf_b + 6)
    calib = {"s_star": s_star, "mu_null": -0.1, "sigma_null": sigma, "n": 20, "her": her, "strengths": {}}
    n._beauty_calibration = lambda: calib
    n._variant_mix = {"未去杂音": {"wavs": wavs, "gains": [1.0] * len(wavs)}}
    variants = [{"name": "未去杂音", "label": "版本 A：未去杂音", "score": 0.97, "pct": 97.0, "recommended": True}]
    if use_asr:
        n.use_asr, n._checker = True, object()
    return n, results, layout, variants, wavs


def test_make_beauty_offers_c_when_every_gate_passes(tmp_path):
    n, results, layout, variants, wavs = _beauty_setup(tmp_path, use_asr=True)
    entry, audio, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is not None and block["offered"] is True and entry["strength"] == 1.0
    assert entry["name"] == eng.VARIANT_BEAUTY and entry["recommended"] is False and entry["base"] == "未去杂音"
    assert block["short_sentences"] == [3]  # 第 3 句人声不到 2 秒：只做静态 EQ，不比像不像
    assert 3 in block["rounds"][-1]["static_only"]
    regions = eng.sentence_regions(layout, SR)
    chk = silence_violations(audio, regions)
    assert chk["outside_nonzero"] == 0 and chk["edges_bad"] == [] and block["g9"]["outside_nonzero"] == 0
    assert len(audio) == len(n._render(layout, [(w, SR) for w in wavs], SR))  # 同一个 layout：时间、字幕完全一样
    mix = n._variant_mix[eng.VARIANT_BEAUTY]
    for w, c, g in zip(wavs, mix["wavs"], mix["gains"]):
        assert len(c) == len(w) and true_peak_db(c * np.float32(g)) <= BT.PEAK_CEILING_DB + 1e-4
    line = entry["note"]
    assert line.startswith("版本 C 轻度美化（润色强度：满）") and "错字没有增加" in line and "第 3 句太短" in line
    assert block["plan"]["target_db"]["max"] <= BT.EQ_LIMIT_DB and block["calibration"]["mu_null"] == -0.1
    assert any(x == line for x in n.notes)


def test_make_beauty_turns_off_dynamics_for_one_sentence_then_halves_to_zero(tmp_path):
    base_keys = {}

    def pct(w):  # 原声 97%，处理过的声音 85%（低 12 个百分点）：每一句都过不了 G2
        return 97.0 if _key(w) in base_keys else 85.0

    n, results, layout, variants, wavs = _beauty_setup(tmp_path, scorer=BeautyScorer(pct))
    base_keys.update({_key(np.trim_zeros(w)): 1 for w in wavs})
    entry, audio, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is None and audio is None and block["offered"] is False
    assert [r["s"] for r in block["rounds"]] == [1.0, 0.5, 0.25]
    assert all(not r["passed"] for r in block["rounds"])
    assert block["rounds"][0]["static_only"] == [1, 2, 3, 4]  # 先关掉去齿音和压缩，只留静态 EQ，还不过才减半
    assert "已经自动减轻到一半" in " ".join(n.notes) and "已经自动减轻到四分之一" in " ".join(n.notes)
    assert any(x.startswith("版本 C（轻度美化）和原声一样") and "减到最轻也没通过检查" in x
               and "低了 12.0%" in x for x in n.notes)


def test_make_beauty_halves_when_the_timbre_gate_fails_at_full_strength(tmp_path, monkeypatch):
    n, results, layout, variants, _ = _beauty_setup(tmp_path)
    orig = BT.gate_timbre

    def strict_at_full(*a, **k):
        out = orig(*a, **k)
        if a[-1] >= 1.0:  # 满强度时假装音色离她更远
            out = dict(out, passed=False, dist_ok=False)
        return out

    monkeypatch.setattr(BT, "gate_timbre", strict_at_full)
    entry, audio, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is not None and entry["strength"] == 0.5 and entry["strength_word"] == "一半"
    assert [r["passed"] for r in block["rounds"]] == [False, True]
    assert any("已经自动减轻到一半" in x and "音色离你本人的录音比原声更远" in x for x in n.notes)
    assert entry["note"].startswith("版本 C 轻度美化（润色强度：一半）")


def test_make_beauty_turns_air_off_first_when_hf_gate_fails(tmp_path, monkeypatch):
    n, results, layout, variants, _ = _beauty_setup(tmp_path)
    orig = BT.gate_hf
    seen = []

    def no_air(*a, **k):
        out = orig(*a, **k)
        seen.append(out)
        return dict(out, passed=len(seen) > 1, hf_ok=len(seen) > 1)  # 第一次（开着空气感）不过

    monkeypatch.setattr(BT, "gate_hf", no_air)
    entry, _, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is not None and entry["strength"] == 1.0 and block["rounds"][0]["air"] is False
    assert block["plan"]["air"] is False


def test_make_beauty_hard_fails_on_program_errors(tmp_path, monkeypatch):
    # G5：整体延迟了 1 个采样（程序错误）→ 不输出 C
    n, results, layout, variants, _ = _beauty_setup(tmp_path)
    orig = BT.beautify_sentence

    def late(wav, sr, plan, dynamics=True, compress_on=True):
        out, info = orig(wav, sr, plan, dynamics, compress_on)
        return np.concatenate([[0.0], out[:-1]]).astype(np.float32), info

    monkeypatch.setattr(BT, "beautify_sentence", late)
    entry, audio, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is None and block["program_error"]["lag"] == 1 and "程序的错误" in block["why"]
    monkeypatch.setattr(BT, "beautify_sentence", orig)
    # G9：拼好的 C 停顿里有一个不是 0 的采样 → 硬失败，不输出 C（原声照常）
    n, results, layout, variants, _ = _beauty_setup(tmp_path / "g9")
    render = n._render

    def leaky(layout_, pieces, sr, gains=None):
        out = render(layout_, pieces, sr, gains)
        out[1] = 1e-6
        return out

    n._render = leaky
    entry, audio, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is None and block["g9"]["outside_nonzero"] == 1 and "数字静音" in block["why"]


def test_make_beauty_not_offered_without_scores_or_calibration(tmp_path):
    n, results, layout, variants, _ = _beauty_setup(tmp_path, s_star=0.0)
    entry, _, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is None and "宁可不美化" in block["why"]
    n, results, layout, variants, _ = _beauty_setup(tmp_path / "b")
    n._judge = None
    entry, _, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is None and "量不出「像你本人 %」" in block["why"]


def test_make_beauty_rejects_more_errors_and_says_cer_unchecked_without_asr(tmp_path):
    base_keys = {}

    def cer(w):
        return 0.0 if _key(w) in base_keys else 0.2

    n, results, layout, variants, wavs = _beauty_setup(tmp_path, scorer=BeautyScorer(cer_fn=cer), use_asr=True)
    base_keys.update({_key(np.trim_zeros(w)): 1 for w in wavs})
    entry, _, block = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is None and block["rounds"][0]["g3_fail"]  # 坏样本：错字变多（原声 ±1 dB 不会）
    assert "错字比原声多" in block["why"]
    n, results, layout, variants, _ = _beauty_setup(tmp_path / "noasr")
    entry, _, _ = n._make_beauty(results, layout, SR, 1.0, variants)
    assert entry is not None and "错字这次没查（这次没有开识别校验）" in entry["note"]
    assert "错字没有增加" not in entry["note"]


def test_beauty_cache_is_reused_and_ignored_after_regeneration(tmp_path):
    n, results, layout, variants, wavs = _beauty_setup(tmp_path)
    p = tmp_path / "seg0.wav"
    from voicetwin.utils.audio import save_audio

    save_audio(p, wavs[0], SR, subtype="FLOAT")
    results[0].path = p
    plan = BT.make_plan(SR, 1.0, _her())
    out1, info1 = n._beautified(results[0], wavs[0], SR, plan, True, True, "未去杂音")
    caches = list(tmp_path.glob("seg0.bt48000.*.wav"))
    assert len(caches) == 1 and plan.sig in caches[0].name and not info1.get("cached")
    out2, info2 = n._beautified(results[0], wavs[0], SR, plan, True, True, "未去杂音")
    assert info2.get("cached") and np.array_equal(out1, out2)
    import os
    import time as _t

    os.utime(p, (_t.time() + 10, _t.time() + 10))  # 这句后来重新生成过：旧缓存不能用
    _, info3 = n._beautified(results[0], wavs[0], SR, plan, True, True, "未去杂音")
    assert not info3.get("cached")
    _, info4 = n._beautified(results[0], wavs[0], SR, plan, False, True, "未去杂音")
    assert not info4.get("cached")  # 动态处理开关不同：另一个缓存


# ============================================================================ 设置和整篇生成
def test_config_default_is_off_and_only_perfect_identical_make_c(tmp_path):
    from test_identical_search import FakeBackend, _project

    import yaml

    cfg, project = _project(tmp_path)
    default = yaml.safe_load((eng.Path(eng.__file__).parent.parent / "default_config.yaml").read_text(encoding="utf-8"))
    assert default["synth"]["beautify"] == {"enabled": False}
    assert eng.Narrator(cfg, project, FakeBackend(), quality="identical", tier="none").want_beauty is False
    assert eng.Narrator(cfg, project, FakeBackend(), quality="identical", tier="none", beautify=True).want_beauty
    assert eng.Narrator(cfg, project, FakeBackend(), quality="perfect", tier="none", beautify=True).want_beauty
    for q in ("fast", "balanced", "best", "max"):
        assert not eng.Narrator(cfg, project, FakeBackend(), quality=q, tier="none", beautify=True).want_beauty, q
    assert not eng.Narrator(cfg, project, FakeBackend(), quality="identical", tier="none", beautify=True,
                            variants=False).want_beauty
    cfg2 = dict(cfg)
    cfg2["synth"] = dict(cfg.get("synth", {}), beautify={"enabled": True})
    assert eng.Narrator(cfg2, project, FakeBackend(), quality="identical", tier="none").want_beauty


def test_narration_without_precise_scoring_says_c_is_same_and_writes_no_c_file(prepared, tmp_path, monkeypatch):
    """整篇走一遍（测试引擎、没有精准声纹打分）：勾了美化也如实说「和原声一样」，不写 C 的文件；A、B 照常。"""
    cfg, project, _ = prepared
    _fake_noisereduce(monkeypatch)
    monkeypatch.setattr(eng, "_vram_tier", lambda: "none")
    res = wf.run_narrate(cfg, project.voice, _uniq("美化整篇第一句，先停一下") + _uniq("第二句也在这里"),
                         out=str(tmp_path / "美化.wav"), candidates=2, beautify=True)
    rep = json.loads(res.report_path.read_text(encoding="utf-8"))
    assert rep["beautify"]["enabled"] is True and rep["beautify"]["offered"] is False
    assert {v["name"] for v in rep["variants"]} == {"未去杂音", "去杂音"}
    assert not list(tmp_path.glob("*轻度美化*"))
    assert any(x.startswith("版本 C（轻度美化）和原声一样") for x in rep["notes"])
    res2 = wf.run_narrate(cfg, project.voice, _uniq("不勾美化第一句，先停一下") + _uniq("第二句"),
                          out=str(tmp_path / "不勾" / "课.wav"), candidates=2)
    rep2 = json.loads(res2.report_path.read_text(encoding="utf-8"))
    assert rep2["beautify"] is None and not any("轻度美化" in x for x in rep2["notes"])


def test_narration_writes_c_file_checked_for_silence_and_not_recommended(prepared, tmp_path, monkeypatch):
    """整篇走一遍，C 通过检查：多一个「…_轻度美化_模型名.wav」，写之前和 A、B 一样查绝对静音；推荐仍在 A、B 之间。"""
    cfg, project, _ = prepared
    _fake_noisereduce(monkeypatch)
    monkeypatch.setattr(eng, "_vram_tier", lambda: "none")
    her_wavs = [_speechy(4.0, 400 + j) for j in range(6)]
    her = BT.her_stats([(w, SR, f"s{j % 3}") for j, w in enumerate(her_wavs)])
    her.update(sigma_her=5.0, hf_p05_db=-80.0, hf_p95_db=20.0, sib_p95_db=0.0)
    calib = {"s_star": 0.5, "mu_null": 0.0, "sigma_null": 0.9, "n": 20, "her": her, "strengths": {}}
    monkeypatch.setattr(eng.Narrator, "_beauty_calibration", lambda self: calib)
    orig_make = eng.Narrator._make_beauty

    def with_judge(self, *a, **k):  # 测试引擎没有精准声纹打分：只在做 C 的时候给一个假的（别的步骤照旧）
        self._judge = FakeJudge()
        try:
            return orig_make(self, *a, **k)
        finally:
            self._judge = None

    monkeypatch.setattr(eng.Narrator, "_make_beauty", with_judge)
    orig_score = eng.Narrator._score_one

    def scored(self, wav, r, sr, mult, use_asr=False):
        s = orig_score(self, wav, r, sr, mult, use_asr=use_asr)
        s.pct, s.pct_raw, s.speech_seconds = 96.0, 96.0, 3.0  # 测试引擎没有声纹分：给一个固定的分数
        return s

    monkeypatch.setattr(eng.Narrator, "_score_one", scored)
    res = wf.run_narrate(cfg, project.voice, _uniq("美化通过第一句，先停一下") + _uniq("第二句也在这里"),
                         out=str(tmp_path / "通过.wav"), candidates=2, beautify=True)
    rep = json.loads(res.report_path.read_text(encoding="utf-8"))
    names = [v["name"] for v in rep["variants"]]
    assert names == ["未去杂音", "去杂音", "轻度美化"]
    vc = rep["variants"][2]
    assert vc["recommended"] is False and vc["final"] is False and "_轻度美化_" in vc["path"]
    assert rep["final"] in ("未去杂音", "去杂音")
    assert rep["silence_check"]["versions"]["轻度美化"]["outside_nonzero"] == 0
    assert rep["beautify"]["offered"] is True and rep["beautify"]["s"] == 0.5
    from test_assembly_identical import _gaps_are_zero

    _gaps_are_zero(vc["path"], rep["segments"])  # 写出的 C 文件：句子之间每个采样都是 0
    # 选 C 当最终版本（网页「最终使用哪个版本」）
    out = wf.choose_variant(cfg, project.voice, str(res.report_path), "轻度美化")
    assert out["final"] == "轻度美化"


# ============================================================================ 网页
def test_web_text_is_honest_and_checkbox_only_for_perfect_identical():
    from voicetwin.webui import app as A

    for must in ("不改音调", "不改语速", "不换声音", "完全的数字静音", "原声版本照常生成", "像你本人 %", "自动减轻或者不用"):
        assert must in A.BEAUTY_HELP, must
    for never in ("录音棚", "更好听", "专业", "保证"):
        assert never not in A.BEAUTY_HELP, never
    assert not re.search(r"\d+(\.\d+)?\s*%", A.BEAUTY_HELP)  # 不写没实测过的百分比
    assert A.BEAUTY_LABEL == "同时做一个「轻度美化」版本（可选）"
    assert A._beauty_visible("identical") and A._beauty_visible("perfect") and A._beauty_visible(None)
    for q in ("fast", "balanced", "best", "max"):
        assert not A._beauty_visible(q), q
    vs = [{"name": "未去杂音", "pct": 97.0, "score": 0.97, "recommended": False, "path": "a.wav"},
          {"name": "去杂音", "pct": 97.5, "score": 0.975, "recommended": True, "path": "b.wav"},
          {"name": "轻度美化", "pct": 96.8, "recommended": False, "path": "c.wav",
           "note": "版本 C 轻度美化（润色强度：一半）：像你本人 96.8%，原声 97.5%"}]
    assert A._variant_letter(vs[2], 2) == "C"
    md = A._variants_md(vs)
    assert "这次做了两个版本" in md and "版本 C：轻度美化（像你本人 96.8%）" in md and "不参与推荐" in md
    c_line = next(x for x in md.splitlines() if "版本 C" in x and "**" in x)
    assert "⭐" not in c_line and md.count("⭐ 推荐：版本 B") == 1
    assert A._variants_md(vs[:2]).count("版本 C") == 0


def test_web_generate_passes_beautify_only_when_checked_and_tier_allows(prepared, tmp_path, monkeypatch):
    import shutil

    from conftest import make_cfg
    from voicetwin.webui import app as A

    _, project, _ = prepared
    shutil.copytree(project.root, tmp_path / "ws" / project.voice)
    cfg = make_cfg(tmp_path / "ws")
    seen = {}

    def fake_stream(kind, label, voice, fn, *args, stages=None, hint="", note="", **kwargs):
        seen.update(kwargs=kwargs)
        yield "", {"done": True, "bar": "", "error": RuntimeError("测试"), "friendly": None}

    monkeypatch.setattr(A, "stream_task", fake_stream)
    monkeypatch.setattr(wf, "english_precheck_note", lambda *a, **k: ("", ""))
    ui = A.WebUI(cfg)
    for q, box, want in (("identical", True, True), ("perfect", True, True), ("fast", True, False),
                         ("identical", False, False), (None, True, True)):
        list(ui.do_generate(project.voice, "第一句。第二句。", None, "dummy", q, 0, "", "", "", "wav", box))
        assert seen["kwargs"]["beautify"] is want, (q, box)
    assert "var_c" in A.WebUI.GEN_OUT


def test_web_page_has_the_checkbox_and_hidden_third_player(tmp_path):
    gr = pytest.importorskip("gradio")
    if not str(getattr(gr, "__version__", "")).startswith("4."):
        pytest.skip("这个测试只针对 gradio 4.x（整合包自带 4.24）")
    from conftest import make_cfg
    from voicetwin.webui import app as A

    ui = A.WebUI(make_cfg(tmp_path / "ws"))
    app = ui.build()
    box = ui.c["beauty"]
    assert box.value is False and box.label == A.BEAUTY_LABEL and box.visible is True
    assert ui.c["var_c"].visible is False and ui.c["var_c"].label == "版本 C：轻度美化"
    conf = json.dumps(app.get_config_file(), ensure_ascii=False, default=str)
    assert A.BEAUTY_HELP in conf
