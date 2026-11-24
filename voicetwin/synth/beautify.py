"""「轻度美化」版本 C（research/V5极致/轻度美化设计.md 第 2、3、4 节）。只用 numpy / scipy（没有 GPL 依赖）。

做法（每一步都有上下限）：
- S1+S2 一条静态线性 EQ：目标曲线 T(f) = clamp(s·(C_her(f) + P(f)), ±2.5 dB)，1/3 倍频程平滑，80 Hz 以下、16 kHz 以上
  余弦过渡到 0 dB，再叠加 2 阶 Butterworth 高通（fc = clamp(0.5 × 她 F0 的 p05, 60, 80) Hz），合成一个最小相位 FIR。
  C_her = 她真实录音的长期平均频谱（LTAS）− 这一篇生成的 LTAS（只取说话的帧，先对齐整体电平），限幅 ±1.5 dB；
  P(f) = 温暖（低搁架 250 Hz +1.5 dB）、清楚（峰值 3.2 kHz +1.0 dB）、空气感（高搁架 11 kHz +1.5 dB，两边这一段都高于底噪才开）。
- S3 去齿音（4–9 kHz，阈值 = 她本人这个比值的 p95，最多压 4·s dB）；S4 轻压缩（阈值 = 她本人说话帧电平的 p90，
  最多压 3·s dB，不加补偿增益）；S5 电平由 engine 调回 base 这一句的说话响度（只乘系数，真峰值 ≤ −1 dBTP）。
- 逐句做，在拼接之前：两边补 0.25 秒的 0，处理完裁回原来的长度和位置；原句里连续 ≥ 1 ms 的精确 0 恢复成 0。
- 强度 s ∈ {1, 0.5, 0.25, 0}：s = 0 时原样返回（逐位相同）。s 乘在整条链上（EQ 的 C_her 和 P、去齿音和压缩的上限），
  这样「减半」每一步都更接近原声，减到 0 就是原声（设计里只把 s 乘在 P 上，那样 s = 0.25 时 C_her 还是满的，
  s = 0 时又突然全部去掉，和「s = 0 等于原声」「润色强度：一半」的说法对不上）。

把关（第 4 节）都是纯函数（gate_*），输入是量出来的数，方便用假打分器测试；engine 负责打分、逐句关动态处理、整篇减半。
"""

from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy import signal
from scipy.special import ndtri

from voicetwin.utils.audio import silence_threshold_from_db, speech_level_db, true_peak_db

CALIB_FILE = "beautify_calib.json"
#: 处理链 / 校准方法的版本：改了算法要升，旧的校准和缓存自动作废
CHAIN_VERSION = "1"
STRENGTHS = (1.0, 0.5, 0.25)
STRENGTH_WORDS = {1.0: "满", 0.5: "一半", 0.25: "四分之一"}

EQ_LIMIT_DB = 2.5          # T(f) 的上下限
C_HER_LIMIT_DB = 1.5       # 锚点 C_her 的上下限
EQ_LO_HZ, EQ_HI_HZ = 80.0, 16000.0
TAPER_LO_HZ = EQ_LO_HZ / math.sqrt(2.0)     # 80 Hz 以下半个倍频程余弦过渡到 0 dB（约 56.6 Hz 起就是 0）
TAPER_HI_HZ = EQ_HI_HZ * 2.0 ** 0.25        # 16 kHz 以上四分之一倍频程余弦过渡到 0 dB（约 19 kHz 起就是 0）
#: P(f)（s = 1 时）：(种类, 频率 Hz, 增益 dB, Q)。范围和来源见设计第 2 节的表
TILT: Tuple[Tuple[str, float, float, float], ...] = (
    ("low_shelf", 250.0, 1.5, 0.7),    # 温暖 / 磁性
    ("peak", 3200.0, 1.0, 1.0),        # 清楚
    ("high_shelf", 11000.0, 1.5, 0.7),  # 空气感（air=False 时不加）
)
AIR_BAND = (10000.0, 12000.0)
AIR_SNR_DB = 10.0          # 提议值：这一段说话时比底噪高 10 dB 以上才算「高于底噪」
HPF_RANGE = (60.0, 80.0)
FIR_TAPS = 4096            # 最小相位 FIR 的长度（线性相位原型 2×4096−1 点）
DEESS_BAND = (4000.0, 9000.0)
DEESS_MAX_DB = 4.0
DEESS_WIN_MS, DEESS_ATTACK_MS, DEESS_RELEASE_MS = 5.0, 2.0, 80.0
COMP_WIN_MS, COMP_ATTACK_MS, COMP_RELEASE_MS = 10.0, 20.0, 150.0
COMP_MAX_GR_DB = 3.0
HOP_MS = 1.0               # 动态处理每 1 ms 算一次增益
PAD_S = 0.25
ZERO_RUN_MS = 1.0
PEAK_CEILING_DB = -1.0     # 和 engine.PEAK_CEILING_DB 一样（真峰值）
SV_MIN_SECONDS = 2.0       # 人声不到 2 秒的句子不做「像你本人」的关卡（实测 −13.3 ± 10.8，太不可靠）
LTAS_TOL_DB = 0.5          # G6 每个频段：实测的差和设计的 EQ 响应之间允许的误差（测试里实测线性 EQ 在 0.3 dB 以内）
F0_RATE = 16000            # F0 追踪器（style.prosody.f0_frames，librosa.yin）在 16 kHz 上算
MIN_CALIB_ITEMS = 3        # 校准至少要 3 段没参加训练、人声 ≥ 2 秒的真实录音（和声纹校准的「不够 3 条」一样）
MAX_CALIB_ITEMS = 40       # 最多用 40 段（和声纹校准 max_clips 一样，控制时间）
MAX_HER_CLIPS = 160        # 量她本人的频谱、齿音、电平分布最多读这么多段

#: 1/3 倍频程中心频率（以 1 kHz 为基准，100 Hz–16 kHz，共 23 个）
BAND_CENTERS = 1000.0 * 2.0 ** (np.arange(-10, 13) / 3.0)


# ============================================================================ 频谱
def _speech_mask(frame_db: np.ndarray) -> np.ndarray:
    """说话的帧：电平在自动静音阈值以上（和 speech_level_db 同一个阈值算法）；全是 0 的帧不算。"""
    db = np.asarray(frame_db, dtype=np.float64)
    if db.size == 0 or not np.any(db > -100):
        return np.zeros(db.shape, dtype=bool)
    thr = silence_threshold_from_db(db)
    return (db >= thr) & (db > -100)


def _nfft(sr: int) -> int:
    return int(1 << int(math.ceil(math.log2(max(0.08 * sr, 256.0)))))  # 48 kHz → 4096 点（11.7 Hz 一格）


@dataclass
class Spectrum:
    """说话帧的平均功率谱（累加用）。power：每个频率的功率和；frames：说话帧数；noise：每个频率功率的 p10 换算的底噪。"""
    sr: int
    nfft: int
    power: np.ndarray
    frames: int = 0
    noise_parts: List[np.ndarray] = field(default_factory=list)
    masks: List[np.ndarray] = field(default_factory=list)   # 每段哪些帧算说话（C 用 base 的，同样的帧才能逐段比）

    @property
    def freqs(self) -> np.ndarray:
        return np.fft.rfftfreq(self.nfft, 1.0 / self.sr)

    def mean_power(self) -> np.ndarray:
        return self.power / max(self.frames, 1)


def speech_spectrum(wavs: Iterable[np.ndarray], sr: int, keep_noise: bool = False,
                    masks: Optional[Sequence[np.ndarray]] = None) -> Spectrum:
    """一段或几段声音说话帧的平均功率谱（Welch：汉宁窗、一半重叠）。keep_noise：同时记下每段每个频率功率的 p10
    （「最小统计」估底噪，和 utils.audio.speech_noise_floor_db 同一个思路），判断空气感那一段是不是高于底噪。
    masks：用给定的说话帧（C 和 base 长度一样，用 base 的帧，处理前后比的是同样的帧）。"""
    n = _nfft(sr)
    hop = n // 2
    win = np.hanning(n)
    wnorm = float(np.sum(win ** 2))
    spec = Spectrum(sr, n, np.zeros(n // 2 + 1))
    for j, w in enumerate(wavs):
        x = np.asarray(w, dtype=np.float64).reshape(-1)
        if x.size < n:
            spec.masks.append(np.zeros(0, dtype=bool))
            continue
        k = 1 + (x.size - n) // hop
        frames = x[np.arange(k)[:, None] * hop + np.arange(n)[None, :]]
        if masks is not None and j < len(masks) and len(masks[j]) == k:
            mask = np.asarray(masks[j], dtype=bool)
        else:
            fdb = 10.0 * np.log10(np.mean(frames ** 2, axis=1) + 1e-10)
            mask = _speech_mask(fdb)
        spec.masks.append(mask)
        if not np.any(mask):
            continue
        pw = np.abs(np.fft.rfft(frames[mask] * win, axis=1)) ** 2 / wnorm
        spec.power += pw.sum(axis=0)
        spec.frames += int(mask.sum())
        if keep_noise and pw.shape[0] >= 5:
            spec.noise_parts.append(np.percentile(pw, 10, axis=0) / -math.log(0.9))
    return spec


def band_db(spec: Spectrum, centers: np.ndarray = BAND_CENTERS) -> np.ndarray:
    """1/3 倍频程每个频段的平均功率（dB）。频段超出奈奎斯特频率的 95%、或者没有说话帧时是 NaN。"""
    out = np.full(len(centers), np.nan)
    if spec.frames <= 0:
        return out
    f, p = spec.freqs, spec.mean_power()
    for i, c in enumerate(centers):
        lo, hi = c * 2.0 ** (-1 / 6), c * 2.0 ** (1 / 6)
        if hi > 0.475 * spec.sr:
            continue
        sel = (f >= lo) & (f < hi)
        if np.any(sel):
            out[i] = 10.0 * math.log10(float(np.mean(p[sel])) + 1e-20)
    return out


def ltas_db(wavs: Iterable[np.ndarray], sr: int) -> np.ndarray:
    """LTAS：说话帧的 1/3 倍频程平均频谱（dB，100 Hz–16 kHz）。"""
    return band_db(speech_spectrum(wavs, sr))


def _aligned_diff(a: Sequence[float], b: Sequence[float]) -> np.ndarray:
    """a − b，减去所有有效频段的平均差（先对齐整体电平）；无效的频段是 NaN。"""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    d = a - b
    ok = np.isfinite(d)
    out = np.full(d.shape, np.nan)
    if np.any(ok):
        out[ok] = d[ok] - float(np.mean(d[ok]))
    return out


def ltas_distance(a: Sequence[float], b: Sequence[float]) -> Optional[float]:
    """两个 LTAS 的距离（对齐整体电平后各频段差的均方根，dB）。没有共同的有效频段时 None。"""
    d = _aligned_diff(a, b)
    ok = np.isfinite(d)
    return float(math.sqrt(float(np.mean(d[ok] ** 2)))) if np.any(ok) else None


def band_ratio_db(spec: Spectrum, num: Tuple[float, float], den: Tuple[float, float]) -> Optional[float]:
    """两个频率范围的能量比（dB），用说话帧的平均功率谱。上限超出奈奎斯特频率时截到 0.475 × 采样率。"""
    if spec.frames <= 0:
        return None
    f, p = spec.freqs, spec.mean_power()
    cap = 0.475 * spec.sr

    def energy(lo: float, hi: float) -> float:
        sel = (f >= lo) & (f < min(hi, cap))
        return float(np.sum(p[sel])) if np.any(sel) else 0.0

    a, b = energy(*num), energy(*den)
    if a <= 0 or b <= 0:
        return None
    return 10.0 * math.log10(a / b)


def hf_ratio_db(spec: Spectrum, hi: float = 16000.0) -> Optional[float]:
    """G7：(8–16 kHz) / (1–4 kHz) 能量比（dB）。"""
    return band_ratio_db(spec, (8000.0, hi), (1000.0, 4000.0))


def air_snr_db(spec: Spectrum) -> Optional[float]:
    """空气感那一段（10–12 kHz）说话时比底噪高多少 dB（底噪 = 每段每个频率功率的 p10 换算，取各段中位数）。"""
    if spec.frames <= 0 or not spec.noise_parts or AIR_BAND[1] > 0.475 * spec.sr:
        return None
    f = spec.freqs
    sel = (f >= AIR_BAND[0]) & (f < AIR_BAND[1])
    if not np.any(sel):
        return None
    speech = float(np.sum(spec.mean_power()[sel]))
    noise = float(np.median([float(np.sum(nz[sel])) for nz in spec.noise_parts]))
    if speech <= 0:
        return None
    return 10.0 * math.log10(speech / max(noise, 1e-20))


# ============================================================================ EQ
def _shelf_peak_db(f: np.ndarray, kind: str, f0: float, gain_db: float, q: float) -> np.ndarray:
    """RBJ 均衡器（模拟原型）在频率 f 上的幅度（dB）：low_shelf / high_shelf / peak。和采样率无关。"""
    if gain_db == 0.0:
        return np.zeros_like(f, dtype=np.float64)
    a = 10.0 ** (gain_db / 40.0)
    s = 1j * np.asarray(f, dtype=np.float64) / f0
    if kind == "peak":
        h = (s ** 2 + s * (a / q) + 1.0) / (s ** 2 + s / (a * q) + 1.0)
    elif kind == "low_shelf":
        h = a * (s ** 2 + (math.sqrt(a) / q) * s + a) / (a * s ** 2 + (math.sqrt(a) / q) * s + 1.0)
    elif kind == "high_shelf":
        h = a * (a * s ** 2 + (math.sqrt(a) / q) * s + 1.0) / (s ** 2 + (math.sqrt(a) / q) * s + a)
    else:
        raise ValueError(kind)
    return 20.0 * np.log10(np.abs(h))


def tilt_db(f: np.ndarray, s: float = 1.0, air: bool = True) -> np.ndarray:
    """P(f) × s（dB）：每一项的增益乘 s（Q 不变）。"""
    out = np.zeros_like(np.asarray(f, dtype=np.float64))
    for kind, f0, g, q in TILT:
        if kind == "high_shelf" and not air:
            continue
        out = out + _shelf_peak_db(np.asarray(f, dtype=np.float64), kind, f0, g * float(s), q)
    return out


def hpf_db(f: np.ndarray, fc: float, floor_db: float = -80.0) -> np.ndarray:
    """2 阶 Butterworth 高通的幅度（dB）：1 / sqrt(1 + (fc/f)^4)，最低 floor_db（f = 0 处不取 log 0）。"""
    f = np.maximum(np.asarray(f, dtype=np.float64), 1e-6)
    return np.maximum(-10.0 * np.log10(1.0 + (fc / f) ** 4), floor_db)


def _taper(f: np.ndarray) -> np.ndarray:
    """80 Hz–16 kHz 里是 1，往外余弦过渡到 0（TAPER_LO_HZ 以下、TAPER_HI_HZ 以上是 0）。"""
    f = np.asarray(f, dtype=np.float64)
    w = np.zeros_like(f)
    lf = np.log2(np.maximum(f, 1e-6))
    inside = (f >= EQ_LO_HZ) & (f <= EQ_HI_HZ)
    w[inside] = 1.0
    lo = (f > TAPER_LO_HZ) & (f < EQ_LO_HZ)
    t = (lf[lo] - math.log2(TAPER_LO_HZ)) / (math.log2(EQ_LO_HZ) - math.log2(TAPER_LO_HZ))
    w[lo] = 0.5 - 0.5 * np.cos(np.pi * t)
    hi = (f > EQ_HI_HZ) & (f < TAPER_HI_HZ)
    t = (math.log2(TAPER_HI_HZ) - lf[hi]) / (math.log2(TAPER_HI_HZ) - math.log2(EQ_HI_HZ))
    w[hi] = 0.5 - 0.5 * np.cos(np.pi * t)
    return w


def _third_octave_smooth(f: np.ndarray, db: np.ndarray) -> np.ndarray:
    """在对数频率上做 1/3 倍频程宽的滑动平均（f 必须是单调递增的对数网格）。"""
    lf = np.log2(f)
    step = float(np.median(np.diff(lf)))
    k = max(1, int(round((1.0 / 3.0) / step)))
    if k <= 1:
        return db
    kernel = np.ones(k) / k
    padded = np.concatenate([np.full(k, db[0]), db, np.full(k, db[-1])])
    return np.convolve(padded, kernel, mode="same")[k:-k]


def target_db(f: np.ndarray, s: float, c_her: Optional[Sequence[float]] = None, air: bool = True,
              centers: np.ndarray = BAND_CENTERS) -> np.ndarray:
    """目标曲线 T(f)（dB，不含高通）：clamp(s·(C_her + P), ±2.5)，1/3 倍频程平滑，80 Hz–16 kHz 以外过渡到 0。
    C_her 是每个 1/3 倍频程频段的值（NaN 当 0），频段之间按对数频率线性插值，频段以外取边上的值。"""
    f = np.asarray(f, dtype=np.float64)
    grid = np.geomspace(20.0, max(24000.0, float(np.max(f)) if f.size else 24000.0), 2048)
    inner = tilt_db(grid, 1.0, air)
    if c_her is not None:
        c = np.nan_to_num(np.clip(np.asarray(c_her, dtype=np.float64), -C_HER_LIMIT_DB, C_HER_LIMIT_DB))
        inner = inner + np.interp(np.log2(grid), np.log2(centers), c)
    t = np.clip(float(s) * inner, -EQ_LIMIT_DB, EQ_LIMIT_DB)
    t = np.clip(_third_octave_smooth(grid, t), -EQ_LIMIT_DB, EQ_LIMIT_DB)
    t = t * _taper(grid)
    return np.interp(np.log2(np.maximum(f, 20.0)), np.log2(grid), t) * _taper(f)


def design_eq(sr: int, s: float, c_her: Optional[Sequence[float]] = None, air: bool = True,
              hpf_fc: Optional[float] = None, taps: int = FIR_TAPS) -> np.ndarray:
    """静态 EQ（S1+S2）：目标 T(f) + 高通 → 频率采样设计线性相位 FIR（2×taps−1 点）→ scipy.signal.minimum_phase 转最小相位。

    注意「平方根」陷阱：minimum_phase 默认（homomorphic）得到的是原滤波器幅度响应的**平方根**（scipy 文档写明；
    half=False 只有 scipy ≥ 1.14 才有，老师的整合包是 1.9.3），所以线性相位原型按**平方后的**目标幅度设计。"""
    n_lin = 2 * int(taps) - 1
    grid = np.linspace(0.0, sr / 2.0, 8193)
    db = target_db(grid, s, c_her, air)
    if hpf_fc:
        db = db + hpf_db(grid, float(hpf_fc))
    gain_sq = (10.0 ** (db / 20.0)) ** 2
    h_lin = signal.firwin2(n_lin, grid, gain_sq, fs=sr, window="hann")
    return signal.minimum_phase(h_lin, method="homomorphic").astype(np.float64)


def fir_response_db(h: np.ndarray, sr: int, f: np.ndarray) -> np.ndarray:
    """FIR 在频率 f 上实测的幅度（dB）。"""
    _, resp = signal.freqz(h, worN=np.asarray(f, dtype=np.float64), fs=sr)
    return 20.0 * np.log10(np.maximum(np.abs(resp), 1e-12))


def apply_fir(x: np.ndarray, h: np.ndarray) -> np.ndarray:
    """因果滤波（不补延迟，长度不变）：最小相位 FIR 的能量集中在开头，没有整体延迟。"""
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        return x.copy()
    return signal.oaconvolve(x, h)[: x.size]


# ============================================================================ 动态处理
def _hop(sr: int) -> int:
    return max(1, int(round(sr * HOP_MS / 1000.0)))


def _rms_db_track(x: np.ndarray, sr: int, win_ms: float) -> np.ndarray:
    """每 1 ms 一个点：以这一点为中心、win_ms 宽的 RMS（dB）。"""
    hop = _hop(sr)
    win = max(hop, int(round(sr * win_ms / 1000.0)))
    pad = win // 2
    sq = np.pad(np.asarray(x, dtype=np.float64) ** 2, (pad, pad))
    csum = np.concatenate([[0.0], np.cumsum(sq)])
    n = 1 + (len(x) - 1) // hop if len(x) else 0
    starts = np.arange(n) * hop
    e = (csum[starts + win] - csum[starts]) / win
    return 10.0 * np.log10(e + 1e-12)


def _smooth_db(target: np.ndarray, sr: int, attack_ms: float, release_ms: float) -> np.ndarray:
    """增益衰减（dB，≥ 0）的平滑：变大用 attack，变小用 release（一阶，每 1 ms 一步）。"""
    a = math.exp(-HOP_MS / max(attack_ms, 1e-6))
    r = math.exp(-HOP_MS / max(release_ms, 1e-6))
    out = np.empty_like(target)
    g = 0.0
    for i, t in enumerate(target.tolist()):
        c = a if t > g else r
        g = c * g + (1.0 - c) * t
        out[i] = g
    return out


def _gain_curve(cut_db: np.ndarray, n: int, sr: int) -> np.ndarray:
    """每 1 ms 的衰减（dB）插值到每个采样，换成线性系数。"""
    hop = _hop(sr)
    t = np.arange(len(cut_db)) * hop
    return 10.0 ** (-np.interp(np.arange(n), t, cut_db) / 20.0)


def deess_band(x: np.ndarray, sr: int) -> np.ndarray:
    """齿音带（4–9 kHz，2 阶带通，sosfiltfilt 零相位）。"""
    hi = min(DEESS_BAND[1], 0.45 * sr)
    sos = signal.butter(2, [DEESS_BAND[0], hi], btype="bandpass", fs=sr, output="sos")
    return signal.sosfiltfilt(sos, np.asarray(x, dtype=np.float64))


def sibilance_db(x: np.ndarray, sr: int) -> Tuple[np.ndarray, np.ndarray]:
    """每 1 ms 一个点的「齿音带 RMS / 全带 RMS」（dB，5 ms 窗）和这一点是不是在说话（全带电平在自动静音阈值以上）。
    采样率太低（奈奎斯特 ≤ 4.5 kHz）时没有齿音带，返回空。"""
    if sr <= 2 * DEESS_BAND[0] * 1.125 or len(x) == 0:
        return np.zeros(0), np.zeros(0, dtype=bool)
    full = _rms_db_track(x, sr, DEESS_WIN_MS)
    band = _rms_db_track(deess_band(x, sr), sr, DEESS_WIN_MS)
    return band - full, _speech_mask(full)


def sib_p95(wavs: Iterable[np.ndarray], sr: int) -> Optional[float]:
    """几句话合起来，说话时齿音比的 p95（dB）。G7 用（和她本人的 p95 比）。"""
    parts = []
    for w in wavs:
        ratio, mask = sibilance_db(w, sr)
        if ratio.size and np.any(mask):
            parts.append(ratio[mask])
    return float(np.percentile(np.concatenate(parts), 95)) if parts else None


def deess(x: np.ndarray, sr: int, threshold_db: Optional[float], max_cut_db: float = DEESS_MAX_DB,
          attack_ms: float = DEESS_ATTACK_MS, release_ms: float = DEESS_RELEASE_MS
          ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """S3 去齿音：齿音比超过阈值（她本人的 p95）的地方，齿音带压低超出的部分，最多 max_cut_db。
    输出 = 其余部分 + g(t)·齿音带。整句都没超过阈值时原样返回（逐位相同）。"""
    x = np.asarray(x, dtype=np.float64)
    info = {"max_cut_db": 0.0, "ms_cut": 0}
    if threshold_db is None or max_cut_db <= 0 or x.size == 0:
        return x, info
    ratio, speech = sibilance_db(x, sr)
    if ratio.size == 0:
        return x, info
    # 把齿音带压低 c dB 以后比值正好回到阈值：ρ = 带 / 全带（功率），τ = 阈值，c = 10·log10(ρ(1−τ) / (τ(1−ρ)))
    # （全带里也有齿音带，只压「超出的 dB 数」压不够）
    rho = np.clip(10.0 ** (ratio / 10.0), 1e-12, 1.0 - 1e-9)
    tau = min(max(10.0 ** (float(threshold_db) / 10.0), 1e-12), 1.0 - 1e-9)
    need = 10.0 * np.log10(rho * (1.0 - tau) / (tau * (1.0 - rho)))
    target = np.clip(np.where(speech & (ratio > float(threshold_db)), need, 0.0), 0.0, float(max_cut_db))
    if not np.any(target > 0):
        return x, info
    cut = np.minimum(_smooth_db(target, sr, attack_ms, release_ms), float(max_cut_db))
    band = deess_band(x, sr)
    rest = x - band
    out = rest + _gain_curve(cut, x.size, sr) * band
    info = {"max_cut_db": round(float(np.max(cut)), 3), "ms_cut": int(np.count_nonzero(cut > 0.01))}
    return out, info


def compress(x: np.ndarray, sr: int, threshold_db: Optional[float], ratio: float, max_gr_db: float = COMP_MAX_GR_DB,
             attack_ms: float = COMP_ATTACK_MS, release_ms: float = COMP_RELEASE_MS
             ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """S4 轻压缩（前馈，10 ms RMS 检测）：超过阈值的部分按比例压，增益衰减最多 max_gr_db；不加补偿增益。
    整句没有任何衰减时原样返回（逐位相同）。"""
    x = np.asarray(x, dtype=np.float64)
    info = {"max_gr_db": 0.0, "ms_gr": 0}
    if threshold_db is None or ratio <= 1.0 or max_gr_db <= 0 or x.size == 0:
        return x, info
    level = _rms_db_track(x, sr, COMP_WIN_MS)
    over = level - float(threshold_db)
    target = np.clip(np.where(over > 0, over * (1.0 - 1.0 / float(ratio)), 0.0), 0.0, float(max_gr_db))
    if not np.any(target > 0):
        return x, info
    gr = np.minimum(_smooth_db(target, sr, attack_ms, release_ms), float(max_gr_db))
    out = x * _gain_curve(gr, x.size, sr)
    info = {"max_gr_db": round(float(np.max(gr)), 3), "ms_gr": int(np.count_nonzero(gr > 0.01))}
    return out, info


def zero_runs(x: np.ndarray, sr: int, min_ms: float = ZERO_RUN_MS) -> List[Tuple[int, int]]:
    """连续 ≥ min_ms 的精确 0：[(起点, 终点)]。"""
    z = np.asarray(x) == 0
    if not np.any(z):
        return []
    n = max(1, int(round(sr * min_ms / 1000.0)))
    d = np.diff(np.concatenate([[0], z.astype(np.int8), [0]]))
    starts, ends = np.flatnonzero(d == 1), np.flatnonzero(d == -1)
    return [(int(a), int(b)) for a, b in zip(starts, ends) if b - a >= n]


# ============================================================================ 计划和逐句处理
@dataclass
class Plan:
    """一篇用的处理计划（每句都一样，句子之间音色不跳）。"""
    s: float
    sr: int
    fir: Optional[np.ndarray] = None
    hpf_fc: float = HPF_RANGE[0]
    air: bool = False
    c_her: Optional[List[float]] = None
    deess_thr_db: Optional[float] = None
    deess_max_db: float = 0.0
    comp_rel_db: Optional[float] = None
    comp_ratio: float = 1.0
    comp_max_db: float = 0.0
    sig: str = ""

    def target_bands(self) -> List[float]:
        """T(f) 在每个 1/3 倍频程中心频率上的值（报告用）。"""
        return [round(float(v), 2) for v in target_db(BAND_CENTERS, self.s, self.c_her, self.air)]

    def expected_change(self, spec: Spectrum) -> np.ndarray:
        """按这条 EQ（T + 高通，用 FIR 实测的响应）和 base 自己在每个频段里的频谱，算出线性滤波以后每个 1/3 倍频程
        频段应该变多少 dB（Σ P·|H|² / Σ P）。G6 用它比实测：频段里能量分布不均匀（例如 100 Hz 附近几乎没有声音）时，
        直接拿频段中心的设计值比会差出零点几 dB。没有 EQ 时全是 0。"""
        out = np.zeros(len(BAND_CENTERS))
        if self.fir is None or spec.frames <= 0:
            return out
        f, p = spec.freqs, spec.mean_power()
        h2 = 10.0 ** (fir_response_db(self.fir, spec.sr, f) / 10.0)
        for i, c in enumerate(BAND_CENTERS):
            lo, hi = c * 2.0 ** (-1 / 6), c * 2.0 ** (1 / 6)
            sel = (f >= lo) & (f < hi)
            if hi > 0.475 * spec.sr or not np.any(sel) or float(np.sum(p[sel])) <= 0:
                out[i] = np.nan
                continue
            out[i] = 10.0 * math.log10(float(np.sum(p[sel] * h2[sel])) / float(np.sum(p[sel])))
        return out

    def summary(self) -> Dict[str, Any]:
        t = self.target_bands()
        return {"s": self.s, "hpf_hz": round(self.hpf_fc, 1), "air": self.air,
                "target_db": {"hz": [round(float(c)) for c in BAND_CENTERS], "db": t,
                              "max": max(t) if t else 0.0, "min": min(t) if t else 0.0},
                "c_her_db": None if self.c_her is None else [None if not np.isfinite(v) else round(float(v), 2)
                                                           for v in self.c_her],
                "deess": {"threshold_db": _r(self.deess_thr_db), "max_cut_db": self.deess_max_db},
                "compress": {"threshold_rel_db": _r(self.comp_rel_db), "ratio": self.comp_ratio,
                             "max_gr_db": self.comp_max_db}, "sig": self.sig}


def _r(v: Any, nd: int = 2) -> Any:
    return None if v is None or not np.isfinite(float(v)) else round(float(v), nd)


def comp_ratio(s: float) -> float:
    """压缩比：s ≤ 0.5 时 1.5:1，s = 1 时 2:1（设计第 2 节）。"""
    return 2.0 if s > 0.5 else 1.5


def make_plan(sr: int, s: float, her: Dict[str, Any], c_her: Optional[Sequence[float]] = None,
              air: bool = True) -> Plan:
    """按强度 s 和她本人的统计量（her：hpf_hz、air_ok、sib_p95_db、comp_rel_p90_db）做一篇的处理计划。"""
    s = float(s)
    if s <= 0:
        return Plan(s=0.0, sr=int(sr), sig="off")
    hpf = float(np.clip(float(her.get("hpf_hz") or HPF_RANGE[0]), *HPF_RANGE))
    air = bool(air and her.get("air_ok") and sr > 2 * AIR_BAND[1] * 1.05)
    c = None if c_her is None else [float(v) if v is not None and np.isfinite(float(v)) else float("nan")
                                    for v in c_her]
    plan = Plan(s=s, sr=int(sr), hpf_fc=hpf, air=air, c_her=c,
                deess_thr_db=_num(her.get("sib_p95_db")), deess_max_db=round(DEESS_MAX_DB * s, 3),
                comp_rel_db=_num(her.get("comp_rel_p90_db")), comp_ratio=comp_ratio(s),
                comp_max_db=round(COMP_MAX_GR_DB * s, 3))
    key = json.dumps([CHAIN_VERSION, plan.s, plan.sr, round(hpf, 3), air,
                      None if c is None else [None if not np.isfinite(v) else round(v, 4) for v in c],
                      _r(plan.deess_thr_db, 4), plan.deess_max_db, _r(plan.comp_rel_db, 4), plan.comp_ratio,
                      plan.comp_max_db])
    plan.sig = hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]
    plan.fir = design_eq(int(sr), s, c, air, hpf)
    return plan


def _num(v: Any) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if np.isfinite(f) else None


def beautify_sentence(wav: np.ndarray, sr: int, plan: Plan, dynamics: bool = True, compress_on: bool = True
                      ) -> Tuple[np.ndarray, Dict[str, Any]]:
    """一句话走 C（拼接之前）：两边补 0.25 秒的 0 → 静态 EQ → 去齿音 → 轻压缩 → 裁回原来的长度和位置 →
    原句里连续 ≥ 1 ms 的精确 0 恢复成 0。s = 0 时原样返回（逐位相同）。电平（S5）由 engine 统一调。
    dynamics=False：只做静态 EQ（G2 / G3 不过的句子）；compress_on=False：不压缩（G8 不过的句子）。"""
    x = np.asarray(wav, dtype=np.float32)
    info: Dict[str, Any] = {"changed": False, "deess": None, "compress": None}
    if plan.s <= 0 or plan.fir is None or x.size == 0:
        return x, info
    pad = int(round(PAD_S * sr))
    n = x.size
    y = np.concatenate([np.zeros(pad), x.astype(np.float64), np.zeros(pad)])
    y = apply_fir(y, plan.fir)
    if dynamics:
        y, info["deess"] = deess(y, sr, plan.deess_thr_db, plan.deess_max_db)
        if compress_on and plan.comp_rel_db is not None:
            lvl = speech_level_db(x, sr)
            if lvl is not None:
                y, info["compress"] = compress(y, sr, lvl + plan.comp_rel_db, plan.comp_ratio, plan.comp_max_db)
    out = y[pad:pad + n].astype(np.float32)
    for a, b in zero_runs(x, sr):
        out[a:b] = 0.0
    info["changed"] = not np.array_equal(out, x)
    return out, info


# ============================================================================ 关卡（第 4 节）
def null_mean_floor(mu: float, sigma: float, n: int) -> float:
    """G1 和校准：mean(Δ) 不能低于 μ_null − 2·σ_null/√n（和单纯改音量在统计上分不开，单侧约 95%）。"""
    return float(mu) - 2.0 * float(sigma) / math.sqrt(max(int(n), 1))


def null_each_floor(mu: float, sigma: float, n: int, alpha: float = 0.05) -> float:
    """G2：每句 Δ_i 不能低于 μ_null − z(1 − α/N)·σ_null（Bonferroni：单纯改音量的整篇约 95% 概率全部通过）。"""
    z = float(ndtri(1.0 - alpha / max(int(n), 1)))
    return float(mu) - z * float(sigma)


def gate_identity(pct_base: Sequence[Optional[float]], pct_c: Sequence[Optional[float]],
                  seconds: Sequence[Optional[float]], mu: float, sigma: float, min_pct: float
                  ) -> Dict[str, Any]:
    """G1（整篇）+ G2（逐句）。只看人声 ≥ 2 秒、两边都量得出的句子（短句子的分数太不可靠，只查别的关卡）。
    G2 另外要求原来达到 min_pct 的句子美化后也不低于 min_pct（原来就低于的句子只看 Δ：
    设计写的是 pct_C ≥ min_pct，但原声那一句本来就不到 85% 时，C 再好也过不了，会让整篇白白减到 0）。"""
    idx = [i for i, (a, b, s) in enumerate(zip(pct_base, pct_c, seconds))
           if a is not None and b is not None and s is not None and float(s) >= SV_MIN_SECONDS]
    short = [i for i, s in enumerate(seconds) if s is None or float(s) < SV_MIN_SECONDS]
    deltas = {i: float(pct_c[i]) - float(pct_base[i]) for i in idx}  # type: ignore[arg-type]
    n = len(idx)
    out: Dict[str, Any] = {"n": n, "short": short, "deltas": {i: round(d, 2) for i, d in deltas.items()}}
    if n == 0:
        out.update(g1=None, g2_fail=[], mean_delta=None, g1_floor=None, g2_floor=None)
        return out
    mean_d = float(np.mean(list(deltas.values())))
    f1 = null_mean_floor(mu, sigma, n)
    f2 = null_each_floor(mu, sigma, n)
    fails = [i for i in idx if deltas[i] < f2
             or (float(pct_base[i]) >= min_pct and float(pct_c[i]) < min_pct)]  # type: ignore[arg-type]
    out.update(g1=mean_d >= f1, g2_fail=fails, mean_delta=round(mean_d, 3), g1_floor=round(f1, 3),
               g2_floor=round(f2, 3))
    return out


def gate_content(cer_base: Optional[float], cer_c: Optional[float],
                 cer_null: Sequence[Optional[float]] = ()) -> Optional[bool]:
    """G3：CER_C ≤ CER_base；升高了时看空白对照（base 音量 ±1 dB 再识别）：base 自己也会跳到这个错误率的不算。
    量不出来（没有识别模型）返回 None（报告里如实写没查）。"""
    if cer_base is None or cer_c is None:
        return None
    if float(cer_c) <= float(cer_base) + 1e-9:
        return True
    return any(v is not None and float(v) >= float(cer_c) - 1e-9 for v in cer_null)


def f0_floor_st(f0_hz: float) -> float:
    """F0 追踪器的分辨率（半音）：16 kHz 上差一个采样的周期，F0 就差 12·log2(1 + F0/16000) 个半音。"""
    return 12.0 * math.log2(1.0 + max(float(f0_hz), 1.0) / F0_RATE)


def gate_pitch(n_base: int, n_c: int, f0_base: Optional[float], f0_c: Optional[float],
               f0_null: Sequence[Optional[float]] = ()) -> Dict[str, Any]:
    """G4：样本数完全相同（时长不变）；F0 中位数偏移 ≤ 同一个追踪器的抖动：空白对照（base 改音量）的偏移和
    追踪器分辨率（f0_floor_st）里大的那个。线性滤波按原理不改 F0，这一关查的是程序错误。"""
    out: Dict[str, Any] = {"same_length": int(n_base) == int(n_c), "shift_st": None, "allow_st": None}
    ok = out["same_length"]
    if f0_base and f0_c and f0_base > 0 and f0_c > 0:
        shift = abs(12.0 * math.log2(float(f0_c) / float(f0_base)))
        jit = [abs(12.0 * math.log2(float(v) / float(f0_base))) for v in f0_null if v and v > 0]
        allow = max([f0_floor_st(f0_base)] + jit)
        out.update(shift_st=round(shift, 4), allow_st=round(allow, 4))
        ok = ok and shift <= allow + 1e-9
    out["passed"] = bool(ok)
    return out


def xcorr_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> int:
    """b 相对 a 的整体延迟（互相关峰值的滞后，采样数；正数 = b 晚）。"""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    n = min(a.size, b.size)
    if n == 0 or not np.any(a) or not np.any(b):
        return 0
    c = signal.correlate(b[:n], a[:n], mode="full", method="fft")
    mid = n - 1
    lo, hi = max(0, mid - max_lag), min(c.size, mid + max_lag + 1)
    return int(np.argmax(c[lo:hi]) + lo - mid)


def chain_lag(base: np.ndarray, out: np.ndarray, plan: "Plan", sr: int, max_ms: float = 20.0) -> int:
    """G5：处理链有没有整体延迟（程序错误：例如没补线性相位 FIR 的群延迟、裁剪时错了补零的长度）。
    拿「base 只过这条静态 EQ」当参照，和输出做互相关，峰值的滞后必须是 0。
    设计写的是直接拿 base 和输出比互相关，但最小相位的高通（2 阶 Butterworth 本身就是最小相位）在 150 Hz 让基频
    超前约 30°，音高很稳的说话声互相关的峰会移十几个采样（测试引擎的声音实测 −11、−13），整体其实一点没延迟。
    EQ 本身的相位是设计的一部分（每条人声链的高通都这样），不是延迟。"""
    if plan.fir is None:
        return xcorr_lag(base, out, int(max_ms * sr / 1000.0))
    return xcorr_lag(apply_fir(np.asarray(base, dtype=np.float64), plan.fir), out, int(max_ms * sr / 1000.0))


def gate_timbre(ltas_base: Sequence[float], ltas_c: Sequence[float], ltas_her: Sequence[float],
                sigma_her: float, designed: Sequence[float], s: float) -> Dict[str, Any]:
    """G6：① D(C, 她) ≤ D(base, 她) + σ_她（离她不能比她自己隔天录音的差别更远）；
    ② 每个频段 LTAS_C − LTAS_base 和这条 EQ 按设计应该带来的变化（designed：Plan.expected_change，|T| ≤ 2.5 dB 加高通）
    之间的差 ≤ 误差：LTAS_TOL_DB，加上动态处理按设计最多能带来的变化（压缩是随时间变的整体增益，任何频段最多变
    3·s dB；齿音带再加去齿音最多压的 4·s dB）。也就是处理链没做设计以外的事（例如 +12 dB 的搁架是坏样本）。
    C 和 base 要用同样的说话帧量（speech_spectrum 的 masks）。"""
    d_base, d_c = ltas_distance(ltas_her, ltas_base), ltas_distance(ltas_her, ltas_c)
    dev = np.asarray(ltas_c, dtype=np.float64) - np.asarray(ltas_base, dtype=np.float64) - np.asarray(designed)
    ok = np.isfinite(dev)
    worst = None
    bands_ok = True
    if np.any(ok):
        dev = dev - float(np.median(dev[ok]))  # 整体电平由 S5 另外调，不算
        base_tol = LTAS_TOL_DB + COMP_MAX_GR_DB * float(s)
        tol = np.where((BAND_CENTERS >= DEESS_BAND[0] * 2 ** (-1 / 6)) & (BAND_CENTERS <= DEESS_BAND[1] * 2 ** (1 / 6)),
                       base_tol + DEESS_MAX_DB * float(s), base_tol)
        excess = np.where(ok, np.abs(dev) - tol, -np.inf)
        k = int(np.argmax(excess))
        worst = {"hz": round(float(BAND_CENTERS[k])), "dev_db": round(float(dev[k]), 2),
                 "tol_db": round(float(tol[k]), 2)}
        bands_ok = bool(np.all(excess[ok] <= 1e-9))
    dist_ok = d_base is None or d_c is None or d_c <= d_base + float(sigma_her) + 1e-9
    return {"passed": bool(dist_ok and bands_ok), "dist_ok": bool(dist_ok), "bands_ok": bands_ok,
            "d_base": _r(d_base), "d_c": _r(d_c), "sigma_her": _r(sigma_her), "worst_band": worst}


def _within(v: Optional[float], lo: Optional[float], hi: Optional[float]) -> bool:
    return v is not None and (lo is None or v >= lo - 1e-9) and (hi is None or v <= hi + 1e-9)


def _outside_by(v: float, lo: Optional[float], hi: Optional[float]) -> float:
    if lo is not None and v < lo:
        return lo - v
    if hi is not None and v > hi:
        return v - hi
    return 0.0


def gate_hf(hf_base: Optional[float], hf_c: Optional[float], her_p05: Optional[float], her_p95: Optional[float],
            sib_base_p95: Optional[float], sib_c_p95: Optional[float], her_sib_p95: Optional[float]
            ) -> Dict[str, Any]:
    """G7（8 kHz 以上，声纹和识别的前端都是 16 kHz，看不到这一段）：
    ① (8–16 kHz)/(1–4 kHz) 能量比在她本人的 p05–p95 之内；② 齿音比的 p95 ≤ 她本人的 p95。
    base 本来就在范围外时，要求 C 不比 base 离得更远（不然原声的问题会让美化白白减到 0）。量不出来的那一项不判。"""
    out: Dict[str, Any] = {"hf_ok": True, "sib_ok": True}
    if hf_c is not None and (her_p05 is not None or her_p95 is not None):
        if hf_base is not None and not _within(hf_base, her_p05, her_p95):
            out["hf_ok"] = _outside_by(hf_c, her_p05, her_p95) <= _outside_by(hf_base, her_p05, her_p95) + 1e-9
        else:
            out["hf_ok"] = _within(hf_c, her_p05, her_p95)
    if sib_c_p95 is not None and her_sib_p95 is not None:
        cap = max(float(her_sib_p95), float(sib_base_p95)) if sib_base_p95 is not None else float(her_sib_p95)
        out["sib_ok"] = float(sib_c_p95) <= cap + 1e-9
    out["passed"] = bool(out["hf_ok"] and out["sib_ok"])
    return out


def gate_noise(floor_base: Optional[float], floor_c: Optional[float], tp_c: Optional[float],
               floor_db: float = -60.0, floor_eq: Optional[float] = None) -> Dict[str, Any]:
    """G8：说话部分底噪 ≤ −60 dBFS（base 本来就高于 −60 的句子：不高于 base）；真峰值 ≤ −1 dBTP。
    floor_eq：base 只过静态 EQ（同样调回说话响度）以后的底噪。base 本来就有一点底噪时，静态 EQ 抬高高频也会把底噪
    抬高零点几 dB（设计写「不高于 base」，按字面这样的句子永远过不了，整篇白白减到 0）；静态 EQ 是全篇一样的音色，
    由 G1、G6 把关，「空气感」也只在高于底噪时才开。这一关真正要抓的是压缩以后调回响度把底噪抬高（设计：
    不过就关掉这一句的压缩），所以上限取 −60、base、只过静态 EQ 三个里最高的。"""
    floor_ok = True
    if floor_c is not None:
        caps = [floor_db] + [float(v) for v in (floor_base, floor_eq) if v is not None]
        floor_ok = float(floor_c) <= max(caps) + 1e-6
    peak_ok = tp_c is None or not np.isfinite(tp_c) or float(tp_c) <= PEAK_CEILING_DB + 1e-6
    return {"passed": bool(floor_ok and peak_ok), "floor_ok": bool(floor_ok), "peak_ok": bool(peak_ok)}


def next_strength(s: float) -> float:
    """整篇减半：1 → 0.5 → 0.25 → 0。"""
    for a, b in zip(STRENGTHS, STRENGTHS[1:] + (0.0,)):
        if s >= a - 1e-9:
            return b
    return 0.0


def strength_word(s: float) -> str:
    return STRENGTH_WORDS.get(float(s), f"{s:g}")


# ============================================================================ 她本人的统计量和一次性校准（第 4.1 节）
def her_stats(clips: Sequence[Tuple[np.ndarray, int, str]], f0_fn: Optional[Callable] = None) -> Dict[str, Any]:
    """她本人真实录音的统计量（clips：(波形, 采样率, 来自哪个文件)）：
    LTAS（C_her 的锚点）、σ_她（按文件分组的 LTAS 距离的标准差）、齿音比 p95（去齿音阈值、G7）、
    (8–16k)/(1–4k) 的 p05/p95（G7）、说话帧电平相对整句的 p90（压缩阈值）、空气感那一段比底噪高多少、F0 p05（高通）。"""
    if not clips:
        return {"n_clips": 0}
    sr = int(max(c[1] for c in clips))
    wavs = []
    groups: Dict[str, List[np.ndarray]] = {}
    sib_all: List[np.ndarray] = []
    rel_all: List[np.ndarray] = []
    hf: List[float] = []
    f0s: List[np.ndarray] = []
    for wav, wsr, src in clips:
        w = np.asarray(wav, dtype=np.float32)
        if wsr != sr:
            from voicetwin.utils.audio import resample

            w = resample(w, wsr, sr)
        wavs.append(w)
        groups.setdefault(str(src or ""), []).append(w)
        ratio, mask = sibilance_db(w, sr)
        if ratio.size and np.any(mask):
            sib_all.append(ratio[mask])
        lvl = speech_level_db(w, sr)
        if lvl is not None:
            tr = _rms_db_track(w, sr, COMP_WIN_MS)
            m = _speech_mask(tr)
            if np.any(m):
                rel_all.append(tr[m] - lvl)
        sp = speech_spectrum([w], sr)
        v = hf_ratio_db(sp)
        if v is not None and sp.frames * (sp.nfft // 2) / sr >= SV_MIN_SECONDS:
            hf.append(v)
        if f0_fn is not None:
            try:
                f = np.asarray(f0_fn(w, sr), dtype=np.float64)
                if f.size:
                    f0s.append(f)
            except Exception:  # noqa: BLE001 - 量不出 F0 就用最保守的高通
                pass
    spec = speech_spectrum(wavs, sr, keep_noise=True)
    ltas = band_db(spec)
    g_ltas = [ltas_db(ws, sr) for ws in groups.values()]
    dists = [d for d in (ltas_distance(g, ltas) for g in g_ltas) if d is not None]
    sigma = float(np.std(dists, ddof=1)) if len(dists) >= 3 else None
    sib = np.concatenate(sib_all) if sib_all else np.zeros(0)
    rel = np.concatenate(rel_all) if rel_all else np.zeros(0)
    f0 = np.concatenate(f0s) if f0s else np.zeros(0)
    f0_p05 = float(np.percentile(f0, 5)) if f0.size >= 10 else None
    snr = air_snr_db(spec)
    return {"n_clips": len(clips), "sr": sr, "n_groups": len(groups),
            "ltas_db": [None if not np.isfinite(v) else round(float(v), 3) for v in ltas],
            "sigma_her": _r(sigma, 4), "sigma_groups": len(dists),
            "sib_p95_db": _r(np.percentile(sib, 95)) if sib.size else None,
            "comp_rel_p90_db": _r(np.percentile(rel, 90)) if rel.size else None,
            "hf_p05_db": _r(np.percentile(hf, 5)) if len(hf) >= 3 else None,
            "hf_p95_db": _r(np.percentile(hf, 95)) if len(hf) >= 3 else None,
            "air_snr_db": _r(snr), "air_ok": bool(snr is not None and snr >= AIR_SNR_DB),
            "f0_p05_hz": _r(f0_p05, 1),
            "hpf_hz": round(float(np.clip(0.5 * f0_p05, *HPF_RANGE)), 1) if f0_p05 else HPF_RANGE[0]}


def level_back(out: np.ndarray, ref: np.ndarray, sr: int) -> Tuple[np.ndarray, float]:
    """S5（校准用）：调回 ref 的说话响度（speech_level_db），真峰值超过 −1 dBTP 再单独调低。只乘系数。"""
    la, lb = speech_level_db(ref, sr), speech_level_db(out, sr)
    g = (la - lb) if la is not None and lb is not None else 0.0
    tp = true_peak_db(out)
    if math.isfinite(tp) and tp + g > PEAK_CEILING_DB:
        g = PEAK_CEILING_DB - tp
    return (np.asarray(out, dtype=np.float32) * np.float32(10.0 ** (g / 20.0))), g


def null_version(wav: np.ndarray) -> np.ndarray:
    """空白对照：只改音量（和 research/sv_benchmark「音量放到最大」同法：放到真峰值 −1 dBTP）。"""
    tp = true_peak_db(wav)
    if not math.isfinite(tp):
        return np.asarray(wav, dtype=np.float32)
    return np.asarray(wav, dtype=np.float32) * np.float32(10.0 ** ((PEAK_CEILING_DB - tp) / 20.0))


def calibrate(items: Sequence[Tuple[np.ndarray, int]], her: Dict[str, Any],
              pct_fn: Callable[[np.ndarray, int], Tuple[Optional[float], Optional[float]]],
              progress: Optional[Callable[[float, str], None]] = None) -> Dict[str, Any]:
    """一次性校准 s*（第 4.1 节）。items：她没参加训练的真实录音；pct_fn(wav, sr) → (像你本人 %（不封顶）, 人声秒数)。
    1. 空白对照 D_null：只改音量再打分，配对差 → μ_null、σ_null；
    2. s = 1、0.5、0.25 各跑一遍处理链（C_her = 0：她自己就是锚点），配对差 Δ_s；
    3. mean(Δ_s) ≥ μ_null − 2·σ_null/√n 的最大 s 就是 s*；都不过 s* = 0。"""
    def p(frac: float, msg: str) -> None:
        if progress is not None:
            progress(max(0.0, min(1.0, frac)), msg)

    base: List[Tuple[np.ndarray, int, float]] = []
    for k, (wav, sr) in enumerate(items):
        p(0.25 * k / max(len(items), 1), "给你没参加训练的真实录音打分")
        pct, sec = pct_fn(wav, sr)
        if pct is not None and sec is not None and sec >= SV_MIN_SECONDS:
            base.append((np.asarray(wav, dtype=np.float32), int(sr), float(pct)))
    out: Dict[str, Any] = {"version": CHAIN_VERSION, "n": len(base), "her": her,
                           "evaluated_at": time.strftime("%Y-%m-%d %H:%M"), "strengths": {}}
    if len(base) < MIN_CALIB_ITEMS:
        out.update(s_star=0.0, mu_null=None, sigma_null=None,
                   why=f"没参加训练、人声有 2 秒以上的真实录音只有 {len(base)} 段（至少要 {MIN_CALIB_ITEMS} 段），"
                       "没法量美化会不会让声音不像你")
        return out
    d_null = []
    for k, (wav, sr, p0) in enumerate(base):
        p(0.25 + 0.15 * k / len(base), "量空白对照（只改音量）")
        v, _ = pct_fn(null_version(wav), sr)
        if v is not None:
            d_null.append(float(v) - p0)
    if len(d_null) < MIN_CALIB_ITEMS:
        out.update(s_star=0.0, mu_null=None, sigma_null=None, why="空白对照打不出分")
        return out
    mu = float(np.mean(d_null))
    sigma = float(np.std(d_null, ddof=1))
    n = len(d_null)
    floor = null_mean_floor(mu, sigma, n)
    out.update(mu_null=round(mu, 3), sigma_null=round(sigma, 3), n_null=n, floor=round(floor, 3))
    s_star = 0.0
    plans: Dict[Tuple[float, int], Plan] = {}
    for j, s in enumerate(STRENGTHS):
        deltas = []
        for k, (wav, sr, p0) in enumerate(base):
            p(0.40 + 0.6 * (j + k / len(base)) / len(STRENGTHS), f"试轻度美化（强度 {strength_word(s)}）")
            plan = plans.get((s, sr)) or plans.setdefault((s, sr), make_plan(sr, s, her))
            y, _ = beautify_sentence(wav, sr, plan)
            y, _ = level_back(y, wav, sr)
            v, _ = pct_fn(y, sr)
            if v is not None:
                deltas.append(float(v) - p0)
        mean_d = float(np.mean(deltas)) if deltas else None
        ok = mean_d is not None and len(deltas) >= MIN_CALIB_ITEMS and mean_d >= floor
        out["strengths"][str(s)] = {"mean_delta": _r(mean_d, 3), "sd": _r(np.std(deltas, ddof=1), 3)
                                    if len(deltas) >= 2 else None, "n": len(deltas), "passed": bool(ok)}
        if ok and s > s_star:
            s_star = s
    out["s_star"] = s_star
    if s_star <= 0:
        out["why"] = "最轻（四分之一）的美化也比只改音量让「像你本人」降得多（超过测量误差）"
    return out


def load_calibration(root: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads((Path(root) / CALIB_FILE).read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except Exception:  # noqa: BLE001
        return None


def calib_signature(judge_sig: str, item_ids: Sequence[str], her_ids: Sequence[str]) -> str:
    key = json.dumps([CHAIN_VERSION, judge_sig, sorted(map(str, item_ids)), sorted(map(str, her_ids))])
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]
