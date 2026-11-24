"""老师自己的录音（research/训练实测/录音.txt，一共约 2.5 小时）在 GitHub 的处理器上真的训练 V5，
再用声音分身自己的打分实测比较不同的设置（音色 LoRA 练几轮、语气 GPT 练几轮、cfg_rate、没训练的 V5 底模），
实测最好的设置以后当程序的默认设置。GitHub 上一台机器最多跑 6 小时，所以分成一步一步的，每一步都能在新的机器上
接着上一步做（状态用 pack / unpack 搬，见 .github/workflows/train-real.yml）。

子命令：
  prep    下载录音 → 声音分身「准备素材」（不识别文字：切片和识别无关，id 和 逐句文字.json 一样，这里核对）→
          按 id 填上核对好的文字（对齐质量是「好」开头的；其他的不用来训练，原因记下来）→「✅ 确认训练素材」→ 数一数
  train   接着上次的进度训练（和网页「② 训练模型」一样：wf.run_train，「一模一样」，这里不挑选）；
          时间快到了：刚存完一轮、下一轮练不完时就停（停止按钮同一个办法：request_cancel → TaskCancelled），
          GPT-SoVITS 下次从最后存的那一轮接着练；全部练完写 DONE.json
  eval    一组候选（音色轮数 × 语气轮数 × cfg_rate，还有没训练的 V5 底模 + 练好的语气）用没参加训练的录音
          （考试题）生成、用声音分身自己的打分量（像你本人 %、和真实录音的声纹相似度、错字率、节奏）；
          --warm 只做准备（声纹模型、参考录音库、按句子种类的校准）并试生成一句，打包给各个分片用
  plan    第二轮要试的候选（按第一轮的实测排名）
  report  合并打分 → 结果.md / 结果.json（排名用程序自己的规则 rank_results，只写实测的数字）+ 每一步用时
  progress  训练中途把进度和日志写进结果文件夹（GitHub 上每个任务结束时提交）
  pack / unpack  打包 / 还原状态：tar 按原来的绝对路径；修改时间按纳秒还原（训练素材的指纹里有录音文件的修改时间，
                 差一点就会被当成「素材改过」从头练）

本机检查（不用真的 GPT-SoVITS）见 tests/test_train_exp.py：用 tests/fake_gptsovits.py 的替身把 prep → train（中途停、
接着练）→ eval → plan → report 整个走一遍。
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import re
import shlex
import shutil
import statistics
import subprocess
import sys
import threading
import time
import traceback
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT))

TEXTS = HERE / "逐句文字.json"
URLS = HERE / "录音.txt"
#: 以前转录时每个录音的字节数和 sha256（切片 id 里有录音的「绝对路径 + 字节数」，文件不对 id 就对不上）
RUN_INFOS = [ROOT / "research/新录音转录/结果/运行信息.json"] + sorted(
    (ROOT / "research/新录音转录/以前的课/结果").glob("*/运行信息.json"))
#: 以前转录时录音放在这里（transcribe.py：<work>/录音/<文件名>，work = /mnt/w/tr）；切片 id 和这个路径有关
SRC_DIR = "/mnt/w/tr/录音"
VOICE = "老师"
OUT_ROOT = HERE / "结果"

#: 要比较的训练方式。default 和老师的笔记本（RTX 5070 Laptop，8 GB 显卡）上程序「一模一样」训练 V5 的默认设置一样
#: （plan_lora / plan_training 按 8 GB 显存算：音色 LoRA 每批 1 条、8 轮、每轮存一个；语气 GPT 每批 4 条、20 轮、每轮存一个；
#: 老师电脑上实测出来的每批条数不一样时，程序会跟着改）。这台机器没有 N 卡，程序自己算会是语气每批 1 条（app_cpu），
#: 每轮的更新次数是老师电脑上的 4 倍，「练几轮最好」就没法直接搬到老师电脑上，所以默认按老师电脑的每批条数练。
VARIANTS: Dict[str, Dict[str, Any]] = {
    "default": {"train": {"batch_size": 8, "lora_batch_size": 1},
                "说明": "和老师的 8 GB 显卡上程序的默认练法一样：音色 V5 LoRA 每批 1 条、8 轮；语气 GPT 每批 4 条、20 轮"
                        "（处理器上是全精度，程序会把填的每批数量减半，所以填 8；只练 V5 时「每批数量」也会用在音色 LoRA 上，"
                        "所以 LoRA 明确写 1 条）"},
    "app_cpu": {"train": {}, "说明": "什么都不改：程序在没有 N 卡的电脑上自己算的练法（语气 GPT 每批 1 条）"},
    "lora16": {"train": {"batch_size": 8, "lora_batch_size": 1, "lora_epochs": 16},
               "说明": "音色 LoRA 练 16 轮（程序允许的最多轮数），其他和 default 一样"},
    "rank64": {"train": {"batch_size": 8, "lora_batch_size": 1, "lora_rank": 64},
               "说明": "音色 LoRA 的秩 64（默认 32），其他和 default 一样"},
}

#: 第一轮的候选：音色 LoRA 轮数 × 语气 GPT 轮数（cfg_rate 用程序默认 auto = 官方的 1.30），再加 V5 底模 × 两个语气轮数。
#: 为什么是这些（不是全部 5 × 4 + 4 = 24 个）：LoRA 按 1、2、4、8 翻倍取（1 = 程序「一模一样」从第 1 轮起试，
#: 2 = 官方默认，8 = 程序默认练到的轮数），6 在 4 和 8 中间，第一轮最好的是 4 或 8 时第二轮再补；GPT 5、10、15、20
#: 都试（15 = 官方默认，20 = 程序默认）；底模只配 10、20 两个语气轮数（GPT 的影响已经在 LoRA 那几行量过了，
#: 这里只看「音色不训练」行不行）。一共 18 个，每个候选单独一台机器（同时做；生成一句要几分钟，一台做一个最稳）。
R1_LORA = (1, 2, 4, 8)
R1_GPT = (5, 10, 15, 20)
R1_BASE_GPT = (10, 20)
#: 第二轮：第一轮排名前 2 的组合各试这几个 cfg_rate（auto = 1.30 第一轮已经量过；第 1 名再明确发一次 1.3，核对 auto 就是 1.30）
CFG_SWEEP = (0.0, 1.0, 1.6, 2.0)
SPLIT_BYTES = "1900m"


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S"), msg, flush=True)


def gh_output(key: str, value: str) -> None:
    """GitHub Actions 的步骤输出（本机运行时只打印）。"""
    path = os.environ.get("GITHUB_OUTPUT")
    if path:
        with open(path, "a", encoding="utf-8") as f:
            f.write(f"{key}={value}\n")
    log(f"输出 {key}={value}")


def read_json(path: Path, default: Any = None) -> Any:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    os.replace(tmp, path)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ============================================================================ 路径、设置
class Ctx:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.work = Path(a.work).resolve()
        self.exp = self.work / "exp"
        self.ws = self.work / "workspace"
        self.variant = a.variant
        if self.variant not in VARIANTS:
            raise SystemExit(f"不认识的训练方式 {self.variant}，可以用：{'、'.join(VARIANTS)}")
        gsv = a.gsv or os.environ.get("GSV") or next(iter(sorted(glob.glob("/mnt/w/g/GPT-SoVITS-*"))), "")
        self.gsv = Path(gsv).resolve() if gsv else None
        self.sv_dir = Path(a.sv_dir).resolve()
        self.extra = json.loads(a.extra_config) if a.extra_config else {}

    def cfg(self) -> Any:
        from voicetwin.config import deep_merge, load_config

        gsv = {"root": str(self.gsv or "/mnt/w/g/GPT-SoVITS-v5"), "python": sys.executable, "port": int(self.a.port),
               "is_half": False, "startup_timeout": 900, "version": "v5dev",
               "train": dict(VARIANTS[self.variant]["train"])}
        over = {"workspace": str(self.ws), "backend": "gptsovits",
                # 不识别文字（文字是核对好的，按切片 id 填进去）；不自动查错字（同一个原因）
                "prepare": {"asr": {"engine": "none"}, "proofcheck": "off"},
                "similarity": {"model_dir": str(self.sv_dir)},
                "backends": {"gptsovits": gsv}}
        over = deep_merge(over, self.extra) if self.extra else over
        return load_config(overrides=over, user_config=False)

    def backend(self, cfg: Any = None) -> Tuple[Any, Any, Any]:
        from voicetwin import workflows as wf
        from voicetwin.backends.base import get_backend

        cfg = cfg if cfg is not None else self.cfg()
        project = wf.open_project(cfg, VOICE, must_exist=True)
        return cfg, project, get_backend("gptsovits", cfg, project)

    def out_dir(self) -> Path:
        if self.a.out:
            return Path(self.a.out)
        return OUT_ROOT if self.variant == "default" else OUT_ROOT / self.variant


# ============================================================================ 打包 / 还原
def _files_under(paths: List[Path]) -> List[Path]:
    out: List[Path] = []
    for p in paths:
        if p.is_file():
            out.append(p)
        elif p.is_dir():
            for dp, _dn, fn in os.walk(p):
                out += [Path(dp) / f for f in fn]
    return out


def pack_paths(ctx: Ctx, kind: str) -> Tuple[List[Path], List[str]]:
    """要打包的东西（绝对路径）和不要的（tar 的 --exclude，相对于 /）。
    prep：声音分身的工作文件夹；state：再加 GPT-SoVITS 的训练进度（logs/<实验名>：特征、断点）和存下的模型；
    models：工作文件夹 + 存下的模型（打分用，不要特征和断点）。"""
    paths = [ctx.work]
    excl = [str(ctx.exp / "evalcache").lstrip("/")]
    if kind in ("state", "models") and ctx.gsv is not None:
        from voicetwin.backends.gptsovits import _exp_name

        exp = _exp_name(VOICE)
        if kind == "state":
            paths.append(ctx.gsv / "logs" / exp)
        elif ctx.sv_dir.is_dir():   # 准备打分时下载好的声纹模型：各个分片不用再下载
            paths.append(ctx.sv_dir)
        for d in sorted(ctx.gsv.glob("SoVITS_weights_*")) + sorted(ctx.gsv.glob("GPT_weights_*")):
            paths += sorted(d.glob(f"{exp}_e*.pth")) + sorted(d.glob(f"{exp}-e*.ckpt"))
    return [p for p in paths if p.exists()], excl


def pack(ctx: Ctx, kind: str, out: Path) -> Dict[str, Any]:
    """tar（POSIX 格式）按 / 下面的相对路径打包，切成不超过 1.9 GB 的几份；另外记下每个文件的修改时间（纳秒）。"""
    t0 = time.time()
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    paths, excl = pack_paths(ctx, kind)
    files = _files_under(paths)
    manifest = {"kind": kind, "made": time.strftime("%Y-%m-%d %H:%M:%S"),
                "files": {str(f): [f.stat().st_mtime_ns, f.stat().st_size] for f in files
                          if not any(str(f).lstrip("/").startswith(e) for e in excl)}}
    write_json(out / "mtimes.json", manifest)
    rel = " ".join(shlex.quote(str(p).lstrip("/")) for p in paths)
    ex = " ".join(f"--exclude={shlex.quote(e)}" for e in excl)
    cmd = f"set -o pipefail; tar --format=posix {ex} -C / -cf - {rel} | split -b {SPLIT_BYTES} -d -a 3 - {shlex.quote(str(out / 'state.tar.'))}"
    subprocess.run(["bash", "-c", cmd], check=True)
    size = sum(p.stat().st_size for p in out.glob("state.tar.*"))
    info = {"kind": kind, "files": len(manifest["files"]), "bytes": size, "seconds": round(time.time() - t0, 1),
            "parts": len(list(out.glob("state.tar.*")))}
    log(f"打包 {kind}：{info['files']} 个文件，{size / 1e9:.2f} GB，{info['parts']} 份，{info['seconds']} 秒")
    return info


def unpack(src: Path, drop: bool = False) -> Dict[str, Any]:
    """还原：解开 tar，再按记下的纳秒还原每个文件的修改时间，核对大小。"""
    t0 = time.time()
    src = Path(src)
    parts = sorted(src.rglob("state.tar.*"))
    man_path = next(iter(sorted(src.rglob("mtimes.json"))), None)
    if not parts or man_path is None:
        raise SystemExit(f"{src} 里没有打包好的状态（state.tar.* 和 mtimes.json）")
    cmd = "set -o pipefail; cat " + " ".join(shlex.quote(str(p)) for p in parts) + " | tar -xf - -C /"
    subprocess.run(["bash", "-c", cmd], check=True)
    man = read_json(man_path, {})
    bad = []
    for f, (ns, size) in (man.get("files") or {}).items():
        p = Path(f)
        try:
            if p.stat().st_size != size:
                bad.append(f"{f}：大小 {p.stat().st_size} ≠ {size}")
            os.utime(p, ns=(ns, ns))
        except OSError as exc:
            bad.append(f"{f}：{exc}")
    if bad:
        raise SystemExit("还原出来的文件不对：\n" + "\n".join(bad[:20]))
    info = {"kind": man.get("kind"), "files": len(man.get("files") or {}), "seconds": round(time.time() - t0, 1),
            "made": man.get("made")}
    log(f"还原 {info['kind']}：{info['files']} 个文件（{man.get('made')} 打包），{info['seconds']} 秒")
    if drop:   # 解开以后不再要，省硬盘（GitHub 上下载下来的那份）
        for p in parts:
            p.unlink(missing_ok=True)
    return info


# ============================================================================ prep
def expected_files() -> Dict[str, Dict[str, Any]]:
    out: Dict[str, Dict[str, Any]] = {}
    for p in RUN_INFOS:
        for f in (read_json(p, {}) or {}).get("files") or []:
            out[f["name"]] = {"bytes": int(f["bytes"]), "sha256": f["sha256"]}
    return out


def _fetch(url: str, dst: Path) -> None:
    for i in range(5):
        try:
            tmp = dst.with_name(dst.name + ".part")
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst)
            return
        except Exception as exc:  # noqa: BLE001
            log(f"下载 {url} 第 {i + 1} 次没成功：{exc}")
            time.sleep(10 * (i + 1))
    raise SystemExit(f"下载不下来：{url}")


def download_recordings(src: Path, want: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    """录音.txt 里的地址：zip 只解出 wav（不要文件夹），其他的直接下载；每个都核对字节数和 sha256。"""
    src.mkdir(parents=True, exist_ok=True)
    tmp = src.parent / "_下载"
    tmp.mkdir(parents=True, exist_ok=True)
    for line in URLS.read_text(encoding="utf-8").splitlines():
        u = line.strip()
        if not u or u.startswith("#"):
            continue
        name = urllib.parse.unquote(u.rsplit("/", 1)[-1])
        if name.lower().endswith(".zip"):
            zp = tmp / name
            log(f"下载 {name}")
            _fetch(u, zp)
            with zipfile.ZipFile(zp) as z:
                for info in z.infolist():
                    base = Path(info.filename).name
                    if base.lower().endswith(".wav") and base in want and not (src / base).exists():
                        with z.open(info) as fi, open(src / base, "wb") as fo:
                            shutil.copyfileobj(fi, fo, 1 << 20)
            zp.unlink()
        elif not (src / name).exists():
            log(f"下载 {name}")
            _fetch(u, src / name)
    shutil.rmtree(tmp, ignore_errors=True)
    return check_recordings(src, want)


def check_recordings(src: Path, want: Dict[str, Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows, bad = [], []
    for name, w in sorted(want.items()):
        p = src / name
        row = {"name": name, "bytes": p.stat().st_size if p.exists() else None}
        row["ok"] = bool(p.exists() and row["bytes"] == w["bytes"] and sha256(p) == w["sha256"])
        if not row["ok"]:
            bad.append(name)
        rows.append(row)
    if bad:
        raise SystemExit(f"这几个录音和以前转录时的不一样（字节数或 sha256 不对），切片 id 会对不上：{'、'.join(bad)}")
    return rows


def match_texts(records: List[Dict[str, Any]], texts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """逐句文字（id、录音、开始秒、秒）→ 这次切出来的片段。先按 id（还要核对开始的时间和长短都一样）；
    id 对不上的按「同一个录音、开始和长短都差不到 0.05 秒」对；都对不上的不用。"""
    by_id = {r["id"]: r for r in records}
    by_file: Dict[str, List[Dict[str, Any]]] = {}
    for r in records:
        by_file.setdefault(Path(str(r.get("source_file") or "")).name, []).append(r)
    pairs: List[Tuple[Dict[str, Any], Dict[str, Any], str]] = []
    unmatched, span_diff = [], []
    used = set()
    for t in texts:
        r = by_id.get(t["id"])
        if r is not None and r["id"] not in used and abs(float(r["start"]) - float(t["开始秒"])) <= 0.011 \
                and abs(float(r["duration"]) - float(t["秒"])) <= 0.011:
            pairs.append((t, r, "id"))
            used.add(r["id"])
            continue
        if r is not None:
            span_diff.append(t["id"])
        cands = [x for x in by_file.get(t["录音"], []) if x["id"] not in used
                 and abs(float(x["start"]) - float(t["开始秒"])) <= 0.05 and abs(float(x["duration"]) - float(t["秒"])) <= 0.05]
        if cands:
            x = min(cands, key=lambda x: abs(float(x["start"]) - float(t["开始秒"])))
            pairs.append((t, x, "time"))
            used.add(x["id"])
        else:
            unmatched.append(t["id"])
    per_file: Dict[str, Counter] = {}
    for t, _r, how in pairs:
        per_file.setdefault(t["录音"], Counter())[how] += 1
    for tid in unmatched:
        f = next((t["录音"] for t in texts if t["id"] == tid), "?")
        per_file.setdefault(f, Counter())["unmatched"] += 1
    for f, rs in by_file.items():
        per_file.setdefault(f, Counter())["clips"] += len(rs)
    for t in texts:
        per_file.setdefault(t["录音"], Counter())["texts"] += 1
    return {"pairs": pairs, "unmatched": unmatched, "span_diff": span_diff,
            "clips_without_text": sorted(set(by_id) - used),
            "per_file": {f: dict(c) for f, c in sorted(per_file.items())}}


def cmd_prep(ctx: Ctx) -> int:
    from voicetwin import workflows as wf
    from voicetwin.data import review
    from voicetwin.utils.textutil import clean_transcript

    a = ctx.a
    t0 = time.time()
    res: Dict[str, Any] = {"kind": "prep", "variant_independent": True, "started": time.strftime("%Y-%m-%d %H:%M:%S"),
                           "timing_s": {}}
    src = Path(a.src or SRC_DIR)
    want = expected_files()
    t = time.time()
    if a.texts:   # 本机检查：自己给的录音和文字（不核对 sha256）
        res["recordings"] = [{"name": p.name, "bytes": p.stat().st_size} for p in sorted(src.glob("*.wav"))]
        texts = read_json(Path(a.texts))
    else:
        res["recordings"] = download_recordings(src, want)
        texts = read_json(TEXTS)
    res["timing_s"]["下载录音"] = round(time.time() - t, 1)
    cfg = ctx.cfg()
    t = time.time()
    summary = wf.run_prepare(cfg, VOICE, [str(src)])
    res["timing_s"]["准备素材（不识别文字）"] = round(time.time() - t, 1)
    res["prepare_summary"] = {k: v for k, v in (summary or {}).items() if isinstance(v, (int, float, str))}
    project = wf.open_project(cfg, VOICE, must_exist=True)
    records = project.load_manifest()
    m = match_texts(records, texts)
    good = [(t_, r, how) for t_, r, how in m["pairs"] if str(t_.get("对齐质量") or "").startswith("好")
            and str(t_.get("文字") or "").strip()]
    good_ids = {r["id"] for _t, r, _h in good}
    not_good = [{"id": r["id"], "文字id": t_["id"], "why": f"对齐质量：{t_.get('对齐质量') or '（没有文字）'}"}
                for t_, r, _ in m["pairs"] if r["id"] not in good_ids]
    res["match"] = {"texts": len(texts), "clips": len(records),
                    "by_id": sum(1 for _, _, h in m["pairs"] if h == "id"),
                    "by_time": sum(1 for _, _, h in m["pairs"] if h == "time"),
                    "id_same_but_span_differs": len(m["span_diff"]), "unmatched_texts": len(m["unmatched"]),
                    "unmatched_text_ids": m["unmatched"][:50], "clips_without_text": len(m["clips_without_text"]),
                    "clips_without_text_ids": m["clips_without_text"][:50], "per_file": m["per_file"],
                    "same_ids_as_transcription": not m["unmatched"] and not m["span_diff"]
                    and len(m["clips_without_text"]) == 0}
    log("切片和逐句文字对照：" + json.dumps({k: v for k, v in res["match"].items() if not isinstance(v, (list, dict))},
                                     ensure_ascii=False))
    # 和网页上一样：改文字（草稿）→「✅ 确认训练素材」（先保存、重新统计、留出考试题、挑参考录音，再记下用哪些句子）
    with review._LOCK:
        recs = {r["id"]: r for r in project.load_manifest()}
        draft = review.load_draft(project)
        for t_, r, _how in good:
            review.put_text(draft, recs[r["id"]], clean_transcript(str(t_["文字"])))
        review.save_draft(project, draft)
    res["applied_texts"] = len(good)
    res["not_used"] = not_good + [{"id": i, "why": "逐句文字里没有这一段"} for i in m["clips_without_text"]]
    t = time.time()
    conf = wf.review_confirm(cfg, VOICE)
    res["timing_s"]["确认训练素材（重新统计、声纹检查、留考试题）"] = round(time.time() - t, 1)
    records = project.load_manifest()
    res["confirmed"] = bool(conf.get("confirmed"))
    res["counts"] = review.material_counts(records)
    res["drop_reasons_with_text"] = dict(Counter(str(r.get("drop_reason") or "?") for r in records
                                                 if str(r.get("text") or "").strip() and not r.get("keep", True)))
    val = [r for r in records if r.get("keep", True) and r.get("split") == "val" and r.get("text")]
    res["val"] = [{"id": r["id"], "秒": r.get("duration"), "lang": r.get("lang"), "文字": r.get("text")} for r in val]
    res["train_minutes"] = round(sum(float(r.get("duration") or 0) for r in records
                                     if review.is_material(r) and r.get("split") != "val") / 60.0, 1)
    res["seconds"] = round(time.time() - t0, 1)
    res["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    write_json(ctx.exp / "prep.json", res)
    log("素材：" + json.dumps(res["counts"], ensure_ascii=False) + f"；训练 {res['train_minutes']} 分钟")
    usable = res["match"]["by_id"] + res["match"]["by_time"]
    if not res["confirmed"] or usable < 0.9 * len(texts):
        raise SystemExit(f"只对上了 {usable}/{len(texts)} 句文字（或者没能确认训练素材），先看 prep.json 再训练")
    if a.pack:
        res["pack"] = pack(ctx, "prep", Path(a.pack))
        write_json(ctx.exp / "prep.json", res)
    return 0


# ============================================================================ train
def _weight_epochs(ctx: Ctx, backend: Any) -> Dict[str, Dict[int, float]]:
    """现在存下的模型：{"lora": {轮: 修改时间}, "gpt": {...}}（这个声音的；按文件名里的轮数）。"""
    from voicetwin.backends.gptsovits import _epoch

    out: Dict[str, Dict[int, float]] = {"lora": {}, "gpt": {}}
    exp = backend.exp_name
    for kind, pat in (("lora", f"SoVITS_weights_{backend.main_version}/{exp}_e*_s*.pth"),
                      ("gpt", f"GPT_weights_{backend.main_version}/{exp}-e*.ckpt")):
        for p in Path(backend.root).glob(pat):
            try:
                m = p.stat().st_mtime
            except OSError:
                continue
            e = _epoch(p)
            out[kind][e] = max(m, out[kind].get(e, 0.0))
    return out


class StopGuard(threading.Thread):
    """训练时盯着存下来的模型：时间到了就停（停止按钮同一个办法）；刚存完一轮、按实测的每轮用时下一轮在时间到以前
    练不完，就马上停（不白练半轮）。全部轮数都存下来以后不再停（后面只剩整理结果，几秒钟）。"""

    def __init__(self, ctx: Ctx, backend: Any, deadline: float, total: Dict[str, Optional[int]],
                 hist: Dict[str, List[float]], job_start: float, poll: float):
        super().__init__(name="train-exp-guard", daemon=True)
        self.ctx, self.backend, self.deadline, self.total, self.hist = ctx, backend, deadline, total, hist
        self.job_start, self.poll = job_start, poll
        self.halt = threading.Event()
        self.why = ""
        self.events: List[Dict[str, Any]] = []
        #: 这台机器上每一步（音色 / 语气）开始练的时间（程序的进度说明「训练音色……共 N 轮」「训练语气和节奏（GPT），共 N 轮」）：
        #: 一轮要练两三个小时时，这台机器上看不到连着存的两轮，就用「开始练 → 存下第一轮」估计每轮多久（多算了加载模型的
        #: 一两分钟，宁可早停）
        self.stage_started: Dict[str, float] = {}

    def note_progress(self, msg: str) -> None:
        if "共" in msg and "轮" in msg:
            if msg.startswith("训练音色"):
                self.stage_started.setdefault("lora", time.time())
            elif msg.startswith("训练语气和节奏"):
                self.stage_started.setdefault("gpt", time.time())

    def _next_stage(self, eps: Dict[str, Dict[int, float]]) -> Optional[str]:
        L, G = self.total.get("lora"), self.total.get("gpt")
        lmax, gmax = max(eps["lora"], default=0), max(eps["gpt"], default=0)
        if L and lmax < L:
            return "lora"
        if G and gmax < G:
            return "gpt"
        if not L or not G:
            return "gpt" if eps["gpt"] or (L and lmax >= L) else "lora"
        return None

    def _request_stop(self, why: str) -> None:
        from voicetwin.utils.progress import request_cancel

        self.why = why
        log(f"⏹ 停下训练：{why}")
        request_cancel()

    def run(self) -> None:
        seen = _weight_epochs(self.ctx, self.backend)
        while not self.halt.wait(self.poll):
            now = time.time()
            eps = _weight_epochs(self.ctx, self.backend)
            new = False
            for kind in ("lora", "gpt"):
                for e, m in sorted(eps[kind].items()):
                    if e in seen[kind] and seen[kind][e] == m:
                        continue
                    new = True
                    ev = {"stage": kind, "epoch": e, "saved_at": round(m, 1)}
                    prev = eps[kind].get(e - 1)
                    first = not any(x["stage"] == kind for x in self.events)
                    if prev is not None and prev >= self.job_start and m > prev:   # 这台机器上连着存的两轮：一轮的用时
                        ev["epoch_seconds"] = round(m - prev, 1)
                        self.hist.setdefault(kind, []).append(ev["epoch_seconds"])
                    elif first and kind in self.stage_started and m > self.stage_started[kind]:
                        ev["epoch_seconds"] = round(m - self.stage_started[kind], 1)   # 开始练 → 存下第一轮（含加载）
                        ev["from_stage_start"] = True
                        self.hist.setdefault(kind, []).append(ev["epoch_seconds"])
                    self.events.append(ev)
                    log(f"存下了 {'音色 LoRA' if kind == 'lora' else '语气 GPT'} 第 {e} 轮"
                        + (f"（这一轮 {ev['epoch_seconds'] / 60:.1f} 分钟）" if "epoch_seconds" in ev else ""))
            seen = eps
            nxt = self._next_stage(eps)
            if nxt is None:
                continue   # 都存下来了：不停
            if now >= self.deadline:
                self._request_stop("这台机器的时间到了（没练完的这一轮下次从最后存的那一轮接着练）")
                return
            if new:
                h = self.hist.get(nxt) or []
                est = statistics.median(h[-5:]) if h else None
                if est and now + 1.1 * est > self.deadline:
                    self._request_stop(f"刚存完一轮；{'音色' if nxt == 'lora' else '语气'}每轮实测大约 {est / 60:.0f} 分钟，"
                               f"这台机器剩下 {(self.deadline - now) / 60:.0f} 分钟，练不完下一轮")
                    return


def cmd_train(ctx: Ctx) -> int:
    from voicetwin import workflows as wf
    from voicetwin.utils.log import setup_logging
    from voicetwin.utils.progress import TaskCancelled, clear_cancel

    a = ctx.a
    started = time.time()
    since = float(a.since or started)
    deadline = since + float(a.budget_min) * 60.0
    restored = unpack(Path(a.restore), drop=a.drop_restored) if a.restore else None
    done_path = ctx.exp / "DONE.json"
    if done_path.exists():
        log("已经练完了（有 DONE.json），这一步什么都不做")
        gh_output("done", "true")
        return 0
    setup_logging()
    cfg, project, backend = ctx.backend()
    plan = backend.training_plan(mode="identical")
    total = {"lora": (plan.get("lora") or {}).get("epochs"), "gpt": plan.get("gpt_epochs")}
    runs = read_json(ctx.exp / "train_runs.json", []) or []
    hist = read_json(ctx.exp / "epoch_seconds.json", {}) or {}
    before = _weight_epochs(ctx, backend)
    rec: Dict[str, Any] = {"job": a.job or f"train-{len(runs) + 1}", "variant": ctx.variant,
                           "started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)),
                           "job_started": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(since)),
                           "budget_min": float(a.budget_min), "restore": restored, "plan": plan.get("summary"),
                           "plan_state": plan.get("state"), "plan_note": plan.get("state_note"),
                           "planned_epochs": total, "batch": {"lora": (plan.get("lora") or {}).get("batch_size"),
                                                              "gpt": plan.get("gpt_batch_size")},
                           "epochs_before": {k: max(v, default=0) for k, v in before.items()}}
    log(f"{rec['job']}：{plan.get('summary')}")
    log(f"  {plan.get('state_note') or ''} 现在存下的：音色第 {rec['epochs_before']['lora']} 轮、语气第 {rec['epochs_before']['gpt']} 轮；"
        f"最晚 {time.strftime('%H:%M', time.localtime(deadline))} 停")
    guard = StopGuard(ctx, backend, deadline, total, hist, since, float(a.poll))
    last = {"t": 0.0, "msg": ""}

    def progress(frac: float, msg: str = "") -> None:
        guard.note_progress(str(msg or ""))
        now = time.time()
        if msg and msg != last["msg"] and (now - last["t"] >= 120 or "完成" in msg or "共" in msg):
            last.update(t=now, msg=msg)
            log(f"  {frac:5.1%} {msg}")

    clear_cancel()
    guard.start()
    info: Dict[str, Any] = {}
    try:
        info = wf.run_train(cfg, VOICE, "gptsovits", progress=progress, select=False, mode="identical")
        rec["result"] = "done"
    except TaskCancelled:
        rec["result"] = "stopped"
    except BaseException as exc:  # noqa: B036 - 真的出错：记下来再报错（这一步算失败，GitHub 上可以「重新运行」接着练）
        rec["result"] = "error"
        rec["error"] = f"{type(exc).__name__}: {exc}"[:2000]
        rec["traceback"] = traceback.format_exc()[-4000:]
        raise
    finally:
        guard.halt.set()
        guard.join(timeout=30)
        clear_cancel()
        after = _weight_epochs(ctx, backend)
        rec.update(why_stopped=guard.why, saves=guard.events, finished=time.strftime("%Y-%m-%d %H:%M:%S"),
                   seconds=round(time.time() - started, 1),
                   epochs_after={k: max(v, default=0) for k, v in after.items()})
        params = (info or {}).get("params") or {}
        if params:
            rec["vt_timing"] = params.get("timing")
            rec["trained_counts"] = params.get("trained_counts")
            rec["report"] = params.get("report")
        runs.append(rec)
        write_json(ctx.exp / "train_runs.json", runs)
        write_json(ctx.exp / "epoch_seconds.json", hist)
    done = rec["result"] == "done"
    if done:
        write_json(done_path, {"kind": "done", "finished": rec["finished"], "epochs": rec["epochs_after"], "planned": total,
                               "plan": plan.get("summary"), "variant": ctx.variant})
    log(f"{rec['job']}：{'全部练完了' if done else '这次先练到这里'}（音色第 {rec['epochs_after']['lora']} 轮、"
        f"语气第 {rec['epochs_after']['gpt']} 轮），用了 {rec['seconds'] / 60:.0f} 分钟")
    if a.pack:
        rec["pack"] = pack(ctx, "models" if done else "state", Path(a.pack) / ("models" if done else "state"))
        write_json(ctx.exp / "train_runs.json", runs)
    gh_output("done", "true" if done else "false")
    return 0


# ============================================================================ 候选
SPEC = re.compile(r"^(?:s(\d+)|(base))-g(\d+)(?:@(auto|\d+(?:\.\d+)?))?$")


class Cand:
    def __init__(self, spec: str):
        m = SPEC.match(spec.strip())
        if not m:
            raise SystemExit(f"候选写法不对：{spec}（例：s8-g20、base-g10、s4-g15@1.6）")
        self.base = bool(m.group(2))
        self.lora = 0 if self.base else int(m.group(1))
        self.gpt = int(m.group(3))
        c = m.group(4)
        self.cfg: Optional[float] = None if c in (None, "auto") else float(c)
        self.pair = f"{'base' if self.base else f's{self.lora}'}-g{self.gpt}"
        self.spec = self.pair + ("" if self.cfg is None else f"@{self.cfg:g}")
        self.id = self.pair + ("" if self.cfg is None else f"-c{self.cfg:g}")


def round1_specs(lora_eps: List[int], gpt_eps: List[int]) -> List[str]:
    """第一轮的候选（按实际存下的轮数：练的轮数和默认的不一样时，最后一轮一定在内）。"""
    L, G = max(lora_eps, default=0), max(gpt_eps, default=0)
    lp = sorted({e for e in R1_LORA if e in lora_eps} | ({L} if L else set()))
    gp = sorted({e for e in R1_GPT if e in gpt_eps} | ({G} if G else set()))
    bp = sorted({e for e in R1_BASE_GPT if e in gpt_eps} | ({G} if G else set()))
    return [f"s{l}-g{g}" for l in lp for g in gp] + [f"base-g{g}" for g in bp]


def resolve(ctx: Ctx, backend: Any, c: Cand) -> Optional[Dict[str, Any]]:
    from voicetwin.backends.gptsovits import BASE_LABEL, PRETRAINED_SOVITS, _epoch

    exp, ver = backend.exp_name, backend.main_version
    root = Path(backend.root)

    def newest(pattern: str, epoch: int) -> Optional[Path]:
        files = [p for p in root.glob(pattern) if _epoch(p) == epoch]
        return max(files, key=lambda p: p.stat().st_mtime) if files else None

    gpt = newest(f"GPT_weights_{ver}/{exp}-e*.ckpt", c.gpt)
    sov = root / PRETRAINED_SOVITS[ver] if c.base else newest(f"SoVITS_weights_{ver}/{exp}_e*_s*.pth", c.lora)
    if gpt is None or sov is None or not Path(sov).exists():
        return None
    ck = {"id": c.id, "sovits": str(sov), "gpt": str(gpt), "sovits_epoch": c.lora, "gpt_epoch": c.gpt, "version": ver}
    if c.base:
        ck.update(base=True, label=BASE_LABEL)
    return ck


# ============================================================================ eval
def _setup_eval(ctx: Ctx) -> Tuple[Any, Any, Any, Dict[str, Any]]:
    from voicetwin.utils.log import setup_logging

    setup_logging()
    cfg = ctx.cfg()
    info: Dict[str, Any] = {}
    t = time.time()
    try:
        from voicetwin.eval import sv_models

        if ctx.a.no_sv_download:
            raise RuntimeError("这次不下载（--no-sv-download）")
        got = sv_models.download(cfg)
        info["sv_download"] = {"ok": True, "files": [str(x) for x in got]}
    except Exception as exc:  # noqa: BLE001 - 下载不了：「像你本人 %」量不出来，结果里如实写
        info["sv_download"] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"[:500]}
    info["sv_download"]["seconds"] = round(time.time() - t, 1)
    cfg, project, backend = ctx.backend(cfg)
    return cfg, project, backend, info


def _load_selector(cfg: Any, project: Any, backend: Any, items: int, info: Dict[str, Any], asr: bool = True) -> Any:
    from voicetwin.synth.select import _DeepSelect

    t = time.time()
    # use_asr=None：和程序挑模型时一样，能加载识别模型就查错字（中文 Paraformer、有英文的句子 Whisper），加载不了就不查
    sel = _DeepSelect(cfg, project, backend, max_items=items, use_asr=None if asr else False)
    sel.load(with_tests=False)
    info["load_seconds"] = round(time.time() - t, 1)
    info["judge"] = sel.sim.info()
    info["asr"] = None if sel.checker is None else {
        "zh": sel.checker.engine_name("zh", "中文句子"), "en": sel.checker.engine_name("en", "English sentence")}
    info["calibration_ok"] = bool((sel.calib or {}).get("ok"))
    info["items"] = [{"key": it.key, "group": it.group, "lang": it.lang, "text": it.text} for it in sel.items]
    return sel


def _measure(ctx: Ctx, sel: Any, backend: Any, c: Cand, ck: Dict[str, Any], deadline: float) -> Dict[str, Any]:
    """一个候选：每句考试题用第 1 条参考录音生成 1 次（和挑选第三步一样打分），句子之间按时间看要不要停。"""
    from voicetwin.synth.select import _num

    backend.bcfg.setdefault("infer", {})["cfg_rate"] = "auto" if c.cfg is None else c.cfg
    sel.cache_dir = ctx.exp / "evalcache" / c.id     # 缓存的键里没有 cfg_rate：每个候选单独一份
    per_item: Dict[str, Any] = {}
    secs: List[float] = []
    for it in sel.items:
        if time.time() >= deadline:
            break
        t = time.time()
        job = sel.job("C", ck, it, 0, 1)
        recs = sel.run_jobs([job], 0.0, 1.0, c.id)
        rows = sel.rows(recs.get(job.key))
        chosen = sel.pick(rows)
        secs.append(time.time() - t)
        per_item[it.key] = {"S": sel.item_S(it, chosen), "pct": sel.item_pct(it, chosen), "cer": sel.item_cer(chosen),
                            "group": it.group, "w": it.w, "kind": it.kind, "lang": it.lang, "failed": chosen is None,
                            "rows": rows, "ratio": _num((chosen or {}).get("dur_ratio")), "m": chosen or {},
                            "seconds": round(secs[-1], 1)}
    res = sel.summarize(ck, per_item)
    res["seconds_per_item"] = round(statistics.mean(secs), 1) if secs else None
    return res


def _extra_metrics(res: Dict[str, Any]) -> Dict[str, Any]:
    """表里另外几列（都是实测的）：和你那句真实录音的声纹相似度、长短比、音调走向相关、没生成出来的句数。"""
    import math

    ms = [v.get("m") or {} for v in (res.get("per_item") or {}).values()]
    vals = lambda k: [float(m[k]) for m in ms if isinstance(m.get(k), (int, float))]
    tr, dr, rf = vals("to_real"), vals("dur_ratio"), vals("r_f0")
    return {"to_real": round(statistics.mean(tr), 4) if tr else None,
            "dur_ratio_median": round(statistics.median(dr), 3) if dr else None,
            "rhythm_dev": round(statistics.mean(abs(math.log(x)) for x in dr if x > 0), 4) if dr else None,
            "r_f0": round(statistics.mean(rf), 3) if rf else None,
            "n_items": len(ms), "failed": int(res.get("failed") or 0)}


def cmd_eval(ctx: Ctx) -> int:
    from voicetwin.synth.select import _jsonable

    a = ctx.a
    started = time.time()
    since = float(a.since or started)
    deadline = since + float(a.budget_min) * 60.0
    restored = unpack(Path(a.restore), drop=a.drop_restored) if a.restore else None
    out = Path(a.eval_out or (ctx.exp / "eval" / ("warm" if a.warm else f"r{a.round}")))
    out.mkdir(parents=True, exist_ok=True)
    cfg, project, backend, info = _setup_eval(ctx)
    info.update(kind="eval_shard", round=0 if a.warm else a.round, shard=a.shard, shards=a.shards, variant=ctx.variant,
                warm=bool(a.warm), restore=restored,
                started=time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(started)))
    eps = _weight_epochs(ctx, backend)
    if a.warm:
        specs = [f"s{max(eps['lora'], default=0)}-g{max(eps['gpt'], default=0)}"]
    elif a.cands:
        specs = [s for s in a.cands.split(",") if s.strip()]
    else:
        specs = round1_specs(sorted(eps["lora"]), sorted(eps["gpt"]))
    mine = specs if a.warm else specs[a.shard::max(1, a.shards)]
    cands = [Cand(s) for s in mine]
    info["cands"] = [c.spec for c in cands]
    log(f"这一片要试：{'、'.join(info['cands']) or '（没有）'}")
    sel = _load_selector(cfg, project, backend, int(a.items) if not a.warm else 1, info, asr=not a.no_asr)
    shard_path = out / (f"warm.json" if a.warm else f"shard_r{a.round}_{a.shard}.json")
    write_json(shard_path, info)
    per_cand: List[float] = []
    try:
        for c in cands:
            path = out / f"{c.id}.json"
            old = read_json(path, {}) or {}
            if old.get("kind") == "eval_candidate" and old.get("status") == "ok" and old.get("n_planned") == len(sel.items):
                log(f"{c.id} 以前量过了，跳过")
                continue
            # 准备那一步只试生成一句（核对整套能用、量一句要多久），不算进比较
            entry: Dict[str, Any] = {"kind": "eval_warm" if a.warm else "eval_candidate", "round": a.round,
                                     "variant": ctx.variant, "id": c.id,
                                     "spec": c.spec, "pair": c.pair, "lora_epoch": c.lora, "gpt_epoch": c.gpt,
                                     "cfg_rate": c.cfg, "base": c.base, "shard": a.shard}
            ck = resolve(ctx, backend, c)
            if ck is None:
                entry.update(status="missing", why="没有这一轮的模型文件")
                write_json(path, entry)
                continue
            entry["files"] = {"sovits": Path(ck["sovits"]).name, "gpt": Path(ck["gpt"]).name}
            est = statistics.median(per_cand) if per_cand else None
            if est and time.time() + est > deadline:
                entry.update(status="skipped", why=f"这台机器剩下的时间不够（每个候选实测大约 {est / 60:.0f} 分钟）")
                write_json(path, entry)
                continue
            t = time.time()
            log(f"▶ {c.id}：{entry['files']}，cfg_rate {'auto' if c.cfg is None else c.cfg}")
            res = _measure(ctx, sel, backend, c, ck, deadline)
            secs = time.time() - t
            per_cand.append(secs)
            n_done = len(res.get("per_item") or {})
            entry.update(status="ok" if n_done == len(sel.items) else "incomplete", seconds=round(secs, 1),
                         n_planned=len(sel.items), metrics=_extra_metrics(res), result=_jsonable(res))
            if entry["status"] != "ok":
                entry["why"] = f"时间到了，只量了 {n_done}/{len(sel.items)} 句"
            write_json(path, entry)
            log(f"✅ {c.id}：像你本人 {res.get('pct')}%、综合分 {res.get('total')}、错字率 {res.get('cer')}、"
                f"{secs / 60:.1f} 分钟")
    finally:
        try:
            backend.stop()
        except Exception:  # noqa: BLE001
            pass
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["seconds"] = round(time.time() - started, 1)
    write_json(shard_path, info)
    # 打分这台机器的日志：只要生成和打分的（训练的日志跟着打包过来了，训练时已经提交过），每个留最后 200 KB
    copy_logs(ctx, out.parent / "日志" / ("打分-准备" if a.warm else f"打分-第{a.round}轮-{a.shard}"),
              names=("voicetwin.log", "gptsovits_api.log"), limit=200_000)
    if a.warm and a.pack:
        pack(ctx, "models", Path(a.pack))
    return 0


# ============================================================================ 合并、排名
def load_results(folder: Path) -> Dict[str, Any]:
    cands: Dict[str, Dict[str, Any]] = {}
    shards, runs, prep, done = [], [], None, None
    for p in sorted(Path(folder).rglob("*.json")):
        d = read_json(p)
        if isinstance(d, dict) and d.get("kind") == "eval_candidate":
            old = cands.get(d["id"])
            if old is None or (old.get("status") != "ok" and d.get("status") == "ok"):
                cands[d["id"]] = d
        elif isinstance(d, dict) and d.get("kind") == "eval_shard":
            shards.append(d)
        elif isinstance(d, dict) and d.get("kind") == "prep":
            prep = d
        elif p.name == "train_runs.json" and isinstance(d, list) and len(d) >= len(runs):
            runs = d
        elif isinstance(d, dict) and d.get("kind") == "done":
            done = d
    return {"cands": cands, "shards": shards, "runs": runs, "prep": prep, "done": done}


def rank(cands: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """程序自己挑模型的排名规则（rank_results：综合总评分、错字明显多的不能第一、某一类句子明显差的不能第一……）；
    只排量完了的（同一批句子）。"""
    from voicetwin.synth.select import rank_results

    full = [c for c in cands.values() if c.get("status") == "ok"]
    if not full:
        return {"order": [], "notes": [], "tie": None}
    n = max(len((c["result"].get("per_item") or {})) for c in full)
    keys = None
    ok = []
    for c in full:
        k = set((c["result"].get("per_item") or {}).keys())
        if len(k) == n and (keys is None or k == keys):
            keys = k
            r = dict(c["result"], id=c["id"])
            ok.append(r)
    rk = rank_results(ok)
    return {"order": [r["id"] for r in rk["order"]], "notes": rk.get("notes") or [], "tie": rk.get("tie"),
            "tradeoff": rk.get("tradeoff"), "keep_two": rk.get("keep_two"), "four_ok": rk.get("four_ok"),
            "ci_p": rk.get("ci_p"), "gated": rk.get("gated"), "n_items": n}


def round2_specs(cands: Dict[str, Dict[str, Any]]) -> List[str]:
    rk = rank({k: v for k, v in cands.items() if v.get("round") == 1})
    top = [cands[i] for i in rk["order"][:2]]
    if not top:
        return []
    specs: List[str] = []
    for c in top:
        specs += [f"{c['pair']}@{x:g}" for x in CFG_SWEEP]
    specs.append(f"{top[0]['pair']}@1.3")
    b = top[0]
    if not b.get("base") and b.get("lora_epoch") in (4, 8):
        specs.append(f"s6-g{b['gpt_epoch']}")
    done = {c["spec"] for c in cands.values()}
    return [s for s in dict.fromkeys(specs) if Cand(s).spec not in done]


def cmd_plan(ctx: Ctx) -> int:
    a = ctx.a
    res = load_results(Path(a.results))
    specs = round2_specs(res["cands"])
    shards = max(1, len(specs))   # 一台做一个
    matrix = [{"shard": i, "cands": ",".join(specs[i::shards])} for i in range(shards)] if specs else []
    log(f"第二轮：{'、'.join(specs) or '（没有，第一轮没有量完的候选）'}")
    gh_output("matrix", json.dumps({"include": matrix}, ensure_ascii=False))
    gh_output("has", "true" if matrix else "false")
    if a.plan_out:
        write_json(Path(a.plan_out), {"specs": specs, "matrix": matrix})
    return 0


def _fmt(v: Any, nd: int = 1, pct: bool = False) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{100 * v:.{nd}f}%" if pct else f"{v:.{nd}f}"
    return str(v)


def _minutes(s: Any) -> str:
    return "—" if s is None else f"{float(s) / 60:.0f} 分钟"


def build_report(res: Dict[str, Any], variant: str) -> Tuple[Dict[str, Any], str]:
    cands, runs, prep, done = res["cands"], res["runs"], res["prep"], res["done"]
    rk = rank(cands)
    pos = {cid: i + 1 for i, cid in enumerate(rk["order"])}
    rows = []
    for cid, c in cands.items():
        r = c.get("result") or {}
        four = (r.get("four") or {}).get("groups") or {}
        rows.append({"名次": pos.get(cid), "候选": cid, "第几轮实测": c.get("round"), "音色LoRA轮数": c.get("lora_epoch") or "底模",
                     "语气GPT轮数": c.get("gpt_epoch"), "cfg_rate": "auto" if c.get("cfg_rate") is None else c.get("cfg_rate"),
                     "状态": c.get("status"), "说明": c.get("why"),
                     "像你本人%": r.get("pct"), "分组像你本人%": {g: v.get("mean") for g, v in four.items() if v.get("status") == "ok"},
                     "综合分S": r.get("total"), "S误差": r.get("se"), "错字率": r.get("cer"),
                     **{k: v for k, v in (c.get("metrics") or {}).items()},
                     "每句秒数": r.get("seconds_per_item"), "用时秒": c.get("seconds"), "模型文件": c.get("files")})
    rows.sort(key=lambda x: (x["名次"] is None, x["名次"] or 0, str(x["候选"])))
    timing = {"prep": (prep or {}).get("timing_s"), "prep_total_s": (prep or {}).get("seconds"),
              "train_jobs": [{k: r.get(k) for k in ("job", "result", "why_stopped", "seconds", "epochs_before",
                                                       "epochs_after", "started", "finished", "vt_timing", "error")}
                             for r in runs],
              "eval_shards": [{k: s.get(k) for k in ("round", "shard", "cands", "load_seconds", "seconds", "asr")}
                              for s in res["shards"]]}
    saves = [ev for r in runs for ev in (r.get("saves") or []) if ev.get("epoch_seconds") and not ev.get("from_stage_start")]
    for kind in ("lora", "gpt"):
        xs = [ev["epoch_seconds"] for ev in saves if ev["stage"] == kind]
        timing[f"{kind}_epoch_seconds_median"] = round(statistics.median(xs), 1) if xs else None
        timing[f"{kind}_epoch_seconds_n"] = len(xs)
    out = {"variant": variant, "variant_note": VARIANTS.get(variant, {}).get("说明"), "made": time.strftime("%Y-%m-%d %H:%M:%S"),
           "prep": {k: (prep or {}).get(k) for k in ("counts", "match", "train_minutes", "applied_texts", "drop_reasons_with_text",
                                                      "confirmed", "recordings")} if prep else None,
           "done": done, "ranking": rk, "rows": rows, "timing": timing}
    L = ["# 老师的录音：V5 真实训练实测（GitHub，只有处理器）", "",
         f"训练方式：{variant}（{out['variant_note']}）。生成于 {out['made']}。表里都是实测的数字。", ""]
    if prep:
        c, m = prep.get("counts") or {}, prep.get("match") or {}
        L += ["## 素材", "",
              f"- 切片 {m.get('clips')} 段，逐句文字 {m.get('texts')} 句：按 id 对上 {m.get('by_id')} 句、按时间对上 {m.get('by_time')} 句、"
              f"对不上 {m.get('unmatched_texts')} 句；和以前转录时的切片{'完全一样' if m.get('same_ids_as_transcription') else '不完全一样（见 结果.json 的 prep.match）'}",
              f"- 填上核对好的文字 {prep.get('applied_texts')} 句；用来训练 {c.get('train')} 句（{prep.get('train_minutes')} 分钟）、"
              f"考试题 {c.get('val')} 句；程序判断不能用 {c.get('unusable')} 段"
              + (f"（有文字但没用的原因：{json.dumps(prep.get('drop_reasons_with_text'), ensure_ascii=False)}）"
                 if prep.get("drop_reasons_with_text") else ""), ""]
    L += ["## 训练", ""]
    if runs:
        r0 = runs[0]
        L += [f"- {str(r0.get('plan') or '').strip()}", ""]
        L += ["| 任务 | 结果 | 开始时存下的（音色/语气） | 结束时存下的 | 用时 | 为什么停 |", "|---|---|---|---|---|---|"]
        for r in runs:
            b, e = r.get("epochs_before") or {}, r.get("epochs_after") or {}
            L.append(f"| {r.get('job')} | {r.get('result')} | {b.get('lora')} / {b.get('gpt')} | {e.get('lora')} / {e.get('gpt')} | "
                     f"{_minutes(r.get('seconds'))} | {r.get('why_stopped') or (r.get('error') or '')[:80]} |")
        L += ["", f"- 每轮用时（实测，同一台机器上连着存的两轮之间）：音色 LoRA {_minutes(timing['lora_epoch_seconds_median'])}"
                  f"（{timing['lora_epoch_seconds_n']} 次）、语气 GPT {_minutes(timing['gpt_epoch_seconds_median'])}"
                  f"（{timing['gpt_epoch_seconds_n']} 次）", f"- {'全部练完了' if done else '还没练完'}", ""]
    else:
        L += ["还没有训练记录。", ""]
    if rows:
        L += ["## 实测比较（没参加训练的录音，每句生成 1 次，用声音分身自己的打分）", "",
              "名次按程序自己挑模型的规则（综合总评分；错字明显多、某一类句子明显差的不能排第一）。"
              "像你本人 % 按你自己的真实录音校准；声纹相似度 = 和你那一句真实录音比；长短比 = 生成的说话时长 / 你本人说这句的时长；"
              "节奏偏差 = |ln 长短比| 的平均（越小越像）；音调走向 = 和你那句录音的音调走向的相关（越大越像）。", "",
              "| 名次 | 候选 | 音色轮数 | 语气轮数 | cfg_rate | 像你本人 % | 声纹相似度 | 错字率 | 长短比 | 节奏偏差 | 音调走向 | 综合分 S | 句数 | 每句秒数 | 状态 |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for x in rows:
            L.append(f"| {x['名次'] or '—'} | {x['候选']} | {x['音色LoRA轮数']} | {x['语气GPT轮数']} | {x['cfg_rate']} | "
                     f"{_fmt(x['像你本人%'])} | {_fmt(x.get('to_real'), 3)} | {_fmt(x['错字率'], 1, pct=True)} | "
                     f"{_fmt(x.get('dur_ratio_median'), 2)} | {_fmt(x.get('rhythm_dev'), 3)} | {_fmt(x.get('r_f0'), 2)} | "
                     f"{_fmt(x['综合分S'], 3)}" + (f" ± {x['S误差']:.3f}" if x.get('S误差') is not None else "")
                     + f" | {x.get('n_items', '—')} | {_fmt(x['每句秒数'], 0)} | {x['状态']}{'：' + x['说明'] if x.get('说明') else ''} |")
        if rk["notes"] or rk.get("tie") is not None:
            L += [""] + [f"- {n}" for n in rk["notes"]]
            if rk.get("tie"):
                L.append("- 第 1、2 名的差别在误差范围内（分不出来）")
            if rk.get("tradeoff"):
                L.append(f"- 第 2 名有一类句子明显更好：{rk['tradeoff']}")
        L.append("")
    if res["shards"]:
        L += ["## 打分用时", "", "| 第几轮 | 分片 | 候选 | 准备打分（秒） | 一共（分钟） | 查错字用的识别 |", "|---|---|---|---|---|---|"]
        for s in sorted(res["shards"], key=lambda s: (s.get("round") or 0, s.get("shard") or 0)):
            L.append(f"| {s.get('round')} | {s.get('shard')} | {'、'.join(s.get('cands') or [])} | {s.get('load_seconds')} | "
                     f"{_minutes(s.get('seconds'))} | {json.dumps(s.get('asr'), ensure_ascii=False)} |")
        L.append("")
    return out, "\n".join(L) + "\n"


def cmd_report(ctx: Ctx) -> int:
    a = ctx.a
    res = load_results(Path(a.results or ctx.exp))
    out_dir = ctx.out_dir()
    data, md = build_report(res, ctx.variant)
    write_json(out_dir / "结果.json", data)
    (out_dir / "结果.md").write_text(md, encoding="utf-8")
    # 每个候选的逐句结果另外存（结果.json 只放汇总）
    for cid, c in res["cands"].items():
        write_json(out_dir / "逐个候选" / f"{cid}.json", c)
    for name, obj in (("prep.json", res["prep"]), ("train_runs.json", res["runs"])):
        if obj:
            write_json(out_dir / "过程" / name, obj)
    if a.results:   # 各台机器的日志（打分的分片打包在结果里）
        for d in Path(a.results).rglob("日志"):
            if d.is_dir() and d.resolve() != (out_dir / "日志").resolve():
                shutil.copytree(d, out_dir / "日志", dirs_exist_ok=True)
    log(f"写好了 {out_dir / '结果.md'}")
    print(md)
    return 0


def cmd_progress(ctx: Ctx) -> int:
    """训练中途：把现在的进度写进结果文件夹，再把这次的日志（每个最多留最后 1 MB）放进 日志/<任务名>/。"""
    cmd_report(ctx)
    # 日志跟着状态一台一台传下来（越来越长）：只留一份最新的（以前的在提交历史里）
    copy_logs(ctx, ctx.out_dir() / "日志" / "准备素材和训练")
    return 0


def copy_logs(ctx: Ctx, dst: Path, names: Optional[Tuple[str, ...]] = None, limit: int = 1_000_000) -> None:
    """声音分身的日志（每个最多留最后 limit 字节）。"""
    logs = ctx.ws / VOICE / "logs"
    if not logs.is_dir():
        return
    dst.mkdir(parents=True, exist_ok=True)
    for p in logs.glob("*.log"):
        if names is None or p.name in names:
            (dst / p.name).write_bytes(p.read_bytes()[-limit:])


# ============================================================================ GitHub 工作流
#: 训练最多接几台机器（估计 7~8 台就练完；多写几台不花时间：练完以后后面的都跳过）
N_TRAIN = 12
#: 第一轮打分分几片同时做（18 个候选，每片 1 个；免费账号最多同时 20 台）
R1_SHARDS = 18
WF_HEAD = """name: train-real

# 由 research/训练实测/train_exp.py workflow 生成（改那里，再运行
#   python research/训练实测/train_exp.py workflow > .github/workflows/train-real.yml），不要直接改这个文件。
#
# 老师自己的录音（research/训练实测/录音.txt，约 2.5 小时，文字逐句核对过）在 GitHub 的处理器上真的训练 V5，
# 再用声音分身自己的打分实测比较不同的设置；实测最好的设置以后当程序的默认设置。
# 一台机器最多 6 小时，所以：准备素材 → 训练（一台接一台：快到时间就在存完一轮以后停下、打包，下一台接着练；
# 练完写 DONE.json，后面的跳过）→ 准备打分（下载声纹模型、整理参考录音库、校准，试生成一句）→ 第一轮打分（几台同时）→
# 第二轮（按第一轮的排名定：cfg_rate、补一个轮数）→ 汇总，提交到 research/训练实测/结果/。
# 每一步结束时把进度和日志提交到结果文件夹（边做边备份）。
on:
  workflow_dispatch:
    inputs:
      variant:
        description: "训练方式：default / app_cpu / lora16 / rank64（见 train_exp.py 的 VARIANTS）"
        default: default
  push:
    branches: ["claude/**"]
    paths:
      - "research/训练实测/train_exp.py"
      - "research/训练实测/ci.sh"
      - ".github/workflows/train-real.yml"

permissions:
  contents: write

# 一次要跑一两天。只有改了这三个文件才会触发（别的推送不会）；触发了就是有意从头再练：新的一次取消正在跑的
# （11-24：训练用的文字处理改了——英文后面的「，」「、」不再被丢掉——要用新代码重练）
concurrency:
  group: train-real-${{ github.ref }}
  cancel-in-progress: true

env:
  VARIANT: ${{ inputs.variant || 'default' }}
  OUT: ${{ (inputs.variant && inputs.variant != 'default') && format('research/训练实测/结果/{0}', inputs.variant) || 'research/训练实测/结果' }}
  EXP: research/训练实测/train_exp.py
  CI_SH: research/训练实测/ci.sh

jobs:
"""


def _steps_head(env: str) -> List[str]:
    return ["    steps:",
            "      - run: echo \"T0=$(date +%s)\" >> \"$GITHUB_ENV\"",
            "      - uses: actions/checkout@v4",
            "      - uses: actions/setup-python@v5",
            "        with:",
            "          python-version: \"3.12\"",
            f"      - run: bash \"$CI_SH\" {env}"]


def _upload(name: str, path: str, cond: str = "", days: int = 5, files: bool = False) -> List[str]:
    out = ["      - uses: actions/upload-artifact@v4"]
    if cond:
        out.append(f"        if: {cond}")
    out += ["        with:", f"          name: {name}", f"          path: {path}"]
    if not files:
        out.append("          compression-level: 0")
    out += [f"          retention-days: {days}", "          overwrite: true", "          if-no-files-found: ignore"]
    return out


def _progress(job: str) -> List[str]:
    return _upload(f"log-{job}", "/mnt/w/vt/exp/*.json", cond="always()", days=10, files=True) + [
        "      - if: always()",
        # 提交进度不能让这一步算失败（不然后面的训练就不接着做了）
        f"        run: (python \"$EXP\" progress --variant \"$VARIANT\" --out \"$OUT\" --job {job} && "
        f"bash \"$CI_SH\" commit \"记录：训练实测 {job}（GitHub 自动运行）\" \"$OUT\") || echo \"::warning::进度没能提交\""]


def workflow_yaml() -> str:
    L: List[str] = []
    # 准备素材
    L += ["  prep:", "    runs-on: ubuntu-latest", "    timeout-minutes: 240"] + _steps_head("prep-env") + [
        "      - run: python \"$EXP\" prep --variant \"$VARIANT\" --pack /mnt/w/pack/prep"] + \
        _upload("prep", "/mnt/w/pack/prep") + _progress("prep") + [""]
    # 训练：一台接一台
    for n in range(1, N_TRAIN + 1):
        prev = "prep" if n == 1 else f"train-{n - 1}"
        art = "prep" if n == 1 else f"state-{n - 1}"
        L += [f"  train-{n}:", f"    needs: {prev}"]
        if n > 1:
            L.append(f"    if: needs.{prev}.outputs.done == 'false'")
        L += ["    runs-on: ubuntu-latest", "    timeout-minutes: 355", "    outputs:",
              "      done: ${{ steps.train.outputs.done }}"] + _steps_head("gsv-env train") + [
            "      - uses: actions/download-artifact@v4", "        with:", f"          name: {art}",
            "          path: /mnt/w/dl",
            "      - id: train",
            f"        run: python \"$EXP\" train --variant \"$VARIANT\" --job train-{n} --restore /mnt/w/dl --drop-restored "
            "--pack /mnt/w/pack --since \"$T0\" --budget-min 300"] + \
            _upload(f"state-{n}", "/mnt/w/pack/state", cond="steps.train.outputs.done == 'false'", days=4) + \
            _upload("models", "/mnt/w/pack/models", cond="steps.train.outputs.done == 'true'", days=10) + \
            _progress(f"train-{n}") + [""]
    trains = ", ".join(f"train-{n}" for n in range(1, N_TRAIN + 1))
    eval_steps = lambda art: ["      - uses: actions/download-artifact@v4", "        with:", f"          name: {art}",
                              "          path: /mnt/w/dl"]
    # 准备打分
    L += ["  warm:", f"    needs: [{trains}]", "    if: ${{ !cancelled() && contains(needs.*.outputs.done, 'true') }}",
          "    runs-on: ubuntu-latest", "    timeout-minutes: 340"] + _steps_head("gsv-env eval") + eval_steps("models") + [
        "      - run: python \"$EXP\" eval --warm --variant \"$VARIANT\" --restore /mnt/w/dl --drop-restored "
        "--pack /mnt/w/pack/warm --eval-out /mnt/w/out/warm --since \"$T0\" --budget-min 300"] + \
        _upload("models-warm", "/mnt/w/pack/warm", days=10) + \
        _upload("eval-warm", "/mnt/w/out", cond="always()", days=10, files=True) + [""]
    # 两轮打分
    shards = ", ".join(str(i) for i in range(R1_SHARDS))
    L += ["  eval1:", "    needs: warm", "    runs-on: ubuntu-latest", "    timeout-minutes: 355", "    strategy:",
          "      fail-fast: false", "      matrix:", f"        shard: [{shards}]"] + _steps_head("gsv-env eval") + \
        eval_steps("models-warm") + [
        "      - run: python \"$EXP\" eval --round 1 --shard ${{ matrix.shard }} --shards " + str(R1_SHARDS) +
        " --variant \"$VARIANT\" --restore /mnt/w/dl --drop-restored --eval-out /mnt/w/out/r1 --since \"$T0\" --budget-min 320"] + \
        _upload("eval-r1-${{ matrix.shard }}", "/mnt/w/out", cond="always()", days=10, files=True) + [""]
    L += ["  plan2:", "    needs: eval1", "    if: ${{ !cancelled() && needs.eval1.result != 'skipped' }}",
          "    runs-on: ubuntu-latest", "    timeout-minutes: 30", "    outputs:",
          "      matrix: ${{ steps.plan.outputs.matrix }}", "      has: ${{ steps.plan.outputs.has }}"] + \
        _steps_head("light-env") + [
        "      - uses: actions/download-artifact@v4", "        continue-on-error: true", "        with:",
        "          pattern: eval-r1-*",
        "          merge-multiple: true", "          path: /mnt/w/res",
        "      - id: plan", "        run: python \"$EXP\" plan --variant \"$VARIANT\" --results /mnt/w/res", ""]
    L += ["  eval2:", "    needs: plan2", "    if: needs.plan2.outputs.has == 'true'", "    runs-on: ubuntu-latest",
          "    timeout-minutes: 355", "    strategy:", "      fail-fast: false",
          "      matrix: ${{ fromJSON(needs.plan2.outputs.matrix) }}"] + _steps_head("gsv-env eval") + \
        eval_steps("models-warm") + [
        "      - run: python \"$EXP\" eval --round 2 --shard ${{ matrix.shard }} --cands \"${{ matrix.cands }}\" "
        "--variant \"$VARIANT\" --restore /mnt/w/dl --drop-restored --eval-out /mnt/w/out/r2 --since \"$T0\" --budget-min 320"] + \
        _upload("eval-r2-${{ matrix.shard }}", "/mnt/w/out", cond="always()", days=10, files=True) + [""]
    # 汇总、提交
    L += ["  report:", f"    needs: [prep, {trains}, warm, eval1, plan2, eval2]", "    if: ${{ !cancelled() }}",
          "    runs-on: ubuntu-latest", "    timeout-minutes: 30"] + _steps_head("light-env") + [
        "      - uses: actions/download-artifact@v4", "        continue-on-error: true", "        with:",
        "          pattern: eval-*", "          path: /mnt/w/res",
        "      - uses: actions/download-artifact@v4", "        continue-on-error: true", "        with:",
        "          pattern: log-*", "          path: /mnt/w/res",
        "      - run: python \"$EXP\" report --variant \"$VARIANT\" --results /mnt/w/res --out \"$OUT\"",
        "      - run: bash \"$CI_SH\" commit \"记录：训练实测结果（GitHub 自动运行）\" \"$OUT\""]
    return WF_HEAD + "\n".join(L) + "\n"


# ============================================================================ 命令行
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=["prep", "train", "eval", "plan", "report", "progress", "pack", "unpack", "workflow"])
    ap.add_argument("--work", default="/mnt/w/vt", help="工作文件夹（声音分身的 workspace 和这个实验的记录）")
    ap.add_argument("--gsv", default="", help="GPT-SoVITS 整合包解压出来的文件夹（默认环境变量 GSV 或 /mnt/w/g/GPT-SoVITS-*）")
    ap.add_argument("--sv-dir", default="/mnt/w/sv", help="声纹模型放在哪里")
    ap.add_argument("--variant", default="default", help="训练方式：" + "、".join(VARIANTS))
    ap.add_argument("--port", type=int, default=9880)
    ap.add_argument("--extra-config", default="", help="（本机检查用）再合并进设置的 JSON")
    ap.add_argument("--out", default="", help="结果文件夹（默认 research/训练实测/结果[/训练方式]）")
    ap.add_argument("--job", default="", help="任务名（记录里用）")
    ap.add_argument("--since", type=float, default=0.0, help="这台机器开始的时间（秒），时间预算从这里算")
    ap.add_argument("--budget-min", type=float, default=320.0, help="最晚在开始以后多少分钟停（默认 320）")
    ap.add_argument("--poll", type=float, default=20.0, help="训练时多久看一次存下的模型（秒）")
    ap.add_argument("--restore", default="", help="先还原这个文件夹里打包好的状态")
    ap.add_argument("--pack", default="", help="做完以后打包到这个文件夹")
    ap.add_argument("--drop-restored", action="store_true", help="还原以后删掉打包的文件（GitHub 上省硬盘）")
    # prep
    ap.add_argument("--src", default="", help="录音放在哪里（默认 /mnt/w/tr/录音，切片 id 和这个路径有关）")
    ap.add_argument("--texts", default="", help="（本机检查用）逐句文字的 JSON，给了就不下载录音")
    # eval
    ap.add_argument("--round", type=int, default=1)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--cands", default="", help="要试的候选，逗号分开（例：s8-g20,base-g10,s4-g15@1.6）；不给就按第一轮的")
    ap.add_argument("--items", type=int, default=20, help="每个候选试几句考试题（最多就是留出来的那些）")
    ap.add_argument("--warm", action="store_true", help="只准备打分（声纹模型、参考录音库、校准）并试生成一句")
    ap.add_argument("--no-sv-download", action="store_true", help="（本机检查用）不下载声纹模型")
    ap.add_argument("--no-asr", action="store_true", help="（本机检查用）不查错字（不加载识别模型）")
    ap.add_argument("--eval-out", default="", help="打分结果放在哪里（默认 <work>/exp/eval/r<轮>）")
    # plan / report
    ap.add_argument("--results", default="", help="打分结果（各个分片的 JSON）在哪个文件夹")
    ap.add_argument("--plan-out", default="")
    # pack / unpack
    ap.add_argument("--kind", default="state", choices=["prep", "state", "models"])
    a = ap.parse_args(argv)
    if a.cmd == "workflow":
        sys.stdout.write(workflow_yaml())
        return 0
    ctx = Ctx(a)
    if a.cmd == "pack":
        pack(ctx, a.kind, Path(a.pack))
        return 0
    if a.cmd == "unpack":
        unpack(Path(a.restore))
        return 0
    return {"prep": cmd_prep, "train": cmd_train, "eval": cmd_eval, "plan": cmd_plan, "report": cmd_report,
            "progress": cmd_progress}[a.cmd](ctx)


if __name__ == "__main__":
    sys.exit(main())
