"""把老师新录的课用声音分身自己的方法转录一遍（和老师在网页上点「开始准备素材」→「📝 一键全部文字校正」一样），
把每一句两个识别引擎各自听出来的字、自动查错字的结果、一键校正以后的字都存下来，给人逐句核对。

  1. 准备素材：切成一句一句，faster-whisper large-v3（通用，中英文都行）识别；
  2. 自动查错字（proofcheck: on）：FunASR paraformer（中文专用）把每一句再听一遍，两边对不上的地方标红；
  3. 一键全部文字校正：按老师的母本标准库（程序自带）改确定的错字、采用有把握的建议。

用法：python transcribe.py --list 要转录.txt --work <临时目录> --out <结果目录>
"""
import argparse
import csv
import hashlib
import json
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

T0 = time.time()


def say(frac, msg=""):
    print(f"[{time.time() - T0:7.0f}s] {float(frac or 0):5.1%} {msg}", flush=True)


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--list", required=True)
    ap.add_argument("--work", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    work, out = Path(a.work), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    src = work / "录音"
    src.mkdir(parents=True, exist_ok=True)
    info = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "files": [], "timing_s": {}}
    for line in Path(a.list).read_text(encoding="utf-8").splitlines():
        u = line.strip()
        if not u or u.startswith("#"):
            continue
        dst = src / urllib.parse.unquote(u.rsplit("/", 1)[-1])
        say(0, f"下载 {dst.name}")
        urllib.request.urlretrieve(u, dst)
        info["files"].append({"name": dst.name, "bytes": dst.stat().st_size, "sha256": sha256(dst)})

    from voicetwin import workflows as wf
    from voicetwin.config import load_config
    from voicetwin.data import review

    cfg = load_config(overrides={
        "workspace": str(work / "workspace"),
        "prepare": {"asr": {"engine": "faster-whisper", "model": "large-v3", "device": "cpu"}, "proofcheck": "on"},
    }, user_config=False)
    voice = "新录音"
    t = time.time()
    summary = wf.run_prepare(cfg, voice, [str(src)], progress=say)
    info["timing_s"]["准备素材（识别 + 查错字）"] = round(time.time() - t)
    info["prepare_summary"] = {k: v for k, v in (summary or {}).items() if isinstance(v, (int, float, str, list))}
    project = wf.open_project(cfg, voice, must_exist=True)
    recs = project.load_manifest()
    asr_text = {r["id"]: str(r.get("text") or "") for r in recs}
    fun = {}
    for name in ("proofcheck_funasr.json", "proofcheck_faster-whisper.json", "proofcheck_faster-whisper-words.json"):
        p = Path(project.cache_dir) / name
        if p.exists():
            items = (json.loads(p.read_text(encoding="utf-8")) or {}).get("items") or {}
            for k, v in items.items():
                fun.setdefault(k, {"engine": name[11:-5], "text": (v or {}).get("text", "")})
    recs = project.load_manifest()   # 查错字以后的（带 suspect）
    t = time.time()
    fix = wf.run_transcript_fix(cfg, voice, adopt_all=True)
    info["timing_s"]["一键全部文字校正"] = round(time.time() - t)
    info["textfix"] = {k: v for k, v in (fix or {}).items() if isinstance(v, (int, float, str))}
    draft = review.load_draft(project)

    rows = []
    for r in recs:
        cid = r["id"]
        sus = r.get("suspect") or {}
        d = draft.get(cid) or {}
        rows.append({
            "行": len(rows) + 1, "id": cid, "开始秒": r.get("start"), "秒": r.get("duration"),
            "语言": r.get("lang"), "用来训练": int(bool(r.get("keep", True))), "不用的原因": r.get("drop_reason") or "",
            "whisper识别": asr_text.get(cid, ""),
            "第二个引擎": (fun.get(cid) or {}).get("text", ""), "第二个引擎是": (fun.get(cid) or {}).get("engine", ""),
            "查错字标红的原因": "；".join(sus.get("reasons") or []), "查错字的建议": sus.get("alt") or "",
            "一键校正以后": d.get("text", asr_text.get(cid, "")),
        })
    with open(out / "逐句对照.csv", "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["行"])
        w.writeheader()
        w.writerows(rows)
    (out / "逐句对照.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    info["counts"] = {"句子": len(rows), "用来训练": sum(x["用来训练"] for x in rows),
                      "第二个引擎听过": sum(bool(x["第二个引擎"]) for x in rows),
                      "查错字标红": sum(bool(x["查错字标红的原因"]) for x in rows),
                      "一键校正改过": sum(x["一键校正以后"] != x["whisper识别"] for x in rows)}
    (out / "运行信息.json").write_text(json.dumps(info, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    full = "\n".join(f"{x['行']}\t{x['一键校正以后']}" for x in rows)
    (out / "全文_一键校正以后.txt").write_text(full + "\n", encoding="utf-8")
    say(1, json.dumps(info["counts"], ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
