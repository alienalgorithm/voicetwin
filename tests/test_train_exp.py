"""research/训练实测/train_exp.py（老师的录音在 GitHub 上真的训练 V5、实测比较设置）的整套流程，用 tests/fake_gptsovits.py 的
替身走一遍（不联网、不用真的 GPT-SoVITS）：准备素材（按 id 填核对好的文字）→ 打包 → 每次在「新的机器」上还原、训练，
时间到了在存完一轮以后停下、下一次接着练（不从头练）→ 练完写 DONE.json → 准备打分 → 两个分片打分 → 第二轮的候选 →
汇总。还核对 .github/workflows/train-real.yml 就是脚本生成的那份。"""

import importlib.util
import json
import os
import shutil
from pathlib import Path

import pytest

from conftest import make_lecture
from fake_gptsovits import build_fake_root
from test_gptsovits_fake import needs_fake_python

ROOT = Path(__file__).resolve().parents[1]
EXP_PATH = ROOT / "research" / "训练实测" / "train_exp.py"


def _load():
    spec = importlib.util.spec_from_file_location("train_exp", EXP_PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_workflow_file_is_generated_by_the_script():
    exp = _load()
    wf_path = ROOT / ".github" / "workflows" / "train-real.yml"
    assert wf_path.read_text(encoding="utf-8") == exp.workflow_yaml(), \
        "改了 train_exp.py 的工作流：运行 python research/训练实测/train_exp.py workflow > .github/workflows/train-real.yml"
    import yaml

    jobs = yaml.safe_load(wf_path.read_text(encoding="utf-8"))["jobs"]
    trains = [f"train-{n}" for n in range(1, exp.N_TRAIN + 1)]
    assert jobs["train-1"]["needs"] == "prep"
    for a, b in zip(trains, trains[1:]):   # 一台接一台；上一台练完（done=true）以后的都跳过
        assert jobs[b]["needs"] == a and jobs[b]["if"] == f"needs.{a}.outputs.done == 'false'"
        assert jobs[b]["timeout-minutes"] < 360
    assert set(trains) <= set(jobs["warm"]["needs"]) and "contains(needs.*.outputs.done, 'true')" in jobs["warm"]["if"]
    assert jobs["eval1"]["strategy"]["matrix"]["shard"] == list(range(exp.R1_SHARDS))
    assert len(exp.round1_specs(list(range(1, 9)), list(range(1, 21)))) == exp.R1_SHARDS   # 默认：一台做一个候选


def test_default_variant_matches_the_teachers_8gb_plan():
    """default 训练方式（处理器上）和老师的 8 GB 显卡上程序自己算的一样：音色 LoRA 每批 1 条、8 轮；语气 GPT 每批 4 条、20 轮，
    都是每轮存一个（素材按这次的大小：约 1160 条、145 分钟）。app_cpu 什么都不改：处理器上程序自己算的语气每批 1 条。"""
    from voicetwin.backends import gptsovits as g

    exp = _load()

    def plan(gpu, half, user):
        lora = g.plan_lora("v5dev", gpu[0], gpu[1], is_half=half, mode="identical", n_clips=1160, user=user)
        p = g.plan_training(1160, 145.0, gpu[0], gpu[1], is_half=half, mode="identical", user=user, lora=lora,
                            mixed_text=True, n_val=20)
        return (lora["batch_size"], lora["epochs"], lora["save_every"], p["gpt_batch_size"], p["gpt_epochs"],
                p["gpt_save_every"])

    teacher = plan((7.96, 6.8), True, {})
    assert teacher == (1, 8, 1, 4, 20, 1)
    assert plan((0.0, None), False, exp.VARIANTS["default"]["train"]) == teacher
    assert plan((0.0, None), False, exp.VARIANTS["app_cpu"]["train"])[3] == 1


def test_candidates_and_round_one_grid():
    exp = _load()
    specs = exp.round1_specs(list(range(1, 9)), list(range(1, 21)))
    assert len(specs) == 18 and "s8-g20" in specs and "s1-g5" in specs and "base-g10" in specs and "s6-g20" not in specs
    c = exp.Cand("s4-g15@1.6")
    assert (c.lora, c.gpt, c.cfg, c.id, c.pair, c.base) == (4, 15, 1.6, "s4-g15-c1.6", "s4-g15", False)
    b = exp.Cand("base-g10")
    assert b.base and b.cfg is None and b.id == "base-g10" and exp.Cand("s8-g20@auto").spec == "s8-g20"
    with pytest.raises(SystemExit):
        exp.Cand("s8")
    # 练的轮数和默认不一样时：最后一轮一定在内
    assert exp.round1_specs([1, 2, 3], [1, 2, 3, 4]) == ["s1-g4", "s2-g4", "s3-g4", "base-g4"]


@pytest.mark.skipif(os.name == "nt", reason="打包用 bash、tar、split（只在 GitHub 的 Linux 机器上用）")
@needs_fake_python
def test_prep_train_resume_eval_report(tmp_path, monkeypatch):
    from voicetwin import workflows as wf
    from voicetwin.backends.gptsovits import GPTSoVITSBackend
    from voicetwin.config import load_config
    from voicetwin.data.subtitles import parse_subtitles
    from voicetwin.utils.progress import CANCEL

    monkeypatch.setenv("EXPECT_VERSION", "v5dev")
    monkeypatch.setenv("FAKE_GSV_EPOCH_SLEEP", "0.8")
    monkeypatch.setenv("FAKE_GSV_CLIP_SLEEP", "0.001")
    monkeypatch.setattr(GPTSoVITSBackend, "ensure_users_pth", lambda self: None)   # 不往这个 Python 里写 users.pth
    monkeypatch.setattr(GPTSoVITSBackend, "_gpu_memory", lambda self, quick=False: (0.0, None, "test"))  # 和 GitHub 一样没有显卡
    monkeypatch.setattr(GPTSoVITSBackend, "POLL_SECONDS", 0.05)
    exp = _load()
    gsv = build_fake_root(tmp_path / "g" / "GPT-SoVITS-fake")
    src = tmp_path / "src"
    make_lecture(src / "第1课.wav", repeats=2, with_srt=False)
    make_lecture(tmp_path / "cues" / "第1课.wav", repeats=2, with_srt=True)   # 同一段录音的字幕（只用来编「核对好的文字」）
    cues = parse_subtitles(tmp_path / "cues" / "第1课.srt")

    # 「以前转录时」切出来的片段（id 和这个路径有关）→ 逐句文字（和 build_texts.py 的字段一样）
    cfg0 = load_config(overrides={"workspace": str(tmp_path / "ws0"), "speaker_encoder": "mfcc",
                                  "prepare": {"asr": {"engine": "none"}, "proofcheck": "off"}}, user_config=False)
    wf.run_prepare(cfg0, "probe", [str(src)])
    texts = []
    for r in wf.open_project(cfg0, "probe", must_exist=True).load_manifest():
        s, e = r["start"], r["start"] + r["duration"]
        words = "".join(c.text for c in cues if min(e, c.end) - max(s, c.start) > 0.3 * (c.end - c.start))
        texts.append({"id": r["id"], "录音": "第1课.wav", "开始秒": r["start"], "秒": r["duration"], "文字": words,
                      "来源": "测试", "对齐质量": "好" if words else "没有对上"})
    texts[1]["对齐质量"] = "不连续"                       # 对齐不好的不用来训练
    texts[2]["id"] = texts[2]["id"][:-4] + "9999"           # id 对不上：按时间对上
    (tmp_path / "texts.json").write_text(json.dumps(texts, ensure_ascii=False), encoding="utf-8")

    work = tmp_path / "vt"
    extra = {"speaker_encoder": "mfcc",
             "backends": {"gptsovits": {"startup_timeout": 60, "train": {"lora_epochs": 3, "gpt_epochs": 4}}}}
    common = ["--work", str(work), "--gsv", str(gsv), "--sv-dir", str(tmp_path / "sv"), "--port", "19941",
              "--extra-config", json.dumps(extra), "--out", str(tmp_path / "out")]
    run = lambda *a: exp.main([*a, *common])

    assert run("prep", "--src", str(src), "--texts", str(tmp_path / "texts.json"), "--pack", str(tmp_path / "p/prep")) == 0
    prep = json.loads((work / "exp" / "prep.json").read_text(encoding="utf-8"))
    assert prep["match"]["by_id"] == len(texts) - 1 and prep["match"]["by_time"] == 1 and prep["match"]["unmatched_texts"] == 0
    assert prep["confirmed"] and prep["applied_texts"] == sum(1 for t in texts if t["对齐质量"] == "好")
    assert any(x["why"] == "对齐质量：不连续" for x in prep["not_used"])
    assert prep["counts"]["val"] >= 1 and prep["counts"]["train"] >= 8

    def fresh_machine():
        shutil.rmtree(work, ignore_errors=True)
        for d in ("logs", "SoVITS_weights_v5dev", "GPT_weights_v5dev"):
            shutil.rmtree(gsv / d, ignore_errors=True)

    prev = tmp_path / "p/prep"
    for n in range(1, 11):
        fresh_machine()
        assert run("train", "--job", f"train-{n}", "--restore", str(prev), "--pack", str(tmp_path / f"p/t{n}"),
                   "--budget-min", "0.06", "--poll", "0.2") == 0
        assert not CANCEL.is_set()
        if (work / "exp" / "DONE.json").exists():
            prev = tmp_path / f"p/t{n}/models"
            break
        prev = tmp_path / f"p/t{n}/state"
    runs = json.loads((work / "exp" / "train_runs.json").read_text(encoding="utf-8"))
    assert runs[-1]["result"] == "done" and len(runs) >= 2 and all(r["result"] == "stopped" for r in runs[:-1])
    assert runs[0]["plan_state"] != "continue" and all(r["plan_state"] == "continue" for r in runs[1:])  # 接着练，没有从头练
    assert all(r["why_stopped"] for r in runs[:-1])
    for a, b in zip(runs, runs[1:]):   # 每一台都接着上一台存下的那一轮
        assert b["epochs_before"] == a["epochs_after"]
    assert runs[-1]["epochs_after"] == {"lora": 3, "gpt": 4}
    assert not list((gsv / "logs").rglob("old_runs"))
    s2 = (work / "workspace" / "老师" / "logs" / "gsv_s2_train_v5dev.log").read_text(encoding="utf-8")
    starts = [int(x.rsplit(" ", 1)[1]) for x in s2.splitlines() if x.startswith("start training from epoch")]
    assert starts[0] == 1 and len(starts) == len(runs) and starts == sorted(starts)
    assert "你指定的" in runs[0]["plan"]   # 训练方式的设置用上了（素材这么少时每批条数被素材条数封顶，见下一个测试）
    models = json.loads((work / "workspace" / "老师" / "models.json").read_text(encoding="utf-8"))["gptsovits"]
    assert len(models["sovits"]) == 3 and len(models["gpt"]) == 4

    # 准备打分（打包给分片用）→ 第一轮两个分片（各在新的机器上）→ 第二轮 → 汇总
    fresh_machine()
    assert run("eval", "--warm", "--restore", str(prev), "--pack", str(tmp_path / "p/warm"), "--no-sv-download",
               "--no-asr", "--eval-out", str(tmp_path / "res/warm")) == 0
    for s in range(2):
        fresh_machine()
        assert run("eval", "--round", "1", "--shard", str(s), "--shards", "2", "--restore", str(tmp_path / "p/warm"),
                   "--no-sv-download", "--no-asr", "--eval-out", str(tmp_path / "res/r1")) == 0
    got = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in (tmp_path / "res/r1").glob("*.json")
           if not p.name.startswith("shard_")}
    assert set(got) == {"s1-g4", "s2-g4", "s3-g4", "base-g4"} and all(c["status"] == "ok" for c in got.values())
    assert all(c["result"]["n_items"] == prep["counts"]["val"] for c in got.values())
    assert got["base-g4"]["base"] and got["base-g4"]["files"]["sovits"] == "s2Gv5dev.pth"
    assert run("plan", "--results", str(tmp_path / "res/r1"), "--plan-out", str(tmp_path / "res/plan2.json")) == 0
    plan = json.loads((tmp_path / "res/plan2.json").read_text(encoding="utf-8"))
    assert plan["specs"] and all("@" in s or s.startswith("s6-") for s in plan["specs"])
    m = plan["matrix"][0]
    fresh_machine()
    assert run("eval", "--round", "2", "--shard", "0", "--cands", m["cands"], "--restore", str(tmp_path / "p/warm"),
               "--no-sv-download", "--no-asr", "--eval-out", str(tmp_path / "res/r2")) == 0
    # 第二轮的 cfg_rate 真的发给了引擎（第一轮是 auto：不发，引擎用官方默认）
    calls = [json.loads(x)["req"] for x in (gsv / "_fake_api_calls.jsonl").read_text(encoding="utf-8").splitlines()]
    want = {exp.Cand(c).cfg for c in m["cands"].split(",") if not exp.Cand(c).base} - {None}
    assert want <= {c.get("cfg_rate") for c in calls}
    shutil.copytree(work / "exp", tmp_path / "res" / "exp", dirs_exist_ok=True)
    assert run("report", "--results", str(tmp_path / "res")) == 0
    data = json.loads((tmp_path / "out" / "结果.json").read_text(encoding="utf-8"))
    ranked = [r for r in data["rows"] if r["名次"]]
    assert len(ranked) >= 5 and ranked[0]["名次"] == 1 and data["done"]["epochs"] == {"lora": 3, "gpt": 4}
    md = (tmp_path / "out" / "结果.md").read_text(encoding="utf-8")
    assert "| 1 |" in md and "train-1" in md and "按 id 对上" in md
