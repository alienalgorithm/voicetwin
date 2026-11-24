"""GPT-SoVITS V5（v5dev / v5turbo）：默认用 v5dev、只练它（老师 2026-11-24：这次只要 V5）。
想同时练 v2ProPlus 一起实测比较时，config.yaml 写 train.compare_versions: v2ProPlus。

V5 的音色模型是 LoRA 微调（官方 s2_train_v3_lora.py）：不用 s2D，模型文件名多一个 _l{rank}，开头 2 字节 07 / 08。"""

import sys
from pathlib import Path

import pytest

from voicetwin import workflows as wf
from voicetwin.backends import gptsovits as g
from voicetwin.backends.base import get_backend
from voicetwin.config import load_config

from conftest import confirm_material, make_cfg
from fake_gptsovits import build_fake_root
from test_gptsovits_fake import Rec, needs_fake_python, no_users_pth  # noqa: F401  (pytest 夹具)


def _v5cfg(project, root, port, version="v5dev", **train):
    return make_cfg(project.root.parent, backend="gptsovits", backends={"gptsovits": {
        "root": str(root), "python": sys.executable, "port": port, "startup_timeout": 60, "is_half": True,
        "version": version, "train": train,
    }})


def _own_project(project, tmp_path):
    """这个测试自己的一份声音（复制出来）：练出来的 V5 模型不留在别的测试共用的那份声音里。"""
    import shutil

    ws = tmp_path / "ws"
    shutil.copytree(project.root, ws / project.voice)
    return wf.open_project(make_cfg(ws), project.voice, must_exist=True)


def test_shipped_default_is_v5dev_only():
    cfg = load_config(user_config=False)
    assert cfg["backends"]["gptsovits"]["version"] == "v5dev" == g.DEFAULT_VERSION == g.LATEST_VERSION
    assert cfg["backends"]["gptsovits"]["train"]["compare_versions"] == "none" and g.COMPARE_VERSIONS == ()
    assert g.is_lora_version("v5dev") and g.is_lora_version("v5turbo") and not g.is_lora_version("v2ProPlus")
    assert g.PRETRAINED_SOVITS["v5dev"].endswith("gsv-v5-pretrained/s2Gv5dev.pth")
    assert g.VOCODER["v5dev"] == g.VOCODER["v5turbo"] == "GPT_SoVITS/pretrained_models/gsv-v5-pretrained/vocoder.pth"
    assert g.PRETRAINED_GPT["v5dev"].endswith("s1v3.ckpt")


def test_detect_v5_files(tmp_path):
    for head, ver in ((b"07", "v5dev"), (b"08", "v5turbo")):
        p = tmp_path / f"{ver}.pth"
        p.write_bytes(head + b"\0" * 100)
        got = g.detect_sovits_version(p)
        assert got["version"] == ver and got["lora"] is True
    # 官方 process_ckpt.py 的底模 MD5（开头 8192 字节）
    assert g.SOVITS_HASH_VERSION["c4ce1f839a0d271e41963f9fb16ade76"] == "v5dev"
    assert g.SOVITS_HASH_VERSION["6bc5cb2a195cc8ec22b2e723cd4d6f47"] == "v5turbo"


def test_plan_lora_batch_epochs_and_user_settings():
    p = g.plan_lora("v5dev", 11.99, 11.2, mode="identical", n_clips=200)
    assert (p["batch_size"], p["epochs"], p["save_every"], p["rank"]) == (1, g.LORA_EPOCHS_DEEP, 1, 32)
    assert g.plan_lora("v5dev", 23.99, 23.0, mode="standard", n_clips=200)["batch_size"] == 3
    assert g.plan_lora("v5dev", 23.99, 23.0, mode="standard", n_clips=200)["epochs"] == g.LORA_EPOCHS_STANDARD
    assert g.plan_lora("v5dev", 23.99, 23.0, is_half=False, n_clips=200)["batch_size"] == 1
    assert g.plan_lora("v5dev", 47.99, 47.0, n_clips=8)["batch_size"] == 2       # 不超过素材条数的 1/4
    u = g.plan_lora("v5dev", 11.99, 11.2, user={"lora_batch_size": 2, "lora_epochs": 7, "lora_save_every": 2,
                                                 "lora_rank": 64})
    assert (u["batch_size"], u["epochs"], u["save_every"], u["rank"]) == (2, 8, 2, 64)  # 最后一轮一定存下来
    assert g.plan_lora("v5dev", 11.99, 11.2, user={"lora_epochs": 99})["epochs"] == g.LORA_MAX_EPOCHS
    assert g.plan_lora("v5dev", 11.99, 11.2, user={"lora_rank": 48})["rank"] == 32
    assert "LoRA" in p["summary"] and "最像" in p["summary"]


def test_compare_versions_setting(prepared, tmp_path):
    _, project, _ = prepared
    root = build_fake_root(tmp_path / "GSV-cmp")
    vers = lambda **t: get_backend("gptsovits", _v5cfg(project, root, 19893, **t), project).train_versions()
    assert vers() == ["v5dev"] and vers(compare_versions="auto") == ["v5dev"]
    assert vers(compare_versions="v2ProPlus") == ["v5dev", "v2ProPlus"]
    assert vers(compare_versions="none") == ["v5dev"]
    assert vers(compare_versions="v5turbo，v2ProPlus 不存在的") == ["v5dev", "v5turbo", "v2ProPlus"]
    b = get_backend("gptsovits", _v5cfg(project, root, 19893, version="v2ProPlus"), project)
    assert b.train_versions() == ["v2ProPlus"]   # 主版本不是 V4 / V5：auto 不加对比


@needs_fake_python
def test_compare_setting_trains_v5dev_and_v2proplus_then_picks_by_measurement(prepared, tmp_path, monkeypatch, no_users_pth):
    monkeypatch.setenv("FAKE_GSV_CLIP_SLEEP", "0.03")
    monkeypatch.setenv("EXPECT_VERSION", "v5dev")    # 1A 处理文字时和官方 webui 一样传 v5dev
    project = _own_project(prepared[1], tmp_path)
    root = build_fake_root(tmp_path / "GPT-SoVITS-v5")
    gcfg = _v5cfg(project, root, 19891, sovits_epochs=8, gpt_epochs=10, batch_size=2, lora_epochs=4,
                  compare_versions="v2ProPlus")
    confirm_material(gcfg, project.voice)
    rec = Rec()
    info = wf.run_train(gcfg, project.voice, "gptsovits", select=True, progress=rec, sovits_save_every=4,
                        gpt_save_every=5, mode="standard")
    assert info["versions"] == ["v5dev", "v2ProPlus"]

    backend = get_backend("gptsovits", gcfg, project)
    exp = backend.exp_name
    opt_dir = root / "logs" / exp
    # v5dev 用 LoRA 脚本：配置里没有 s2D，断点在 logs_s2_v5dev_lora_32
    assert (opt_dir / "logs_s2_v5dev_lora_32").is_dir()
    assert (opt_dir / "7-sv_cn").is_dir()             # 对比的 v2ProPlus 要声纹
    # 语义 token：和官方一样，每个版本用自己的底模提取一份；练完 6-name2semantic.tsv 是设置里那个版本（v5dev）的（GPT 用它练）
    mark = lambda f: str(sum(Path(g.PRETRAINED_SOVITS[f]).name.encode()) % 997)
    for ver in ("v5dev", "v2ProPlus"):
        rows = (opt_dir / g.SEMANTIC_COPY.format(ver)).read_text(encoding="utf-8").splitlines()[1:]
        assert rows and all(r.split()[-1] == mark(ver) for r in rows)
    assert (opt_dir / "6-name2semantic.tsv").read_bytes() == (opt_dir / g.SEMANTIC_COPY.format("v5dev")).read_bytes()
    assert (project.logs_dir / "gsv_1c_semantic_v2ProPlus.log").exists()
    models = project.load_models()["gptsovits"]
    v5 = [p for p in models["sovits"] if "SoVITS_weights_v5dev" in p.replace("\\", "/")]
    pp = [p for p in models["sovits"] if "SoVITS_weights_v2ProPlus" in p.replace("\\", "/")]
    assert v5 and pp and all(p.endswith("_l32.pth") for p in v5)
    assert [int(p.rsplit("_e", 1)[1].split("_")[0]) for p in v5] == [1, 2, 3, 4]   # lora_epochs=4，每轮存一个
    for p in v5:
        assert g.detect_sovits_version(p)["version"] == "v5dev"
    # 两个版本都进了实测；挑出来的是实测里的一个
    ids = {c["id"] for c in backend.checkpoints()}
    assert any(i.startswith("v5dev-") for i in ids) and any(i.startswith("v2ProPlus-") for i in ids)
    assert models["selected"]["id"] in ids
    tested = {r.get("version") or r["id"].split("-")[0] for r in models["selection"]["results"]}
    assert {"v5dev", "v2ProPlus"} <= tested
    assert models["params"]["lora"]["rank"] == 32 and models["params"]["versions"] == ["v5dev", "v2ProPlus"]
    # 详细日志：v5dev 的音色训练用官方 LoRA 脚本
    lora_log = (project.logs_dir / "gsv_s2_train_v5dev.log").read_text(encoding="utf-8")
    assert "s2_train_v3_lora.py" in lora_log and "INFO:" in lora_log
    msgs = [m for _, m in rec.train_part()]
    assert any("v5dev" in m and "LoRA" in m for m in msgs) and msgs[-1] == "GPT-SoVITS 训练完成"

    res = wf.run_narrate(gcfg, project.voice, "大家好，这是用 V5 生成的一句话。Hello!",
                         out=str(tmp_path / "v5.wav"), backend_name="gptsovits", quality="fast")
    assert res.audio_path.exists() and res.duration > 1.0


@needs_fake_python
def test_v5_only_default_uses_the_same_page_settings(prepared, tmp_path, monkeypatch, no_users_pth):
    """默认只练 V5：网页高级设置里的「音色训练轮数」「每批数量」照样管用（用在 V5 的 LoRA 上），不练 v2ProPlus、不要声纹。"""
    monkeypatch.setenv("FAKE_GSV_CLIP_SLEEP", "0.03")
    monkeypatch.setenv("EXPECT_VERSION", "v5dev")
    project = _own_project(prepared[1], tmp_path)
    root = build_fake_root(tmp_path / "GPT-SoVITS-v5only")
    gcfg = _v5cfg(project, root, 19895, gpt_epochs=4)
    confirm_material(gcfg, project.voice)
    rec = Rec()
    # 和网页上「② 训练模型 → 高级设置」一样：音色训练轮数 3、每批数量 2
    info = wf.run_train(gcfg, project.voice, "gptsovits", select=True, progress=rec, mode="standard",
                        sovits_epochs=3, batch_size=2)
    assert info["versions"] == ["v5dev"]
    backend = get_backend("gptsovits", gcfg, project)
    opt_dir = root / "logs" / backend.exp_name
    models = project.load_models()["gptsovits"]
    assert all("SoVITS_weights_v5dev" in p.replace("\\", "/") for p in models["sovits"])
    assert [int(p.rsplit("_e", 1)[1].split("_")[0]) for p in models["sovits"]] == [1, 2, 3]
    lora = models["params"]["lora"]
    assert (lora["epochs"], lora["batch_size"]) == (3, 2)
    s = models["params"]["summary"]
    assert "v5dev（官方的 LoRA 练法）3 轮（你指定的）" in s and "音色每批 2 条（你指定的）" in s and "SoVITS" not in s
    assert not (opt_dir / "7-sv_cn").exists()            # 只练 V5：不用声纹（v2Pro 才要）
    assert not (opt_dir / g.SEMANTIC_COPY.format("v2ProPlus")).exists()
    assert {c["id"] for c in backend.checkpoints()} and models["selected"]["id"].startswith("s")
    msgs = [m for _, m in rec.train_part()]
    assert msgs[-1] == "GPT-SoVITS 训练完成"
    res = wf.run_narrate(gcfg, project.voice, "大家好，这是只用 V5 生成的一句话。", out=str(tmp_path / "v5only.wav"),
                         backend_name="gptsovits", quality="fast")
    assert res.audio_path.exists() and res.audio_path.name.endswith("_v5dev.wav")


def test_v5_does_not_use_aux_refs(prepared, tmp_path, monkeypatch):
    """官方 V5 合成时只用第一条参考录音：不发辅助参考，「一模一样」也不把「带 3 条 / 不带」当成两种试法。
    按正在用的模型文件判断（_batch_version 读模型文件本身），不看设置。"""
    from voicetwin.synth import search as S

    _, project, _ = prepared
    b = get_backend("gptsovits", _v5cfg(project, build_fake_root(tmp_path / "GSV-aux"), 19897), project)
    for ver, want in (("v5dev", False), ("v5turbo", False), ("v4", False), ("v2ProPlus", True), ("v2Pro", True)):
        monkeypatch.setattr(b, "_batch_version", lambda ver=ver: ver)
        assert b.supports_aux_refs is want, ver
    refs = [{"id": "r1"}, {"id": "r2"}]
    arms = S.plan_arms("mid", refs, {"r1": [], "r2": []}, n_presets=2, aux_options=(3, 0))  # 不支持：一条都不给
    assert len(arms) == len(set((a.ref_id, a.preset_idx, a.aux_n) for a in arms)) and all(a.aux_n == 0 for a in arms)


# ---------------------------------------------------------------------------- 8 GB 显卡：每批 1 条也不够时打开梯度检查点
def _lora_oom_setup(prepared, tmp_path, monkeypatch, name, port):
    """只练 V5、8 GB 显卡（空闲 6.8 GB，和老师的 RTX 5070 Laptop 一样）：官方公式算出来 LoRA 每批 1 条。"""
    from voicetwin.backends.gptsovits import GPTSoVITSBackend

    monkeypatch.setenv("EXPECT_VERSION", "v5dev")
    monkeypatch.setattr(GPTSoVITSBackend, "_gpu_memory", lambda self, quick=False: (7.96, 6.8, "test"))
    project = _own_project(prepared[1], tmp_path)
    root = build_fake_root(tmp_path / name)
    gcfg = _v5cfg(project, root, port, lora_epochs=2, gpt_epochs=2)
    confirm_material(gcfg, project.voice)
    return gcfg, project, root


def _s2_cfg(project):
    import json

    return json.loads((project.models_dir / "gptsovits" / "tmp_s2.json").read_text(encoding="utf-8"))


@needs_fake_python
def test_v5_lora_oom_at_batch_one_retries_with_gradient_checkpointing(prepared, tmp_path, monkeypatch, no_users_pth):
    """每批 1 条显存也不够（不开梯度检查点就练不下）：打开官方的「梯度检查点」再练一次，练完；训练报告里告诉老师。"""
    gcfg, project, root = _lora_oom_setup(prepared, tmp_path, monkeypatch, "GSV-ckpt", 19898)
    monkeypatch.setenv("FAKE_GSV_LORA_NEED_CKPT", "1")
    rec = Rec()
    info = wf.run_train(gcfg, project.voice, "gptsovits", select=False, progress=rec, mode="standard")
    lora = info["params"]["lora"]
    assert (lora["batch_size"], lora["batch_size_used"], lora["grad_ckpt"]) == (1, 1, True)
    assert _s2_cfg(project)["train"]["grad_ckpt"] is True
    assert [int(p.rsplit("_e", 1)[1].split("_")[0]) for p in info["sovits"]] == [1, 2]   # 2 轮都练完、存下
    msgs = [m for _, m in rec.train_part()]
    assert msgs.count("训练音色（v5dev，LoRA 微调）：每批 1 条显存也不够：打开官方的「梯度检查点」再练一次"
                      "（更省显存，但每轮会慢一些；已经练好的部分会接着用）") == 1
    report = [ln for ln in info["params"]["report"] if "梯度检查点" in ln]
    assert len(report) == 1 and report[0].startswith(
        "音色（v5dev）：显存不够，每批 1 条也练不下，这次打开了官方的「梯度检查点」才练成：每轮会慢一些")
    assert "实测每轮" in report[0]              # 用时是这次量出来的（这台电脑量不出显存，就不写显存）
    assert project.load_models()["gptsovits"]["params"]["lora"]["grad_ckpt"] is True
    assert msgs[-1] == "GPT-SoVITS 训练完成"


@needs_fake_python
def test_v5_lora_normal_training_keeps_gradient_checkpointing_off(prepared, tmp_path, monkeypatch, no_users_pth):
    """显存够的时候和官方默认一样不开梯度检查点，报告里也不提。"""
    gcfg, project, root = _lora_oom_setup(prepared, tmp_path, monkeypatch, "GSV-ckpt-off", 19899)
    info = wf.run_train(gcfg, project.voice, "gptsovits", select=False, mode="standard")
    assert "grad_ckpt" not in info["params"]["lora"] and _s2_cfg(project)["train"]["grad_ckpt"] is False
    assert not [ln for ln in info["params"]["report"] if "梯度检查点" in ln]


@needs_fake_python
def test_v5_lora_still_oom_with_gradient_checkpointing_gives_the_clear_error(prepared, tmp_path, monkeypatch,
                                                                           no_users_pth):
    """开了梯度检查点还是不够：只多试这一次，然后和以前一样报「显存不够」，告诉老师怎么办。"""
    from voicetwin.backends.base import TrainStepError
    from voicetwin.errors import explain

    gcfg, project, root = _lora_oom_setup(prepared, tmp_path, monkeypatch, "GSV-ckpt-oom", 19900)
    monkeypatch.setenv("FAKE_GSV_OOM_ABOVE", "0")
    rec = Rec()
    with pytest.raises(TrainStepError) as ei:
        wf.run_train(gcfg, project.voice, "gptsovits", select=False, progress=rec, mode="standard")
    assert _s2_cfg(project)["train"]["grad_ckpt"] is True       # 最后一次是开着梯度检查点试的
    assert sum("梯度检查点" in m for m in rec.msgs) == 1          # 只多试一次
    assert explain(ei.value).key == "gpu_oom"
    assert str(ei.value).startswith("「训练音色（v5dev）」这一步出错了：显卡内存（显存）不够")
