"""英文读音预检（voicetwin/backends/gsv_english.py）：每条规则一个会读错的写法（必须抓到 / 必须改）和一个
不该动的写法（必须放过），整合包词典的读法，GPT-SoVITS 真正发出去的文字，生成前 / 训练前给老师看的话，
老师自带母本（core_corpus.tsv）上实测出来的结果。规则和实测见 research/V5极致/英文读音预检.md。"""

import json
import pickle
import sys
from pathlib import Path

import pytest

from conftest import make_cfg
from voicetwin import workflows as wf
from voicetwin.backends import gsv_english as G
from voicetwin.backends.base import SynthRequest
from voicetwin.project import Project

CORPUS = Path(__file__).resolve().parent.parent / "voicetwin" / "data" / "lexicon" / "core_corpus.tsv"


def _dict(tmp_path, words=(), fast=(), hot=(), names=None):
    """仿整合包的 GPT_SoVITS/text：cmudict.rep 前 56 行是说明（english.py 从第 57 行读起）。"""
    d = tmp_path / "GPT_SoVITS" / "text"
    d.mkdir(parents=True, exist_ok=True)
    head = [f";;; header line {i}" for i in range(1, 57)]
    head[10] = "HEADERWORD  HH EH1 D"  # 第 57 行以前的不算词典里的词
    (d / "cmudict.rep").write_text("\n".join(head + [f"{w.upper()}  X" for w in words]) + "\n", encoding="utf-8")
    (d / "cmudict-fast.rep").write_text("".join(f"{w} X\n" for w in fast), encoding="utf-8")
    (d / "engdict-hot.rep").write_text("".join(f"{w} X\n" for w in hot), encoding="utf-8")
    if names is not None:
        (d / "namedict_cache.pickle").write_bytes(names if isinstance(names, bytes) else pickle.dumps(names))
    return d


def _kinds(text, dictionary=None):
    return [(f.kind, f.word) for f in G.find(text, dictionary)]


# ============================================================================ 只改发给引擎的文字
@pytest.mark.parametrize("before, after", [
    # 英文里的 ; : → 逗号（引擎会把它们删掉）；句末的冒号 → 句号；句末的分号引擎自己会补句号，不动
    ("However, the data show otherwise; we must act.", "However, the data show otherwise, we must act."),
    ("Note: this is important.", "Note, this is important."),
    ("he said \"no\"; then left", "he said \"no\", then left"),
    ("翻译成 the answer：答案", "翻译成 the answer,答案"),
    ("the sentence is as follows:", "the sentence is as follows."),
    # 带点的缩写
    ("You can call him Dr.Banner or Hulk.", "You can call him Doctor Banner or Hulk."),
    ("叫他Dr. Banner", "叫他Doctor Banner"),
    ("Mr. Smith and Mrs. Lee met Ms. Wang.", "Mr Smith and Mrs Lee met Ms Wang."),
    ("He is No. 1 in the class.", "He is number 1 in the class."),
    ("the U.S. government", "the U S government"),
    ("He moved to the U.S.", "He moved to the U S."),
    ("at 9 a.m. every day", "at 9 A M every day"),
    ("Tom vs. Jerry", "Tom versus Jerry"),
    # 年代（引擎把数字后面的 s 当成「秒」）
    ("in the 1990s", "in the nineteen nineties"),
    ("the 1900s and the 2000s", "the nineteen hundreds and the two thousands"),
    ("the 2010s", "the twenty tens"),
    ("back in the ’90s", "back in the nineties"),
    # 词中间的弯引号（会被删掉：I’m → Im）
    ("I’m sure it’s fine.", "I'm sure it's fine."),
])
def test_engine_text_fixes(before, after):
    assert G.engine_text(before) == after
    assert G.engine_text(after) == after  # 改第二遍不再变


@pytest.mark.parametrize("text", [
    "10:30 is the time.",               # 时间：引擎自己会读
    "比例是 3:2。",                     # 两边都是数字
    "例如：我们来看一个句子。",         # 冒号前面是汉字：中文的读法会把它当成逗号
    "注意：apple 是苹果。",
    "The data show otherwise;",         # 句末的分号：引擎自己补句号
    "e.g. this one, i.e. that one",     # 引擎自己会换成 for example / that is
    "Drive carefully.",                 # 不是 Dr
    "No, I won't.",                     # No 后面不是数字
    "Wait 10s please.",                 # 10s 可能就是 10 秒：只提醒，不改
    "In 1990 we met.",
    "‘hello’ he said",                  # 引号不在词中间
    "今天我们学习定语从句；然后做练习。",
    "AND的第二种连接方式",              # 全大写：只提醒（缩写本来就按字母读）
    "",
])
def test_engine_text_leaves_alone(text):
    assert G.engine_text(text) == text


def test_rewrite_reports_each_fix():
    new, fixes = G.rewrite("the 1990s; and Dr. Smith said: no")
    assert new == "the nineteen nineties, and Doctor Smith said, no"
    assert [(f.kind, f.before, f.after) for f in fixes] == [
        ("punct", ";", ","), ("punct", ":", ","), ("abbr", "Dr.", "Doctor"), ("decade", "1990s", "nineteen nineties")]
    assert G.rewrite("纯中文的句子。") == ("纯中文的句子。", [])


# ============================================================================ 找可能读错的词
def test_find_caps_split_and_letters_without_dictionary():
    # 没有词典：全大写的词都报（分不出是单词还是缩写），大小写混着的都报会拆开
    assert _kinds("AND的第二种连接方式") == [("caps", "AND")]
    assert _kinds("VR和AI") == [("caps", "VR"), ("caps", "AI")]
    assert _kinds("McDonald's") == [("split", "McDonald's")]
    assert _kinds("and的用法、as这个关系代词") == []
    assert _kinds("A good student works hard.") == [("letter_a", "A")]
    assert _kinds("a good student, I am fine.") == []
    assert _kinds("选 A 还是 B") == []                          # 选项字母：后面不是英文单词
    assert _kinds(G.engine_text("the U.S.A. government")) == []  # U S A：不是冠词


def test_find_with_dictionary(tmp_path):
    d = G.load_dictionary(_dict(tmp_path, words=["and", "jim", "who", "donald", "banner", "doctor", "the", "wait", "st"],
                                fast=["paul"], names={"meng": [["M", "EH1", "NG"]]}))
    assert _kinds("AND的第二种连接方式", d) == [("caps", "AND")]
    assert "写成小写 and" in G.find("AND", d)[0].detail and "A-N-D" in G.find("AND", d)[0].detail
    assert _kinds("VR和AI", d) == [("acronym", "VR"), ("acronym", "AI")]   # 词典里没有：缩写，按字母读是对的
    assert _kinds("JimWho", d) == []                                     # Jim、Who 都在词典里：拆开读也对
    f = G.find("McDonald's", d)
    assert [(x.kind, x.word) for x in f] == [("split", "McDonald's")] and "Mc 会按字母读" in f[0].detail
    assert _kinds("LvMeng", d) == [("split", "LvMeng")]                  # Lv 不在词典里（Meng 是人名词典里的）
    assert G.find("LvMeng", d)[0].detail.endswith("分开读，Lv 会按字母读")
    assert _kinds("Meng", d) == [] and _kinds("meng", d) == [("guess", "meng")]  # 人名词典只查首字母大写的词
    assert _kinds("postgraduate", d) == [("guess", "postgraduate")]
    assert _kinds("Lv", d) == [("short", "Lv")]
    assert _kinds("Banner's", d) == []                                   # 所有格：查 banner
    assert _kinds("Doctor Banner", d) == []
    assert _kinds("St. Paul", d) == [("street", "St.")]
    assert _kinds("wait 10s", d) == [("num_s", "10s")]


def test_find_heteronyms_as_v5_reads_them():
    assert _kinds("This will lead to a crisis.") == [("fixed_reading", "lead")]
    assert "/led/（铅）" in G.find("lead")[0].detail and "/liːd/（带领）" in G.find("lead")[0].detail
    assert _kinds("live本身属于是不及物动词") == [("fixed_reading", "live")]
    assert _kinds("the wind, a tear") == [("fixed_reading", "wind"), ("fixed_reading", "tear")]
    assert _kinds("I object to this plan.") == [("swapped", "object")]
    assert _kinds("Please read it.") == [("read", "read")]
    assert _kinds("She lives in a small city.") == []  # lives：词典的第一个读音就是动词，对的


def test_precheck_generation_lines():
    texts = ["AND的第二种连接方式。", "You can call him Dr.Banner or Hulk.", "没有英文。", "This will lead to a crisis; "
             "we must act.", "以上是AND只连接了两个句子，AND前后。"]
    rep = G.precheck(texts)
    risk, fixed = rep.risk_line(), rep.fix_line()
    assert risk.startswith("英文读音预检（生成前查的）：这些英文可能读错：")
    assert "AND（第 1、5 句，共 3 处：全是大写，会一个字母一个字母地读成 A-N-D" in risk
    assert "lead（第 4 句：引擎总是读成 /led/（铅），读不出 /liːd/（带领））" in risk
    assert "没找到整合包的英文词典" in risk                        # 如实说：这次没查词典里没有的词
    assert "Dr." not in risk                                          # 自动改好的不再算可能读错
    assert fixed.startswith("英文读音预检：已经自动改好（只改发给引擎的文字，字幕和你看到的文字都不变）：")
    assert "Dr. → Doctor（第 2 句）" in fixed and "分号、冒号改成逗号" in fixed and "（第 4 句）" in fixed
    data = rep.to_json()
    assert data["texts"] == 5 and data["auto_fix"] is True and data["dictionary"] is None
    assert {(r["kind"], r["word"], r["count"]) for r in data["items"]} == {("caps", "AND", 3),
                                                                          ("fixed_reading", "lead", 1)}
    # 没有要说的：两句都是空的
    clean = G.precheck(["今天我们学习定语从句。", "We can add a condition at the end."])
    assert clean.risk_line() == "" and clean.fix_line() == ""


def test_precheck_lists_at_most_six_words():
    rep = G.precheck(["lead live wind wound tear bow bass sow"])
    risk = rep.risk_line()
    assert risk.count("引擎总是读成") == G.MAX_ITEMS and "还有 2 个（都写在结果报告里）" in risk


def test_precheck_training_lists_what_is_not_changed():
    rep = G.precheck(["AND的第二种连接方式。", "You can call him Dr.Banner or Hulk.", "otherwise; we act",
                      "in the 1990s"], auto_fix=False)
    assert rep.fixes == [] and rep.fix_line() == ""
    line = rep.training_line()
    assert line.startswith("英文读音预检：训练素材里这些英文会按下面的读法去学，可能和你说的不一样：")
    assert "AND（1 处：" in line and "Dr.（1 处：会读成 drive）" in line
    assert "英文里的 ; :（1 处：会被删掉，该停的地方不停）" in line and "1990s（1 处：数字后面的 s 会读成 seconds（秒））" in line
    assert "训练用的文字照原样不改" in line
    assert line.count("1990s") == 1  # 不重复报成 num_s
    assert G.precheck(["今天我们学习定语从句。"], auto_fix=False).training_line() == ""


# ============================================================================ 整合包的英文词典
def test_load_dictionary_follows_english_py(tmp_path):
    d = G.load_dictionary(_dict(tmp_path, words=["lead", "ai", "os"], fast=["mc"], hot=["CHATGPT"],
                                names={"aadi": [["AA1", "D", "IY0"]]}))
    assert {"lead", "mc", "chatgpt"} <= d.words
    assert "ai" not in d.words and "os" not in d.words      # english.py 删掉的「读音错误的几个缩写」
    assert "headerword" not in d.words                      # 第 57 行以前的不是词
    assert d.names == frozenset({"aadi"})
    assert G.load_dictionary(tmp_path / "没有这个文件夹") is None and G.load_dictionary(None) is None


def test_name_dictionary_pickle_never_runs_code(tmp_path):
    """人名词典是 pickle：里面引用了任何类或函数（可能执行代码）就整个不读，只当没有人名词典。"""
    evil = pickle.dumps({"x": Path("/")})  # 要找 pathlib 里的类才能读出来
    d = G.load_dictionary(_dict(tmp_path, words=["lead"], names=evil))
    assert d is not None and d.names == frozenset() and "lead" in d.words


def test_dictionary_cache_follows_file_changes(tmp_path):
    import os
    import time

    text = _dict(tmp_path, words=["lead"])
    assert "zebra" not in G.load_dictionary(text).words
    (text / "engdict-hot.rep").write_text("zebra Z IY1 B R AH0\n", encoding="utf-8")
    st = os.stat(text / "engdict-hot.rep")
    os.utime(text / "engdict-hot.rep", ns=(st.st_atime_ns, st.st_mtime_ns + 2_000_000_000))
    time.sleep(0.01)
    assert "zebra" in G.load_dictionary(text).words


# ============================================================================ 老师自带的母本（实测）
def _corpus():
    rows = [ln.rstrip("\n").split("\t", 1) for ln in CORPUS.read_text(encoding="utf-8").splitlines()
            if ln and not ln.startswith("#")]
    return [r for r in rows if len(r) == 2]


def test_teacher_corpus_measured_findings():
    """research/V5极致/英文读音预检.md 第 2 节记下的数（这里不用整合包的词典；带词典的结果写在记录里）。"""
    rows = _corpus()
    texts = [t for _i, t in rows]
    rep = G.precheck(texts)
    got = {(it.kind, it.word): it.count for it in rep.items}
    assert got == {("caps", "AND"): 10, ("caps", "VR"): 5, ("caps", "AI"): 2, ("caps", "PPT"): 2,
                   ("split", "McDonald's"): 2, ("split", "JimWho"): 1, ("fixed_reading", "live"): 2}
    # 英文后面的「，」「、」→「,」28 处（11-24 整合包实测这些全被引擎删掉），Dr. → Doctor 1 处
    assert [(it.kind, it.word, it.count) for it in rep.fixes] == [("comma", "", 28), ("abbr", "Dr.", 1)]
    banner = next(t for t in texts if "Dr.Banner" in t)
    assert "Doctor Banner" in G.engine_text(banner) and "Dr" not in G.engine_text(banner)
    # 发给引擎的文字变了的句子：Dr.Banner 那一句 + 英文后面有「，」「、」的 26 句（一共 27 句；别的句子一个字都不改，旧缓存照常能用）
    assert sum(G.engine_text(t) != t for t in texts) == 27
    train = G.precheck(texts, auto_fix=False).training_line()
    assert "AND（10 处：" in train and "Dr.（1 处：会读成 drive）" in train


# ============================================================================ GPT-SoVITS 真正发出去的文字
def _gsv_backend(tmp_path, monkeypatch, with_dict=False):
    from fake_gptsovits import build_fake_root
    from voicetwin.backends.gptsovits import GPTSoVITSBackend

    root = build_fake_root(tmp_path / "GPT-SoVITS")
    if with_dict:
        _dict(root, words=["and", "this", "will", "to", "a", "crisis", "you", "can", "call", "him", "doctor", "banner",
                           "or", "hulk"])
    monkeypatch.setattr(GPTSoVITSBackend, "ensure_users_pth", lambda self: None)
    cfg = make_cfg(tmp_path / "ws", backend="gptsovits",
                   backends={"gptsovits": {"root": str(root), "python": sys.executable, "version": "v5dev"}})
    project = Project(cfg, "预检")
    project.ensure()
    project.save_manifest([])  # 有 manifest 才算准备过素材的声音
    return cfg, project, GPTSoVITSBackend(cfg, project)


def _req(text):
    return SynthRequest(text=text, lang="zh", ref_audio=Path("ref.wav"), ref_text="参考。", ref_lang="zh")


def test_gptsovits_sends_engine_text(tmp_path, monkeypatch):
    cfg, project, b = _gsv_backend(tmp_path, monkeypatch)
    sent = []
    monkeypatch.setattr(b, "_alive", lambda: True)
    monkeypatch.setattr(b, "_tts_answer", lambda payload: sent.append(payload) or b"RIFF")
    b.synthesize(_req("You can call him Dr.Banner or Hulk; 你可以叫他Banner博士。"), tmp_path / "a.wav")
    assert sent[-1]["text"] == "You can call him Doctor Banner or Hulk, 你可以叫他Banner博士。"
    b.synthesize(_req("今天我们学习定语从句。"), tmp_path / "b.wav")
    assert sent[-1]["text"] == "今天我们学习定语从句。"                 # 没有要改的：一个字都不变
    # 同时要好几个版本：算能不能同时生成、一个一个生成，用的都是改好的文字
    seen = []
    monkeypatch.setattr(b, "_single_reason", lambda req, n: seen.append(req.text) or "测试")
    monkeypatch.setattr(b, "_many_singles", lambda req, n, out: seen.append(req.text) or [])
    b.synthesize_many(_req("in the 1990s; Mr. Smith"), 3, tmp_path / "many")
    assert seen == ["in the nineteen nineties, Mr Smith"] * 2
    assert b.engine_text("Dr.Banner") == "Doctor Banner"


def test_gptsovits_english_dictionary_and_training_plan(tmp_path, monkeypatch):
    cfg, project, b = _gsv_backend(tmp_path, monkeypatch)
    assert b.english_dictionary() is None                     # 整合包里没有 cmudict.rep
    _dict(b.root, words=["and"])
    assert "and" in b.english_dictionary().words
    # 训练预览（「② 训练模型」页）：训练素材里的英文按什么读音去学
    from voicetwin.data import exporters

    recs = [{"text": "AND的第二种连接方式。", "duration": 2.0, "lang": "zh"},
            {"text": "You can call him Dr.Banner.", "duration": 2.0, "lang": "en"}]
    monkeypatch.setattr(exporters, "train_records", lambda project, include_val=False: list(recs))
    monkeypatch.setattr(type(b), "_gpu_memory", lambda self, quick=False: (11.99, 11.2, "test"))
    lines = b.training_plan(quick=True, mode="identical")["audit_lines"]
    assert any(x.startswith("英文读音预检：训练素材里") and "AND（1 处" in x and "Dr.（1 处" in x for x in lines)
    # 标准方式：中文句子里夹着的英文不参加训练，只看全是英文的句子
    std = b.training_plan(quick=True, mode="standard")["audit_lines"]
    line = next(x for x in std if x.startswith("英文读音预检"))
    assert "AND（" not in line and "Dr.（1 处" in line


def test_precheck_note_before_generating(tmp_path, monkeypatch):
    cfg, project, b = _gsv_backend(tmp_path, monkeypatch)
    risk, fixed = wf.english_precheck_note(cfg, project.voice, "AND的用法。\n\nThis will lead to a crisis.\n\n"
                                           "You can call him Dr.Banner.", "gptsovits")
    assert "AND（第 1 句" in risk and "lead（第 2 句" in risk
    assert "Dr. → Doctor（第 3 句）" in fixed
    # 不是 GPT-SoVITS（测试引擎）、讲稿是空的：什么都不说
    assert wf.english_precheck_note(cfg, project.voice, "AND的用法。", "dummy") == ("", "")
    assert wf.english_precheck_note(cfg, project.voice, "", "gptsovits") == ("", "")
    # 读音词典已经把 AND 改成 and：不再报
    project.lexicon_path.write_text("AND => and\n", encoding="utf-8")
    assert wf.english_precheck_note(cfg, project.voice, "AND的用法。", "gptsovits") == ("", "")


# ============================================================================ 生成引擎：提醒、小结、报告、缓存键
class _PrecheckDummy:
    """测试引擎 + GPT-SoVITS 的英文读音预检（生成引擎只看有没有 engine_text / english_precheck）。"""

    @staticmethod
    def wrap(backend):
        backend.engine_text = G.engine_text
        backend.english_precheck = lambda texts, training=False: G.precheck(texts, None, auto_fix=not training)
        return backend


def test_narrator_shows_precheck_and_keeps_other_cache_keys(prepared, tmp_path):
    from voicetwin.backends.base import get_backend
    from voicetwin.synth import engine as eng
    from voicetwin.synth.script import ScriptSegment

    cfg, project, _ = prepared
    plain = eng.Narrator(cfg, project, get_backend("dummy", cfg, project), quality="fast")
    pre = eng.Narrator(cfg, project, _PrecheckDummy.wrap(get_backend("dummy", cfg, project)), quality="fast")

    def seg(text):
        return ScriptSegment(text=text, display=text, lang="zh", index=0)

    ok, fix = "今天我们学习as这个关系代词。", "You can call him Dr.Banner or Hulk."
    assert pre._plan(seg(ok)).key == plain._plan(seg(ok)).key       # 没有要改的句子：缓存键一个字节都不变
    assert pre._plan(seg(fix)).key != plain._plan(seg(fix)).key     # 发给引擎的文字改了：旧缓存不能再用
    ident = eng.Narrator(cfg, project, _PrecheckDummy.wrap(get_backend("dummy", cfg, project)), quality="identical",
                         tier="mid")
    ident0 = eng.Narrator(cfg, project, get_backend("dummy", cfg, project), quality="identical", tier="mid")
    assert ident._plan(seg(ok)).key == ident0._plan(seg(ok)).key
    p1, p0 = ident._plan(seg(fix)), ident0._plan(seg(fix))
    assert p1.key != p0.key and p1.pool_key != p0.pool_key

    out = tmp_path / "out" / "预检.wav"
    res = pre.narrate("AND的第二种连接方式。\n\nYou can call him Dr.Banner or Hulk.", out, subtitles=False)
    risk = next(w for w in res.warnings if w.startswith("英文读音预检（生成前查的）"))
    assert "AND（第 1 句" in risk
    assert any("Dr. → Doctor（第 2 句）" in n for n in res.notes)
    report = json.loads(Path(res.report_path).read_text(encoding="utf-8"))
    assert [(r["kind"], r["word"]) for r in report["english_precheck"]["items"]] == [("caps", "AND")]
    seg2 = report["segments"][1]
    assert seg2["text"] == "You can call him Dr.Banner or Hulk." and seg2["tts_text"] == seg2["text"]
    assert seg2["engine_text"] == "You can call him Doctor Banner or Hulk."
    assert "engine_text" not in report["segments"][0]
    # 没有英文读音预检的引擎：报告里是 null，不多写提醒
    res0 = plain.narrate("AND的第二种连接方式。", tmp_path / "out" / "plain.wav", subtitles=False)
    assert not any(w.startswith("英文读音预检") for w in res0.warnings)
    assert json.loads(Path(res0.report_path).read_text(encoding="utf-8"))["english_precheck"] is None


# ============================================================================ 网页：点「生成」时马上在结果区说
def test_web_generate_shows_precheck_while_running(prepared, tmp_path, monkeypatch):
    import shutil

    from voicetwin.webui import app as A

    _, project, _ = prepared
    shutil.copytree(project.root, tmp_path / "ws" / project.voice)
    cfg = make_cfg(tmp_path / "ws")
    monkeypatch.setattr(wf, "english_precheck_note",
                        lambda cfg, voice, source, backend=None: ("英文读音预检（生成前查的）：这些英文可能读错：AND（第 1 句）",
                                                                  "英文读音预检：已经自动改好：Dr. → Doctor（第 2 句）"))

    def fake_stream(kind, label, voice, fn, *args, stages=None, hint="", note="", **kwargs):
        yield "日志", {"done": False, "bar": "进度"}
        yield "日志", {"done": False, "bar": "进度"}
        yield "日志", {"done": True, "bar": "", "error": RuntimeError("测试"), "friendly": None}

    monkeypatch.setattr(A, "stream_task", fake_stream)
    ui = A.WebUI(cfg)
    outs = [dict(zip(ui.GEN_OUT, o)) for o in ui.do_generate(project.voice, "AND的用法。", None, "gptsovits", "fast", 0,
                                                             "", "", "", "wav")]
    md = outs[0]["gen_md"]
    assert md.startswith("> ⚠️ 英文读音预检（生成前查的）") and "AND（第 1 句）" in md
    assert "<small>英文读音预检：已经自动改好：Dr. → Doctor（第 2 句）</small>" in md
    assert outs[1]["gen_md"].get("__type__") == "update"           # 只发一次，后面「不变」
