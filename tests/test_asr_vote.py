"""三个识别引擎投票（voicetwin/data/asr_vote.py）：faster-whisper（现在的文字）+ FunASR + SenseVoice。

纯文字，不用模型、不联网。最后一个测试在老师新录的 26 分钟课（204 句，两路独立核对的定稿）上量字错率：
投票以后必须比每一个引擎单独的都好（research/文字校正/三引擎投票/实测.py）。"""

import importlib.util
from pathlib import Path

import pytest

from voicetwin.data import asr_vote as av

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "research" / "文字校正" / "三引擎投票" / "实测.py"


def _vote(whisper, funasr, sensevoice, **kw):
    return av.vote(whisper, {"funasr": funasr, "sensevoice": sensevoice}, **kw)


def test_two_engines_agreeing_win_and_are_sure():
    w = "名词性冲剧是三大冲剧里面最难的。"
    edits = _vote(w, "名 词 性 从 句 是 三 大 从 句 里 面 最 难 的", "名词性从句是三大从句里面最难的。")
    assert [(w[e.start:e.end], e.rep, e.sure) for e in edits] == [("冲剧", "从句", True), ("冲剧", "从句", True)]
    assert edits[0].reason == "另外两个识别引擎都听成「从句」" and set(edits[0].engines) == {"funasr", "sensevoice"}
    assert av.apply(w, edits) == "名词性从句是三大从句里面最难的。"  # 标点照旧是现在的文字的


def test_one_engine_alone_does_not_change_the_text():
    # 两票对一票：现在的文字赢（FunASR 不一样的地方照旧由查错字标红，老师听录音决定）
    assert _vote("我们来看定语从句。", "我们来看定于从句", "我们来看定语从句。") == []
    assert _vote("我们来看定语从句。", "我们来看定语从句", "我们来看定于从句。") == []


def test_three_different_answers_decide_nothing_but_a_grammar_term_does():
    w = "什么样的名词英从句引导词"
    assert _vote(w, "什么样的名词性从句引导词", "什么样的名词型从句引导词", terms=()) == []
    edits = _vote(w, "什么样的名词性从句引导词", "什么样的名词型从句引导词")  # 程序自带的语法术语
    assert [(w[e.start:e.end], e.rep, e.sure) for e in edits] == [("英", "性", True)]
    assert "语法术语「名词性从句」" in edits[0].reason and edits[0].engines == ("funasr",)
    # 两票对一票、少数那个凑成了术语：也按术语改
    w2 = "把这个名词型从句给删掉"
    edits = _vote(w2, "把这个名词性从句给删掉", "把这个名词型从句给删掉")
    assert av.apply(w2, edits) == "把这个名词性从句给删掉"
    assert _vote(w2, "把这个名词性从句给删掉", "把这个名词型从句给删掉", terms=["名词性从句"])
    assert _vote(w2, "把这个名词性从句给删掉", "把这个名词型从句给删掉", terms=["性从"]) == []  # 太短的不算


def test_homophones_that_are_both_common_are_left_alone():
    # 他 / 它、地 / 的、在 / 再：读音一样，听录音分不出来
    assert _vote("这部分他只能去做主语", "这部分它只能去做主语", "这部分它只能去做主语。") == []
    assert _vote("名正言顺地去做主语", "名正言顺的去做主语", "名正言顺的去做主语") == []
    assert _vote("我们在来看一下", "我们再来看一下", "我们再来看一下") == []


def test_english_keeps_whisper_unless_the_other_two_spell_it_the_same():
    w = "一个是chaos,一个是ace,所以说"
    edits = _vote(w, "一个是 chaos 一个是 is 所以说", "一个是kios，一个是is。所以说")
    assert [(w[e.start:e.end], e.rep, e.sure) for e in edits] == [("ace", "is", True)]
    # FunASR、SenseVoice 拼得不一样（kilols / chaos）：保留 faster-whisper 的 kills
    assert _vote("最后再看smoking kills,这部分", "最后再看 smoking kilols 这部分", "最后再看，smoking chaos这部分") == []
    # 英文补空格：中文里夹着的英文换成另一个词
    w3 = "引导词Net就是"
    e3 = _vote(w3, "引导词 that 就是", "引导词that就是")
    assert av.apply(w3, e3) == "引导词that就是"


def test_insertions_numbers_and_long_changes():
    w = "那这个句子里面的系动词就是is。"
    edits = _vote(w, "那这个句子里面的系动词呢就是 is", "那这个句子里面的系动词呢就是is。")
    assert [(e.start, e.end, e.rep, e.sure) for e in edits] == [(11, 11, "呢", True)]
    assert edits[0].reason == "另外两个识别引擎都听到这里还有「呢」"
    assert _vote("我们来看第2个句子", "我们来看第二个句子", "我们来看第二个句子。") == []  # 数字按数值比
    long_w = "今天我们来讲一讲甲乙丙丁戊己庚辛的用法然后做练习"
    other = "今天我们来讲一讲子丑寅卯辰巳午未的用法然后做练习"
    edits = _vote(long_w, other, other)
    assert edits and not any(e.sure for e in edits)  # 一处太长：只给建议，不自动采用
    assert av.apply(long_w, edits) == long_w and av.apply(long_w, edits, sure_only=False) == other


def test_needs_two_other_engines_that_heard_the_same_sentence():
    w = "名词性冲剧是三大冲剧"
    assert av.vote(w, {"funasr": "名词性从句是三大从句"}) == []
    assert av.vote(w, {"funasr": "名词性从句是三大从句", "sensevoice": None}) == []
    assert av.vote(w, {"funasr": "名词性从句是三大从句", "sensevoice": ""}) == []
    # 另一个引擎整句差得太多（录音可能有问题）：它不参加投票
    assert av.vote(w, {"funasr": "名词性从句是三大从句", "sensevoice": "今天天气真好啊大家出去玩"}) == []
    assert av.vote("", {"funasr": "你好", "sensevoice": "你好"}) == []
    assert av.vote("？！", {"funasr": "你好", "sensevoice": "你好"}) == []


def _bench():
    spec = importlib.util.spec_from_file_location("three_engine_vote_bench", BENCH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vote_beats_every_single_engine_on_the_teachers_recording():
    mod = _bench()
    if not mod.TRUTH.exists():
        pytest.skip("没有老师新录音的定稿（research/新录音转录/核对结果/逐句定稿.json）")
    res = mod.pure_text()
    cer, bad, _errs = res["三个引擎投票（只改有把握的）"]
    singles = {k: v for k, v in res.items() if k != "三个引擎投票（只改有把握的）"}
    assert {"whisper", "funasr", "sensevoice"} <= set(singles)
    for name, (c, b, _e) in singles.items():
        assert cer < c and bad < b, (name, c, b, cer, bad)
    assert mod.chars("Smoking Kills，is！") == list("smokingkillsis")
    assert mod.score(["我们去公园", "好"], ["我们去了公园。", "好！"]) == (14.29, 1, 1)
