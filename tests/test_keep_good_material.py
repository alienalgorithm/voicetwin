"""V5 统一计划 A2：别把好素材白白丢掉（data/prepare.py 的过滤）。

- 识别引擎没把握（置信度低 / 可能不是语音）、「点赞 / 转发」这类词表，说的都是「识别出来的字可能不对」：
  老师确认过的文字（改过保存了、点过「这句没错」）和母本一致的文字，不再因为这些丢掉；
- 每种去掉的原因按句子种类（中英混合 / 纯中文 / 纯英文）、按来源文件分开数，列出哪些文件自动降噪过，
  写进 prepare_summary.json 和 prepare.log。
"""

import shutil

from conftest import make_cfg

from voicetwin import workflows as wf
from voicetwin.data import review
from voicetwin.data import transcript_fix as tf
from voicetwin.data.prepare import apply_filters, summarize
from voicetwin.project import Project

#: 程序自带母本（老师逐句修缮过的讲课文字）里的一句：voicetwin/data/lexicon/core_corpus.tsv 第 3 行
MOTHER_LINE = "首先，as这个关系代词，它最经常出现在非限定性定语从句中，经常会作为非限定定语从句的引导词而出现。"
LOW_CONF = {"engine": "faster-whisper", "avg_logprob": -1.4, "no_speech_prob": 0.1}
NO_SPEECH = {"engine": "faster-whisper", "avg_logprob": -0.3, "no_speech_prob": 0.8}
PCFG = {"filter": {"speaker_outlier": False}}


def _rec(rid, text, asr=None, source="s1", **extra):
    r = {"id": rid, "text": text, "lang": "zh", "duration": 5.0, "rate": 4.0, "clip_ratio": 0.0, "source": source,
         "source_file": f"/录音/{source}.mp4", "asr": dict(asr or LOW_CONF), "keep": True, "drop_reason": ""}
    r.update(extra)
    return r


def _filter(project, recs):
    apply_filters(project, recs, PCFG, {})
    return {r["id"]: r for r in recs}


def test_mother_line_is_in_builtin_corpus():
    """测试用的那一句真的在程序自带的母本里（母本换了，下面的测试要跟着换一句）。"""
    assert MOTHER_LINE in [x for _rid, x in tf.builtin_mother()]


def test_low_confidence_but_confirmed_clip_is_kept(tmp_path):
    """识别置信度低 / 可能不是语音，但老师点过「这句没错」（文字没再改过）：以前照样丢掉，现在留下。
    没确认过的照旧丢；确认以后文字又改回识别的字（和点「这句没错」时不一样）也照旧丢。"""
    project = Project(make_cfg(tmp_path / "ws"), "确认")
    text = "这个句子识别的时候引擎没什么把握，但老师听过了没错。"
    recs = [
        _rec("ok_low", text, suspect_ok=text),
        _rec("ok_nospeech", text, NO_SPEECH, suspect_ok=text),
        _rec("plain_low", text),
        _rec("plain_nospeech", text, NO_SPEECH),
        _rec("ok_other_text", text, suspect_ok=text + "（改过）"),
    ]
    got = _filter(project, recs)
    assert got["ok_low"]["keep"] and got["ok_low"]["drop_reason"] == ""
    assert got["ok_nospeech"]["keep"] and got["ok_nospeech"]["drop_reason"] == ""
    assert got["plain_low"]["drop_reason"] == "识别置信度低" and not got["plain_low"]["keep"]
    assert got["plain_nospeech"]["drop_reason"] == "可能不是语音" and not got["plain_nospeech"]["keep"]
    assert got["ok_other_text"]["drop_reason"] == "识别置信度低"


def test_low_confidence_clip_matching_mother_is_kept(tmp_path):
    """识别出来的字和母本一模一样（标点、空格不算）：字是对的，置信度低也留下。
    只对上母本一句里很短的一段（常说的话，哪里都有）不算；一句里连着的一长段算。"""
    project = Project(make_cfg(tmp_path / "ws"), "母本")
    no_punct = "首先 as这个关系代词它最经常出现在非限定性定语从句中经常会作为非限定定语从句的引导词而出现"
    recs = [
        _rec("whole", no_punct),
        _rec("whole_nospeech", MOTHER_LINE, NO_SPEECH),
        _rec("part_long", "它最经常出现在非限定性定语从句中"),  # 15 个字：母本一句里连着的一段
        _rec("part_short", "首先，as这个关系代词"),  # 7 个字 / 词：太短，证明不了
        _rec("not_mother", "它最经常出现在限定性定语从句中"),  # 和母本差一个字
    ]
    got = _filter(project, recs)
    assert got["whole"]["keep"] and got["whole"]["drop_reason"] == ""
    assert got["whole_nospeech"]["keep"] and got["whole_nospeech"]["drop_reason"] == ""
    assert got["part_long"]["keep"]
    assert got["part_short"]["drop_reason"] == "识别置信度低"
    assert got["not_mother"]["drop_reason"] == "识别置信度低"


def test_uploaded_mother_cannot_prove_the_row_itself(tmp_path):
    """老师上传的母本也算；但和校对表某一句一模一样的那一行就是这一句自己（比如下载的「改好的文字」），
    一句话不能拿来证明它自己没错（和「📝 文字校正」的规矩一样）：不算。标点不一样的（真正的讲稿）算。"""
    project = Project(make_cfg(tmp_path / "ws"), "上传")
    project.ensure()
    same = "这一句是新课里的话，母本文件里是从校对表下载来的同一句。"
    real = "这一句是新课里的话母本里是老师自己写的讲稿"
    folder = tf.transcript_dir(project)
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "新课.txt").write_text(f"{same}\n这一句是新课里的话，母本里是老师自己写的讲稿。\n", encoding="utf-8")
    got = _filter(project, [_rec("self", same), _rec("real", real)])
    assert got["self"]["drop_reason"] == "识别置信度低"
    assert got["real"]["keep"] and got["real"]["drop_reason"] == ""


def test_confirmed_like_and_share_text_is_kept(tmp_path):
    """「记得点赞转发」是老师真的会说的话：以前「点赞 / 转发」词表排在人工豁免前面，老师改对了也丢（疑似识别错误）。
    现在改过保存了、点过「这句没错」、和母本一致的都留下；识别出来、没人确认过的照旧丢。"""
    project = Project(make_cfg(tmp_path / "ws"), "点赞")
    good = {"engine": "faster-whisper", "avg_logprob": -0.2, "no_speech_prob": 0.05}
    like = "记得点赞转发，我们下节课再见。"
    recs = [
        _rec("edited", like, good, text_edited=True, orig_text="记得点赞转发我们下节课在见"),
        _rec("dismissed", like, good, suspect_ok=like),
        _rec("asr_only", like, good),
        _rec("reverted", like, good, text_edited=True, orig_text=like),  # 改回识别的字：不算改过
    ]
    got = _filter(project, recs)
    assert got["edited"]["keep"] and got["edited"]["drop_reason"] == ""
    assert got["dismissed"]["keep"] and got["dismissed"]["drop_reason"] == ""
    assert got["asr_only"]["drop_reason"] == "疑似识别错误"
    assert got["reverted"]["drop_reason"] == "疑似识别错误"


def test_drop_counts_by_kind_and_file_in_summary(tmp_path):
    """每种去掉的原因按「中英混合 / 纯中文 / 纯英文」和来源文件分开数，列出自动降噪过的文件，都写成中文。"""
    project = Project(make_cfg(tmp_path / "ws"), "计数")
    project.ensure()
    project.write_json(project.sources_path, {
        "s1": {"file": "/录音/旧课.mp4", "done": True, "denoised": True, "snr_before": 18.0},
        "s2": {"file": "/录音/新录的26分钟.wav", "done": True, "snr_before": 40.0},
    })
    recs = [
        _rec("a", "纯中文的句子用来训练。", keep=True),
        _rec("b", "纯中文的句子没有用上。", keep=False, drop_reason="识别置信度低"),
        _rec("c", "这一句夹着 English 单词。", keep=False, drop_reason="识别置信度低", source="s2",
             source_file="/录音/新录的26分钟.wav"),
        _rec("d", "这一句也夹着 relative clause。", keep=False, drop_reason="太短", source="s2",
             source_file="/录音/新录的26分钟.wav"),
        _rec("e", "This one is English only.", lang="en", keep=True, source="s2", source_file="/录音/新录的26分钟.wav"),
    ]
    recs[0]["source_file"] = recs[1]["source_file"] = "/录音/旧课.mp4"
    s = summarize(project, recs, [])
    assert s["dropped_by_kind"]["纯中文"] == {"total": 2, "kept": 1, "dropped": {"识别置信度低": 1}}
    assert s["dropped_by_kind"]["中英混合"] == {"total": 2, "kept": 0, "dropped": {"识别置信度低": 1, "太短": 1}}
    assert s["dropped_by_kind"]["纯英文"] == {"total": 1, "kept": 1, "dropped": {}}
    assert s["dropped_by_file"]["旧课.mp4"] == {"total": 2, "kept": 1, "dropped": {"识别置信度低": 1}, "denoised": True}
    assert s["dropped_by_file"]["新录的26分钟.wav"]["dropped"] == {"识别置信度低": 1, "太短": 1}
    assert s["dropped_by_file"]["新录的26分钟.wav"]["denoised"] is False
    assert s["denoised_files"] == ["旧课.mp4"]
    report = s["drop_report"]
    assert "中英混合的句子：一共 2 段，用来训练 0 段，去掉 2 段（太短 1 段、识别置信度低 1 段）" in report
    assert "纯英文的句子：一共 1 段，用来训练 1 段，一段都没去掉" in report
    assert "文件「旧课.mp4」：一共 2 段，用来训练 1 段，去掉 1 段（识别置信度低 1 段）；这个文件自动降噪过" in report
    assert "自动降噪过的文件（1 个）：「旧课.mp4」" in report


def test_prepare_writes_drop_counts_to_summary_and_log(prepared):
    """真的准备一次素材：分组计数在返回的小结、prepare_summary.json 和 prepare.log 里都有。"""
    _cfg, project, summary = prepared
    saved = project.read_json(project.root / "prepare_summary.json", {})
    for s in (summary, saved):
        assert set(s["dropped_by_file"]) == {"第1课.wav"}
        assert s["dropped_by_file"]["第1课.wav"]["total"] == s["clips_total"]
        assert sum(g["total"] for g in s["dropped_by_kind"].values()) == s["clips_total"]
        assert s["drop_report"] and isinstance(s["denoised_files"], list)
    log_text = (project.logs_dir / "prepare.log").read_text(encoding="utf-8")
    assert "每种去掉的原因（按句子种类、按文件分开数）" in log_text and "文件「第1课.wav」：一共" in log_text


def test_teacher_corrected_like_and_share_is_trained_after_save(prepared, tmp_path):
    """校对表里把一句改成「记得点赞转发……」、点「保存修改」（真的保存流程）：以前保存以后马上被当成识别错误丢掉。"""
    cfg, project, _ = prepared
    ws = tmp_path / "ws"
    shutil.copytree(project.root, ws / project.voice)
    cfg2 = make_cfg(ws)
    p2 = wf.open_project(cfg2, project.voice, must_exist=True)
    rec = next(r for r in p2.load_manifest() if r["text"] == "首先我们看一个最简单的例子。" and r["keep"])
    review.set_draft(p2, rec["id"], text="记得点赞转发，我们看一个最简单的例子。")
    wf.review_save(cfg2, project.voice)
    got = next(r for r in p2.load_manifest() if r["id"] == rec["id"])
    assert got["text"] == "记得点赞转发，我们看一个最简单的例子。"
    assert got["keep"] and got["drop_reason"] == ""
