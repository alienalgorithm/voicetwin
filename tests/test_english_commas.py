"""英文后面的全角「，」「、」换成半角「,」（训练和生成同一个规则）。

11-24 用官方 V5 整合包真的文字处理实测（research/V5极致/文字处理实测/结果/结果.md 第 5 节）：
英文后面的「，」25 处全丢（停顿没了）、「、」3 处全丢，「when who、what」里的 who、what 训练和生成都丢了；同样位置的半角「,」159 处全留下。"""

from voicetwin.backends import gsv_english
from voicetwin.backends.gsv_mixed_text import english_commas


def test_full_width_commas_after_english_become_half_width():
    assert english_commas("类似于when who、what这样的词") == "类似于when who, what这样的词"
    assert english_commas("一个是kills，一个是is。") == "一个是kills,一个是is。"
    assert english_commas("smoking kills ，是一个陈述句") == "smoking kills,是一个陈述句"
    assert english_commas("that smoking kills，") == "that smoking kills,"


def test_chinese_commas_are_untouched():
    for t in ("中文，中文、中文。", "主语、宾语和表语", "", "3，4、5"):
        assert english_commas(t) == t


def test_engine_text_uses_the_same_rule_and_reports_it():
    text, fixes = gsv_english.rewrite("那就是kills，然后再翻译is。")
    assert text == "那就是kills,然后再翻译is。"
    assert [(f.kind, f.before, f.after) for f in fixes] == [("comma", "，", ",")]
    assert gsv_english.engine_text("中文，中文。") == "中文，中文。"


def test_training_script_applies_the_rule():
    from pathlib import Path

    src = (Path(gsv_english.__file__).parent / "gsv_scripts" / "get_text_mixed.py").read_text(encoding="utf-8")
    assert "mixed.english_commas(text)" in src
